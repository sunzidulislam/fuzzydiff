"""Verify the FuzzyDiff processor on the actual SD3 checkpoint (paper: Tab. processor_check,
column "Checkpoint", and Tab. memory, "Measured peak" rows).

Checks
  1. identity_out / identity_grad: stock JointAttnProcessor2_0 vs. FuzzyJointAttnProcessor
     (readout and SAR off) on the full transformer at the target resolution and dtype,
     output and gradient w.r.t. the latent (gradient checkpointing on).
  2. readout_val / share_val / mass_val / readout_grad / fd_grad / sar_r1_out / sar_equiv / sar_uncond: custom
     processor vs. a fully materialized softmax at a reduced resolution in float32
     (materializing attention at 1024^2 is exactly what the method avoids).
  3. token_index: the pipeline's packed text sequence is [CLIP-L||CLIP-G (77), T5], i.e. rows
     77..77+L of prompt_embeds equal the T5 encoder output, and the T5 indices of a test word.
  4. image_identity: full pipeline images with stock vs. custom processors (modulation off),
     same seed; max. absolute difference in 8-bit pixel units.
  5. memory / runtime: peak GPU memory and wall time of an unguided CFG step, a guided-step
     proxy (batched forward with readout + VAE decode + backward to the latent), and a SAR step
     (SAR-free CFG forward + decode + conditional-only forward with the bias; CLIPSeg excluded).

  6. backbone_config: checkpoint-level settings (blocks, heads, VAE scaling/shift, scheduler shift,
     revision, library versions, GPU) read from the loaded pipeline; written to backbone_config.tex,
     which config.tex inputs last, so the paper reports the loaded values instead of typed ones.

Writes checkpoint_check.json, memory_check.tex, backbone_config.tex, and regenerates
processor_check_rows.tex.

Usage (GPU, actual checkpoint):
    python -I verify_checkpoint.py --model stabilityai/stable-diffusion-3-medium-diffusers
Smoke test of all code paths without weights (CPU, tiny random transformer):
    python -I verify_checkpoint.py --tiny
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzydiff_sd3_attention import FuzzyJointAttnProcessor  # noqa: E402
from sd3_backend import checkpoint_vae_resnets  # noqa: E402
from table_rows import write_rows  # noqa: E402
from test_processor import MaterializedReference, rel, set_procs, make_model  # noqa: E402

from diffusers.models.attention_processor import JointAttnProcessor2_0  # noqa: E402

CLIP_TOKENS = 77


def fwd(tr, x, ctx, pooled, t):
    return tr(hidden_states=x, encoder_hidden_states=ctx, pooled_projections=pooled,
              timestep=t, return_dict=False)[0]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def peak_gb(dev):
    return torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else float("nan")


def reset_peak(dev):
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def backbone_config(args, pipe, tr):
    """Checkpoint-level settings read from the loaded pipeline (not typed by hand).

    Returns a dict and writes code/backbone_config.tex, which config.tex inputs last, so
    these values replace the corresponding proposed defaults in the paper."""
    import diffusers
    rev = "unknown"
    try:
        from huggingface_hub import model_info
        rev = model_info(args.model).sha
    except Exception:
        pass
    sch = pipe.scheduler.config
    cfg = {
        "checkpoint": args.model, "revision": rev,
        "diffusers": diffusers.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "blocks": tr.config.num_layers, "heads": tr.config.num_attention_heads,
        "head_dim": tr.config.attention_head_dim,
        "hidden": tr.config.num_attention_heads * tr.config.attention_head_dim,
        "joint_attention_dim": tr.config.joint_attention_dim,
        "qk_norm": getattr(tr.transformer_blocks[0].attn, "norm_q", None) is not None,
        "image_tokens": (args.res // pipe.vae_scale_factor // tr.config.patch_size) ** 2,
        "clip_tokens": CLIP_TOKENS, "t5_tokens": 256,
        "vae_scale": pipe.vae.config.scaling_factor, "vae_shift": pipe.vae.config.shift_factor,
        "scheduler": type(pipe.scheduler).__name__, "shift": sch.get("shift"),
        "resolution": args.res, "steps": args.steps,
    }
    esc = lambda v: str(v).replace("_", "\\_")
    rows = {
        "cfgCheckpoint": f"{esc(args.model)}, revision \\texttt{{{esc(rev)[:12]}}}",
        "cfgDiffusers": esc(cfg["diffusers"]), "cfgGPU": esc(cfg["gpu"]),
        "cfgBlocks": cfg["blocks"], "cfgHeads": cfg["heads"], "cfgHeadDim": cfg["head_dim"],
        "cfgHidden": cfg["hidden"], "cfgImageTokens": cfg["image_tokens"],
        "cfgResolution": f"${args.res}\\times{args.res}$",
        "cfgShift": cfg["shift"], "cfgVAEScale": cfg["vae_scale"], "cfgVAEShift": cfg["vae_shift"],
    }
    out = os.path.join(args.out, "backbone_config.tex")
    with open(out, "w") as f:
        f.write("% Generated by verify_checkpoint.py from the loaded checkpoint; do not edit.\n")
        for k, v in rows.items():
            f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
    return cfg


def copy_table_inputs(out_dir):
    """The table also reads the CPU check results, which live next to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("processor_check.json", "arch_check.json"):
        target = os.path.join(out_dir, name)
        if not os.path.exists(target):
            shutil.copy(os.path.join(here, name), target)


def encoders_to(pipe, dev):
    """In its own function: a loop variable in main() would keep T5 alive after del_text_encoders."""
    for encoder in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        encoder.to(dev)


def del_text_encoders(pipe):
    """T5 and the transformer do not fit on one T4 together; every later check reuses the embeddings."""
    pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tiny", action="store_true", help="smoke test on a tiny random transformer (CPU)")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--ref-res", type=int, default=256, help="resolution for materialized-reference checks")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--prompt", default="a slightly red mug next to a blue teapot")
    ap.add_argument("--word", default="mug")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--vae-device", default=None, help="measure memory with the VAE here, as the pipeline runs it")
    ap.add_argument("--vae-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    res = {}
    torch.manual_seed(0)
    if args.tiny:
        dev, dt = torch.device("cpu"), torch.float64
        pipe = None
        tr = make_model()
        in_ch, joint_dim, pooled_dim, vae_f = 4, 32, 16, 1
        lat = 16
        ctx = torch.randn(2, CLIP_TOKENS // 7 + 8, joint_dim, dtype=dt)      # 11 "CLIP" + 8 "T5" tokens
        pooled = torch.randn(2, pooled_dim, dtype=dt)
        clip_tokens = CLIP_TOKENS // 7
        word_idx = [2, 3]
        res["mode"] = "tiny (random weights, CPU, float64)"
    else:
        from diffusers import StableDiffusion3Pipeline
        dev, dt = torch.device("cuda"), getattr(torch, args.dtype)
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=dt)
        encoders_to(pipe, dev)
        tr = pipe.transformer
        in_ch = tr.config.in_channels
        vae_f = pipe.vae_scale_factor
        lat = args.res // vae_f
        with torch.no_grad():
            pe, npe, pp, npp = pipe.encode_prompt(prompt=args.prompt, prompt_2=None, prompt_3=None,
                                                  device=dev, do_classifier_free_guidance=True,
                                                  negative_prompt="", max_sequence_length=256)
        ctx = torch.cat([pe, npe])
        pooled = torch.cat([pp, npp])
        clip_tokens = CLIP_TOKENS
        # ---- 3. token indexing ----
        with torch.no_grad():
            t5 = pipe._get_t5_prompt_embeds(prompt=args.prompt, num_images_per_prompt=1,
                                            max_sequence_length=256, device=dev)
        packed_ok = torch.equal(pe[0, CLIP_TOKENS:], t5[0].to(pe.dtype))
        enc = pipe.tokenizer_3(args.prompt, return_offsets_mapping=True, max_length=256,
                               truncation=True, padding="max_length")
        s0 = args.prompt.index(args.word); s1 = s0 + len(args.word)
        word_idx = [j for j, (a, b) in enumerate(enc["offset_mapping"]) if a < s1 and b > s0 and b > a]
        n_img = (lat // tr.config.patch_size) ** 2
        res["token_index"] = bool(packed_ok and len(word_idx) > 0)
        res["token_index_detail"] = {"t5_indices": word_idx,
                                     "joint_key_indices": [n_img + CLIP_TOKENS + j for j in word_idx],
                                     "packed_rows_equal_t5": bool(packed_ok)}
        res["qk_norm"] = getattr(tr.transformer_blocks[0].attn, "norm_q", None) is not None
        res["mode"] = f"{args.model}, {args.res}^2, {args.dtype}"
        res["backbone_config"] = backbone_config(args, pipe, tr)
        del_text_encoders(pipe)
        tr.to(dev)
        pipe.vae.to(dev)
    tr.eval()
    for p_ in tr.parameters():
        p_.requires_grad_(False)
    res.setdefault("qk_norm", getattr(tr.transformer_blocks[0].attn, "norm_q", None) is not None)
    t = torch.full((2,), 500.0, dtype=dt, device=dev)
    text_valid = list(range(0, ctx.shape[1]))
    readout = {args.word: [clip_tokens + j for j in word_idx]}

    # ---- 1. identity at full resolution / dtype ----
    try:
        tr.enable_gradient_checkpointing()
        tr.training = True      # diffusers 0.31.0 checkpoints only when the top-level module's flag is set
    except Exception:
        pass
    x = torch.randn(2, in_ch, lat, lat, dtype=dt, device=dev, requires_grad=True)
    w = torch.randn_like(x)
    set_procs(tr, lambda i: JointAttnProcessor2_0())
    y0 = fwd(tr, x, ctx, pooled, t)
    g0, = torch.autograd.grad((y0.float() * w.float()).sum(), x)
    y_again = fwd(tr, x, ctx, pooled, t)
    g_again, = torch.autograd.grad((y_again.float() * w.float()).sum(), x)
    # The same computation repeated: how much a gradient differs here with nothing changed.
    res["identity_grad_noise_floor"] = rel(g_again.float(), g0.float())
    del y_again, g_again
    set_procs(tr, lambda i: FuzzyJointAttnProcessor())
    y1 = fwd(tr, x, ctx, pooled, t)
    g1, = torch.autograd.grad((y1.float() * w.float()).sum(), x)
    res["identity_out"] = rel(y1.float(), y0.float())
    res["identity_grad"] = rel(g1.float(), g0.float())
    del y0, y1, g0, g1

    # ---- 2. reference checks at reduced resolution in float32 (float64 for tiny) ----
    rdt = torch.float64 if args.tiny else torch.float32
    tr_r = tr.to(rdt)
    rl = lat if args.tiny else args.ref_res // vae_f
    xr = torch.randn(2, in_ch, rl, rl, dtype=rdt, device=dev, requires_grad=True)
    ctx_r, pooled_r, t_r = ctx.to(rdt), pooled.to(rdt), t.to(rdt)
    n_img_r = (rl // tr.config.patch_size) ** 2
    n_txt = ctx.shape[1]
    set_procs(tr_r, lambda i: JointAttnProcessor2_0())
    with torch.no_grad():
        yr0 = fwd(tr_r, xr, ctx_r, pooled_r, t_r)
    ours = set_procs(tr_r, lambda i: FuzzyJointAttnProcessor(readout=readout, valid_text=text_valid, chunk=1024))
    fwd(tr_r, xr, ctx_r, pooled_r, t_r)
    mo = {f"{n}:{p}": v for n, pr in ours.items() for p, v in pr.maps.items()}
    so = {f"{n}:{p}": v for n, pr in ours.items() for p, v in pr.shares.items()}
    go, = torch.autograd.grad(sum((v ** 2).sum() for v in mo.values()), xr)
    refs = set_procs(tr_r, lambda i: MaterializedReference(readout=readout, valid_text=text_valid))
    fwd(tr_r, xr, ctx_r, pooled_r, t_r)
    mr = {f"{n}:{p}": v for n, pr in refs.items() for p, v in pr.maps.items()}
    sr = {f"{n}:{p}": v for n, pr in refs.items() for p, v in pr.shares.items()}
    grf, = torch.autograd.grad(sum((v ** 2).sum() for v in mr.values()), xr)
    res["readout_val"] = max((rel(mo[k].detach(), mr[k].detach()) for k in mr), key=lambda z: z[0])
    res["share_val"] = max((rel(so[k].detach(), sr[k].detach()) for k in sr), key=lambda z: z[0])
    res["readout_grad"] = rel(go, grf)
    res["mass_val"] = max((rel(ours[n].text_mass.detach(), refs[n].text_mass.detach()) for n in ours), key=lambda z: z[0])
    # finite-difference directional derivative of a scalar through the full transformer + readout
    procs_fd = set_procs(tr_r, lambda i: FuzzyJointAttnProcessor(readout=readout, valid_text=text_valid, chunk=1024))
    wr = torch.randn_like(xr)

    def scalar(xx):
        y = fwd(tr_r, xx, ctx_r, pooled_r, t_r)
        return (y * wr).sum() + sum((pr.maps[args.word] ** 2).sum() + pr.text_mass.sum() for pr in procs_fd.values())

    dvec = torch.randn_like(xr)
    gfd, = torch.autograd.grad(scalar(xr), xr)
    eps = 1e-6 if rdt == torch.float64 else 1e-3
    with torch.no_grad():
        fdv = (scalar(xr + eps * dvec) - scalar(xr - eps * dvec)) / (2 * eps)
    adv = (gfd * dvec).sum()
    res["fd_grad"] = (abs(fdv - adv).item(), abs(fdv - adv).item() / max(abs(adv).item(), 1e-30))
    with torch.no_grad():
        set_procs(tr_r, lambda i: FuzzyJointAttnProcessor(sar_log_r=torch.zeros(n_img_r, n_txt, dtype=rdt, device=dev)))
        res["sar_r1_out"] = rel(fwd(tr_r, xr, ctx_r, pooled_r, t_r), yr0)
        log_r = torch.log(torch.rand(n_img_r, n_txt, dtype=rdt, device=dev) * 1.8 + 0.2)
        set_procs(tr_r, lambda i: FuzzyJointAttnProcessor(sar_log_r=log_r, sar_batch=[0]))
        ys = fwd(tr_r, xr, ctx_r, pooled_r, t_r)
        set_procs(tr_r, lambda i: MaterializedReference(log_r=log_r, sar_batch=[0]))
        res["sar_equiv"] = rel(ys, fwd(tr_r, xr, ctx_r, pooled_r, t_r))
        res["sar_uncond"] = rel(ys[1], yr0[1])
    tr = tr_r.to(dt)

    # ---- 4. image-level identity (pipeline) ----
    if pipe is not None:
        imgs = []
        for factory in (lambda i: JointAttnProcessor2_0(), lambda i: FuzzyJointAttnProcessor()):
            set_procs(tr, factory)
            g = torch.Generator(device=dev).manual_seed(0)
            im = pipe(prompt_embeds=pe, negative_prompt_embeds=npe, pooled_prompt_embeds=pp,
                      negative_pooled_prompt_embeds=npp, num_inference_steps=args.steps, guidance_scale=7.0,
                      height=args.res, width=args.res, generator=g, output_type="np").images[0]
            imgs.append(torch.from_numpy(im))
        res["image_identity"] = f"{(imgs[0] - imgs[1]).abs().max().item() * 255:.1f}"
    else:
        res["image_identity"] = "skipped (tiny)"

    # ---- 5. memory / runtime ----
    def timed(fn):
        reset_peak(dev); sync(dev); t0 = time.perf_counter(); fn(); sync(dev)
        return time.perf_counter() - t0, peak_gb(dev)

    def decode(z0):
        c = pipe.vae.config
        param = next(pipe.vae.parameters())
        return pipe.vae.decode(z0.to(param.device, param.dtype) / c.scaling_factor + c.shift_factor).sample

    vae_dev = None
    if pipe is not None and args.vae_device:
        vae_dev = torch.device(args.vae_device)
        pipe.vae.to(vae_dev, getattr(torch, args.vae_dtype))
        torch.cuda.empty_cache()
        torch.zeros(1, device=vae_dev)
        reset_peak(vae_dev)

    xm = torch.randn(1, in_ch, lat, lat, dtype=dt, device=dev)
    set_procs(tr, lambda i: JointAttnProcessor2_0())
    with torch.no_grad():
        res["time_unguided_s"], res["mem_unguided_gb"] = timed(lambda: fwd(tr, xm.repeat(2, 1, 1, 1), ctx, pooled, t))

    procs = set_procs(tr, lambda i: FuzzyJointAttnProcessor(readout=readout, valid_text=text_valid))

    def guided():
        z = xm.clone().requires_grad_(True)
        v = fwd(tr, z.repeat(2, 1, 1, 1), ctx, pooled, t)
        v = v[1:] + 7.0 * (v[:1] - v[1:])
        z0 = z - 0.8 * v
        img = decode(z0) if pipe is not None else z0
        loss = img.float().mean().to(z.device) + sum(m.float().mean() for pr in procs.values() for m in pr.maps.values())
        torch.autograd.grad(loss, z)

    if pipe is not None:
        checkpoint_vae_resnets(pipe.vae)
    res["time_guided_s"], res["mem_guided_gb"] = timed(guided)
    if vae_dev is not None:
        res["mem_guided_vae_gpu_gb"] = peak_gb(vae_dev)

    # SAR step as sequenced in the paper (App. B, "Order of operations"):
    #   (1) SAR-free CFG forward at z_k  -> clean prediction for the drift check,
    #   (2) VAE decode (CLIPSeg is not loaded here and is timed separately),
    #   (3) conditional-branch-only forward with the SAR bias; the unconditional output of (1) is reused.
    n_img = (lat // tr.config.patch_size) ** 2
    sar_procs = [FuzzyJointAttnProcessor(), FuzzyJointAttnProcessor(
        sar_log_r=torch.zeros(n_img, ctx.shape[1], dtype=dt, device=dev), sar_batch=[0])]

    def sar_step():
        set_procs(tr, lambda i: sar_procs[0])
        v = fwd(tr, xm.repeat(2, 1, 1, 1), ctx, pooled, t)
        z0 = xm - 0.8 * (v[1:] + 7.0 * (v[:1] - v[1:]))
        if pipe is not None:
            decode(z0)
        set_procs(tr, lambda i: sar_procs[1])
        v_c = fwd(tr, xm, ctx[:1], pooled[:1], t[:1])
        return v[1:] + 7.0 * (v_c - v[1:])

    with torch.no_grad():
        res["time_sar_s"], res["mem_sar_gb"] = timed(sar_step)

    with open(os.path.join(args.out, "checkpoint_check.json" if not args.tiny else "checkpoint_check_tiny.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    if not args.tiny:
        with open(os.path.join(args.out, "memory_check.tex"), "w") as f:
            f.write("% Generated by verify_checkpoint.py; do not edit.\n")
            f.write(f"\\renewcommand{{\\memGuided}}{{{res['mem_guided_gb']:.1f}\\,GB / {res['time_guided_s']:.2f}\\,s}}\n")
            f.write(f"\\renewcommand{{\\memUnguided}}{{{res['mem_unguided_gb']:.1f}\\,GB / {res['time_unguided_s']:.2f}\\,s}}\n")
            f.write(f"\\renewcommand{{\\memSAR}}{{{res['mem_sar_gb']:.1f}\\,GB / {res['time_sar_s']:.2f}\\,s}}\n")
        copy_table_inputs(args.out)
        write_rows(args.out)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
