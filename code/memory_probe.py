"""Tab. memory per image: weights, activations, and the peak and time of one step of each kind, measured on the
GPUs a run uses (SD3-Medium at 1024^2, the transformer on --tf-device, the VAE and estimators on --vae-device).

  python -I code/memory_probe.py --out outputs/memory.json --vae-device cuda:1

Every measurement resets the per-GPU peak counters, runs once to warm up, then --repeats times; it reports the
median wall time, the largest peak allocated (torch.cuda.max_memory_allocated, not nvidia-smi's reserved pool),
and the activation memory: that peak minus what was allocated just before. "Ours" is the code as it runs;
"naive" turns off the transformer's gradient checkpointing and the VAE's per-resnet checkpointing, so its
backward keeps every activation (an out-of-memory error is recorded, not raised).

The guided step runs the real guided_step (forward with readout, decode, estimators, backward, candidate passes)
from mid-schedule on; a step whose constraints are not active yet does no backward, so only steps that took a
gradient count. The SAR step is one Stage-2 step: CFG pass, decode, supports, and the biased conditional pass.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from constraints import new_state  # noqa: E402
from estimators import ClipSegSupport, SigLIPEvidence  # noqa: E402
from grounding import phrase_spans  # noqa: E402
from guidance import Stage1Ctx, _euler, guided_step  # noqa: E402
from measure import EstimatorDeps, Measure  # noqa: E402
from predicates import translate  # noqa: E402
from run_experiment import _tiny_estimators, guide_cfg, sar_cfg  # noqa: E402
from sar import build_log_r  # noqa: E402
from sd3_backend import load_backend, tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

GIB = 2 ** 30
PROMPT = "a kitten moderately larger than a pillow"


def module_gib(obj) -> float:
    """Parameters and buffers of a module, or of every module an object holds (an estimator wraps its model)."""
    modules = [obj] if isinstance(obj, torch.nn.Module) else [v for v in vars(obj).values() if isinstance(v, torch.nn.Module)]
    seen, total = set(), 0
    for m in modules:
        for t in list(m.parameters()) + list(m.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                total += t.numel() * t.element_size()
    return total / GIB


def _gpus():
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def _sync():
    for i in _gpus():
        torch.cuda.synchronize(i)


def _allocated():
    return {f"cuda:{i}": torch.cuda.memory_allocated(i) / GIB for i in _gpus()}


def _peaks():
    return {f"cuda:{i}": torch.cuda.max_memory_allocated(i) / GIB for i in _gpus()}


def _reset():
    for i in _gpus():
        torch.cuda.reset_peak_memory_stats(i)


def _oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def timed(fn, repeats=3, warmup=1):
    """Median time, peak per GPU, and activation per GPU (peak above what was allocated before) of fn()."""
    times, peak, extra = [], {}, {}
    try:
        for _ in range(warmup):
            fn()
        for _ in range(repeats):
            _sync()
            before = _allocated()
            _reset()
            start = time.perf_counter()
            fn()
            _sync()
            times.append(time.perf_counter() - start)
            for d, v in _peaks().items():
                peak[d] = max(peak.get(d, 0.0), v)
                extra[d] = max(extra.get(d, 0.0), v - before.get(d, 0.0))
    except RuntimeError as e:
        if not _oom(e):
            raise
        out = {"oom": True, "peak_gb": _peaks(), "error": str(e).splitlines()[0][:200]}
        torch.cuda.empty_cache()
        return out
    return {"time_s": statistics.median(times), "peak_gb": peak, "activation_gb": extra}


@contextlib.contextmanager
def no_checkpointing(backend):
    """The naive memory path: every activation is kept for the backward pass."""
    tf = backend.transformer
    tf.disable_gradient_checkpointing()
    original = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = lambda fn, *args, **kwargs: fn(*args)    # the VAE's per-resnet wrapper
    try:
        yield
    finally:
        torch.utils.checkpoint.checkpoint = original
        tf.enable_gradient_checkpointing()
        tf.training = True


class _NullLog:
    def write(self, **record):
        pass


def _grad_through(backend, z, sigma, cond, decode):
    """Forward (both CFG branches, with readout) and backward to the latent; with decode, through the VAE too."""
    leaf = z.detach().requires_grad_(True)
    with torch.enable_grad():
        out = backend.velocity(leaf, sigma, cond, readout=True)
        x0 = backend.clean(leaf, sigma, out.v)
        loss = (backend.decode(x0) if decode else x0).float().pow(2).mean()
        loss.backward()
    return leaf.grad


def _decode_backward(backend, z):
    leaf = z.detach().requires_grad_(True)
    with torch.enable_grad():
        backend.decode(leaf).float().mean().backward()


def _sar_step(backend, seg, sar, cond, z, sigma, sigma_next, heads):
    """One Stage-2 step with the bias on every object phrase (the most passes a step can take, rollback aside)."""
    with torch.no_grad():
        first = backend.velocity(z, sigma, cond)
        x_hat = backend.decode(backend.clean(z, sigma, first.v))
        supports = seg(x_hat, heads)
        priors = {q: supports[q.split("#")[0]] for q in cond.phrase_tokens if q.split("#")[0] in supports}
        log_r = build_log_r(priors, cond.phrase_tokens, cond.embeds.shape[1], sar.alpha, sar.beta, set(priors),
                            n_img=backend.n_img)
        v_cond = backend.velocity_cond(z, sigma, cond, log_r)
        v = first.v_uncond + backend.cfg * (v_cond - first.v_uncond)
        return z + (sigma_next - sigma).to(z.device, z.dtype) * v


def probe(backend, seg, siglip, s, prompt=PROMPT, repeats=3, guided_steps=5, text_gib=None):
    calib = ProvisionalCalibration(s)
    cfg, sar = guide_cfg(s), sar_cfg(s)
    program = translate(prompt)
    cond = backend.encode_prompt(prompt, phrase_spans(prompt, program))
    result = {"prompt": prompt, "repeats": repeats,
              "weights_gb": {"mmdit": module_gib(backend.transformer), "vae": module_gib(backend.vae),
                             "text_encoders": text_gib, "siglip": module_gib(siglip), "clipseg": module_gib(seg)}}
    sigmas = backend.sigmas(cfg.steps)
    mid = cfg.steps // 2
    z = backend.noise_like_latent(0)
    with torch.no_grad():                                   # the latent a mid-schedule step really sees
        for i in range(mid):
            z = _euler(z, sigmas[i], sigmas[i + 1], backend.velocity(z, sigmas[i], cond).v)
    sigma, sigma_next = sigmas[mid], sigmas[mid + 1]

    def no_grad(fn):
        def run():
            with torch.no_grad():
                return fn()
        return run

    result["unguided_step"] = timed(no_grad(lambda: _euler(z, sigma, sigma_next, backend.velocity(z, sigma, cond).v)), repeats)
    result["readout_forward"] = timed(no_grad(lambda: backend.velocity(z, sigma, cond, readout=True)), repeats)
    result["vae_decode"] = timed(no_grad(lambda: backend.decode(backend.clean(z, sigma, z))), repeats)
    result["mmdit_backward"] = timed(lambda: _grad_through(backend, z, sigma, cond, decode=False), repeats)
    result["vae_backward"] = timed(lambda: _decode_backward(backend, z), repeats)
    with no_checkpointing(backend):
        result["naive_mmdit_backward"] = timed(lambda: _grad_through(backend, z, sigma, cond, decode=False), 1)
        result["naive_vae_backward"] = timed(lambda: _decode_backward(backend, z), 1)
        result["naive_guided_step"] = timed(lambda: _grad_through(backend, z, sigma, cond, decode=True), 1)

    measure = Measure(EstimatorDeps(seg, siglip, calib, s), prompt, cond)
    ctx = Stage1Ctx(backend, program, cond, calib, measure, cfg, new_state(ramp=cfg.ramp, **cfg.state_kw), _NullLog(),
                    None, prompt, 0)
    steps, zg = [], z
    for i in range(mid, min(mid + guided_steps, cfg.steps)):
        holder = {}

        def run(i=i, zg=zg):
            holder["out"] = guided_step(zg, sigmas[i], sigmas[i + 1], i, ctx)

        m = timed(run, repeats=1, warmup=0)
        if m.get("oom"):
            steps.append({"step": i, **m})
            break
        out = holder["out"]
        took_gradient = "no_active" not in out.rejections
        steps.append({"step": i, "took_gradient": took_gradient, "candidates": out.n_tried,
                      "accepted": out.accepted, **m})
        zg = out.latent_next
    result["guided_steps"] = steps
    took = [st for st in steps if st.get("took_gradient")]
    result["guided_step"] = _summary(took) if took else {"note": "no step took a gradient", "steps": len(steps)}

    heads = sorted(set(program.objects))
    result["sar_step"] = timed(lambda: _sar_step(backend, seg, sar, cond, z, sigma, sigma_next, heads), repeats)
    return result


def _summary(steps):
    keys = sorted({d for st in steps for d in st["peak_gb"]})
    return {"time_s": statistics.median(st["time_s"] for st in steps),
            "peak_gb": {d: max(st["peak_gb"].get(d, 0.0) for st in steps) for d in keys},
            "activation_gb": {d: max(st["activation_gb"].get(d, 0.0) for st in steps) for d in keys},
            "n_steps": len(steps)}


def _fmt(m):
    if m is None:
        return "--"
    if m.get("oom"):
        return "OOM (peak " + ", ".join(f"{d} {v:.2f} GB" for d, v in m["peak_gb"].items()) + ")"
    if "time_s" not in m:
        return m.get("note", "--")
    peak = ", ".join(f"{d} {v:.2f}" for d, v in m["peak_gb"].items()) or "cpu"
    act = ", ".join(f"{d} +{v:.2f}" for d, v in m["activation_gb"].items()) or "cpu"
    return f"{m['time_s']:.2f} s | peak GB {peak} | activation GB {act}"


def report(result) -> str:
    w = result["weights_gb"]
    lines = [f"weights GB: MM-DiT {w['mmdit']:.2f}, VAE {w['vae']:.2f}, text encoders "
             f"{'--' if w['text_encoders'] is None else format(w['text_encoders'], '.2f')} (offloaded after encoding), "
             f"SigLIP {w['siglip']:.2f}, CLIPSeg {w['clipseg']:.2f}"]
    for key, label in (("unguided_step", "unguided CFG step"), ("readout_forward", "forward with image->text readout"),
                       ("vae_decode", "VAE decode (no grad)"), ("mmdit_backward", "MM-DiT fwd+bwd, ours"),
                       ("naive_mmdit_backward", "MM-DiT fwd+bwd, naive"), ("vae_backward", "VAE decode+bwd, ours"),
                       ("naive_vae_backward", "VAE decode+bwd, naive"), ("guided_step", "guided step, ours"),
                       ("naive_guided_step", "guided step (fwd+decode+bwd), naive"), ("sar_step", "Stage 2 SAR step")):
        lines.append(f"{label}: {_fmt(result.get(key))}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--guided-steps", type=int, default=5)
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()
    s = load_settings()
    if args.tiny:
        backend, (seg, siglip), text_gib = tiny_backend(), _tiny_estimators(), None
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), args.resolution, torch.float16)
        pipe = backend._pipe
        text_gib = sum(module_gib(e) for e in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3))
        device = args.vae_device or args.tf_device
        seg = ClipSegSupport.from_pretrained(s.get("cfgCLIPSeg", str), device)
        siglip = SigLIPEvidence.from_pretrained(s.get("cfgSigLIP", str), device)
    backend.encode_prompt(args.prompt, phrase_spans(args.prompt, translate(args.prompt)))
    if not args.tiny:
        backend.unload_text_encoders()
    result = probe(backend, seg, siglip, s, args.prompt, args.repeats, args.guided_steps, text_gib)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(report(result))


if __name__ == "__main__":
    main()
