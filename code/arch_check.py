"""Processor checks at the real SD3-Medium layer dimensions (paper: Tab. processor_check,
column "SD3-M dims").

The unit tests in test_processor.py use a reduced transformer (4 heads of dim. 8). This script
repeats the same checks on a randomly initialized SD3Transformer2DModel with the *layer
dimensions of SD3-Medium* (24 heads x 64 = 1536 hidden, joint_attention_dim 4096,
caption_projection_dim 1536, pooled_projection_dim 2048, 16 latent channels, patch 2,
no QK-norm) and the real text length (77 CLIP + 256 T5 = 333 tokens), but with only
--layers blocks (default 2; the last one is context_pre_only as in the checkpoint) and
256^2 resolution (256 image tokens; 589 joint tokens), in float64 on CPU. It checks the code paths at the
real head count, head width, token counts, and the context_pre_only final block; it does
NOT replace the checkpoint run (trained weights, bf16, 1024^2), which verify_checkpoint.py
performs.

Usage:  python -I arch_check.py [--layers 2] [--res 256] [--out DIR]
Writes arch_check.json and regenerates processor_check_rows.tex.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzydiff_sd3_attention import FuzzyJointAttnProcessor  # noqa: E402
from table_rows import write_rows  # noqa: E402
from test_processor import MaterializedReference, rel, set_procs  # noqa: E402

import diffusers  # noqa: E402
from diffusers import SD3Transformer2DModel  # noqa: E402
from diffusers.models.attention_processor import JointAttnProcessor2_0  # noqa: E402

DT = torch.float64
SD3M = dict(patch_size=2, in_channels=16, out_channels=16, attention_head_dim=64,
            num_attention_heads=24, joint_attention_dim=4096, caption_projection_dim=1536,
            pooled_projection_dim=2048, pos_embed_max_size=192)
CLIP_TOKENS, T5_TOKENS = 77, 256


def run(model, x, ctx, pooled, t):
    return model(hidden_states=x, encoder_hidden_states=ctx, pooled_projections=pooled,
                 timestep=t, return_dict=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    torch.manual_seed(0)
    lat = args.res // 8
    model = SD3Transformer2DModel(sample_size=lat, num_layers=args.layers, **SD3M).to(DT).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_img = (lat // 2) ** 2
    n_txt = CLIP_TOKENS + T5_TOKENS
    x = torch.randn(2, 16, lat, lat, dtype=DT, requires_grad=True)
    ctx = torch.randn(2, n_txt, 4096, dtype=DT)
    pooled = torch.randn(2, 2048, dtype=DT)
    t = torch.full((2,), 500.0, dtype=DT)
    w = torch.randn(2, 16, lat, lat, dtype=DT)
    res = {"layers": args.layers, "res": args.res, "image_tokens": n_img, "text_tokens": n_txt,
           "heads": SD3M["num_attention_heads"], "head_dim": SD3M["attention_head_dim"],
           "last_block_context_pre_only": bool(model.transformer_blocks[-1].context_pre_only)}

    set_procs(model, lambda i: JointAttnProcessor2_0())
    y0 = run(model, x, ctx, pooled, t)
    g0, = torch.autograd.grad((y0 * w).sum(), x)

    set_procs(model, lambda i: FuzzyJointAttnProcessor())
    y1 = run(model, x, ctx, pooled, t)
    g1, = torch.autograd.grad((y1 * w).sum(), x)
    res["identity_out"], res["identity_grad"] = rel(y1, y0), rel(g1, g0)

    set_procs(model, lambda i: FuzzyJointAttnProcessor(sar_log_r=torch.zeros(n_img, n_txt, dtype=DT)))
    res["sar_r1_out"] = rel(run(model, x, ctx, pooled, t), y0)

    # phrase tokens inside the T5 segment (joint key N_I + 77 + j); padding after the prompt
    readout = {"mug": [CLIP_TOKENS + 3, CLIP_TOKENS + 4], "red": [CLIP_TOKENS + 2]}
    valid = list(range(0, 12)) + list(range(CLIP_TOKENS, CLIP_TOKENS + 10))
    ours = set_procs(model, lambda i: FuzzyJointAttnProcessor(readout=readout, valid_text=valid, chunk=1000))
    run(model, x, ctx, pooled, t)
    loss_o = sum((v ** 2).sum() for pr in ours.values() for v in list(pr.maps.values()) + list(pr.shares.values()))
    go, = torch.autograd.grad(loss_o, x)
    refs = set_procs(model, lambda i: MaterializedReference(readout=readout, valid_text=valid))
    run(model, x, ctx, pooled, t)
    loss_r = sum((v ** 2).sum() for pr in refs.values() for v in list(pr.maps.values()) + list(pr.shares.values()))
    gr, = torch.autograd.grad(loss_r, x)
    keys = list(ours.keys())
    res["readout_val"] = max((rel(ours[k].maps[p], refs[k].maps[p]) for k in keys for p in readout), key=lambda z: z[0])
    res["share_val"] = max((rel(ours[k].shares[p], refs[k].shares[p]) for k in keys for p in readout), key=lambda z: z[0])
    res["mass_val"] = max((rel(ours[k].text_mass, refs[k].text_mass) for k in keys), key=lambda z: z[0])
    res["readout_grad"] = rel(go, gr)

    log_r = torch.log(torch.rand(n_img, n_txt, dtype=DT) * 1.8 + 0.2)
    set_procs(model, lambda i: FuzzyJointAttnProcessor(sar_log_r=log_r, sar_batch=[0]))
    y6 = run(model, x, ctx, pooled, t)
    set_procs(model, lambda i: MaterializedReference(log_r=log_r, sar_batch=[0]))
    res["sar_equiv"] = rel(y6, run(model, x, ctx, pooled, t))
    res["sar_uncond"] = rel(y6[1], y0[1])

    procs = set_procs(model, lambda i: FuzzyJointAttnProcessor(readout={"mug": readout["mug"]}, valid_text=valid, chunk=1000))

    def scalar(xx):
        y = run(model, xx, ctx, pooled, t)
        return (y * w).sum() + sum((pr.maps["mug"] ** 2).sum() + pr.text_mass.sum() for pr in procs.values())

    d = torch.randn_like(x)
    g, = torch.autograd.grad(scalar(x), x)
    eps = 1e-6
    with torch.no_grad():
        fd = (scalar(x + eps * d) - scalar(x - eps * d)) / (2 * eps)
    ad = (g * d).sum()
    res["fd_grad"] = (abs(fd - ad).item(), abs(fd - ad).item() / max(abs(ad).item(), 1e-300))
    res["torch"], res["diffusers"] = torch.__version__, diffusers.__version__

    with open(os.path.join(args.out, "arch_check.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    write_rows(args.out)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
