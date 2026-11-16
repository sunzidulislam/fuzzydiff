"""Unit tests for FuzzyJointAttnProcessor (paper: Tab. processor_check).

Runs on a randomly initialized, reduced-size SD3Transformer2DModel in float64
on CPU and compares against diffusers' stock JointAttnProcessor2_0 and against
a fully materialized softmax reference. Writes processor_check_rows.tex (rows
for the paper table) and processor_check.json.

Usage:  python tests/test_processor.py [--out DIR]
For the real checkpoint, repeat with SD3Transformer2DModel.from_pretrained(...)
at full resolution in bf16 and report the same quantities.
"""
import argparse
import json
import os

import diffusers
import torch
from diffusers.models.attention_processor import JointAttnProcessor2_0

from fuzzydiff.checks.reference import MaterializedReference, make_model, rel, set_procs
from fuzzydiff.evaluation.table_rows import write_rows
from fuzzydiff.models.attention_processor import FuzzyJointAttnProcessor, chunked_text_attention

torch.manual_seed(0)
DT = torch.float64


def inputs(batch=2, n_txt=12):
    x = torch.randn(batch, 4, 16, 16, dtype=DT, requires_grad=True)
    ctx = torch.randn(batch, n_txt, 32, dtype=DT)
    pooled = torch.randn(batch, 16, dtype=DT)
    t = torch.full((batch,), 500.0, dtype=DT)
    return x, ctx, pooled, t



def run(model, x, ctx, pooled, t):
    return model(hidden_states=x, encoder_hidden_states=ctx, pooled_projections=pooled,
                 timestep=t, return_dict=False)[0]



def main(out_dir):
    results = {}
    for qk in (None, "rms_norm"):
        tag = "qk-norm" if qk else "no qk-norm"
        model = make_model(qk)
        x, ctx, pooled, t = inputs()
        w = torch.randn(2, 4, 16, 16, dtype=DT)

        set_procs(model, lambda i: JointAttnProcessor2_0())
        y0 = run(model, x, ctx, pooled, t)
        g0, = torch.autograd.grad((y0 * w).sum(), x)

        # 1-2: identity (readout off, SAR off)
        set_procs(model, lambda i: FuzzyJointAttnProcessor())
        y1 = run(model, x, ctx, pooled, t)
        g1, = torch.autograd.grad((y1 * w).sum(), x)
        results[f"identity_out_{tag}"] = rel(y1, y0)
        results[f"identity_grad_{tag}"] = rel(g1, g0)

        # 3: SAR enabled with r = 1 (log r = 0)
        n_img = (16 // 2) ** 2
        n_txt = ctx.shape[1]
        set_procs(model, lambda i: FuzzyJointAttnProcessor(sar_log_r=torch.zeros(n_img, n_txt, dtype=DT)))
        y3 = run(model, x, ctx, pooled, t)
        results[f"sar_r1_out_{tag}"] = rel(y3, y0)

        # 4-5: readout values and gradients vs materialized reference (cond = batch 0)
        readout = {"mug": [5, 6], "red": [8]}
        valid = list(range(0, 10))          # tokens 10, 11 play the role of padding
        ours = set_procs(model, lambda i: FuzzyJointAttnProcessor(readout=readout, valid_text=valid, chunk=7))
        run(model, x, ctx, pooled, t)
        m_ours = {f"{n}:{p}": v for n, pr in ours.items() for p, v in pr.maps.items()}
        s_ours = {f"{n}:{p}": v for n, pr in ours.items() for p, v in pr.shares.items()}
        loss_o = sum((v ** 2).sum() for v in m_ours.values()) + sum((v ** 2).sum() for v in s_ours.values())
        go, = torch.autograd.grad(loss_o, x)
        refs = set_procs(model, lambda i: MaterializedReference(readout=readout, valid_text=valid))
        run(model, x, ctx, pooled, t)
        m_ref = {f"{n}:{p}": v for n, pr in refs.items() for p, v in pr.maps.items()}
        s_ref = {f"{n}:{p}": v for n, pr in refs.items() for p, v in pr.shares.items()}
        loss_r = sum((v ** 2).sum() for v in m_ref.values()) + sum((v ** 2).sum() for v in s_ref.values())
        gr, = torch.autograd.grad(loss_r, x)
        results[f"readout_val_{tag}"] = max((rel(m_ours[k], m_ref[k]) for k in m_ref), key=lambda z: z[0])
        results[f"share_val_{tag}"] = max((rel(s_ours[k], s_ref[k]) for k in s_ref), key=lambda z: z[0])
        results[f"mass_val_{tag}"] = max((rel(ours[n].text_mass, refs[n].text_mass) for n in ours), key=lambda z: z[0])
        results[f"readout_grad_{tag}"] = rel(go, gr)

        # 6: SAR logit bias == renormalized post-softmax reweighting (random r), cond only
        log_r = torch.log(torch.rand(n_img, n_txt, dtype=DT) * 1.8 + 0.2)
        set_procs(model, lambda i: FuzzyJointAttnProcessor(sar_log_r=log_r, sar_batch=[0]))
        y6 = run(model, x, ctx, pooled, t)
        set_procs(model, lambda i: MaterializedReference(log_r=log_r, sar_batch=[0]))
        y6r = run(model, x, ctx, pooled, t)
        results[f"sar_equiv_{tag}"] = rel(y6, y6r)
        # 7: unconditional batch element unaffected by SAR (blocks interact only within a sample)
        results[f"sar_uncond_{tag}"] = rel(y6[1], y0[1])
        results[f"sar_changes_cond_{tag}"] = rel(y6[0], y0[0])

    # 7b: finite-difference directional derivative of the full transformer with readout
    model = make_model()
    x, ctx, pooled, t = inputs()
    readout = {"mug": [5, 6]}
    procs = set_procs(model, lambda i: FuzzyJointAttnProcessor(readout=readout, valid_text=list(range(10)), chunk=7))
    w = torch.randn(2, 4, 16, 16, dtype=DT)

    def scalar(xx):
        y = run(model, xx, ctx, pooled, t)
        # Confidence statistics are detached and must not enter a finite-difference
        # objective: their values change under perturbation but autograd omits them.
        return (y * w).sum() + sum((pr.maps["mug"] ** 2).sum() for pr in procs.values())

    d = torch.randn_like(x)
    g, = torch.autograd.grad(scalar(x), x)
    eps = 1e-6
    with torch.no_grad():
        fd = (scalar(x + eps * d) - scalar(x - eps * d)) / (2 * eps)
    ad = (g * d).sum()
    results["fd_grad_no qk-norm"] = (abs(fd - ad).item(), abs(fd - ad).item() / max(abs(ad).item(), 1e-300))
    results["fd_grad_qk-norm"] = results["fd_grad_no qk-norm"]

    # 8: finite-difference gradient check of the chunked readout itself
    H, NI, NT, d = 2, 6, 5, 4
    q = torch.randn(H, NI, d, dtype=DT, requires_grad=True)
    k = torch.randn(H, NI + NT, d, dtype=DT, requires_grad=True)
    bias = torch.randn(NI, NT, dtype=DT)
    idx = torch.tensor([1, 3])
    ok = torch.autograd.gradcheck(lambda q_, k_: chunked_text_attention(q_, k_, NI, idx, 0.5, bias, chunk=3),
                                  (q, k), eps=1e-6, atol=1e-8)
    results["gradcheck_readout"] = bool(ok)

    results["torch"] = torch.__version__
    results["diffusers"] = diffusers.__version__

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "processor_check.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    write_rows(out_dir)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    main(ap.parse_args().out)
