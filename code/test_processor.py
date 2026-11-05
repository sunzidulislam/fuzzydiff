"""Unit tests for FuzzyJointAttnProcessor (paper: Tab. processor_check).

Runs on a randomly initialized, reduced-size SD3Transformer2DModel in float64
on CPU and compares against diffusers' stock JointAttnProcessor2_0 and against
a fully materialized softmax reference. Writes processor_check_rows.tex (rows
for the paper table) and processor_check.json.

Usage:  python -I test_processor.py [--out DIR]
For the real checkpoint, repeat with SD3Transformer2DModel.from_pretrained(...)
at full resolution in bf16 and report the same quantities.
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzydiff_sd3_attention import FuzzyJointAttnProcessor, chunked_text_attention  # noqa: E402
from table_rows import write_rows  # noqa: E402

import diffusers  # noqa: E402
from diffusers import SD3Transformer2DModel  # noqa: E402
from diffusers.models.attention_processor import JointAttnProcessor2_0  # noqa: E402

torch.manual_seed(0)
DT = torch.float64


def make_model(qk_norm=None):
    kw = dict(sample_size=16, patch_size=2, in_channels=4, num_layers=3, attention_head_dim=8,
              num_attention_heads=4, joint_attention_dim=32, caption_projection_dim=32,
              pooled_projection_dim=16, out_channels=4, pos_embed_max_size=32)
    if qk_norm is not None:
        kw["qk_norm"] = qk_norm
    m = SD3Transformer2DModel(**kw).to(DT)
    # random-init weights are small; scale up so attention is far from uniform
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() > 1:
                p.mul_(3.0)
    return m.eval()


def inputs(batch=2, n_txt=12):
    x = torch.randn(batch, 4, 16, 16, dtype=DT, requires_grad=True)
    ctx = torch.randn(batch, n_txt, 32, dtype=DT)
    pooled = torch.randn(batch, 16, dtype=DT)
    t = torch.full((batch,), 500.0, dtype=DT)
    return x, ctx, pooled, t


def set_procs(model, factory):
    procs = {name: factory(i) for i, name in enumerate(model.attn_processors.keys())}
    model.set_attn_processor(dict(procs))   # diffusers pops entries from the dict it receives
    return procs


def run(model, x, ctx, pooled, t):
    return model(hidden_states=x, encoder_hidden_states=ctx, pooled_projections=pooled,
                 timestep=t, return_dict=False)[0]


class MaterializedReference:
    """Reference: explicit softmax over all joint keys (optional additive bias),
    stores averaged image->text probabilities for the given text indices."""

    def __init__(self, readout=None, readout_batch=0, log_r=None, sar_batch=(0,), mode="logit", valid_text=None):
        self.readout, self.readout_batch, self.log_r = readout, readout_batch, log_r
        self.sar_batch, self.mode, self.maps, self.shares = list(sar_batch), mode, {}, {}
        self.text_mass = None
        self.valid_text = valid_text

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *a, **k):
        B = hidden_states.shape[0]
        q = attn.to_q(hidden_states); kk = attn.to_k(hidden_states); v = attn.to_v(hidden_states)
        hd = kk.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        kk = kk.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None: q = attn.norm_q(q)
        if attn.norm_k is not None: kk = attn.norm_k(kk)
        n_img = q.shape[2]
        eq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
        ek = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
        ev = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
        if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
        q = torch.cat([q, eq], 2); kk = torch.cat([kk, ek], 2); v = torch.cat([v, ev], 2)
        logits = q @ kk.transpose(-1, -2) / math.sqrt(hd)
        A = torch.softmax(logits, -1)
        if self.log_r is not None:
            r = torch.ones_like(A)
            for b in self.sar_batch:
                r[b, :, :n_img, n_img:] = torch.exp(self.log_r)
            A = A * r
            A = A / A.sum(-1, keepdim=True)          # renormalized post-softmax reweighting
        if self.readout is not None:
            b = self.readout_batch
            for ph, idx in self.readout.items():
                self.maps[ph] = A[b, :, :n_img, n_img + torch.tensor(idx)].mean(dim=(0, 2))
                if self.valid_text is not None:
                    mass = A[b, :, :n_img, n_img + torch.tensor(self.valid_text)].sum(-1)
                    self.shares[ph] = (A[b, :, :n_img, n_img + torch.tensor(idx)].sum(-1) / mass).mean(0)
                    self.text_mass = mass.mean(0)
        out = (A @ v).transpose(1, 2).reshape(B, -1, attn.heads * hd)
        h, e = out[:, :hidden_states.shape[1]], out[:, hidden_states.shape[1]:]
        if not attn.context_pre_only:
            e = attn.to_add_out(e)
        h = attn.to_out[1](attn.to_out[0](h))
        return h, e


def rel(a, b):
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-300)


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
        return (y * w).sum() + sum((pr.maps["mug"] ** 2).sum() + pr.text_mass.sum() for pr in procs.values())

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
