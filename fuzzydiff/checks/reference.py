"""Reference attention for the processor checks: an explicitly materialized softmax over all joint keys.

Shared by tests/test_processor.py, arch_check.py and the checkpoint verification."""
import math

import torch
from diffusers import SD3Transformer2DModel

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


def set_procs(model, factory):
    procs = {name: factory(i) for i, name in enumerate(model.attn_processors.keys())}
    model.set_attn_processor(dict(procs))   # diffusers pops entries from the dict it receives
    return procs


class MaterializedReference:
    """Reference: explicit softmax over all joint keys (optional additive bias),
    stores averaged image->text probabilities for the given text indices."""

    def __init__(self, readout=None, readout_batch=0, log_r=None, sar_batch=(0,), mode="logit", valid_text=None):
        self.readout, self.readout_batch, self.log_r = readout, readout_batch, log_r
        self.sar_batch, self.mode, self.maps, self.shares = list(sar_batch), mode, {}, {}
        self.text_mass = None
        self.text_mass_stats = None
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
                    # Match the processor's detached confidence statistics and its
                    # denominator floor, including nonzero mass below 1e-30.
                    mass = A[b, :, :n_img, n_img + torch.tensor(self.valid_text)].sum(-1).detach()
                    self.text_mass_stats = {
                        "head_query_rows": mass.numel(),
                        "zero_rows": int((mass == 0).sum().item()),
                        "floored_rows": int((mass < 1e-30).sum().item()),
                        "nonfinite_rows": int((~torch.isfinite(mass)).sum().item()),
                    }
                    phrase_mass = A[b, :, :n_img, n_img + torch.tensor(idx)].detach().sum(-1)
                    self.shares[ph] = (phrase_mass / mass.clamp_min(1e-30)).mean(0)
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
