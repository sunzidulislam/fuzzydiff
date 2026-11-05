"""Custom SD3 joint-attention processor for FuzzyDiff.

Drop-in replacement for diffusers' ``JointAttnProcessor2_0`` (verified against
diffusers 0.31.0) with two optional additions:

* Readout (Stage 1): for one batch element (the conditional CFG branch), the
  image-to-text block of the joint row-softmax is computed without forming the
  full N x N attention matrix,
      A[i, j] = exp(q_i k_j^T * scale + bias_ij - LSE_i),   j in text keys,
  where LSE_i is the log-sum-exp over *all* joint keys, accumulated over key
  chunks. Only the head- and token-averaged map per phrase is kept
  (Eq. phrase_attention of the paper). The readout is wrapped in activation
  checkpointing so its intermediates are recomputed in the backward pass.

* SAR (Stage 2): an additive logit bias log r_ij on (image query, text key)
  pairs of selected batch elements, applied before a single softmax over all
  joint keys (Eq. spatial_refiner). Rows still sum to one.

With readout and SAR disabled, the processor performs exactly the stock
computation; see test_processor.py.

Joint ordering (diffusers SD3): queries/keys are cat([image, text]); the text
stream is cat([CLIP-L||CLIP-G (77 tokens), T5 (max_sequence_length)]). Hence
T5 token j is joint key  N_I + 77 + j.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_text_attention(
    q_img: torch.Tensor,           # [H, N_I, d]
    k_all: torch.Tensor,           # [H, N, d]   joint keys (image first, then text)
    n_img: int,
    text_idx: torch.Tensor,        # [m] indices into the text part (0-based)
    scale: float,
    bias_text: Optional[torch.Tensor] = None,  # [N_I, N_T] additive bias on text keys
    chunk: int = 1024,
) -> torch.Tensor:
    """Image-to-text joint attention probabilities for selected text keys.

    Returns A[h, i, m] = softmax_j(q_i k_j^T * scale + b_ij)[text_idx[m]],
    the softmax being over all N joint keys, without materializing [H, N_I, N].
    Half-precision inputs are widened: these probabilities are too small for 16 bits.
    """
    if q_img.dtype in (torch.float16, torch.bfloat16):
        q_img, k_all = q_img.float(), k_all.float()
        bias_text = None if bias_text is None else bias_text.float()
    H, n_q, _ = q_img.shape
    n = k_all.shape[1]
    lse = torch.full((H, n_q), -math.inf, dtype=q_img.dtype, device=q_img.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        s = torch.einsum("hid,hjd->hij", q_img, k_all[:, start:stop]) * scale
        if bias_text is not None and stop > n_img:
            t0 = max(start, n_img)
            s[:, :, t0 - start:] = s[:, :, t0 - start:] + bias_text[:, t0 - n_img:stop - n_img]
        lse = torch.logaddexp(lse, torch.logsumexp(s, dim=-1))
    k_sel = k_all[:, n_img + text_idx]                                  # [H, m, d]
    s_sel = torch.einsum("hid,hmd->him", q_img, k_sel) * scale
    if bias_text is not None:
        s_sel = s_sel + bias_text[:, text_idx]
    return torch.exp(s_sel - lse.unsqueeze(-1))


class FuzzyJointAttnProcessor:
    """SD3 joint-attention processor with optional readout and SAR bias.

    Args:
        readout: mapping phrase -> list of text-token indices (0-based within the
            packed text sequence; for T5 token j use 77 + j). None disables readout.
        readout_batch: batch index of the conditional branch.
        valid_text: indices of all valid (non-padding) text tokens. If given, the readout
            also returns, per phrase, its share of the image->text attention mass,
            share_q[i] = mean_h sum_{j in T(q)} A_hij / sum_{j in valid} A_hij,
            used for the confidence test (paper Sec. 3.2).
        sar_log_r: tensor [N_I, N_T] of log r_ij (zeros = identity). None disables SAR.
        sar_batch: batch indices that receive the SAR bias (conditional branch only).
        chunk: key chunk size for the log-sum-exp.
        use_checkpoint: recompute the readout in the backward pass.
    """

    def __init__(
        self,
        readout: Optional[Dict[str, Sequence[int]]] = None,
        readout_batch: int = 0,
        valid_text: Optional[Sequence[int]] = None,
        sar_log_r: Optional[torch.Tensor] = None,
        sar_batch: Sequence[int] = (0,),
        chunk: int = 1024,
        use_checkpoint: bool = True,
    ) -> None:
        self.readout = readout
        self.readout_batch = readout_batch
        self.valid_text = list(valid_text) if valid_text is not None else None
        self.sar_log_r = sar_log_r
        self.sar_batch = list(sar_batch)
        self.chunk = chunk
        self.use_checkpoint = use_checkpoint
        self.maps: Dict[str, torch.Tensor] = {}     # B_q: head- and token-averaged joint probability
        self.shares: Dict[str, torch.Tensor] = {}   # share_q: phrase share of text attention
        self.text_mass: Optional[torch.Tensor] = None  # [N_I] head-averaged joint mass on valid text keys

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # ---- identical to JointAttnProcessor2_0 (diffusers 0.31.0) ----
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        n_img = query.shape[2]
        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)
            query = torch.cat([query, eq], dim=2)
            key = torch.cat([key, ek], dim=2)
            value = torch.cat([value, ev], dim=2)
        n_all = key.shape[2]
        scale = 1.0 / math.sqrt(head_dim)

        # ---- optional SAR: additive bias on (image query, text key) block ----
        mask = None
        if self.sar_log_r is not None and encoder_hidden_states is not None:
            n_txt = n_all - n_img
            assert self.sar_log_r.shape == (n_img, n_txt), "sar_log_r must be [N_I, N_T]"
            mask = torch.zeros(batch_size, 1, n_all, n_all, dtype=query.dtype, device=query.device)
            for b in self.sar_batch:
                mask[b, 0, :n_img, n_img:] = self.sar_log_r.to(query.dtype)

        # ---- optional readout on the conditional branch ----
        if self.readout is not None and encoder_hidden_states is not None:
            b = self.readout_batch
            bias = self.sar_log_r.to(query.dtype) if (self.sar_log_r is not None and b in self.sar_batch) else None
            phrases = list(self.readout.keys())
            if self.valid_text is not None:
                cols = list(self.valid_text)
                pos = {p: [cols.index(j) for j in self.readout[p]] for p in phrases}
            else:
                cols = sorted({j for p in phrases for j in self.readout[p]})
                pos = {p: [cols.index(j) for j in self.readout[p]] for p in phrases}
            cols_t = torch.as_tensor(cols, device=query.device, dtype=torch.long)
            want_share = self.valid_text is not None

            def _read(q, k, _bias=bias):
                a = chunked_text_attention(q, k, n_img, cols_t, scale, _bias, self.chunk)   # [H, N_I, |cols|]
                out = []
                # Confidence statistics only: never differentiated, and a query can have no text mass at
                # all, where 0/0 would otherwise poison the gradient of every map through the shared graph.
                mass = a.sum(-1).detach() if want_share else None                         # [H, N_I]
                for p in phrases:
                    ap = a[:, :, pos[p]]
                    out.append(ap.mean(dim=(0, 2)))                                         # B_q
                    if want_share:
                        out.append((ap.detach().sum(-1) / mass.clamp_min(1e-30)).mean(0))    # share_q
                if want_share:
                    out.append(mass.mean(0))                                                # text mass
                return torch.stack(out)

            q_b, k_b = query[b, :, :n_img], key[b]
            if self.use_checkpoint and torch.is_grad_enabled():
                res = checkpoint(_read, q_b, k_b, use_reentrant=False)
            else:
                res = _read(q_b, k_b)
            step = 2 if want_share else 1
            for n, p in enumerate(phrases):
                self.maps[p] = res[step * n]
                if want_share:
                    self.shares[p] = res[step * n + 1]
            if want_share:
                self.text_mass = res[-1]

        # ---- attention output (fused kernel; same call as stock when mask is None) ----
        if mask is None:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def sar_log_multiplier(prior_img: torch.Tensor, text_idx_by_phrase: Dict[str, List[int]],
                       priors: Dict[str, torch.Tensor], n_text: int,
                       alpha: Dict[str, float], beta: Dict[str, float]) -> torch.Tensor:
    """Build log r_ij (Eq. spatial_refiner) of shape [N_I, N_T].

    priors[q]: [N_I] prior M_q on the image-token grid (already fused/competed);
    keys of phrase q get log((1 + a M)(1 - b (1 - M))); all other keys get 0.
    """
    first = next(iter(priors.values()))
    n_img = prior_img.shape[0] if prior_img is not None else first.shape[0]
    log_r = torch.zeros(n_img, n_text, dtype=first.dtype, device=first.device)
    for q, idx in text_idx_by_phrase.items():
        m = priors[q].clamp(0, 1)
        r = (1 + alpha[q] * m) * (1 - beta[q] * (1 - m))
        log_r[:, idx] = torch.log(r).unsqueeze(-1)
    return log_r


def phrase_confidence(b_q: torch.Tensor, share_q: torch.Tensor, n_phrase: int, n_valid: int,
                      text_mass: Optional[torch.Tensor] = None):
    """Confidence statistics of Sec. 3.2 (computed on raw, un-normalized maps).

    peak_to_mean: max_i B_q / mean_i B_q (scale-invariant).
    abs_text_mass: mean_i of the joint-softmax mass on valid text keys (absolute criterion).
    relative_share: mean_i share_q / (|T(q)| / |T_valid|); equals 1 when the phrase receives
        exactly its uniform share of the text attention, independently of how much mass
        the image keys take.
    """
    peak_to_mean = (b_q.max() / b_q.mean().clamp_min(1e-12)).item()
    relative_share = (share_q.mean() / (n_phrase / n_valid)).item()
    # absolute criterion: mean joint-softmax mass that image queries put on valid text keys.
    # A relative share can look confident while text keys receive almost no mass at all.
    abs_text_mass = float("nan") if text_mass is None else text_mass.mean().item()
    return peak_to_mean, relative_share, abs_text_mass
