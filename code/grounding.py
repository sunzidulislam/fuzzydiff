"""Phrase grounding in joint attention (paper: Sec. 3.2, App. "Attention confidence").

Maps phrases to T5 tokens, combines per-block readouts into localization fields, and judges each map
before min-max normalization, since normalization stretches even a flat map to [0, 1]. Attention is
used only to localize and to test binding, never as evidence of presence, area, or colour.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

CLIP_TOKENS = 77
EPS_A = 1e-6


@dataclass
class GroundedPhrase:
    B: torch.Tensor          # raw map over image tokens
    P: torch.Tensor          # min-max normalized localization field on the token grid
    peak: float              # peak-to-mean ratio of B
    share: float             # relative text share (1 = uniform split of the text mass)
    mass: float              # absolute text mass of the image queries
    informative: bool


def _occurrences(prompt: str) -> Dict[str, List[Tuple[int, int]]]:
    found: Dict[str, List[Tuple[int, int]]] = {}
    for m in re.finditer(r"[\w-]+", prompt):
        found.setdefault(m.group(0), []).append(m.span())
    return found


def _keyed(word: str, occurrences):
    spans = occurrences.get(word, [])
    return {word: spans[0]} if len(spans) == 1 else {f"{word}#{i}": s for i, s in enumerate(spans)}


def _attributes(program):
    return [c.args[0] for c in program.constraints if c.family == "binding"]


def phrase_spans(prompt: str, program) -> Dict[str, Tuple[int, int]]:
    """Character span per phrase. A word that occurs once keeps its name; repeats get #0, #1, ..."""
    occurrences = _occurrences(prompt)
    spans: Dict[str, Tuple[int, int]] = {}
    for word in list(program.objects) + _attributes(program):
        spans.update(_keyed(word, occurrences))
    return spans


def _key_at(word, span, occurrences):
    spans = occurrences[word]
    return word if len(spans) == 1 else f"{word}#{spans.index(span)}"


def binding_keys(prompt: str, program) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """(attribute, object) of each binding constraint -> their phrase keys. The attribute is the
    occurrence closest before the object; with repeated objects, each object takes its own."""
    occurrences = _occurrences(prompt)
    keys = {}
    used = set()
    for c in program.constraints:
        if c.family != "binding":
            continue
        a, q = c.args
        if a not in occurrences or q not in occurrences:
            continue
        q_span = next((s for s in occurrences[q] if (q, s) not in used), occurrences[q][-1])
        used.add((q, q_span))
        before = [s for s in occurrences[a] if s[1] <= q_span[0]]
        a_span = before[-1] if before else occurrences[a][0]
        keys[(a, q)] = (_key_at(a, a_span, occurrences), _key_at(q, q_span, occurrences))
    return keys


def map_tokens(offsets, spans, t5_limit: int = 256):
    """Packed text indices (T5 token j is 77 + j) overlapping each span; phrases with none are reported."""
    tokens, unmapped = {}, []
    for phrase, (start, end) in spans.items():
        idx = [CLIP_TOKENS + j for j, (a, b) in enumerate(offsets[:t5_limit]) if b > a and a < end and b > start]
        if idx:
            tokens[phrase] = idx
        else:
            unmapped.append(phrase)
    return tokens, unmapped


def _to_grid(x):
    side = int(math.isqrt(x.numel()))
    return x.reshape(side, side) if side * side == x.numel() else x


def _wide(x):
    """Weights run in bf16; statistics and losses run in at least float32."""
    return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x


def aggregate_maps(out, cond, read_blocks: Sequence[int], thresholds, weights=None) -> Dict[str, GroundedPhrase]:
    """Eq. phrase_attention and Eq. map_normalization. thresholds = (kappa_pk, m_min, tau_min)."""
    kappa_pk, m_min, tau_min = thresholds
    w = weights or {b: 1 / len(read_blocks) for b in read_blocks}
    result = {}
    for q, per_block in out.maps.items():
        B = sum(w[b] * _wide(per_block[b]) for b in read_blocks)
        share = sum(w[b] * _wide(out.shares[q][b]) for b in read_blocks).mean().detach()
        mass = sum(w[b] * _wide(out.text_mass[b]) for b in read_blocks).mean().detach()
        raw = B.detach()
        peak = float(raw.max() / raw.mean().clamp_min(1e-12))
        relative = float(share / (len(cond.phrase_tokens[q]) / len(cond.valid_text)))
        P = (B - B.min()) / (B.max() - B.min() + EPS_A)
        informative = peak >= kappa_pk and relative >= m_min and float(mass) >= tau_min
        result[q] = GroundedPhrase(B, _to_grid(P), peak, relative, float(mass), informative)
    return result


def overlap(p_a, p_q, eps: float = 1e-6):
    """b(a, q): the mean of a's map over q's map (Eq. attention_stats)."""
    return (p_a * p_q).sum() / (p_q.sum() + eps)


def binding(p_a, p_q, others: List[torch.Tensor], tau: float, eps: float = 1e-6):
    """beta(a, q) = b(a, q) - smax_r b(a, r), with the log-mean-exp soft maximum (Eq. attention_stats)."""
    competitors = torch.stack([overlap(p_a, r, eps) for r in others])
    smax = tau * (torch.logsumexp(competitors / tau, dim=0) - math.log(len(others)))
    return overlap(p_a, p_q, eps) - smax


def missing_phrases(prompt: str, program) -> List[str]:
    """Phrases the program needs that do not occur as a word of the prompt, so they cannot be grounded."""
    occurrences = _occurrences(prompt)
    return [w for w in dict.fromkeys(list(program.objects) + _attributes(program)) if w not in occurrences]
