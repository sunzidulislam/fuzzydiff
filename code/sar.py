"""Stage 2: the Spatial Attention Refiner (paper: Sec. 3.6, App. "Spatial Attention Refiner Details").

The Stage-1 image is re-encoded and re-noised, then refined with a bias on image-to-text logits. Each
step checks a SAR-free prediction before applying the bias, so the check cannot be influenced by the
bias it authorizes. Drift is detected, not prevented: a SAR step after which an object moved is
rolled back, and an output that fails the final check is replaced by the Stage-1 image.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_schedule import bin_of  # noqa: E402
from constraints import freeze_step, in_domain, membership, new_state, score, scorer_for  # noqa: E402
from estimators import area, centroid  # noqa: E402
from fuzzydiff_sd3_attention import sar_log_multiplier  # noqa: E402
from grounding import binding_keys  # noqa: E402

BINS = 4
OVERLAP_WORDS = {"on", "in", "inside", "holding", "wearing", "carrying", "of", "with"}


@dataclass
class SarCfg:
    strength: float
    k: int
    alpha: float
    beta: float
    temp: float
    gamma_min: float
    overlap: float
    nest: float
    confidence: bool = False
    competition: bool = False
    consistency: bool = True                # False: the ablation keeps every prior for all K steps (final check stays)
    # Tab. full ablations (Stage 2); the defaults are the method
    prior: str = "product"                  # "attention": M_q is the Stage-1 attention trace alone
    competition_all: bool = False           # competition between every pair of objects, whatever their relation
    drift_ref: str = "support"              # "prior": the drift check compares against M_q (the old check)
    lagged: bool = False                    # the check reads the previous step's SAR-modified x0, not this step's
    post_softmax: bool = False              # r multiplies the attention probabilities after the softmax, unnormalized
    time_factor: int = 1                    # plain img2img with time_factor x K steps over the same noise range


@dataclass
class Stage2Out:
    image: torch.Tensor
    latent: torch.Tensor
    renoised: torch.Tensor
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    rollbacks: int = 0
    disabled: Set[str] = field(default_factory=set)
    fallback_objects: Set[str] = field(default_factory=set)
    suspended_steps: int = 0
    passes: int = 0
    bias_steps: int = 0
    no_op_reason: Optional[str] = None
    latents: List[torch.Tensor] = field(default_factory=list)
    sar_free: List[torch.Tensor] = field(default_factory=list)


def soft_iou(a, b, eps=1e-6):
    return torch.minimum(a, b).sum() / (torch.maximum(a, b).sum() + eps)


def _overlap_coefficient(a, b, eps=1e-6):
    return torch.minimum(a, b).sum() / (torch.minimum(a.sum(), b.sum()) + eps)


def _resize(p, shape):
    if p.shape == tuple(shape):
        return p
    return F.interpolate(p[None, None], size=tuple(shape), mode="bilinear", align_corners=False)[0, 0]


def relation_types(prompt: str, program) -> Dict[tuple, str]:
    """Exclusive unless a containment or support word joins the two objects."""
    tokens = prompt.split()
    out = {}
    objects = list(dict.fromkeys(program.objects))
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if a in tokens and b in tokens:
                lo, hi = sorted((tokens.index(a), tokens.index(b)))
                between = set(tokens[lo + 1:hi])
                out[(a, b)] = "overlap" if between & OVERLAP_WORDS else "exclusive"
    return out


def _relation(relations, a, b):
    return (relations or {}).get((a, b), (relations or {}).get((b, a)))


def build_priors(traces, supports, cfg: SarCfg, presence=None, relations=None) -> Dict[str, torch.Tensor]:
    """M_q = sqrt(mean attention trace x Stage-1 support), so attention alone places no prior."""
    priors = {}
    aligned = {}
    for q, trace in traces.items():
        head = q.split("#")[0]
        if head in supports:
            s = supports[head]
            aligned[q] = s
            t = _resize(trace.to(s.dtype), s.shape).clamp_min(0)
            if cfg.prior == "attention":
                priors[q] = t / t.max().clamp_min(1e-12)        # attention-only prior, scaled to [0, 1]
            else:
                priors[q] = torch.sqrt(t * s.clamp_min(0))
    if not cfg.competition:
        return priors
    competed = {}
    for q, m in priors.items():
        if cfg.competition_all:
            group = [r for r in priors if r != q]
        else:
            group = [r for r in priors if r != q and _relation(relations, q, r) == "exclusive"
                     and soft_iou(aligned[q], aligned[r]) < cfg.overlap
                     and _overlap_coefficient(aligned[q], aligned[r]) < cfg.nest]
        if group:
            stack = torch.stack([m] + [priors[r] for r in group]) / cfg.temp
            competed[q] = m * torch.softmax(stack, dim=0)[0]
        else:
            competed[q] = m
    return competed


def attribute_priors(prompt, program, priors):
    """An attribute uses the prior of the object it is bound to."""
    return {a_key: priors[q_key] for (a, q), (a_key, q_key) in binding_keys(prompt, program).items() if q_key in priors}


def _to_tokens(prior, n_img):
    if prior.dim() == 1:
        return prior
    side = int(math.isqrt(n_img))
    return F.adaptive_avg_pool2d(prior[None, None], side)[0, 0].reshape(-1)


def build_log_r(priors, phrase_tokens, n_text, alpha, beta, enabled, n_img=None) -> torch.Tensor:
    """Eq. spatial_refiner: log r_ij, zero except on the keys of enabled phrases."""
    size = n_img or next(iter(priors.values())).numel()
    active = {q: _to_tokens(priors[q], size) for q in enabled if q in priors and q in phrase_tokens}
    if not active:
        return torch.zeros(size, n_text)
    a = alpha if isinstance(alpha, dict) else {q: alpha for q in active}
    b = beta if isinstance(beta, dict) else {q: beta for q in active}
    return sar_log_multiplier(None, {q: phrase_tokens[q] for q in active}, active, n_text, a, b)


def stratum(a) -> str:
    a = float(a)
    return "small" if a < 0.02 else ("medium" if a <= 0.10 else "large")


def drift_ok(s_now, s_ref, thresholds) -> bool:
    theta, delta_rel = thresholds
    s_ref = s_ref.to(s_now.device, s_now.dtype)
    a = float(area(s_ref))
    shift = float(torch.linalg.norm(centroid(s_now) - centroid(s_ref)))
    return float(soft_iou(s_now, s_ref)) >= theta and shift <= delta_rel * math.sqrt(a / math.pi)


def _strengths(cfg, priors, attr_of, traces, supports, presence):
    alpha, beta, weak = {}, {}, set()
    for q, m in priors.items():
        gamma = 1.0
        if cfg.confidence:
            head = q.split("#")[0]
            trace = _resize(traces[q], supports[head].shape).to(supports[head].dtype) if q in traces else m
            gamma = float((presence or {}).get(head, 1.0)) * float(soft_iou(trace, supports[head]))
            if gamma < cfg.gamma_min:
                weak.add(q)
        alpha[q], beta[q] = cfg.alpha * gamma, cfg.beta * gamma
    for a_key, q_key in attr_of.items():
        if q_key in alpha:
            alpha[a_key], beta[a_key] = alpha[q_key], beta[q_key]
    return alpha, beta, weak


def _attribute_regressed(stage1, x, program, calib, measure, backend) -> bool:
    """Final test (b): a Stage-1 constraint that held (mu >= 0.5) must still hold."""
    before = {(f["cid"], f["component"]): f["mu"] for f in stage1.finals}
    evidence = measure(backend, x, None, program, BINS)
    for f in freeze_step(program, evidence, calib, BINS, new_state()):
        if before.get((f.cid, f.component), 0.0) < 0.5:
            continue
        if scorer_for(f.constraint).raw(f.constraint, evidence, f.component) is None:
            continue
        if not in_domain(f, evidence, calib):
            return True
        if float(membership(score(f, evidence, calib), f.low, f.high, f.kappa)) < 0.5:
            return True
    return False


def _no_op_reason(objects_ok, priors, out, cond):
    """Why a SAR run applied no bias: it looks like a success, so it must say so."""
    if not objects_ok:
        return "attention_failed"
    if not priors:
        return "no_priors"
    if not any(q in cond.phrase_tokens for q in priors):
        return "no_tokens"
    if out.fallback_objects >= set(priors):
        return "no_thresholds"
    if out.suspended_steps and not out.disabled:
        return "suspended"
    return "all_disabled"


@torch.no_grad()
def run_stage2(backend, stage1, cond, program, prompt, calib, seg, cfg: SarCfg, seed, log, mode="sar", measure=None,
               steps=28, priors=None) -> Stage2Out:
    """mode "sar" or "plain". All modes share the re-noising, so arms differ only in the bias."""
    backend.sar_post_softmax = cfg.post_softmax         # read by the attention processors; reset at the end
    sigmas = backend.sigmas(steps)
    start = steps - cfg.k
    if cfg.time_factor != 1:                # time-matched plain pass: each step of the same noise range split in f
        f, sub = cfg.time_factor, sigmas[start:]
        sigmas = torch.cat([torch.cat([torch.linspace(float(a), float(b), f + 1, dtype=sub.dtype)[:-1]
                                       for a, b in zip(sub[:-1], sub[1:])]), sub[-1:]])
        start, cfg = 0, dataclasses.replace(cfg, k=cfg.k * f, time_factor=1)
    noise = backend.noise_like_latent(1_000_003 + seed)
    latent0 = backend.encode(stage1.image)
    z = (1 - sigmas[start].to(latent0.dtype)) * latent0 + sigmas[start].to(latent0.dtype) * noise
    out = Stage2Out(image=stage1.image, latent=z, renoised=z.clone())

    objects_ok = calib.attention_ok("objects")
    active_mode = mode == "sar" and objects_ok
    object_priors = {}
    if active_mode:
        if priors is not None:
            object_priors = dict(priors)
        else:
            relations = relation_types(prompt, program) if cfg.competition else None
            object_priors = build_priors(stage1.traces, stage1.supports, cfg, stage1.presence, relations)
    attr_of = {a: q for (_, _), (a, q) in binding_keys(prompt, program).items()} if active_mode else {}
    if not calib.attention_ok("attributes"):
        attr_of = {}
    all_priors = dict(object_priors)
    all_priors.update({a: object_priors[q] for a, q in attr_of.items() if q in object_priors})
    alpha, beta, weak = _strengths(cfg, object_priors, attr_of, stage1.traces, stage1.supports, stage1.presence)

    if cfg.drift_ref == "prior":            # old check: drift measured against the prior M_q, scaled to [0, 1]
        refs = {q: m / m.max().clamp_min(1e-12) for q, m in object_priors.items()}
    else:
        refs = {q: stage1.supports[q.split("#")[0]] for q in object_priors}
    enabled = set()
    for q, ref in refs.items():
        if calib.drift_thresholds(stratum(area(ref))) is None:
            out.fallback_objects.add(q)
        elif q not in weak and q in cond.phrase_tokens:
            enabled.add(q)
    checked = set(refs) - out.fallback_objects
    n_text = cond.embeds.shape[1]
    heads = sorted({q.split("#")[0] for q in refs})

    def failing(x, phrases):
        """Phrases whose support moved; a stratum with no valid threshold also counts as failing."""
        supports = seg(x, heads)
        bad = set()
        for q in phrases:
            s = supports[q.split("#")[0]]
            t = calib.drift_thresholds(stratum(area(s)))
            if t is None or not drift_ok(s, refs[q], t):
                bad.add(q)
        return bad

    previous_q: Set[str] = set()
    lagged_x = None
    z0_next = None
    for k in range(cfg.k):
        i = start + k
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        b = int(bin_of(float(sigma), BINS)[0])
        rollback = False
        first = backend.velocity(z, sigma, cond)
        out.passes += 1
        q_k: Set[str] = set()
        suspended = False
        if active_mode and refs:
            if not calib.stage2_valid(b):
                suspended = True
                out.suspended_steps += 1
            elif enabled and not cfg.consistency:
                q_k = set(enabled)
            elif enabled:
                if cfg.lagged and lagged_x is not None:      # ablation: last step's SAR-modified prediction
                    x_hat = lagged_x
                else:
                    x_hat = backend.decode(backend.clean(z, sigma, first.v))
                bad = failing(x_hat, enabled)
                if bad & previous_q:
                    z = z0_next
                    rollback = True
                    out.rollbacks += 1
                    first = backend.velocity(z, sigma, cond)
                    out.passes += 1
                    enabled -= bad
                    out.disabled |= bad
                    x_hat = backend.decode(backend.clean(z, sigma, first.v))
                    bad = failing(x_hat, enabled) if enabled else set()
                enabled -= bad
                out.disabled |= bad
                q_k = set(enabled)
        out.latents.append(z.clone())
        v = first.v
        if q_k:
            phrases = q_k | {a for a, q in attr_of.items() if q in q_k}
            log_r = build_log_r(all_priors, cond.phrase_tokens, n_text, alpha, beta, phrases, n_img=backend.n_img)
            v_cond = backend.velocity_cond(z, sigma, cond, log_r)
            out.passes += 1
            v = first.v_uncond + backend.cfg * (v_cond - first.v_uncond)
        step = (sigma_next - sigma).to(z.device, z.dtype)
        z0_next = z + step * first.v
        out.sar_free.append(z0_next.clone())
        if cfg.lagged:
            lagged_x = backend.decode(backend.clean(z, sigma, v)) if q_k else None
        z = z + step * v
        previous_q = q_k
        out.bias_steps += bool(q_k)
        log.write(type="sar_step", prompt=prompt, seed=seed, step=k, sigma=float(sigma), mode=mode, enabled=sorted(q_k),
                  rollback=rollback, suspended=suspended)

    out.latent = z
    image = backend.decode(z)
    out.image = image
    if mode == "sar" and out.bias_steps == 0:
        out.no_op_reason = _no_op_reason(objects_ok, object_priors, out, cond)
    if active_mode and checked:
        if failing(image, checked):
            out.used_fallback, out.fallback_reason = True, "spatial"
        elif measure is not None and _attribute_regressed(stage1, image, program, calib, measure, backend):
            out.used_fallback, out.fallback_reason = True, "attribute"
        if out.used_fallback:
            out.image = stage1.image
    log.write(type="sar_final", prompt=prompt, seed=seed, mode=mode, used_fallback=out.used_fallback,
              fallback_reason=out.fallback_reason, rollbacks=out.rollbacks, disabled=sorted(out.disabled),
              fallback_objects=sorted(out.fallback_objects), suspended_steps=out.suspended_steps, passes=out.passes,
              bias_steps=out.bias_steps, no_op_reason=out.no_op_reason)
    backend.sar_post_softmax = False
    return out
