"""Stage 1: tiered latent guidance on the sampler-consistent clean prediction.

Paper: Eq. tier_projection, Eq. stable_guided_step, Alg. guided_step. Every guided step fixes its
constraint set at z_t, takes one backward pass per non-empty tier, projects each lower tier onto the
cone of directions that do not oppose the raw gradients above it, and accepts a bounded candidate
only if it passes the domain and loss tests under that same frozen set.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_schedule import bin_of  # noqa: E402
from constraints import (Evidence, charbonnier, freeze_step, in_domain, membership, new_state, score, scorer_for,  # noqa: E402
                         tier_losses, violation)

BINS = 4


@dataclass
class GuideCfg:
    steps: int
    cfg: float
    sigma_lo: float                         # guided on sigma_lo <= sigma <= sigma_hi
    eta0: float
    delta: float
    dmax: float
    eps: Dict[int, float]
    backtrack: int
    ramp: int
    g_min: float = 1e-12
    eps_g: float = 1e-8
    guided: bool = True
    c_sigma: float = 1.0
    log_all: bool = False                   # also measure and log steps without guidance (calibration runs)
    sigma_hi: float = 1.0
    state_kw: Dict[str, float] = field(default_factory=dict)


@dataclass
class Stage1Ctx:
    backend: object
    program: object
    cond: object
    calib: object
    measure: Callable
    cfg: GuideCfg
    state: object
    log: object
    targets: Optional[dict]
    prompt: str
    seed: int


@dataclass
class StepOut:
    latent: torch.Tensor
    latent_next: torch.Tensor
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    v_plain: torch.Tensor
    z_used: torch.Tensor
    accepted: Optional[bool]
    rejections: List[str]
    n_tried: int
    evidence: object


@dataclass
class Stage1Out:
    latent: torch.Tensor
    image: torch.Tensor
    traces: Dict[str, torch.Tensor]
    supports: Dict[str, torch.Tensor]
    finals: List[dict]
    presence: Dict[str, float] = field(default_factory=dict)


def _dot(a, b):
    return (a.double() * b.double()).sum()


def project_tier(g, higher: List[torch.Tensor]):
    """Euclidean projection onto {d: <d, a> >= 0 for every a in higher}; closed form for <= 2 half-spaces."""
    shape = g.shape
    v = g.reshape(-1).double()
    cols = [a.reshape(-1).double() for a in higher if _dot(a, a) > 0]
    if not cols:
        return g
    candidates = [v]
    for a in cols:
        candidates.append(v - (v @ a) / (a @ a) * a)
    if len(cols) == 2:
        A = torch.stack(cols, dim=1)
        candidates.append(v - A @ (torch.linalg.pinv(A.T @ A) @ (A.T @ v)))
    tol = 1e-9 * float(v.norm().clamp_min(1.0))
    feasible = [d for d in candidates if all(float(d @ a) >= -tol * float(a.norm()) for a in cols)]
    best = min(feasible, key=lambda d: float((v - d).norm()))
    return best.reshape(shape).to(g.dtype)


def bounded_step(z, g, loss, cfg: GuideCfg):
    """Eq. stable_guided_step. Returns the candidate and the step size actually used."""
    L = float(loss)
    eta = cfg.eta0 * cfg.c_sigma * L / (L + cfg.delta)
    direction = _direction(g, cfg)
    peak = float(direction.abs().max())
    if peak > 0 and eta * peak > cfg.dmax:
        eta = cfg.dmax / peak
    return z - eta * direction, eta


def _direction(g, cfg):
    rms = g.pow(2).mean().sqrt()
    return g / (rms + cfg.eps_g)


def _bin(sigma):
    return int(bin_of(float(sigma), BINS)[0])


def _euler(z, sigma, sigma_next, v):
    return z + (sigma_next - sigma).to(z.device, z.dtype) * v


def _floats(losses):
    return {str(m): float(l.detach()) for m, l in losses.items()}


def _constraint_losses(frozen, evidence, calib):
    out = {}
    for f in frozen:
        if f.weight > 0:
            d = violation(score(f, evidence, calib), f.low, f.high)
            out[(f.cid, f.component)] = float((f.weight * charbonnier(d, f.kappa)).detach())
    return out


def _measure(ctx, z, sigma, b, readout):
    out = ctx.backend.velocity(z, sigma, ctx.cond, readout=readout, cfg=ctx.cfg.cfg)
    x0 = ctx.backend.decode(ctx.backend.clean(z, sigma, out.v))
    return out, ctx.measure(ctx.backend, x0, out, ctx.program, b)


def _light_evidence(evidence):
    """What a finished step keeps: detached attention for the trace, and no gradient graph."""
    attention = {k: {**a, "P": a["P"].detach()} for k, a in evidence.attention.items()}
    return Evidence(attention=attention)


def guided_step(z, sigma, sigma_next, step, ctx: Stage1Ctx) -> StepOut:
    """One guided step (Alg. guided_step). Falls back to the plain Euler update on any failure."""
    cfg, b = ctx.cfg, _bin(sigma)
    readout = bool(ctx.cond.phrase_tokens)
    leaf = z.detach().requires_grad_(True)
    with torch.enable_grad():
        out, evidence = _measure(ctx, leaf, sigma, b, readout)
        ctx.state.step = step
        frozen = freeze_step(ctx.program, evidence, ctx.calib, b, ctx.state, ctx.targets)
        losses = tier_losses(frozen, evidence, ctx.calib)
    v_plain = out.v.detach()
    plain_next = _euler(z, sigma, sigma_next, v_plain)
    before_terms = _constraint_losses(frozen, evidence, ctx.calib)
    before = _floats(losses)
    raw_fields = [_raw_fields(f, evidence) for f in frozen]
    attention_fields = _attention_fields(evidence)
    rejections, cos, removed = [], {}, {}
    accepted, n_tried, z_used, v_used, used_evidence = None, 0, z, v_plain, _light_evidence(evidence)
    after: Dict[str, float] = {}
    after_terms: Dict = {}

    if not losses:
        rejections.append("no_active")
        accepted = False
        del out, evidence
    else:
        tiers = sorted(losses)
        raw = {}
        for k, m in enumerate(tiers):
            if not losses[m].requires_grad:
                raw[m] = torch.zeros_like(leaf)     # e.g. a term whose evidence has no path to the latent
                continue
            g, = torch.autograd.grad(losses[m], leaf, retain_graph=k < len(tiers) - 1, allow_unused=True)
            raw[m] = torch.zeros_like(leaf) if g is None else g
        total = sum(losses.values()).detach()
        del out, losses, evidence                   # free the graph before the candidate passes
        finite = all(torch.isfinite(raw[m]).all() for m in tiers)
        g = torch.zeros_like(leaf)
        if finite:
            for a in tiers:
                for c in tiers:
                    if a < c and raw[a].norm() > 0 and raw[c].norm() > 0:
                        cos[f"{a}-{c}"] = float(_dot(raw[a], raw[c]) / (raw[a].double().norm() * raw[c].double().norm()))
            for m in tiers:
                projected = project_tier(raw[m], [raw[h] for h in tiers if h < m])
                norm = raw[m].double().norm()
                removed[str(m)] = float(1 - projected.double().norm() / norm) if norm > 0 else 0.0
                g = g + projected
        if not finite:
            rejections.append("nonfinite")
            accepted = False
        elif float(total) == 0:
            rejections.append("target_met")         # every active score is inside its interval
            accepted = False
        elif float(g.double().norm()) < cfg.g_min:
            rejections.append("small_gradient")
            accepted = False
        else:
            _, eta = bounded_step(z, g, total, cfg)
            direction = _direction(g, cfg)
            accepted = False
            for _ in range(cfg.backtrack + 1):
                n_tried += 1
                candidate = z - eta * direction
                with torch.no_grad():
                    out_c, ev_c = _measure(ctx, candidate, sigma, b, readout)
                reason = _reject_reason(frozen, ev_c, ctx.calib, before, cfg)
                if reason is None:
                    accepted, z_used, v_used, used_evidence = True, candidate, out_c.v.detach(), _light_evidence(ev_c)
                    after = _floats(tier_losses(frozen, ev_c, ctx.calib))
                    after_terms = _constraint_losses(frozen, ev_c, ctx.calib)
                    break
                rejections.append(reason)
                eta /= 2

    _log_step(ctx, step, sigma, b, frozen, accepted, before, after, cos, removed, rejections,
              before_terms, after_terms, raw_fields, attention_fields)
    latent_next = plain_next if accepted is not True else _euler(z_used, sigma, sigma_next, v_used)
    return StepOut(z, latent_next.detach(), sigma, sigma_next, v_plain, z_used, accepted, rejections, n_tried,
                   used_evidence)


def _reject_reason(frozen, evidence, calib, before, cfg):
    """Domain first, so no domain-limited function sees an uncalibrated input (App. "Acceptance")."""
    for f in frozen:
        if f.weight > 0 and not in_domain(f, evidence, calib):
            return "domain"
    now = _floats(tier_losses(frozen, evidence, calib))
    for m in (1, 2):
        key = str(m)
        if key in before and now.get(key, 0.0) > before[key] + cfg.eps.get(m, 0.0):
            return f"tier{m}"
    if sum(now.values()) > sum(before.values()):
        return "total"
    return None


def _raw_fields(f, evidence):
    """Raw inputs at this step, so calibration can later fit each noise bin (App. "Data")."""
    c = f.constraint
    fields = {"raw": _raw(f, evidence), "quantity": ":".join(c.quantity) if isinstance(c.quantity, tuple) else c.quantity}
    obj = evidence.objects.get(c.args[0], {}) if c.args else {}
    if c.family == "presence" and "g" in obj:
        fields.update(object=c.args[0], **{"global": float(obj["g"]), "region": float(obj["r"])})
    if c.family == "F8":
        objs = [evidence.objects.get(q) for q in c.args]
        if all(o is not None for o in objs):
            fields.update(objects=list(c.args), orientation=c.orientation, areas=[float(o["A"].detach()) for o in objs])
    return fields


def _attention_fields(evidence):
    out = {}
    for key, a in evidence.attention.items():
        P = a["P"].detach()
        i = int(P.argmax())
        side = P.shape[-1]
        out[key] = {"peak": a["peak"], "share": a["share"], "mass": a["mass"], "peak_xy": [i // side, i % side]}
    return out


def _log_step(ctx, step, sigma, b, frozen, accepted, before, after, cos, removed, rejections, before_terms, after_terms,
              raw_fields, attention_fields):
    for f, fields in zip(frozen, raw_fields):
        key = (f.cid, f.component)
        ctx.log.write(type="constraint", prompt=ctx.prompt, seed=ctx.seed, step=step, sigma=float(sigma), bin=b,
                      family=f.constraint.family, cid=f.cid, component=f.component, tier=f.tier, gates=f.gates,
                      active=f.weight > 0, accepted=accepted if f.weight > 0 else None,
                      dL=after_terms[key] - before_terms[key] if key in after_terms and key in before_terms else None,
                      **fields)
    ctx.log.write(type="step", prompt=ctx.prompt, seed=ctx.seed, step=step, sigma=float(sigma),
                  attention=attention_fields, tiers=sorted({f.tier for f in frozen if f.weight > 0}), loss_before=before,
                  loss_after=after or before, cos=cos, removed=removed, rejections=rejections,
                  accepted=bool(accepted), eps={str(m): e for m, e in ctx.cfg.eps.items()})


def _log_unguided_step(ctx, z, sigma, step):
    """Raw estimator values of a step without guidance, for calibration (App. "Data"). Returns its velocity."""
    b = _bin(sigma)
    out, evidence = _measure(ctx, z, sigma, b, bool(ctx.cond.phrase_tokens))
    ctx.state.step = step
    frozen = freeze_step(ctx.program, evidence, ctx.calib, b, ctx.state, ctx.targets)
    for f in frozen:
        ctx.log.write(type="constraint", prompt=ctx.prompt, seed=ctx.seed, step=step, sigma=float(sigma), bin=b,
                      family=f.constraint.family, cid=f.cid, component=f.component, tier=f.tier, gates=f.gates,
                      active=False, accepted=None, dL=None, **_raw_fields(f, evidence))
    return out.v


def _accumulate_trace(sums, counts, evidence):
    for q, a in evidence.attention.items():
        if a.get("informative"):
            p = a["P"].detach().cpu().double()
            sums[q] = sums[q] + p if q in sums else p
            counts[q] = counts.get(q, 0) + 1


def score_image(program, measure, backend, calib, x, state, targets=None):
    """Scores of every constraint on a finished image, under the last bin's calibration."""
    evidence = measure(backend, x, None, program, BINS)
    records = []
    for f in freeze_step(program, evidence, calib, BINS, state, targets):
        if f.constraint.family == "binding":
            continue                        # attention needs a backbone pass; the final image has none
        ok = in_domain(f, evidence, calib)
        s = score(f, evidence, calib) if ok else None
        c = f.constraint
        records.append({"family": c.family, "cid": f.cid, "component": f.component, "tier": f.tier,
                        "quantity": ":".join(c.quantity) if isinstance(c.quantity, tuple) else c.quantity,
                        "s": float(s.detach()) if ok else None, "low": f.low, "high": f.high,
                        "mu": float(membership(s, f.low, f.high, f.kappa).detach()) if ok else 0.0,
                        "raw": _raw(f, evidence)})
    return records, evidence


def _raw(f, evidence):
    r = scorer_for(f.constraint).raw(f.constraint, evidence, f.component)
    return None if r is None else float(torch.as_tensor(r).detach())


def run_stage1(backend, program, cond, seed, calib, measure, cfg: GuideCfg, log, targets=None, prompt="") -> Stage1Out:
    state = new_state(ramp=cfg.ramp, **cfg.state_kw)
    ctx = Stage1Ctx(backend, program, cond, calib, measure, cfg, state, log, targets, prompt, seed)
    sigmas = backend.sigmas(cfg.steps)
    z = backend.noise_like_latent(seed)
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for i in range(cfg.steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        if cfg.guided and cfg.sigma_lo <= float(sigma) <= cfg.sigma_hi:
            out = guided_step(z, sigma, sigma_next, i, ctx)
            z = out.latent_next
            _accumulate_trace(sums, counts, out.evidence)
        else:
            with torch.no_grad():
                if cfg.log_all:
                    v = _log_unguided_step(ctx, z, sigma, i)
                else:
                    v = backend.velocity(z, sigma, cond, cfg=cfg.cfg).v
                z = _euler(z, sigma, sigma_next, v)
    state.step = cfg.steps
    with torch.no_grad():
        image = backend.decode(z)
        finals, evidence = score_image(program, measure, backend, calib, image, state, targets)
    supports = {q: o["S"].detach().cpu() for q, o in evidence.objects.items() if "S" in o}
    presence = {q: float(o["p"]) for q, o in evidence.objects.items() if "p" in o}
    traces = {q: sums[q] / counts[q] for q in sums}
    return Stage1Out(z, image, traces, supports, finals, presence)
