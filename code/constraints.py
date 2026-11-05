"""Fuzzy memberships, Charbonnier losses, and the constraint set frozen for one guided step.

Paper: Eq. hedge_membership, Eq. aggregate_fuzzy_loss, App. "Frozen active set". A family registers a
Scorer in FAMILY_SCORERS. Weights, intervals, and widths are fixed at z_t and reused at every
candidate, so a loss cannot fall because a constraint was deactivated.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import domain_weight, ramp_weight  # noqa: E402
from predicates import Constraint  # noqa: E402


@dataclass
class Evidence:
    """Everything measured at one latent."""
    x0: Optional[torch.Tensor] = None
    scene: Dict[str, torch.Tensor] = field(default_factory=dict)
    objects: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)
    pairs: Dict[tuple, torch.Tensor] = field(default_factory=dict)
    attention: Dict[str, dict] = field(default_factory=dict)
    raw_ok: Dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Frozen:
    constraint: Constraint
    cid: int
    tier: int
    bin: int
    weight: float
    low: float
    high: float
    kappa: float
    gates: Dict[str, bool] = field(hash=False, default_factory=dict)
    component: str = "main"


@dataclass
class StepState:
    p_gate: float = 0.5
    a_min: float = 0.005
    psi_min: float = 0.5
    ramp: int = 2
    band: float = 0.1
    lam: float = 1.0
    step: int = 0
    first_step: Optional[int] = None
    first_valid: Dict[str, int] = field(default_factory=dict)
    unguided: set = field(default_factory=set)
    size_regime: Dict[int, str] = field(default_factory=dict)


def new_state(**kw) -> StepState:
    return StepState(**kw)


def membership(s, low, high, kappa):
    return torch.exp(-violation(s, low, high) ** 2 / (2 * kappa ** 2))


def violation(s, low, high):
    """d(s, I) = [l - s]_+ + [s - u]_+; infinite ends give half-lines."""
    d = torch.zeros_like(s)
    if not math.isinf(low):
        d = d + (low - s).clamp_min(0)
    if not math.isinf(high):
        d = d + (s - high).clamp_min(0)
    return d


def charbonnier(d, kappa):
    return torch.sqrt(1 + (d / kappa) ** 2) - 1


class Scorer:
    """One family: how to read its raw input, whether that is in domain, and its score."""

    def tapered(self, component):
        """The domain taper protects a domain-limited perceptual map; native scores have none."""
        return False

    def components(self, c: Constraint):
        return ["main"]

    def raw(self, c, evidence, component):
        raise NotImplementedError

    def domain(self, c, calib, bin, component):
        return calib.domain(c.family, c.quantity, bin)

    def score(self, c, evidence, calib, bin, component):
        raise NotImplementedError

    def interval(self, c, calib, component):
        return calib.interval(c.family, c.quantity, c.hedge, c.orientation)

    def gates(self, c, evidence, calib, state, component):
        """Family-specific gates; validity and domain are handled by freeze_step."""
        return {"presence": _presence_gate(c, evidence, state), "support": True, "confidence": True}


def _presence_gate(c, evidence, state):
    if c.tier == 1 or not c.args:
        return True
    obj = evidence.objects.get(c.args[0])
    return True if obj is None or "p" not in obj else float(obj["p"]) >= state.p_gate


def _support_gate(q, evidence, state):
    obj = evidence.objects.get(q)
    return obj is not None and float(obj["A"].detach()) >= state.a_min and float(obj["psi"]) >= state.psi_min


def mix_key(quantity):
    return "mix:" + ":".join(quantity)


class Photometric(Scorer):
    """F2a: scene lightness over the whole image, or lightness and chroma inside one object."""

    def tapered(self, component):
        return True

    def raw(self, c, evidence, component):
        if c.quantity == "scene_lightness":
            return evidence.scene.get(c.quantity)
        return evidence.objects.get(c.args[0], {}).get(c.quantity)

    def score(self, c, evidence, calib, bin, component):
        s = calib.phi(c.family, c.quantity, bin, self.raw(c, evidence, component))
        return s if c.orientation > 0 else 1 - s

    def gates(self, c, evidence, calib, state, component):
        gates = super().gates(c, evidence, calib, state, component)
        if c.quantity != "scene_lightness":
            gates["support"] = _support_gate(c.args[0], evidence, state)
        return gates


class Presence(Scorer):
    def raw(self, c, evidence, component):
        return evidence.objects.get(c.args[0], {}).get("h")

    def score(self, c, evidence, calib, bin, component):
        return evidence.objects[c.args[0]]["h"]


class Binding(Scorer):
    def _key(self, c):
        return ("binding", c.args[0], c.args[1])

    def raw(self, c, evidence, component):
        return evidence.pairs.get(self._key(c))

    def domain(self, c, calib, bin, component):
        return (-1.0, 1.0)

    def score(self, c, evidence, calib, bin, component):
        return evidence.pairs[self._key(c)]

    def gates(self, c, evidence, calib, state, component):
        info = evidence.raw_ok.get(self._key(c), False)
        ok = info and calib.attention_ok("objects") and calib.attention_ok("attributes")
        obj = evidence.objects.get(c.args[1])
        present = obj is None or float(obj["p"]) >= state.p_gate
        return {"presence": present, "support": True, "confidence": ok}


class Leakage(Scorer):
    def _key(self, c):
        return ("leakage", c.args[0], c.args[2])

    def raw(self, c, evidence, component):
        return evidence.pairs.get(self._key(c))

    def domain(self, c, calib, bin, component):
        return (0.0, 1.0)

    def score(self, c, evidence, calib, bin, component):
        return evidence.pairs[self._key(c)]

    def gates(self, c, evidence, calib, state, component):
        other = c.args[2]
        obj = evidence.objects.get(other)
        present = obj is not None and float(obj["p"]) >= state.p_gate
        return {"presence": present, "support": _support_gate(other, evidence, state), "confidence": True}


class Mixture(Scorer):
    """F6: mixture fraction on the colour axis, with upper bounds that define a blend."""

    def tapered(self, component):
        return component == "main"

    def components(self, c):
        return ["main", "e", "varpi", "zeta"]

    def _stats(self, c, evidence):
        return evidence.objects.get(c.args[0], {}).get(mix_key(c.quantity))

    def raw(self, c, evidence, component):
        stats = self._stats(c, evidence)
        return None if stats is None else stats["rho" if component == "main" else component]

    def domain(self, c, calib, bin, component):
        return calib.domain(c.family, c.quantity, bin) if component == "main" else (-math.inf, math.inf)

    def score(self, c, evidence, calib, bin, component):
        value = self.raw(c, evidence, component)
        return calib.phi(c.family, c.quantity, bin, value) if component == "main" else value

    def interval(self, c, calib, component):
        if component == "main":
            return calib.interval(c.family, c.quantity, c.hedge, c.orientation)
        high, kappa = calib.bound(component)
        return (-math.inf, high, kappa)

    def gates(self, c, evidence, calib, state, component):
        gates = super().gates(c, evidence, calib, state, component)
        gates["support"] = _support_gate(c.args[0], evidence, state)
        return gates


FAMILY_SCORERS: Dict[str, Scorer] = {"F2a": Photometric(), "presence": Presence(), "binding": Binding(),
                                     "leakage": Leakage(), "F6": Mixture()}


def scorer_for(c: Constraint) -> Optional[Scorer]:
    return FAMILY_SCORERS.get(c.family)


def _inside(raw, lo, hi):
    if raw is None:
        return False
    value = torch.as_tensor(raw).detach()
    return bool(torch.isfinite(value)) and lo <= float(value) <= hi


def freeze_step(program, evidence, calib, bin, state: StepState, targets=None) -> List[Frozen]:
    """Fix active set, tiers, weights, intervals, and bin for this step (Alg. guided_step)."""
    if state.first_step is None:
        state.first_step = state.step
    out = []
    for cid, c in enumerate(program.constraints):
        sc = scorer_for(c)
        if sc is None:
            state.unguided.add(c.family)
            continue
        valid = calib.valid(c.family, bin)
        if valid and c.family not in state.first_valid:
            state.first_valid[c.family] = state.step
        for component in sc.components(c):
            gates = {"validity": valid, **sc.gates(c, evidence, calib, state, component)}
            raw = sc.raw(c, evidence, component)
            lo, hi = sc.domain(c, calib, bin, component)
            gates["domain"] = _inside(raw, lo, hi)
            low, high, kappa = sc.interval(c, calib, component)
            if targets and cid in targets:
                low, high = targets[cid]
            weight = 0.0
            if all(gates.values()):
                dom = domain_weight(torch.as_tensor(raw).detach(), lo, hi, state.band) if sc.tapered(component) else 1.0
                since = state.first_valid[c.family]
                ramp = 1.0 if since == state.first_step else ramp_weight(state.step - since, state.ramp)
                weight = state.lam * dom * ramp * _support_weight(c, evidence)
            out.append(Frozen(c, cid, c.tier, bin, float(weight), low, high, kappa, gates, component))
    return out


def _support_weight(c, evidence):
    obj = evidence.objects.get(c.args[0]) if c.args else None
    return 1.0 if obj is None else float(obj.get("psi", 1.0))


def score(f: Frozen, evidence, calib):
    return scorer_for(f.constraint).score(f.constraint, evidence, calib, f.bin, f.component)


def in_domain(f: Frozen, evidence, calib) -> bool:
    """Raw inputs inside D_k, checked at a candidate before any domain-limited function runs."""
    sc = scorer_for(f.constraint)
    lo, hi = sc.domain(f.constraint, calib, f.bin, f.component)
    return _inside(sc.raw(f.constraint, evidence, f.component), lo, hi)


def tier_losses(frozen: List[Frozen], evidence, calib) -> Dict[int, torch.Tensor]:
    """L^(m) = sum lambda_k L_k over positively weighted constraints of each tier."""
    losses: Dict[int, torch.Tensor] = {}
    for f in frozen:
        if f.weight > 0:
            term = f.weight * charbonnier(violation(score(f, evidence, calib), f.low, f.high), f.kappa)
            losses[f.tier] = losses[f.tier] + term if f.tier in losses else term
    return losses
