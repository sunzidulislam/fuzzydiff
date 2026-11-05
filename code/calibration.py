"""One calibration interface for the guidance loop, provisional or fitted.

ProvisionalCalibration uses simple scales and configs/provisional.yaml, so the pipeline can run
before annotations exist. Its outputs are tagged and refused by reporting.py. The loop never checks
which version it has: fitted calibration replaces it without changing guidance or SAR.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Protocol, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_stats import srgb_to_oklab  # noqa: E402

PERCEPTUAL_100 = {"lightness", "scene_lightness", "chroma"}


class Calibration(Protocol):
    tag: str

    def phi(self, family, quantity, bin, e): ...
    def domain(self, family, quantity, bin) -> Tuple[float, float]: ...
    def valid(self, family, bin) -> bool: ...
    def interval(self, family, quantity, hedge, orientation) -> Tuple[float, float, float]: ...
    def bound(self, name) -> Tuple[float, float]: ...
    def leakage_scale(self) -> float: ...
    def attention_thresholds(self, kind) -> Tuple[float, float, float]: ...
    def presence_logit(self, bin, global_logit, region_logit): ...
    def size_quantiles(self, features) -> Tuple[float, float]: ...
    def prototype(self, term): ...
    def psi(self, phrase, features) -> float: ...
    def drift_thresholds(self, stratum): ...
    def attention_ok(self, kind) -> bool: ...
    def stage2_valid(self, bin) -> bool: ...


def domain_weight(e, lo, hi, band):
    """Eq. appA_domgate: 0 at and beyond the boundary, cosine ramp inside the band, then 1."""
    e = float(e)
    if not lo <= e <= hi:
        return 0.0
    w = band * (hi - lo)
    dist = min(e - lo, hi - e)
    return 1.0 if dist >= w else 0.5 * (1 - math.cos(math.pi * dist / w))


def ramp_weight(steps_since_valid, ramp):
    return 1.0 if ramp <= 0 else min(steps_since_valid / ramp, 1.0)


class ProvisionalCalibration:
    tag = "provisional"

    def __init__(self, settings):
        self.s = settings
        self.centers = settings.get("hedge_centers", dict)

    def _center(self, hedge):
        return self.centers[hedge] if hedge is not None else self.s.get("default_center")

    def phi(self, family, quantity, bin, e):
        return e / 100.0 if quantity in PERCEPTUAL_100 else e

    def domain(self, family, quantity, bin):
        if quantity in PERCEPTUAL_100:
            return (0.0, 100.0)
        return (-0.5, 1.5) if family == "F6" else (-math.inf, math.inf)

    def valid(self, family, bin):
        return True

    def interval(self, family, quantity, hedge, orientation):
        if family == "presence":
            p = self.s.get("cfgPresTarget")
            return (math.log(p / (1 - p)), math.inf, self.s.get("cfgPresKappa"))
        if family == "binding":
            return (self.s.get("cfgBindMargin"), math.inf, self.s.get("binding_kappa"))
        if family == "leakage":
            return (-math.inf, self.s.get("leakage_max"), self.s.get("leakage_kappa"))
        center = self._center(hedge)
        if family == "F8":
            return (center, 2 * center, self.s.get("size_kappa"))
        half = self.s.get("hedge_halfwidth")
        return (max(center - half, 0.0), min(center + half, 1.0), self.s.get("hedge_kappa"))

    def bound(self, name):
        """Upper bounds of the F6 blend definition (e, varpi, zeta)."""
        return (self.s.get(f"f6_{name}_max"), self.s.get("f6_bound_kappa"))

    def leakage_scale(self):
        return self.s.get("leakage_tau")

    def attention_thresholds(self, kind):
        return (self.s.get("cfgPeak"), self.s.get("cfgMassMin"), self.s.get("cfgTauMin"))

    def presence_logit(self, bin, global_logit, region_logit):
        return (global_logit + region_logit) / 2

    def size_quantiles(self, features):
        return (self.s.get("size_c68"), self.s.get("size_c95"))

    def prototype(self, term):
        rgb = self.s.get("prototypes", dict)[term]
        return srgb_to_oklab(torch.tensor(rgb, dtype=torch.float64).view(3, 1, 1)).view(3)

    def psi(self, phrase, features):
        return self.s.get("psi")

    def drift_thresholds(self, stratum):
        return (self.s.get("theta_drift"), self.s.get("cfgDeltaRel"))

    def attention_ok(self, kind):
        return True

    def stage2_valid(self, bin):
        return True
