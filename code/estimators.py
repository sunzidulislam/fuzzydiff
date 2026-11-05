"""Frozen visual estimators on the clean prediction (paper: Tab. estimators).

Analytic colour statistics take images in [0, 1] as [3, H, W] or [1, 3, H, W]. Support weights enter
photometric statistics with stop-gradient, so guidance changes pixel colours inside the object and
not the mask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS_C = 1e-6
_RGB_TO_XYZ = [[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]]
_WHITE_D65 = (0.95047, 1.0, 1.08883)


def _chw(x):
    return x[0] if x.dim() == 4 else x


def pool_grid(x, size=64):
    return F.adaptive_avg_pool2d(_chw(x).unsqueeze(0), size)[0]


def srgb_to_cielab(x):
    """sRGB -> CIELAB (D65), differentiable everywhere (both power laws have linear toes).

    No clamp here: Backend.decode already clamps with a straight-through gradient, and a second
    clamp would zero the gradient of saturated pixels."""
    x = _chw(x)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)
    m = torch.as_tensor(_RGB_TO_XYZ, dtype=x.dtype, device=x.device)
    xyz = torch.einsum("ij,jhw->ihw", m, lin)
    white = torch.as_tensor(_WHITE_D65, dtype=x.dtype, device=x.device).view(3, 1, 1)
    t = xyz / white
    d = 6 / 29
    f = torch.where(t > d ** 3, t.clamp_min(d ** 3) ** (1 / 3), t / (3 * d ** 2) + 4 / 29)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def scene_lightness(x):
    """Mean L* of the whole image on the 64^2 grid (F2a scene lightness)."""
    return srgb_to_cielab(pool_grid(x))[0].mean()


def object_photometric(x, s_core, quantity):
    """Core-weighted mean L* ("lightness") or C*_ab ("chroma") on the 64^2 grid."""
    w = s_core.detach().to(x.device, x.dtype)
    lab = srgb_to_cielab(pool_grid(x, w.shape[-1]))
    if quantity == "lightness":
        value = lab[0]
    elif quantity == "chroma":
        value = torch.sqrt(lab[1] ** 2 + lab[2] ** 2 + 1e-8)
    else:
        raise ValueError(f"unknown photometric quantity {quantity}")
    return (value * w).sum() / (w.sum() + EPS_C)


def erode_core(s, k=3):
    """S_core = -maxpool_k(-S): excludes boundary and background bleed."""
    return -F.max_pool2d(-s[None, None], k, stride=1, padding=k // 2)[0, 0]
