"""Differentiable color-mixture statistics of F6 (paper: Eq. color_mix, App. A F6).

  * sRGB -> OKLab (Ottosson 2020), differentiable;
  * trimmed core mean: soft weights that suppress the darkest and brightest 10% of core pixels
    in L (quantiles held fixed by stop-gradient);
  * mixture fraction rho and off-axis residual e along the prototype axis C1 -> C2;
  * pattern index varpi: normalized variance of pixel positions along the C1 -> C2 axis
    (Bhatia-Davis bound: in [0, 1], = 1 only for a pure two-tone surface), computed after a
    Gaussian blur at the perceptual integration scale sigma_p (fine mottling that the eye
    integrates counts as a blend, coarse patches do not);
  * orthogonal dispersion zeta: pixel-level spread perpendicular to the axis (two-tone surfaces
    whose colors differ across the axis have zero axial spread but large zeta);
  * all statistics use the same blurred image and the same weights (mixture_stats);
  * the axis lies in the chromatic (a, b) plane when the colors differ in hue/chroma, so shading
    (a change of L) does not move rho.
All functions take float tensors in [0, 1], images as [3, H, W] and masks as [H, W].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_M1 = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]], dtype=torch.float64)
_M2 = torch.tensor([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]], dtype=torch.float64)
_M2I = torch.tensor([[1.0, 0.3963377774, 0.2158037573],
                     [1.0, -0.1055613458, -0.0638541728],
                     [1.0, -0.0894841775, -1.2914855480]], dtype=torch.float64)
_M1I = torch.tensor([[4.0767416621, -3.3077115913, 0.2309699292],
                     [-1.2684380046, 2.6097574011, -0.3413193965],
                     [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=torch.float64)


def _lin(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055).clamp_min(1e-12) ** 2.4)


def _gam(c):
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def srgb_to_oklab(img):
    """img [3, ...] in [0,1] -> OKLab [3, ...]."""
    sh = img.shape
    x = _lin(img.reshape(3, -1).to(torch.float64))
    lms = (_M1.to(x) @ x).clamp_min(1e-12) ** (1 / 3)
    return (_M2.to(x) @ lms).reshape(sh).to(img.dtype)


def oklab_to_srgb(lab):
    sh = lab.shape
    lms = (_M2I.to(torch.float64) @ lab.reshape(3, -1).to(torch.float64)) ** 3
    return _gam(_M1I @ lms).reshape(sh).to(lab.dtype)


def _blur(lab, sigma):
    if sigma <= 0:
        return lab
    r = int(math.ceil(3 * sigma))
    xs = torch.arange(-r, r + 1, dtype=lab.dtype, device=lab.device)
    k = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = lab.unsqueeze(0)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    return x[0]


def trimmed_mean(lab, w, trim=0.1, temp=0.02):
    """Core-weighted mean with soft trimming of the darkest/brightest `trim` share in L."""
    L = lab[0].flatten()
    ww = w.flatten()
    with torch.no_grad():
        order = torch.argsort(L)
        cw = torch.cumsum(ww[order], 0) / ww.sum().clamp_min(1e-12)
        lo = L[order][torch.searchsorted(cw, torch.tensor(trim, dtype=cw.dtype, device=cw.device)).clamp(max=len(L) - 1)]
        hi = L[order][torch.searchsorted(cw, torch.tensor(1 - trim, dtype=cw.dtype, device=cw.device)).clamp(max=len(L) - 1)]
    wt = ww * torch.sigmoid((L - lo) / temp) * torch.sigmoid((hi - L) / temp)
    return (lab.reshape(3, -1) * wt).sum(1) / (wt.sum() + 1e-8)


def _axis(c1, c2, chroma_min=0.5):
    """Mixture axis: the chromatic (a, b) plane when the two colors differ mainly in hue or chroma
    (|v_ab| >= chroma_min |v|), which makes the axis insensitive to shading; full OKLab otherwise
    (lightness mixtures such as gray-white)."""
    v = c2 - c1
    dims = slice(1, 3) if torch.linalg.norm(v[1:]) >= chroma_min * torch.linalg.norm(v) else slice(0, 3)
    return dims


def _weights(lb, w, trim=0.1, temp=0.02):
    """Common weighting distribution omega: core support x soft lightness-trim weights (quantiles of
    the blurred L held fixed), normalized to sum to one. Used by EVERY F6 statistic."""
    L = lb[0].flatten()
    ww = w.flatten()
    core = ww > ww.max() * 1e-3            # quantiles describe the object's core, not a tiny weight elsewhere
    if trim > 0 and core.any():
        with torch.no_grad():
            lo, hi = torch.quantile(L[core], trim), torch.quantile(L[core], 1 - trim)
        ww = ww * torch.sigmoid((L - lo) / temp) * torch.sigmoid((hi - L) / temp)
    return ww / ww.sum().clamp_min(1e-12)


def mixture_stats(lab, w, c1, c2, sigma_p=2.0, trim=0.1, eps=1e-6):
    """All F6 statistics from ONE representation: the OKLab image blurred at the integration scale
    sigma_p, C~ = B_sigma_p * C, and ONE weighting distribution omega (core x trim).

      s_i   = <C~_i - C1, v> / |v|^2            unclipped axial position
      t_i   = clip(s_i, 0, 1)                    clipped position (pattern index)
      r_i   = (C~_i - C1) - s_i v                orthogonal residual of pixel i
      Cbar  = sum omega_i C~_i                   mean color
      rho   = <Cbar - C1, v>/|v|^2 = sum omega_i s_i      mixture fraction
      e     = |Cbar - C1 - rho v| / |v| = |sum omega_i r_i| / |v|   off-axis mean
      varpi = sum omega_i (t_i - tbar)^2 / (tbar (1 - tbar) + eps) in [0, 1)   axial dispersion
      zeta  = sum omega_i |r_i|^2 / |v|^2 >= 0   orthogonal (pixel-level) dispersion
    varpi misses two-tone surfaces whose colors differ perpendicular to the axis (equal s_i);
    zeta catches them. e only sees the mean residual, which cancels for symmetric off-axis pairs."""
    lb = _blur(lab, sigma_p)
    omega = _weights(lb, w, trim)
    d = _axis(c1, c2)
    v = (c2 - c1)[d]
    vv = v @ v
    X = lb.reshape(3, -1)[d] - c1[d].view(-1, 1)                 # [k, N]
    s_ = (X * v.view(-1, 1)).sum(0) / vv                         # axial positions
    R = X - s_.unsqueeze(0) * v.view(-1, 1)                      # orthogonal residuals
    cbar = (lb.reshape(3, -1) * omega).sum(1)
    rho = (omega * s_).sum()
    e = torch.linalg.norm((R * omega).sum(1)) / torch.sqrt(vv)
    t = s_.clamp(0, 1)
    tb = (omega * t).sum()
    varpi = (omega * (t - tb) ** 2).sum() / (tb * (1 - tb) + eps)
    zeta = (omega * (R ** 2).sum(0)).sum() / vv
    return {"rho": rho, "e": e, "varpi": varpi, "zeta": zeta, "cbar": cbar}


def mixture(lab, w, c1, c2, trim=0.1, sigma_p=2.0):
    """rho and off-axis residual e (Eq. color_mix), from the common representation."""
    st = mixture_stats(lab, w, c1, c2, sigma_p, trim)
    return st["rho"], st["e"], st["cbar"]


def pattern_index(lab, w, c1, c2, sigma_p=2.0, trim=0.1, eps=1e-6):
    """Axial pattern index varpi, Eq. (pattern_index); see mixture_stats."""
    return mixture_stats(lab, w, c1, c2, sigma_p, trim, eps)["varpi"]


def orthogonal_dispersion(lab, w, c1, c2, sigma_p=2.0, trim=0.1):
    """Pixel-level orthogonal dispersion zeta, Eq. (orth_dispersion); see mixture_stats."""
    return mixture_stats(lab, w, c1, c2, sigma_p, trim)["zeta"]
