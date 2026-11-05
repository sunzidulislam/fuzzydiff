"""Synthetic control for the F6 color statistics (paper: Tab. color_synthetic).

Patches with known ground truth are rendered in OKLab with realistic nuisances (shading gradient,
specular highlight, pixel noise, sRGB gamut clipping) for six color pairs:
  U  uniform blend at fraction rho in [0.2, 0.8]                         (should read as blend)
  T  two-tone: straight split or stripes (6-24 px), area fraction f        (should read as pattern)
  M  mottled: thresholded smooth noise with grain scale s in {1,2,4,8} px  (scale-dependent)
  O  off-axis two-tone: two colors Cbar +/- delta n with n perpendicular to the axis, so both
     project to the same axial position (the case the axial index varpi cannot see)
All statistics come from colour_stats.mixture_stats (one blurred image, one weighting).
Measured: AUROC of the pattern index varpi for U vs T; mean varpi of M per grain scale; the
mean-only failure (rho of T equals that of a blend with rho = f); and rho accuracy on U with and
without the soft trimming of shadows/highlights. These are controls of the statistic, not of
perception; the human blend / two-tone / mottled labels of App. A (F6) decide validity.
Usage: python -I test_color_unimodality.py [--n 40] [--out DIR]
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_stats import _axis, _blur, mixture, mixture_stats, oklab_to_srgb, pattern_index, srgb_to_oklab  # noqa: E402

PAIRS = {  # (C1, C2) as sRGB; C1 is the dominant/base color of the term
    "greenish-gray": ((0.50, 0.50, 0.50), (0.10, 0.60, 0.20)),
    "turquoise-green": ((0.10, 0.60, 0.20), (0.25, 0.88, 0.82)),
    "coral-orange": ((1.00, 0.55, 0.00), (1.00, 0.50, 0.31)),
    "bluish-purple": ((0.50, 0.20, 0.60), (0.10, 0.30, 0.90)),
    "pinkish-white": ((0.95, 0.95, 0.95), (1.00, 0.75, 0.80)),
    "reddish-brown": ((0.45, 0.27, 0.10), (0.80, 0.10, 0.10)),
}
H = W = 96


def auroc(neg, pos):
    n = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return n / (len(pos) * len(neg))


def nuisance(lab, g):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, dtype=lab.dtype), torch.linspace(-1, 1, W, dtype=lab.dtype), indexing="ij")
    th = torch.rand(1, generator=g).item() * 2 * math.pi
    shade = 0.92 + 0.15 * (math.cos(th) * xx + math.sin(th) * yy)          # shading gradient
    cx, cy = (torch.rand(2, generator=g) * 1.2 - 0.6).tolist()
    spec = 0.6 * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02)       # specular highlight
    out = lab.clone()
    out[0] = out[0] * shade
    out[0] = out[0] + spec * (1 - out[0])
    out[1:] = out[1:] * (1 - spec)
    out = out + 0.01 * torch.randn(out.shape, generator=g, dtype=lab.dtype)
    return srgb_to_oklab(oklab_to_srgb(out))                                # gamut clipping


def mask():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, dtype=torch.float64), torch.linspace(-1, 1, W, dtype=torch.float64), indexing="ij")
    return ((xx / 0.85) ** 2 + (yy / 0.75) ** 2 <= 1).to(torch.float64)


def field(kind, c1, c2, g, param):
    lab = torch.empty(3, H, W, dtype=torch.float64)
    if kind == "U":
        col = c1 + param * (c2 - c1)
        lab[:] = col.view(3, 1, 1)
        return lab, param
    if kind == "T":
        f, stripe = param
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
        th = torch.rand(1, generator=g).item() * math.pi
        u = math.cos(th) * xx + math.sin(th) * yy
        sel = ((u / stripe) % 1.0) < f if stripe else (u - u.min()) / (u.max() - u.min()) < f
    else:  # mottled
        f, s = param
        z = _blur(torch.randn(3, H, W, generator=g, dtype=torch.float64), s)[0]
        sel = z < torch.quantile(z, f)
    lab[:] = torch.where(sel.unsqueeze(0), c2.view(3, 1, 1), c1.view(3, 1, 1))
    return lab, sel.double().mean().item()


def check_bounds(trials=300):
    """Property test of the bound: varpi in [0, 1] for arbitrary images, weights, and prototypes;
    varpi ~ 1 for a binary two-tone field and = 0 for a constant field."""
    g = torch.Generator().manual_seed(1)
    for _ in range(trials):
        lab = srgb_to_oklab(torch.rand(3, 24, 24, generator=g, dtype=torch.float64))
        w = torch.rand(24, 24, generator=g, dtype=torch.float64)
        c1 = srgb_to_oklab(torch.rand(3, 1, 1, generator=g, dtype=torch.float64)).view(3)
        c2 = srgb_to_oklab(torch.rand(3, 1, 1, generator=g, dtype=torch.float64)).view(3)
        if torch.linalg.norm(c2 - c1) < 1e-3:
            continue
        v = pattern_index(lab, w, c1, c2, sigma_p=float(torch.rand(1, generator=g)) * 3)
        assert -1e-12 <= v.item() <= 1.0, v.item()
    c1 = srgb_to_oklab(torch.tensor([0.1, 0.6, 0.2], dtype=torch.float64).view(3, 1, 1)).view(3)
    c2 = srgb_to_oklab(torch.tensor([0.9, 0.2, 0.2], dtype=torch.float64).view(3, 1, 1)).view(3)
    half = torch.zeros(3, 24, 24, dtype=torch.float64)
    half[:, :, :12] = c1.view(3, 1, 1)
    half[:, :, 12:] = c2.view(3, 1, 1)
    ones = torch.ones(24, 24, dtype=torch.float64)
    assert pattern_index(half, ones, c1, c2, sigma_p=0, trim=0.0).item() > 0.99
    const = (c1 + 0.4 * (c2 - c1)).view(3, 1, 1).expand(3, 24, 24).clone()
    assert pattern_index(const, ones, c1, c2, sigma_p=0, trim=0.0).item() < 1e-9
    return trials


def offaxis(c1, c2, g, rho, delta_rel=0.5):
    """Two-tone surface whose two colors lie symmetrically off the axis at axial position rho."""
    d = _axis(c1, c2)
    v = (c2 - c1)
    n = torch.zeros(3, dtype=torch.float64)
    vd = v[d]
    if vd.numel() == 2:                       # chromatic plane: rotate v_ab by 90 degrees
        n[1], n[2] = -vd[1], vd[0]
    else:                                     # full space: any direction orthogonal to v
        n = torch.linalg.cross(v, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    n = n / torch.linalg.norm(n) * delta_rel * torch.linalg.norm(vd)
    center = c1 + rho * v
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    th = torch.rand(1, generator=g).item() * math.pi
    sel = ((math.cos(th) * xx + math.sin(th) * yy) / 16 % 1.0) < 0.5
    lab = torch.where(sel.unsqueeze(0), (center + n).view(3, 1, 1), (center - n).view(3, 1, 1))
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--sigma-p", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    m = mask()
    res = {"sigma_p": a.sigma_p, "n_per_class": a.n, "pairs": {}, "bound_property_trials": check_bounds()}
    allU, allT, mott = [], [], {1: [], 2: [], 4: [], 8: []}
    allUz, allTz, allO, allOz = [], [], [], []
    by_stripe = {0: [], 6: [], 12: [], 24: []}
    rho_err_trim, rho_err_raw, mean_only = [], [], []
    for name, (s1, s2) in PAIRS.items():
        c1 = srgb_to_oklab(torch.tensor(s1, dtype=torch.float64).view(3, 1, 1)).view(3)
        c2 = srgb_to_oklab(torch.tensor(s2, dtype=torch.float64).view(3, 1, 1)).view(3)
        U, T, O, Uz, Tz, Oz = [], [], [], [], [], []
        for i in range(a.n):
            rho = 0.2 + 0.6 * torch.rand(1, generator=g).item()
            lab, _ = field("U", c1, c2, g, rho)
            lab = nuisance(lab, g)
            st = mixture_stats(lab, m, c1, c2, a.sigma_p)
            U.append(st["varpi"].item())
            Uz.append(st["zeta"].item())
            r_t, _, _ = mixture(lab, m, c1, c2, trim=0.1)
            r_r, _, _ = mixture(lab, m, c1, c2, trim=0.0)
            rho_err_trim.append(abs(r_t.item() - rho))
            rho_err_raw.append(abs(r_r.item() - rho))
            f = 0.3 + 0.4 * torch.rand(1, generator=g).item()
            stripe = [0, 6, 12, 24][i % 4]
            lab, frac = field("T", c1, c2, g, (f, stripe))
            lab = nuisance(lab, g)
            st = mixture_stats(lab, m, c1, c2, a.sigma_p)
            T.append(st["varpi"].item())
            Tz.append(st["zeta"].item())
            by_stripe[stripe].append(T[-1])
            lab = nuisance(offaxis(c1, c2, g, 0.2 + 0.6 * torch.rand(1, generator=g).item()), g)
            st = mixture_stats(lab, m, c1, c2, a.sigma_p)
            O.append(st["varpi"].item())
            Oz.append(st["zeta"].item())
            r_t, _, _ = mixture(lab, m, c1, c2, trim=0.1)
            labU = nuisance(field("U", c1, c2, g, frac)[0], g)
            r_u, _, _ = mixture(labU, m, c1, c2, trim=0.1)
            mean_only.append(abs(r_t.item() - r_u.item()))
        for s in mott:
            for i in range(a.n // 4):
                lab, _ = field("M", c1, c2, g, (0.5, s))
                mott[s].append(pattern_index(nuisance(lab, g), m, c1, c2, a.sigma_p).item())
        res["pairs"][name] = {"auroc_blend_vs_twotone": auroc(U, T), "varpi_U_mean": sum(U) / len(U),
                              "varpi_T_mean": sum(T) / len(T),
                              "auroc_varpi_blend_vs_offaxis": auroc(U, O),
                              "auroc_zeta_blend_vs_offaxis": auroc(Uz, Oz),
                              "axis_length_ab": float(torch.linalg.norm((c2 - c1)[1:]))}
        allU += U
        allT += T
        allUz += Uz
        allTz += Tz
        allO += O
        allOz += Oz
    # acceptance as a blend with upper bounds at the 95th percentile of the uniform blends (in-sample)
    q95 = lambda v: sorted(v)[int(0.95 * len(v)) - 1]
    uv, uz = q95(allU), q95(allUz)
    acc = lambda vs, zs, use_z: sum((x <= uv) and (not use_z or y <= uz) for x, y in zip(vs, zs)) / len(vs)
    res["acceptance_as_blend"] = {
        "varpi_only": {"U": acc(allU, allUz, False), "T": acc(allT, allTz, False), "O": acc(allO, allOz, False)},
        "varpi_and_zeta": {"U": acc(allU, allUz, True), "T": acc(allT, allTz, True), "O": acc(allO, allOz, True)}}
    res["overall"] = {"auroc_blend_vs_twotone": auroc(allU, allT), "varpi_U_mean": sum(allU) / len(allU),
                      "auroc_varpi_blend_vs_offaxis": auroc(allU, allO), "auroc_zeta_blend_vs_offaxis": auroc(allUz, allOz),
                      "varpi_T_mean": sum(allT) / len(allT),
                      "mottled_varpi_by_grain_px": {str(s): sum(v) / len(v) for s, v in mott.items()},
                      "twotone_varpi_by_stripe_px": {("split" if k == 0 else str(k)): sum(v) / len(v) for k, v in by_stripe.items()},
                      "rho_mae_trimmed": sum(rho_err_trim) / len(rho_err_trim),
                      "rho_mae_untrimmed": sum(rho_err_raw) / len(rho_err_raw),
                      "mean_only_rho_gap_twotone_vs_blend": sum(mean_only) / len(mean_only)}

    rows = []
    for name, r in res["pairs"].items():
        rows.append(f"{name} & {r['axis_length_ab']:.2f} & {r['varpi_U_mean']:.2f} & {r['varpi_T_mean']:.2f} & {r['auroc_blend_vs_twotone']:.2f} & "
                    f"{r['auroc_varpi_blend_vs_offaxis']:.2f} & {r['auroc_zeta_blend_vs_offaxis']:.2f}\\\\")
    o = res["overall"]
    rows.append(f"\\midrule All pairs & -- & {o['varpi_U_mean']:.2f} & {o['varpi_T_mean']:.2f} & {o['auroc_blend_vs_twotone']:.2f} & "
                f"{o['auroc_varpi_blend_vs_offaxis']:.2f} & {o['auroc_zeta_blend_vs_offaxis']:.2f}\\\\")
    ab = res["acceptance_as_blend"]
    pc = lambda x: f"{100 * x:.0f}\\%"
    mrow = " / ".join(f"{v:.2f}" for v in o["mottled_varpi_by_grain_px"].values())
    trow = " / ".join(f"{v:.2f}" for v in o["twotone_varpi_by_stripe_px"].values())
    with open(os.path.join(a.out, "color_synthetic.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out, "color_synthetic_rows.tex"), "w") as f:
        f.write("% Generated by test_color_unimodality.py; do not edit.\n")
        f.write("\\def\\colorSynthRows{%\n" + "\n".join(rows) + "\n}\n")
        f.write(f"\\def\\colorSynthMottled{{{mrow}}}\n")
        f.write(f"\\def\\colorSynthStripes{{{trow}}}\n")
        f.write(f"\\def\\colorSynthRhoTrim{{{o['rho_mae_trimmed']:.3f}}}\n")
        f.write(f"\\def\\colorSynthRhoRaw{{{o['rho_mae_untrimmed']:.3f}}}\n")
        f.write(f"\\def\\colorSynthMeanGap{{{o['mean_only_rho_gap_twotone_vs_blend']:.3f}}}\n")
        f.write(f"\\def\\colorSynthN{{{a.n}}}\n")
        f.write(f"\\def\\colorSynthAccVarpi{{U {pc(ab['varpi_only']['U'])}, T {pc(ab['varpi_only']['T'])}, O {pc(ab['varpi_only']['O'])}}}\n")
        f.write(f"\\def\\colorSynthAccBoth{{U {pc(ab['varpi_and_zeta']['U'])}, T {pc(ab['varpi_and_zeta']['T'])}, O {pc(ab['varpi_and_zeta']['O'])}}}\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
