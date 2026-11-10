"""Turn annotations and run logs into a fitted calibration file.

Usage:
  python -I code/fit_calibration.py --input calibration_input.json --out calibration.json

The input file has one list per section. `rows_from_run` builds the sections that come from run logs
joined with annotations (ratings, presence, size, attention). Sections that have no annotation yet
leave their family not valid, so it gets no guidance: nothing is filled in silently.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_fit import (FittedCalibration, contiguous, cross_fit, fit_gamma, fit_hedge, fit_phi, fit_presence,  # noqa: E402
                             fit_size_quantiles, oof_size_coverage, pool_orientations, select_attention, select_binding,
                             select_drift, validity_flags, _auroc)
from color_stats import srgb_to_oklab  # noqa: E402
from prompts import base_prompt  # noqa: E402
from reporting import LeakageError, cluster_bootstrap, ece, read_jsonl, spearman  # noqa: E402
from settings import load_settings  # noqa: E402

BINS = ("1", "2", "3", "4")
EPS = 1e-6


def _is_perceptual(r):
    """Only a score with a perceptual map: F6's upper bounds are native quantities, not degrees."""
    return r["family"] not in ("presence", "binding", "leakage", "F8") and r.get("component", "main") == "main"


def _n_images(rows):
    """Validity counts images, not per-step rows."""
    return len({r.get("item", i) for i, r in enumerate(rows)})


def rows_from_run(run_dir, ratings=None, presence_labels=None, masks=None):
    """Join per-step raw values with annotations of the final image of each item (arm/image id)."""
    ratings, presence_labels, masks = ratings or {}, presence_labels or {}, masks or {}
    out = {"ratings": [], "presence": [], "size": [], "attention_rows": []}
    for log_path in Path(run_dir).glob("*/*/log.jsonl"):
        item = f"{log_path.parts[-3]}/{log_path.parts[-2]}"
        with open(log_path.parent / "meta.json") as f:
            meta = json.load(f)
        if "failed" in meta:
            continue
        base = base_prompt(meta["prompt"])
        human = masks.get(item, {})
        for r in read_jsonl(log_path):
            if r.get("split") == "test":
                raise LeakageError(f"fitting may not read test records: {item}")
            if r["type"] == "final" and r.get("raw") is not None and item in ratings and _is_perceptual(r):
                out["ratings"].append({"family": r["family"], "quantity": r.get("quantity"), "bin": r["bin"],
                                       "e": r["raw"], "rating": ratings[item], "base": base, "item": item})
            elif r["type"] == "constraint" and r.get("raw") is not None:
                if item in ratings and _is_perceptual(r):
                    out["ratings"].append({"family": r["family"], "quantity": r.get("quantity"), "bin": r["bin"],
                                           "e": r["raw"], "rating": ratings[item], "base": base, "item": item})
                if r["family"] == "presence" and r.get("object") in presence_labels.get(item, {}):
                    out["presence"].append({"bin": r["bin"], "global": r["global"], "region": r["region"],
                                            "label": presence_labels[item][r["object"]], "base": base, "item": item})
                if r["family"] == "F8" and r.get("objects", [None])[0] in human and r["objects"][1] in human:
                    a, b = human[r["objects"][0]], human[r["objects"][1]]
                    m_human = r["orientation"] * (np.log(a.mean() + EPS) - np.log(b.mean() + EPS))
                    out["size"].append({"A_q": r["areas"][0], "A_r": r["areas"][1], "bin": r["bin"], "estimate": r["raw"],
                                        "human": float(m_human), "abs_error": abs(r["raw"] - float(m_human)), "base": base, "item": item})
            elif r["type"] == "step":
                for phrase, a in r.get("attention", {}).items():
                    head = phrase.split("#")[0]
                    if head in human:
                        y, x = a["peak_xy"]
                        out["attention_rows"].append({"base": base, "kind": "objects", "config": "read_blocks",
                                                      "peak": a["peak"], "share": a["share"], "mass": a["mass"],
                                                      "correct": bool(human[head][y, x] > 0.5)})
    return out


def _qkey(q):
    return ":".join(q) if isinstance(q, (list, tuple)) else str(q)


def _spearman_lb(rows, x, y):
    if len(rows) < 4:
        return -1.0
    lo, _ = cluster_bootstrap(rows, lambda rs: spearman([r[x] for r in rs], [r[y] for r in rs]), key="base", n=300)
    return lo


def _fit_perceptual(rows, s, min_images):
    phi, valid = {}, {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], _qkey(r["quantity"]), str(r["bin"]))].append(r)
    cells = defaultdict(dict)
    eps = s.get("cfgEpsPhi")
    for (family, quantity, b), cell in groups.items():
        def fit(train):
            return fit_phi([r["e"] for r in train], [r["rating"] for r in train], [r["base"] for r in train], eps)

        def predict(model, test):
            lo, hi = model.domain
            return [{"pred": float(model(torch.tensor(min(max(r["e"], lo), hi), dtype=torch.float64))),
                     "rating": r["rating"], "base": r["base"]} for r in test]

        oof = cross_fit(cell, fit, predict)
        cells[family].setdefault(quantity, {})[b] = {"n": _n_images(cell), "rho_lb": _spearman_lb(oof, "pred", "rating")}
        phi.setdefault(family, {}).setdefault(quantity, {})[b] = fit(cell).to_dict()
    for family, per_quantity in cells.items():
        flags_all = {b: True for b in BINS}
        for stats in per_quantity.values():
            flags = validity_flags(stats, min_images, s.get("cfgNposMin"), s.get("cfgRhoMin"), s.get("cfgAUCMin"),
                                   s.get("cfgECEMax"))
            for b in BINS:
                flags_all[b] = flags_all[b] and flags.get(b, False)
        valid[family] = contiguous(flags_all)
    return phi, valid, {f: dict(q) for f, q in cells.items()}


def _fit_presence_section(rows, s, min_images):
    models, stats = {}, {}
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[str(r["bin"])].append(r)
    for b, cell in by_bin.items():
        arrays = ([r["global"] for r in cell], [r["region"] for r in cell], [r["label"] for r in cell], [r["base"] for r in cell])
        if len(set(arrays[2])) < 2:
            continue
        models[b] = fit_presence(*arrays).to_dict()

        def fit(train):
            return fit_presence([r["global"] for r in train], [r["region"] for r in train], [r["label"] for r in train],
                                [r["base"] for r in train])

        def predict(model, test):
            return [{"p": float(torch.sigmoid(torch.tensor(model.logit(r["global"], r["region"])))), "label": r["label"],
                     "base": r["base"]} for r in test]

        oof = cross_fit(cell, fit, predict)
        auc_lb, _ = cluster_bootstrap(oof, lambda rs: _auroc([r["p"] for r in rs], [r["label"] for r in rs]), key="base", n=300)
        _, ece_ub = cluster_bootstrap(oof, lambda rs: ece([r["p"] for r in rs], [r["label"] for r in rs]), key="base", n=300)
        images = {r.get("item", i): r["label"] for i, r in enumerate(cell)}
        stats[b] = {"n": len(images), "npos": sum(images.values()), "nneg": len(images) - sum(images.values()),
                    "auc_lb": auc_lb, "ece_ub": ece_ub}
    flags = validity_flags(stats, min_images, s.get("cfgNposMin"), s.get("cfgRhoMin"), s.get("cfgAUCMin"), s.get("cfgECEMax"))
    return models, contiguous({b: flags.get(b, False) for b in BINS})


def _fit_acceptance(rows, sigma_m, s):
    """Intervals [l, u, kappa_eff] for guidance, and per interval the rest of its fit (plateau, one-sided,
    held-out likelihood, gamma) for reporting."""
    intervals, details = {}, {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], _qkey(r["quantity"]), str(r["hedge"]), r["orientation"])].append(r)
    fits = {}
    for key, cell in groups.items():
        values = [r["s"] for r in cell]
        scale = (0.0, 1.0) if key[0] != "F8" else (0.0, max(values))
        fits[key] = (fit_hedge(values, [r["accept"] for r in cell], scale=scale), cell, scale)
    for (family, quantity, hedge, orientation), (fit, cell, scale) in fits.items():
        gamma = 0.0
        # F8 is widened at run time by the pair's own c68 (constraints.Size.resolve); widening it here too
        # would count the size estimator's error twice
        if family in sigma_m and family != "F8" and all("s_hat" in r for r in cell):
            gamma = fit_gamma([r["s_hat"] for r in cell], [r["accept"] for r in cell], fit, sigma_m[family],
                              [r["base"] for r in cell], s=[r["s"] for r in cell], scale=scale)
        kappa = float(np.sqrt(fit.kappa ** 2 + (gamma * sigma_m.get(family, 0.0)) ** 2))
        key = f"{family}|{quantity}|{hedge}|{orientation}"
        intervals[key] = [fit.l, fit.u, kappa]
        details[key] = {"plateau": fit.plateau, "one_sided": fit.one_sided, "heldout_ll": fit.heldout_ll, "gamma": gamma}
        other = fits.get((family, quantity, hedge, -orientation))
        if family == "F8" and orientation == 1 and other is not None:
            data = ([r["s"] for r in cell], [r["accept"] for r in cell])
            data2 = ([r["s"] for r in other[1]], [r["accept"] for r in other[1]])
            arrays = (tuple(map(np.asarray, data)), tuple(map(np.asarray, data2)))
            if pool_orientations(fit, other[0], arrays):
                joined = fit_hedge(np.concatenate([arrays[0][0], arrays[1][0]]), np.concatenate([arrays[0][1], arrays[1][1]]),
                                   scale=(0.0, max(np.max(arrays[0][0]), np.max(arrays[1][0]))))
                pooled = f"{family}|{quantity}|{hedge}|pooled"
                intervals[pooled] = [joined.l, joined.u, kappa]
                details[pooled] = {"plateau": joined.plateau, "one_sided": joined.one_sided,
                                   "heldout_ll": joined.heldout_ll, "gamma": gamma}
    return intervals, details


def _heuristics(s):
    half, kappa = s.get("hedge_halfwidth"), s.get("hedge_kappa")
    out = {h: [max(c - half, 0.0), min(c + half, 1.0), kappa] for h, c in s.get("hedge_centers", dict).items()}
    d = s.get("default_center")
    out["None"] = [max(d - half, 0.0), min(d + half, 1.0), kappa]
    return out


def fit_main(input_path, out_path, settings=None, min_images=None):
    s = settings or load_settings()
    with open(input_path) as f:
        data = json.load(f)
    n_min = min_images if min_images is not None else s.get("cfgNmin")
    parts = {"valid": {}, "stage2_valid": {b: False for b in BINS},
             "fallback_intervals": _heuristics(s),
             "bounds": {name: [s.get(f"f6_{name}_max"), s.get("f6_bound_kappa")] for name in ("e", "varpi", "zeta", "dl")},
             "leakage_scale": s.get("leakage_tau"), "psi": s.get("psi"),
             "size": {"c68": s.get("size_c68"), "c95": s.get("size_c95")},
             "prototypes": {t: srgb_to_oklab(torch.tensor(rgb, dtype=torch.float64).view(3, 1, 1)).view(3).tolist()
                            for t, rgb in s.get("prototypes", dict).items()},
             "drift": {"small": None, "medium": None, "large": None},
             "attention": {k: {"feasible": False, "kappa_pk": s.get("cfgPeak"), "m_min": s.get("cfgMassMin"),
                               "tau_min": s.get("cfgTauMin")} for k in ("objects", "attributes")},
             "binding_ok": False,
             "default_intervals": {"presence": [float(np.log(s.get("cfgPresTarget") / (1 - s.get("cfgPresTarget")))), None,
                                                s.get("cfgPresKappa")],
                                   "binding": [s.get("cfgBindMargin"), None, s.get("binding_kappa")],
                                   "leakage": [None, s.get("leakage_max"), s.get("leakage_kappa")]}}
    parts["prototypes"].update(data.get("prototypes", {}))
    parts["bounds"].update(data.get("bounds", {}))
    if "leakage_scale" in data:
        parts["leakage_scale"] = data["leakage_scale"]
    if "psi" in data:
        parts["psi"] = data["psi"]
    parts["provisional_parts"] = [k for k in ("prototypes", "bounds", "leakage_scale", "psi", "size") if k not in data]
    parts["phi"], parts["cells"] = {}, {}
    if data.get("ratings"):
        parts["phi"], parts["valid"], parts["cells"] = _fit_perceptual(data["ratings"], s, n_min)
    if data.get("leakage_cells"):
        flags = validity_flags(data["leakage_cells"], n_min, s.get("cfgNposMin"), s.get("cfgRhoMin"), s.get("cfgAUCMin"), s.get("cfgECEMax"))
        parts["valid"]["leakage"] = contiguous({b: flags.get(b, False) for b in BINS})
    if data.get("presence"):
        parts["presence"], parts["valid"]["presence"] = _fit_presence_section(data["presence"], s, n_min)
    if data.get("acceptance"):
        parts["intervals"], parts["interval_fits"] = _fit_acceptance(data["acceptance"], data.get("sigma_m", {}), s)
    models = None
    if data.get("size"):
        rows = data["size"]
        features, errors, bases = [[r["A_q"], r["A_r"]] for r in rows], [r["abs_error"] for r in rows], [r["base"] for r in rows]
        models = fit_size_quantiles(features, errors, bases)
        by_bin = defaultdict(list)
        for r in rows:
            by_bin[str(r["bin"])].append(r)
        bins = np.array([str(r["bin"]) for r in rows])
        parts["size_coverage"] = oof_size_coverage(features, errors, bases, {f"bin {b}": bins == b for b in by_bin})
        stats = {b: {"n": _n_images(c), "rho_lb": _spearman_lb(c, "estimate", "human")} for b, c in by_bin.items()}
        flags = validity_flags(stats, n_min, s.get("cfgNposMin"), s.get("cfgRhoMin"), s.get("cfgAUCMin"), s.get("cfgECEMax"))
        parts["valid"]["F8"] = contiguous({b: flags.get(b, False) for b in BINS})
    if data.get("attention_rows"):
        chosen = select_attention(data["attention_rows"], data.get("attention_candidates", []), s.get("attention_required"))
        for kind, c in chosen.items():
            parts["attention"][kind] = {"feasible": c.feasible, "kappa_pk": c.kappa_pk or s.get("cfgPeak"),
                                        "m_min": c.m_min or s.get("cfgMassMin"), "tau_min": c.tau_min if c.tau_min is not None else s.get("cfgTauMin")}
    if data.get("binding"):
        margin, ok = select_binding(data["binding"], s.get("cfgBindAUROC"))
        parts["binding_ok"] = ok
        parts["default_intervals"]["binding"][0] = margin
        parts["valid"]["binding"] = {b: ok for b in BINS}
    if data.get("drift"):
        chosen = select_drift(data["drift"], s.get("cfgFDmax"))
        parts["drift"].update({k: None if c is None else [c.theta, c.delta_rel] for k, c in chosen.items()})
    if data.get("stage2_cells"):
        flags = validity_flags(data["stage2_cells"], n_min, s.get("cfgNposMin"), s.get("cfgRhoMin"), s.get("cfgAUCMin"), s.get("cfgECEMax"))
        parts["stage2_valid"] = contiguous({b: flags.get(b, False) for b in BINS})
    parts["not_valid"] = sorted(f for f in ("F2a", "F6", "F8", "presence", "binding", "leakage")
                                if not any(parts["valid"].get(f, {}).values()))
    calib = FittedCalibration(parts, models)
    calib.save(out_path)
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    calib = fit_main(args.input, args.out)
    print(json.dumps({"tag": calib.tag, "provisional_parts": calib.parts["provisional_parts"],
                      "not_valid": calib.parts["not_valid"], "valid": calib.parts["valid"], "drift": calib.parts["drift"],
                      "attention": calib.parts["attention"], "binding_ok": calib.parts["binding_ok"]}, indent=1))


if __name__ == "__main__":
    main()
