"""Tests of calibration_fit.py on synthetic data with known answers (python -I test_fit.py)."""
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_fit import (FittedCalibration, PhiMap, contiguous, coverage, cross_fit, fit_gamma, fit_hedge,  # noqa: E402
                             fit_phi, fit_presence, fit_size_quantiles, pool_orientations, select_attention,
                             select_binding, select_drift, validity_flags)
from reporting import ece, spearman  # noqa: E402


def true_sigmoid(x):
    return torch.sigmoid((x - 50.0) / 12.0)


def synthetic_sigmoid_ratings(n, seed=0):
    r = np.random.default_rng(seed)
    e = r.uniform(0, 100, n)
    y = np.clip(torch.sigmoid(torch.tensor((e - 50.0) / 12.0)).numpy() + r.normal(0, 0.05, n), 0, 1)
    return e, y, np.array([f"base{i % 200}" for i in range(n)])


def test_phi_recovers_monotone_map_and_slope():
    e, y, g = synthetic_sigmoid_ratings(2000)
    m = fit_phi(e, y, g, 0.02)
    grid = torch.linspace(*m.domain, 50, dtype=torch.float64)
    v = m(grid)
    assert (v[1:] > v[:-1]).all() and 0 <= v.min() and v.max() <= 1
    assert (v - true_sigmoid(grid)).abs().mean() < 0.05


def test_phi_is_differentiable_with_positive_slope_and_roundtrips():
    m = fit_phi(*synthetic_sigmoid_ratings(1000), 0.02)
    e = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    m(e).backward()
    assert e.grad > 0
    again = PhiMap.from_dict(json.loads(json.dumps(m.to_dict())))
    assert abs(again(torch.tensor(50.0, dtype=torch.float64)) - m(torch.tensor(50.0, dtype=torch.float64))) < 1e-12


def test_phi_refuses_out_of_domain():
    m = fit_phi(*synthetic_sigmoid_ratings(500), 0.02)
    try:
        m(torch.tensor(m.domain[1] + 1.0))
        assert False
    except ValueError:
        pass


def _stats(n=500, rho_lb=0.9, **k):
    return {"n": n, "rho_lb": rho_lb, **k}


def test_validity_insufficient_data_and_contiguous():
    assert validity_flags({1: _stats(n=10)}, 200, 50, .6, .85, .05)[1] is False
    assert validity_flags({1: _stats()}, 200, 50, .6, .85, .05)[1] is True
    assert validity_flags({1: _stats(rho_lb=.5)}, 200, 50, .6, .85, .05)[1] is False
    assert contiguous({1: True, 2: False, 3: True, 4: True}) == {1: False, 2: False, 3: True, 4: True}


def test_validity_of_presence_needs_counts_auroc_and_ece():
    ok = {"n": 500, "npos": 100, "nneg": 100, "auc_lb": 0.9, "ece_ub": 0.03}
    assert validity_flags({1: ok}, 200, 50, .6, .85, .05)[1] is True
    assert validity_flags({1: {**ok, "npos": 10}}, 200, 50, .6, .85, .05)[1] is False
    assert validity_flags({1: {**ok, "auc_lb": 0.7}}, 200, 50, .6, .85, .05)[1] is False
    assert validity_flags({1: {**ok, "ece_ub": 0.2}}, 200, 50, .6, .85, .05)[1] is False


def test_cross_fit_never_shares_groups():
    rows = [{"base": f"b{i % 50}", "x": i} for i in range(500)]
    seen = []

    def fit(train):
        seen.append({r["base"] for r in train})
        return {r["base"] for r in train}

    def predict(model, test):
        assert not ({r["base"] for r in test} & model)
        return [{"x": r["x"]} for r in test]

    out = cross_fit(rows, fit, predict, k=5)
    assert len(seen) == 5 and sorted(o["x"] for o in out) == list(range(500))


def synthetic_presence(n, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    g = r.normal(y * 3 - 1.5, 1.0)
    reg = r.normal(y * 3 - 1.5, 1.0)
    return g, reg, y, np.array([f"b{i % 100}" for i in range(n)])


def test_presence_calibrates():
    g, reg, y, groups = synthetic_presence(3000)
    m = fit_presence(g, reg, y, groups)
    gt, regt, yt, _ = synthetic_presence(3000, seed=1)
    logits = m.logit(torch.tensor(gt), torch.tensor(regt))
    p = torch.sigmoid(logits).tolist()
    assert ece(p, yt.tolist()) < 0.05
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    pos = yt == 1
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())
    assert auc > 0.85


def sample_acceptance(l, u, kappa, plateau, n, seed=0):
    r = np.random.default_rng(seed)
    s = r.uniform(0, 1, n)
    d = np.maximum(l - s, 0) + np.maximum(s - u, 0)
    p = plateau * np.exp(-d ** 2 / (2 * kappa ** 2))
    return s, (r.uniform(0, 1, n) < p).astype(int)


def test_hedge_recovers_known_curve():
    s, a = sample_acceptance(l=.3, u=.5, kappa=.08, plateau=.9, n=4000)
    f = fit_hedge(s, a)
    assert abs(f.l - .3) < .05 and abs(f.u - .5) < .05 and abs(f.plateau - .9) < .05 and abs(f.kappa - .08) < .04
    assert math.isfinite(f.heldout_ll)


def test_hedge_one_sided_at_scale_end():
    assert fit_hedge(*sample_acceptance(l=.85, u=1., kappa=.05, plateau=.95, n=4000)).one_sided
    assert not fit_hedge(*sample_acceptance(l=.3, u=.5, kappa=.08, plateau=.9, n=4000)).one_sided


def _gamma_data(noise, n=6000, seed=0):
    r = np.random.default_rng(seed)
    s = r.uniform(0, 1, n)
    d = np.maximum(.4 - s, 0) + np.maximum(s - .6, 0)
    a = (r.uniform(0, 1, n) < .9 * np.exp(-d ** 2 / (2 * .05 ** 2))).astype(int)
    s_hat = s + r.normal(0, noise, n) if noise else s
    f = fit_hedge(s, a)
    return s_hat, a, f, max(noise, 0.1), np.array([f"b{i % 100}" for i in range(n)])


def test_gamma_zero_when_no_noise_positive_with_noise():
    assert fit_gamma(*_gamma_data(0.0)) < 0.2 and fit_gamma(*_gamma_data(.1)) > 0.5


def test_gamma_is_chosen_on_held_out_folds_with_the_curve_refit_on_the_others():
    s_hat, a, f, sigma, groups = _gamma_data(.1)
    s = np.random.default_rng(0).uniform(0, 1, len(s_hat))     # the true scores _gamma_data draws first
    assert fit_gamma(s_hat, a, f, sigma, groups, s=s) > 0.5
    assert fit_gamma(s, a, f, sigma, groups, s=s) < 0.2   # estimates equal to the truth need no widening


def test_acceptance_fit_keeps_plateau_and_does_not_widen_size_twice():
    from fit_calibration import _fit_acceptance
    rows = []
    for family, quantity in (("F2a", "lightness"), ("F8", "area")):
        s, a = sample_acceptance(.3, .5, .08, .9, 1500)
        noisy = s + np.random.default_rng(3).normal(0, .1, len(s))
        rows += [{"family": family, "quantity": quantity, "hedge": "slightly", "orientation": 1, "s": float(x),
                  "s_hat": float(y), "accept": int(z), "base": f"b{i % 100}"} for i, (x, y, z) in enumerate(zip(s, noisy, a))]
    intervals, details = _fit_acceptance(rows, {"F2a": 0.1, "F8": 0.1}, None)
    light, size = intervals["F2a|lightness|slightly|1"], intervals["F8|area|slightly|1"]
    assert details["F2a|lightness|slightly|1"]["gamma"] > 0 and light[2] > size[2] - 1e-9
    assert details["F8|area|slightly|1"]["gamma"] == 0.0
    assert abs(details["F2a|lightness|slightly|1"]["plateau"] - .9) < .06


def test_size_coverage_is_measured_out_of_fold_and_per_stratum():
    from calibration_fit import oof_size_coverage
    f, e, g = _size_errors(3000)
    bins = np.array([i % 2 for i in range(len(e))])
    cov = oof_size_coverage(f, e, g, {"bin 1": bins == 0, "bin 2": bins == 1})
    assert cov["all"]["n"] == len(e) and abs(cov["all"]["c68"] - .68) < .05 and abs(cov["all"]["c95"] - .95) < .04
    assert cov["bin 1"]["n"] + cov["bin 2"]["n"] == len(e)


def test_pool_orientations():
    fit_a = fit_hedge(*sample_acceptance(.3, .5, .08, .9, 3000, seed=1))
    data_a = sample_acceptance(.3, .5, .08, .9, 3000, seed=1)
    same = sample_acceptance(.3, .5, .08, .9, 3000, seed=2)
    other = sample_acceptance(.7, .9, .08, .9, 3000, seed=3)
    assert pool_orientations(fit_a, fit_hedge(*same), (data_a, same)) is True
    assert pool_orientations(fit_a, fit_hedge(*other), (data_a, other)) is False


def _size_errors(n, seed=0):
    r = np.random.default_rng(seed)
    area = r.uniform(0.005, 0.3, n)
    features = np.stack([area, r.uniform(0.005, 0.3, n)], axis=1)
    err = np.abs(r.normal(0, 0.05 + 0.02 / np.sqrt(area)))
    return features, err, np.array([f"b{i % 100}" for i in range(n)])


def test_size_quantile_coverage():
    f, e, g = _size_errors(5000)
    q = fit_size_quantiles(f, e, g)
    ft, et, _ = _size_errors(5000, seed=1)
    cov = coverage(q, ft, et, {"small": ft[:, 0] < 0.02})
    assert abs(cov["all"]["c68"] - .68) < .04 and abs(cov["all"]["c95"] - .95) < .03 and "small" in cov
    assert (q.predict_c95(ft) >= q.predict_c68(ft)).all()         # no crossing


def _attention_rows(obj_ok=True, attr_ok=True, bad_folds=0):
    rows = []
    rng = random.Random(0)
    bases = [f"b{i}" for i in range(100)]
    bad = set(_fold_members(bases, bad_folds))
    for kind, good in (("objects", obj_ok), ("attributes", attr_ok)):
        for b in bases:
            for _ in range(4):
                confident = rng.random() < 0.6
                works = good and b not in bad
                correct = rng.random() < (0.95 if (confident and works) else 0.3)
                rows.append({"base": b, "kind": kind, "config": "c1", "peak": 3.0 if confident else 1.2,
                             "share": 1.0, "mass": 0.5, "correct": correct})
    return rows


def _fold_members(bases, n_folds):
    import hashlib
    def fold(b):
        return int(hashlib.sha256(b.encode()).hexdigest()[:8], 16) % 5
    return [b for b in bases if fold(b) < n_folds]


CANDIDATES = [{"config": "c1", "kappa_pk": k, "m_min": 0.5, "tau_min": 0.0} for k in (1.0, 2.0)]


def test_attention_selects_confident_threshold():
    c = select_attention(_attention_rows(), CANDIDATES, required=.8)
    assert c["objects"].feasible and c["objects"].kappa_pk == 2.0


def test_attention_infeasible_disables_only_that_kind():
    c = select_attention(_attention_rows(attr_ok=False), CANDIDATES, required=.8)
    assert c["objects"].feasible and not c["attributes"].feasible and "binding" in c["attributes"].disabled


def test_attention_needs_four_of_five_folds():
    assert not select_attention(_attention_rows(bad_folds=2), CANDIDATES, required=.8)["objects"].feasible
    assert select_attention(_attention_rows(bad_folds=1), CANDIDATES, required=.7)["objects"].feasible


def test_binding_gate():
    rng = random.Random(0)
    good = [{"beta": rng.gauss(0.2 if c else -0.2, 0.05), "correct": c} for c in [True, False] * 200]
    margin, ok = select_binding(good, 0.70)
    assert ok and -0.1 < margin < 0.1
    bad = [{"beta": rng.gauss(0, 0.1), "correct": c} for c in [True, False] * 200]
    assert select_binding(bad, 0.70)[1] is False


def _drift_rows(bad=None):
    rng = random.Random(0)
    rows = []
    for stratum_name in ("small", "medium", "large"):
        for i in range(600):
            moved = i % 3 == 0
            if stratum_name == bad:
                iou, shift = rng.uniform(0, 1), rng.uniform(0, 2)
            else:
                iou = rng.gauss(0.3, 0.06) if moved else rng.gauss(0.85, 0.05)
                shift = rng.gauss(1.0, 0.1) if moved else rng.gauss(0.1, 0.03)
            rows.append({"base": f"b{i % 120}", "stratum": stratum_name, "iou": iou, "shift_rel": shift, "moved": moved})
    return rows


def test_drift_respects_false_disable_bound():
    d = select_drift(_drift_rows(), fd_max=.10)
    assert d["medium"] is not None and d["medium"].fd_upper <= .10 and d["medium"].miss < 0.1


def test_stratum_without_valid_threshold_gets_none():
    d = select_drift(_drift_rows(bad="small"), fd_max=.10)
    assert d["small"] is None and d["large"] is not None


def _parts():
    phi = fit_phi(*synthetic_sigmoid_ratings(400), 0.02)
    return {
        "phi": {"F2a": {"scene_lightness": {"4": phi.to_dict()}}},
        "valid": {"F2a": {"1": False, "2": True, "3": True, "4": True}},
        "intervals": {"F2a|scene_lightness|slightly|-1": [0.1, 0.3, 0.09]},
        "default_intervals": {"presence": [2.2, None, 1.0], "binding": [0.05, None, 0.05], "leakage": [None, 0.2, 0.1]},
        "bounds": {"e": [0.4, 0.1], "varpi": [0.5, 0.1], "zeta": [0.2, 0.1]},
        "presence": {"4": {"w_global": 0.5, "w_region": 0.5, "bias": 0.1, "range": [-9.0, 9.0]}},
        "attention": {"objects": {"feasible": True, "kappa_pk": 2.0, "m_min": 0.5, "tau_min": 0.1},
                      "attributes": {"feasible": False, "kappa_pk": 2.0, "m_min": 0.5, "tau_min": 0.1}},
        "binding_ok": True, "drift": {"small": None, "medium": [0.5, 0.5], "large": [0.5, 0.5]},
        "stage2_valid": {"1": False, "2": True, "3": True, "4": True}, "prototypes": {"red": [0.5, 0.2, 0.1]},
        "leakage_scale": 0.07, "size": {"c68": 0.1, "c95": 0.2},
    }


def test_fitted_interface_and_roundtrip():
    d = tempfile.mkdtemp()
    try:
        FittedCalibration(_parts()).save(Path(d) / "c.json")
        c = FittedCalibration.load(Path(d) / "c.json")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert c.tag == "fitted" and c.valid("F2a", 4) and not c.valid("F2a", 1) and not c.valid("F6", 4)
    assert c.interval("F2a", "scene_lightness", "slightly", -1) == (0.1, 0.3, 0.09)
    assert c.domain("F2a", "scene_lightness", 4) == tuple(fit_phi(*synthetic_sigmoid_ratings(400), 0.02).domain)
    assert c.interval("presence", None, None, 1)[1] == math.inf and c.interval("leakage", None, None, -1)[0] == -math.inf
    assert c.drift_thresholds("small") is None and c.drift_thresholds("medium") == (0.5, 0.5)
    assert c.attention_ok("objects") and not c.attention_ok("attributes") and not c.stage2_valid(1)
    assert c.prototype("red").shape == (3,) and c.size_quantiles({}) == (0.1, 0.2) and c.leakage_scale() == 0.07
    assert c.attention_thresholds("objects") == (2.0, 0.5, 0.1)


def test_missing_acceptance_data_uses_recorded_heuristic_or_fails_loudly():
    parts = _parts()
    c = FittedCalibration(parts)
    try:
        c.interval("F2a", "chroma", "very", 1)
        assert False, "with no fallback table, a missing interval must not pass silently"
    except KeyError:
        pass
    parts["fallback_intervals"] = {"very": [0.75, 0.95, 0.1]}
    c = FittedCalibration(parts)
    assert c.interval("F2a", "chroma", "very", 1) == (0.75, 0.95, 0.1)
    assert c.heuristic_used == {"F2a|chroma|very|1"}


def test_family_without_a_map_is_out_of_domain_not_a_crash():
    c = FittedCalibration(_parts())
    lo, hi = c.domain("F2a", "chroma", 4)
    assert lo > hi                                                # an empty domain: no value is inside it
    assert not c.valid("F2a", 3) or c.domain("F2a", "scene_lightness", 3)[0] > c.domain("F2a", "scene_lightness", 3)[1]


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("fit: pass")


if __name__ == "__main__":
    main()
