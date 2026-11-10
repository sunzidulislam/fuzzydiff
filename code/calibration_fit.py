"""Fit calibration from annotations (paper: App. "Calibration"), and the calibration that uses the fits.

Every fit takes plain arrays, so it is tested on synthetic data with known answers. FittedCalibration
implements the same interface as ProvisionalCalibration; guidance and SAR do not change.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FOLDS = 5
Z_95 = 1.96


def fold_of(base: str, k: int = FOLDS) -> int:
    return int(hashlib.sha256(base.encode()).hexdigest()[:8], 16) % k


def _group_weights(groups):
    """Each base prompt counts once, however many images it has."""
    values, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def cross_fit(rows: List[dict], fit, predict, k: int = FOLDS, group: str = "base") -> List[dict]:
    """Grouped k-fold: fit on the other folds, predict on the held-out one, pool the predictions."""
    out = []
    for i in range(k):
        train = [r for r in rows if fold_of(r[group], k) != i]
        test = [r for r in rows if fold_of(r[group], k) == i]
        if test:
            out.extend(predict(fit(train), test))
    return out


# -------------------------------------------------------------------------------- perceptual map
class PhiMap:
    """Eq. appA_phi: PCHIP of isotonic knot means plus a small slope, defined on its domain only."""

    def __init__(self, xs, ys, ds, lo, hi, eps):
        self.xs, self.ys, self.ds = list(map(float, xs)), list(map(float, ys)), list(map(float, ds))
        self.domain = (float(lo), float(hi))
        self.eps = float(eps)

    def _pchip(self, e):
        xs = torch.as_tensor(self.xs, dtype=e.dtype, device=e.device)
        ys = torch.as_tensor(self.ys, dtype=e.dtype, device=e.device)
        ds = torch.as_tensor(self.ds, dtype=e.dtype, device=e.device)
        clipped = torch.minimum(torch.maximum(e, xs[0]), xs[-1])
        i = torch.clamp(torch.searchsorted(xs, clipped.detach(), right=True) - 1, 0, len(self.xs) - 2)
        h = xs[i + 1] - xs[i]
        t = (clipped - xs[i]) / h
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        return h00 * ys[i] + h10 * h * ds[i] + h01 * ys[i + 1] + h11 * h * ds[i + 1]

    def __call__(self, e):
        lo, hi = self.domain
        value = e.detach()
        if bool((value < lo).any()) or bool((value > hi).any()):
            raise ValueError(f"perceptual map is not defined outside its domain [{lo}, {hi}]")
        # eps acts on the normalized raw output, so the map does not depend on the raw units
        return (self._pchip(e) + self.eps * (e - lo) / (hi - lo)) / (1 + self.eps)

    def to_dict(self):
        return {"xs": self.xs, "ys": self.ys, "ds": self.ds, "lo": self.domain[0], "hi": self.domain[1], "eps": self.eps}

    @classmethod
    def from_dict(cls, d):
        return cls(d["xs"], d["ys"], d["ds"], d["lo"], d["hi"], d["eps"])


def fit_phi(e, ratings, groups, eps_phi) -> PhiMap:
    e, ratings = np.asarray(e, dtype=float), np.asarray(ratings, dtype=float)
    lo, hi = np.percentile(e, [1, 99])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(e, ratings, sample_weight=_group_weights(groups))
    xs, ys = np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_)
    keep = np.concatenate([[True], np.diff(xs) > 0])
    xs, ys = xs[keep], ys[keep]
    ds = PchipInterpolator(xs, ys).derivative()(xs)
    return PhiMap(xs, ys, ds, lo, hi, eps_phi)


# -------------------------------------------------------------------------------- presence
@dataclass
class PresenceModel:
    w_global: float
    w_region: float
    bias: float
    range: tuple

    def logit(self, global_logit, region_logit):
        return self.w_global * global_logit + self.w_region * region_logit + self.bias

    def to_dict(self):
        return {"w_global": self.w_global, "w_region": self.w_region, "bias": self.bias, "range": list(self.range)}


def fit_presence(global_logits, region_logits, labels, groups) -> PresenceModel:
    x = np.stack([np.asarray(global_logits, float), np.asarray(region_logits, float)], axis=1)
    model = LogisticRegression(C=1e3, max_iter=1000).fit(x, labels, sample_weight=_group_weights(groups))
    h = model.decision_function(x)
    lo, hi = np.percentile(h, [0.5, 99.5])
    return PresenceModel(float(model.coef_[0, 0]), float(model.coef_[0, 1]), float(model.intercept_[0]), (float(lo), float(hi)))


# -------------------------------------------------------------------------------- validity
def validity_flags(cell_stats: Dict, nmin, npos_min, rho_min, auc_min, ece_max) -> Dict:
    """Decisions use lower confidence bounds, and too little data means not valid."""
    flags = {}
    for b, st in cell_stats.items():
        ok = st.get("n", 0) >= nmin
        if "auc_lb" in st:
            ok = (ok and st.get("npos", 0) >= npos_min and st.get("nneg", 0) >= npos_min
                  and st["auc_lb"] >= auc_min and st.get("ece_ub", 1.0) <= ece_max)
        else:
            ok = ok and st.get("rho_lb", -1.0) >= rho_min
        flags[b] = bool(ok)
    return flags


def contiguous(flags: Dict) -> Dict:
    """nu~_b = nu_b and ... and nu_B: the active window ends at the final step, without gaps."""
    out, ok = {}, True
    for b in sorted(flags, key=int, reverse=True):
        ok = ok and flags[b]
        out[b] = ok
    return out


# -------------------------------------------------------------------------------- hedges
@dataclass
class HedgeFit:
    l: float
    u: float
    kappa: float
    plateau: float
    one_sided: bool
    heldout_ll: float


def _accept_prob(s, l, u, kappa, plateau):
    d = np.maximum(l - s, 0) + np.maximum(s - u, 0)
    return np.clip(plateau * np.exp(-d ** 2 / (2 * kappa ** 2)), 1e-6, 1 - 1e-6)


def _log_likelihood(s, a, l, u, kappa, plateau):
    p = _accept_prob(s, l, u, kappa, plateau)
    return float(np.sum(a * np.log(p) + (1 - a) * np.log(1 - p)))


def _fit_params(s, a, scale):
    span = scale[1] - scale[0]

    def nll(t):
        l = t[0]
        u = l + abs(t[1])
        kappa = abs(t[2]) + 1e-3 * span
        plateau = float(np.clip(t[3], 0.05, 1.0))
        return -_log_likelihood(s, a, l, u, kappa, plateau)

    best = None
    centers = np.linspace(scale[0], scale[1], 7)
    edges = np.linspace(scale[0], scale[1], 8)
    rates = [a[(s >= edges[i]) & (s < edges[i + 1])].mean() if ((s >= edges[i]) & (s < edges[i + 1])).any() else 0.0
             for i in range(7)]
    order = np.argsort(rates)[::-1][:3]
    for i in order:
        start = [centers[i] - 0.1 * span, 0.2 * span, 0.08 * span, max(rates[i], 0.3)]
        r = minimize(nll, start, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    t = best.x
    return t[0], t[0] + abs(t[1]), abs(t[2]) + 1e-3 * span, float(np.clip(t[3], 0.05, 1.0))


def fit_hedge(s, accept, scale=(0.0, 1.0), seed=0) -> HedgeFit:
    """Eq. acceptance_curve by maximum likelihood; 20% of the judgments are held out."""
    s, a = np.asarray(s, float), np.asarray(accept, float)
    order = np.random.default_rng(seed).permutation(len(s))
    cut = int(0.8 * len(s))
    train, test = order[:cut], order[cut:]
    l, u, kappa, plateau = _fit_params(s[train], a[train], scale)
    span = scale[1] - scale[0]
    one_sided = u >= scale[1] - 0.03 * span or l <= scale[0] + 0.03 * span
    return HedgeFit(float(l), float(u), float(kappa), plateau, bool(one_sided),
                    _log_likelihood(s[test], a[test], l, u, kappa, plateau) / max(len(test), 1))


def fit_gamma(s_hat, accept, hedge: HedgeFit, sigma_m, groups, grid=None, s=None, scale=(0.0, 1.0)) -> float:
    """gamma_r in kappa_eff = sqrt(kappa^2 + (gamma sigma_m)^2): out-of-fold likelihood on estimated scores.

    Each gamma is scored on the held-out fold only, summed over folds. With the true scores s, the
    acceptance curve is refit on the other folds first, so no held-out judgment shapes the curve."""
    s_hat, a = np.asarray(s_hat, float), np.asarray(accept, float)
    folds = np.array([fold_of(str(g)) for g in groups])
    grid = np.linspace(0, 3, 31) if grid is None else grid
    total = np.zeros(len(grid))
    for i in range(FOLDS):
        test = folds == i
        if not test.any() or test.all():
            continue
        h = hedge if s is None else fit_hedge(np.asarray(s, float)[~test], a[~test], scale=scale)
        total += [_log_likelihood(s_hat[test], a[test], h.l, h.u, math.sqrt(h.kappa ** 2 + (g * sigma_m) ** 2),
                                  h.plateau) for g in grid]
    return float(grid[int(np.argmax(total))])


def _ll_of(fit: HedgeFit, data):
    return _log_likelihood(np.asarray(data[0], float), np.asarray(data[1], float), fit.l, fit.u, fit.kappa, fit.plateau)


def pool_orientations(fit_pos: HedgeFit, fit_neg: HedgeFit, data) -> bool:
    """Likelihood-ratio test: pool "larger" and "smaller" unless equality is rejected."""
    (s_a, a_a), (s_b, a_b) = data
    pooled = fit_hedge(np.concatenate([s_a, s_b]), np.concatenate([a_a, a_b]))
    separate = _ll_of(fit_pos, (s_a, a_a)) + _ll_of(fit_neg, (s_b, a_b))
    together = _ll_of(pooled, (s_a, a_a)) + _ll_of(pooled, (s_b, a_b))
    return bool(2 * (separate - together) < chi2.ppf(0.95, df=4))


# -------------------------------------------------------------------------------- size
@dataclass
class SizeQuantiles:
    m68: object
    m95: object

    def predict_c68(self, features):
        return self.m68.predict(np.asarray(features))

    def predict_c95(self, features):
        return np.maximum(self.m95.predict(np.asarray(features)), self.predict_c68(features))


def fit_size_quantiles(features, abs_error, groups) -> SizeQuantiles:
    """Gradient-boosted conditional quantiles of the pairwise error; c95 never crosses below c68."""
    w = _group_weights(groups)
    models = []
    for alpha in (0.68, 0.95):
        m = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=100, max_depth=3, random_state=0)
        models.append(m.fit(np.asarray(features), np.asarray(abs_error), sample_weight=w))
    return SizeQuantiles(*models)


def coverage(q: SizeQuantiles, features, abs_error, strata: Dict[str, np.ndarray]) -> Dict[str, dict]:
    err = np.asarray(abs_error)
    c68, c95 = q.predict_c68(features), q.predict_c95(features)

    def rates(mask):
        return {"c68": float((err[mask] <= c68[mask]).mean()), "c95": float((err[mask] <= c95[mask]).mean()),
                "n": int(mask.sum())}

    out = {"all": rates(np.ones(len(err), dtype=bool))}
    for name, mask in strata.items():
        if np.asarray(mask).any():
            out[name] = rates(np.asarray(mask))
    return out


def oof_size_coverage(features, abs_error, groups, strata: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Coverage of c68 and c95 on held-out folds (grouped by base prompt), so it is not judged on its own
    training data. strata, e.g. one mask per noise bin, show where the pooled quantiles miss."""
    x, err, groups = np.asarray(features, float), np.asarray(abs_error, float), np.asarray(groups)
    folds = np.array([fold_of(str(g)) for g in groups])
    in68, in95, seen = np.zeros(len(err), bool), np.zeros(len(err), bool), np.zeros(len(err), bool)
    for i in range(FOLDS):
        test = folds == i
        if not test.any() or test.all():
            continue
        q = fit_size_quantiles(x[~test], err[~test], groups[~test])
        in68[test], in95[test], seen[test] = err[test] <= q.predict_c68(x[test]), err[test] <= q.predict_c95(x[test]), True

    def rates(mask):
        mask = mask & seen
        return {"c68": float(in68[mask].mean()), "c95": float(in95[mask].mean()), "n": int(mask.sum())}

    out = {"all": rates(np.ones(len(err), dtype=bool))}
    for name, mask in strata.items():
        if (np.asarray(mask) & seen).any():
            out[name] = rates(np.asarray(mask))
    return out


# -------------------------------------------------------------------------------- attention thresholds
@dataclass
class AttentionChoice:
    feasible: bool
    config: Optional[str] = None
    kappa_pk: Optional[float] = None
    m_min: Optional[float] = None
    tau_min: Optional[float] = None
    accuracy: Optional[float] = None
    folds_passed: int = 0
    disabled: List[str] = field(default_factory=list)


def _accuracy(rows, c):
    kept = [r for r in rows if r["config"] == c["config"] and r["peak"] >= c["kappa_pk"]
            and r["share"] >= c["m_min"] and r["mass"] >= c["tau_min"]]
    return (sum(r["correct"] for r in kept) / len(kept) if kept else 0.0), len(kept)


def _best_candidate(rows, candidates, required):
    """Highest coverage among candidates that reach the required pointing accuracy."""
    scored = [(c, *_accuracy(rows, c)) for c in candidates]
    usable = [t for t in scored if t[1] >= required and t[2] > 0]
    return max(usable, key=lambda t: t[2]) if usable else None


def select_attention(records, candidates, required, k=FOLDS, min_folds=4) -> Dict[str, AttentionChoice]:
    """Selection and validation by grouped cross-fitting; a failed kind disables what depends on it."""
    out = {}
    for kind in ("objects", "attributes"):
        rows = [r for r in records if r["kind"] == kind]
        passed = 0
        for i in range(k):
            train = [r for r in rows if fold_of(r["base"], k) != i]
            test = [r for r in rows if fold_of(r["base"], k) == i]
            best = _best_candidate(train, candidates, required)
            if best is not None and _accuracy(test, best[0])[0] >= required:
                passed += 1
        final = _best_candidate(rows, candidates, required)
        if final is None or passed < min_folds:
            disabled = ["binding"] if kind == "attributes" else ["binding", "sar_priors"]
            out[kind] = AttentionChoice(False, folds_passed=passed, disabled=disabled)
        else:
            c = final[0]
            out[kind] = AttentionChoice(True, c["config"], c["kappa_pk"], c["m_min"], c["tau_min"], final[1], passed)
    return out


def _auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def select_binding(records, auc_min):
    """Binding gate: the margin that best separates bound from swapped pairs, if AUROC clears the gate."""
    beta = np.array([r["beta"] for r in records], float)
    correct = np.array([r["correct"] for r in records], bool)
    best_margin, best_j = 0.0, -1.0
    for m in np.quantile(beta, np.linspace(0.05, 0.95, 37)):
        j = (beta[correct] >= m).mean() - (beta[~correct] >= m).mean()
        if j > best_j:
            best_margin, best_j = float(m), float(j)
    return best_margin, bool(_auroc(beta, correct) >= auc_min)


# -------------------------------------------------------------------------------- drift thresholds
@dataclass
class DriftChoice:
    theta: float
    delta_rel: float
    fd_upper: float
    miss: float


def _wilson_upper(k, n):
    if n == 0:
        return 1.0
    p = k / n
    z2 = Z_95 ** 2
    centre = p + z2 / (2 * n)
    margin = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n ** 2))
    return (centre + margin) / (1 + z2 / n)


def _rates(rows, theta, delta):
    disabled = [r["iou"] < theta or r["shift_rel"] > delta for r in rows]
    fixed = [d for d, r in zip(disabled, rows) if not r["moved"]]
    moved = [d for d, r in zip(disabled, rows) if r["moved"]]
    fd_upper = _wilson_upper(sum(fixed), len(fixed))
    miss = 1 - sum(moved) / len(moved) if moved else 0.0
    return fd_upper, miss


def _best_theta(rows, delta, fd_max, thetas):
    usable = [(t, *_rates(rows, t, delta)) for t in thetas]
    usable = [u for u in usable if u[1] <= fd_max]
    return min(usable, key=lambda u: u[2]) if usable else None


def select_drift(records, fd_max, k=FOLDS, thetas=None, deltas=None) -> Dict[str, Optional[DriftChoice]]:
    """Per stratum: least miss rate with the false-disable upper bound at most fd_max, for a common
    delta_rel. A stratum with no valid setting in any training split gets None, meaning no SAR."""
    thetas = np.linspace(0.1, 0.9, 17) if thetas is None else thetas
    deltas = np.array([0.25, 0.5, 0.75, 1.0, 1.5]) if deltas is None else deltas
    strata = sorted({r["stratum"] for r in records})
    by = {s: [r for r in records if r["stratum"] == s] for s in strata}

    def total_miss(delta):
        found = [_best_theta(by[s], delta, fd_max, thetas) for s in strata]
        return sum(f[2] if f else 1.0 for f in found) + sum(f is None for f in found)

    delta = float(min(deltas, key=total_miss))
    out: Dict[str, Optional[DriftChoice]] = {}
    for s in strata:
        trains_ok = all(_best_theta([r for r in by[s] if fold_of(r["base"], k) != i], delta, fd_max, thetas) is not None
                        for i in range(k))
        best = _best_theta(by[s], delta, fd_max, thetas)
        out[s] = DriftChoice(float(best[0]), delta, float(best[1]), float(best[2])) if best and trains_ok else None
    return out


# -------------------------------------------------------------------------------- the fitted calibration
def _qkey(quantity):
    return ":".join(quantity) if isinstance(quantity, tuple) else str(quantity)


def _end(value, default):
    return default if value is None else value


class FittedCalibration:
    """Same interface as ProvisionalCalibration; everything comes from the fitted parts.

    A family or bin with no fit is simply not valid, so it gets no guidance. A valid family with no
    acceptance data uses `fallback_intervals` (manuscript's pre-registered heuristic intervals), and
    each such use is recorded in `heuristic_used` so it can be reported separately."""

    def __init__(self, parts, size_quantile_models: Optional[SizeQuantiles] = None):
        self.parts = parts
        self.tag = "provisional" if parts.get("provisional_parts") else "fitted"
        self.size_models = size_quantile_models
        self.heuristic_used = set()
        self._phi = {f: {q: {b: PhiMap.from_dict(d) for b, d in bins.items()} for q, bins in qs.items()}
                     for f, qs in parts.get("phi", {}).items()}

    def _map(self, family, quantity, bin):
        return self._phi.get(family, {}).get(_qkey(quantity), {}).get(str(bin))

    def phi(self, family, quantity, bin, e):
        return self._map(family, quantity, bin)(e)

    def domain(self, family, quantity, bin):
        m = self._map(family, quantity, bin)
        if m is not None:
            return m.domain
        if family == "presence":
            p = self.parts.get("presence", {}).get(str(bin))
            return tuple(p["range"]) if p else (math.inf, -math.inf)
        if family in self._phi or family == "F6":
            return (math.inf, -math.inf)
        return (-math.inf, math.inf)

    def valid(self, family, bin):
        return bool(self.parts.get("valid", {}).get(family, {}).get(str(bin), False))

    def interval(self, family, quantity, hedge, orientation):
        if family in self.parts.get("default_intervals", {}):
            l, u, k = self.parts["default_intervals"][family]
            return (_end(l, -math.inf), _end(u, math.inf), k)
        key = f"{family}|{_qkey(quantity)}|{hedge}|{orientation}"
        fitted = self.parts.get("intervals", {})
        if key in fitted:
            return tuple(fitted[key])
        pooled = f"{family}|{_qkey(quantity)}|{hedge}|pooled"
        if pooled in fitted:
            return tuple(fitted[pooled])
        fallback = self.parts.get("fallback_intervals", {}).get(str(hedge))
        if fallback is None:
            raise KeyError(key)
        self.heuristic_used.add(key)
        return tuple(fallback)

    def bound(self, name):
        return tuple(self.parts["bounds"][name])

    def leakage_scale(self):
        return self.parts["leakage_scale"]

    def attention_thresholds(self, kind):
        a = self.parts["attention"][kind]
        return (a["kappa_pk"], a["m_min"], a["tau_min"])

    def presence_logit(self, bin, global_logit, region_logit):
        p = self.parts.get("presence", {}).get(str(bin))
        if p is None:
            return torch.full_like(global_logit, float("nan"))   # not fitted: out of domain, below the gate
        return p["w_global"] * global_logit + p["w_region"] * region_logit + p["bias"]

    def size_quantiles(self, features):
        if self.size_models is not None:
            x = [[float(features["A_q"]), float(features["A_r"])]]
            return (float(self.size_models.predict_c68(x)[0]), float(self.size_models.predict_c95(x)[0]))
        return (self.parts["size"]["c68"], self.parts["size"]["c95"])

    def prototype(self, term):
        return torch.tensor(self.parts["prototypes"][term], dtype=torch.float64)

    def psi(self, phrase, features):
        return self.parts.get("psi", 1.0)

    def drift_thresholds(self, stratum):
        d = self.parts["drift"].get(stratum)
        return None if d is None else tuple(d)

    def attention_ok(self, kind):
        ok = self.parts["attention"][kind]["feasible"]
        return ok and self.parts.get("binding_ok", True) if kind == "attributes" else ok

    def stage2_valid(self, bin):
        return bool(self.parts["stage2_valid"].get(str(bin), False))

    def save(self, path):
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.parts, f, indent=1)
        if self.size_models is not None:
            joblib.dump(self.size_models, path.with_suffix(".size.joblib"))

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path) as f:
            parts = json.load(f)
        models_path = path.with_suffix(".size.joblib")
        return cls(parts, joblib.load(models_path) if models_path.exists() else None)
