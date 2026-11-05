"""Run logs -> paper tables, with the selection/evaluation separation enforced in code.

Two kinds of functions:
  * SELECTION (decide_support, choose_sigma_lo): may only read calibration (cal), intervention
    (int), system-validation (sys), or development (dev) data. They raise LeakageError on any
    record whose split is 'test'. Their outputs are frozen in the support manifest
    (freeze_manifest.py) before test images are generated.
  * REPORTING (activation_rows, tier_rows, transfer_rows, shift_rows, interval_rows): compute
    table rows on any split; on 'test' they are confirmatory and never change the manifest.

Log schema (JSONL, one object per line, written by the guidance loop; see `StepLog`):
  {"type": "constraint", "split", "prompt", "seed", "step", "sigma", "bin", "family", "cid",
   "tier", "gates": {"validity": bool, "presence": bool, "support": bool, "domain": bool,
   "confidence": bool}, "active": bool, "accepted": bool|None, "dL": float|None}
  {"type": "step", "split", "prompt", "step", "tiers": [1,2,3], "loss_before": {m: L},
   "loss_after": {m: L} (accepted candidate, frozen set), "cos": {"1-2": c, "1-3": c, "2-3": c},
   "removed": {m: frac}, "rejections": [reason, ...], "accepted": bool, "eps": {m: eps_m}}
  {"type": "final", "split", "prompt", "seed", "family", "cid", "tier",
   "mu_unguided": float, "mu_guided": float}
Paired transfer measurements (CSV or list of dicts): split, prompt, family, ds, dy, direction,
  mu_est, human_accept, n_active, conflict.
Calibration-shift records: split, source (unguided | guided | variant), family, s, rating,
  accept, mu, in_domain.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from statistics import median
from typing import Dict, Iterable, List

SELECTION_SPLITS = {"cal", "int", "sys", "dev"}
GATES = ("validity", "presence", "support", "domain", "confidence")


class LeakageError(RuntimeError):
    pass


def _check_selection(records: Iterable[dict]):
    recs = list(records)
    bad = {r.get("split") for r in recs} - SELECTION_SPLITS
    if bad:
        raise LeakageError(f"selection functions may not read splits {sorted(bad)}")
    return recs


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class StepLog:
    """Minimal writer used by the guidance loop."""

    def __init__(self, path, split):
        self.f, self.split = open(path, "a"), split

    def write(self, **rec):
        rec["split"] = self.split
        self.f.write(json.dumps(rec) + "\n")

    def close(self):
        self.f.close()


# ------------------------------------------------------------------------- statistics helpers
def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def slope_through_origin(ds, dy):
    den = sum(d * d for d in ds)
    return sum(a * b for a, b in zip(ds, dy)) / den if den else float("nan")


def cluster_bootstrap(rows, stat, key="prompt", n=2000, seed=0):
    """95% percentile CI of stat(rows) resampling clusters (base prompts)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    keys = list(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in groups[k]]
        v = stat(sample)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    if not vals:
        return (float("nan"), float("nan"))
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def ece(pred, obs, bins=10):
    """Expected calibration error with equal-mass bins."""
    order = sorted(range(len(pred)), key=lambda i: pred[i])
    out, n = 0.0, len(pred)
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if idx:
            out += len(idx) / n * abs(sum(pred[i] for i in idx) / len(idx) - sum(obs[i] for i in idx) / len(idx))
    return out


# ---------------------------------------------------------------------------------- reporting
def activation_rows(records: List[dict], n_act: int):
    """Tab. activation (bottom). Per family: first valid bin, median active steps per constraint,
    share of constraints with >= n_act active steps, most frequent blocking gate, accepted share
    on active steps, mean realized tier-3 loss change per accepted step, mean mu unguided -> guided."""
    cons = [r for r in records if r["type"] == "constraint"]
    fin = [r for r in records if r["type"] == "final"]
    out = {}
    for fam in sorted({r["family"] for r in cons}):
        rs = [r for r in cons if r["family"] == fam]
        valid_bins = [r["bin"] for r in rs if r["gates"].get("validity")]
        per = defaultdict(int)
        for r in rs:
            per[(r["prompt"], r.get("seed"), r["cid"])] += int(r["active"])
        counts = list(per.values())
        blocked = Counter(g for r in rs if not r["active"] for g in GATES if not r["gates"].get(g, True))
        act = [r for r in rs if r["active"]]
        acc = [r for r in act if r.get("accepted")]
        dls = [r["dL"] for r in acc if r.get("dL") is not None]
        f = [r for r in fin if r["family"] == fam]
        out[fam] = {
            "b0": min(valid_bins) if valid_bins else None,
            "active_median": median(counts) if counts else 0,
            "share_ge_nact": sum(c >= n_act for c in counts) / len(counts) if counts else 0.0,
            "block": blocked.most_common(1)[0][0] if blocked else "--",
            "accepted_share": len(acc) / len(act) if act else float("nan"),
            "dL3_per_accepted": sum(dls) / len(dls) if dls else float("nan"),
            "mu_unguided": sum(r["mu_unguided"] for r in f) / len(f) if f else float("nan"),
            "mu_guided": sum(r["mu_guided"] for r in f) / len(f) if f else float("nan"),
        }
    return out


def tier_rows(records: List[dict]):
    """Tab. tier diagnostics. Per tier m: violation rate (mu < 0.5) at the first and the last
    guided step, conflict share (raw cosine < 0 with a higher tier), mean removed fraction,
    and *finite-step failures*: accepted steps on which the frozen tier-m loss rose by more than
    its tolerance although the first-order projection predicted no increase. Plus the rejection
    rate and the distribution of rejection reasons."""
    steps = [r for r in records if r["type"] == "step"]
    cons = [r for r in records if r["type"] == "constraint"]
    fin = [r for r in records if r["type"] == "final"]
    out = {}
    for m in (1, 2, 3):
        sm = [s for s in steps if m in s["tiers"]]
        conf = [s["cos"][k] < 0 for s in sm for k in s.get("cos", {}) if k.endswith(str(m)) and k[0] != str(m)]
        rem = [s["removed"][str(m)] for s in sm if str(m) in s.get("removed", {})]
        acc = [s for s in sm if s["accepted"]]
        fail = [s for s in acc if s["loss_after"][str(m)] > s["loss_before"][str(m)] + s.get("eps", {}).get(str(m), 0.0)]
        f = [r for r in fin if r["tier"] == m]
        out[m] = {
            "viol_unguided": sum(r["mu_unguided"] < 0.5 for r in f) / len(f) if f else float("nan"),
            "viol_guided": sum(r["mu_guided"] < 0.5 for r in f) / len(f) if f else float("nan"),
            "conflict_share": sum(conf) / len(conf) if conf else float("nan"),
            "removed_mean": sum(rem) / len(rem) if rem else float("nan"),
            "finite_step_failures": len(fail) / len(acc) if acc else float("nan"),
        }
    reasons = Counter(x for s in steps for x in s.get("rejections", []))
    n_cand = sum(1 + len(s.get("rejections", [])) for s in steps)
    out["rejections"] = {"rate": sum(len(s.get("rejections", [])) for s in steps) / n_cand if n_cand else float("nan"),
                         "reasons": dict(reasons)}
    out["n_constraint_steps"] = len(cons)
    return out


def transfer_rows(pairs: List[dict], s_min=0.05, n_boot=2000):
    """Tab. transfer_full. Directional agreement, through-origin slope with cluster-bootstrap CI,
    slope by number of active constraints, exploitation rate."""
    out = {}
    for fam in sorted({p["family"] for p in pairs}):
        ps = [p for p in pairs if p["family"] == fam]
        moved = [p for p in ps if abs(p["ds"]) >= s_min]
        dir_acc = (sum((p["dy"] * p["direction"]) > 0 for p in moved) / len(moved)) if moved else float("nan")
        st = lambda rows: slope_through_origin([r["ds"] * r["direction"] for r in rows if abs(r["ds"]) >= s_min],
                                               [r["dy"] * r["direction"] for r in rows if abs(r["ds"]) >= s_min])
        strata = {}
        for name, cond in (("1", lambda p: p["n_active"] == 1), ("2", lambda p: p["n_active"] == 2),
                           (">=3", lambda p: p["n_active"] >= 3)):
            strata[name] = st([p for p in ps if cond(p)])
        expl = [p for p in ps if p["mu_est"] >= 0.5]
        out[fam] = {"n": len(ps), "dir_acc": dir_acc, "slope": st(ps),
                    "slope_ci": cluster_bootstrap(ps, st, n=n_boot), "slope_by_active": strata,
                    "exploitation": (sum(not p["human_accept"] for p in expl) / len(expl)) if expl else float("nan"),
                    "moved_share": len(moved) / len(ps) if ps else float("nan")}
    return out


def shift_rows(records: List[dict]):
    """Tab. shift. Per family and image source: estimator reliability (Spearman of calibrated
    score with ratings), membership calibration (ECE of pi*mu vs. acceptance), and the share of
    raw estimator values outside the calibrated domain."""
    out = {}
    for fam in sorted({r["family"] for r in records}):
        for src in sorted({r["source"] for r in records if r["family"] == fam}):
            rs = [r for r in records if r["family"] == fam and r["source"] == src]
            ind = [r for r in rs if r["in_domain"]]
            out[(fam, src)] = {
                "n": len(rs),
                "spearman": spearman([r["s"] for r in ind], [r["rating"] for r in ind]) if len(ind) > 2 else float("nan"),
                "ece": ece([r["mu"] for r in ind], [float(r["accept"]) for r in ind]) if ind else float("nan"),
                "out_of_domain": 1 - len(ind) / len(rs),
            }
    return out


def interval_rows(judgments: List[dict]):
    """Tab. interval_ablation. judgments: variant, prompt, verdict in {weak, ok, strong},
    unguided_verdict, dy_ok (change of the independent measure on images acceptable unguided),
    pair_correct (hedge-ladder pair ordered correctly, or None), nfe, seconds."""
    out = {}
    for v in sorted({j["variant"] for j in judgments}):
        js = [j for j in judgments if j["variant"] == v]
        ok0 = [j for j in js if j["unguided_verdict"] == "ok"]
        pairs = [j["pair_correct"] for j in js if j.get("pair_correct") is not None]
        out[v] = {"pair": sum(pairs) / len(pairs) if pairs else float("nan"),
                  "acc": sum(j["verdict"] == "ok" for j in js) / len(js),
                  "over": sum(j["verdict"] == "strong" for j in js) / len(js),
                  "under": sum(j["verdict"] == "weak" for j in js) / len(js),
                  "ofa": sum(j["verdict"] != "ok" for j in ok0) / len(ok0) if ok0 else float("nan"),
                  "change_on_ok": sum(abs(j["dy_ok"]) for j in ok0) / len(ok0) if ok0 else float("nan"),
                  "nfe": sum(j["nfe"] for j in js) / len(js), "seconds": sum(j["seconds"] for j in js) / len(js)}
    return out


# ---------------------------------------------------------------------------------- selection
def decide_support(validity: Dict[str, dict], single: Dict[str, dict], system_records: List[dict],
                   system_pairs: List[dict], n_act: int, act_share: float, dir_min: float, slope_min: float,
                   unsupported: Iterable[str] = ()):
    """Freeze the status of every predicate family from non-test data only.

    validity[fam] = {"final_bin_valid": bool}               (G_cal, App. A.4)
    single[fam]   = {"dir_lb": float, "slope_lb": float}     (G_int, single-family interventions)
    system_records / system_pairs                             (G_sys, full multi-constraint runs)
    Status: 'degree-guided' | 'presence/binding only' | 'unsupported', with the failed criterion."""
    _check_selection(system_records)
    _check_selection(system_pairs)
    act = activation_rows(system_records, n_act)
    tr = transfer_rows(system_pairs)
    status = {}
    for fam in sorted(set(validity) | set(single) | set(act)):
        reasons = []
        if not validity.get(fam, {}).get("final_bin_valid", False):
            reasons.append("estimator not valid at final bin")
        s = single.get(fam)
        if not s or s["dir_lb"] < dir_min or s["slope_lb"] < slope_min:
            reasons.append("single-family gradient validity")
        a = act.get(fam)
        if not a or a["share_ge_nact"] < act_share:
            reasons.append("activation rule")
        t = tr.get(fam)
        if not t or t["dir_acc"] < dir_min or t["slope_ci"][0] < slope_min:
            reasons.append("full-system transfer")
        status[fam] = {"status": "degree-guided" if not reasons else "presence/binding only", "failed": reasons}
    for fam in unsupported:
        status[fam] = {"status": "unsupported", "failed": ["no estimator"]}
    return status


def choose_sigma_lo(records_by_sigma: Dict[float, List[dict]], families: Iterable[str], n_act: int, act_share: float):
    """Per-family guided windows (G_sys/dev only). For each family r, sigma_lo[r] is the LARGEST
    grid value whose window {t: sigma_t >= sigma_lo} gives r at least n_act active steps on at least
    act_share of the prompts; None if no grid value does (the family is then not degree-guided).
    A family that becomes valid only at low noise therefore gets a low sigma_lo[r] instead of being
    excluded by a window chosen for the others. records_by_sigma[lo] are logs of runs with that window."""
    out = {}
    acts = {}
    for lo in sorted(records_by_sigma, reverse=True):
        acts[lo] = activation_rows(_check_selection(records_by_sigma[lo]), n_act)
    for f in families:
        out[f] = next((lo for lo in sorted(acts, reverse=True)
                       if f in acts[lo] and acts[lo][f]["share_ge_nact"] >= act_share), None)
    return out
