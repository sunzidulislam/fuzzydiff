"""Tests of reporting.py and freeze_manifest.py on synthetic logs with known answers.
No numbers from these tests appear in the paper; they check the code that will produce them.
Usage: python -I test_reporting.py
"""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import (LeakageError, ProvisionalError, StepLog, read_jsonl, activation_rows, choose_sigma_lo, decide_support, ece,  # noqa: E402
                       interval_rows, shift_rows, slope_through_origin, spearman, tier_rows, transfer_rows)
from freeze_manifest import freeze  # noqa: E402


def logs(split, fam, n_prompts, active_steps, accepted=True, dl=-0.1):
    out = []
    for p in range(n_prompts):
        for t in range(6):
            act = t < active_steps
            out.append({"type": "constraint", "split": split, "prompt": f"{fam}{p}", "seed": 0, "step": t,
                        "sigma": 1 - t / 6, "bin": 1 + t // 2, "family": fam, "cid": 0, "tier": 3,
                        "gates": {"validity": act, "presence": True, "support": True, "domain": True,
                                  "confidence": True},
                        "active": act, "accepted": accepted if act else None, "dL": dl if act else None})
        out.append({"type": "final", "split": split, "prompt": f"{fam}{p}", "seed": 0, "family": fam, "cid": 0,
                    "tier": 3, "mu_unguided": 0.3, "mu_guided": 0.8})
    return out


def pairs(split, fam, slope, n=60, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ds = rng.uniform(0.1, 0.5)
        out.append({"split": split, "prompt": f"{fam}{i // 3}", "family": fam, "ds": ds,
                    "dy": slope * ds + rng.gauss(0, 0.01), "direction": 1, "mu_est": 0.8,
                    "human_accept": slope > 0.3, "n_active": 1 + i % 3, "conflict": False})
    return out


def test_provisional_refused():
    recs = logs("sys", "F2a", 3, 4)
    for r in recs:
        r["calibration"] = "provisional"
    prs = pairs("sys", "F2a", 0.8)
    for r in prs:
        r["calibration"] = "provisional"
    for fn in (lambda: tier_rows(recs), lambda: activation_rows(recs, 3), lambda: transfer_rows(prs),
               lambda: shift_rows(recs), lambda: interval_rows(recs),
               lambda: choose_sigma_lo({0.5: recs}, ["F2a"], 3, 0.8),
               lambda: decide_support({"F2a": {"final_bin_valid": True}}, {"F2a": {"dir_lb": 1, "slope_lb": 1}},
                                      recs, [], 3, 0.8, 0.8, 0.5)):
        try:
            fn()
            assert False, "provisional records must be refused"
        except ProvisionalError:
            pass


def test_steplog_tags_every_record():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "a.jsonl")
        log = StepLog(path, "dev", "provisional")
        log.write(type="step")
        log.close()
        assert read_jsonl(path)[0]["calibration"] == "provisional"


def main():
    test_provisional_refused()
    test_steplog_tags_every_record()
    # statistics helpers
    assert abs(slope_through_origin([1, 2, 3], [2, 4, 6]) - 2) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert ece([0.0, 1.0], [0.0, 1.0]) == 0.0

    # activation: family A active 4 of 6 steps, family B 1 of 6
    rec = logs("sys", "A", 10, 4) + logs("sys", "B", 10, 1)
    a = activation_rows(rec, n_act=3)
    assert a["A"]["share_ge_nact"] == 1.0 and a["B"]["share_ge_nact"] == 0.0
    assert a["A"]["active_median"] == 4 and a["B"]["block"] == "validity"
    assert abs(a["A"]["dL3_per_accepted"] + 0.1) < 1e-12 and a["A"]["mu_guided"] == 0.8

    # transfer: slope recovered, exploitation depends on human acceptance
    t = transfer_rows(pairs("sys", "A", 0.9) + pairs("sys", "B", 0.05), n_boot=200)
    assert abs(t["A"]["slope"] - 0.9) < 0.02 and t["A"]["slope_ci"][0] > 0.85
    assert t["B"]["exploitation"] == 1.0 and t["A"]["exploitation"] == 0.0

    # tier diagnostics: one accepted step raises tier-2 loss beyond tolerance -> finite-step failure
    steps = [{"type": "step", "split": "sys", "prompt": "x", "step": 0, "tiers": [1, 2, 3],
              "loss_before": {"1": 1.0, "2": 1.0, "3": 1.0}, "loss_after": {"1": 0.9, "2": 1.2, "3": 0.5},
              "cos": {"1-2": -0.3, "1-3": 0.1, "2-3": 0.2}, "removed": {"2": 0.4, "3": 0.0},
              "rejections": ["tier2"], "accepted": True, "eps": {"1": 0.0, "2": 0.1, "3": 0.0}},
             {"type": "step", "split": "sys", "prompt": "x", "step": 1, "tiers": [1, 2, 3],
              "loss_before": {"1": 0.9, "2": 1.0, "3": 0.5}, "loss_after": {"1": 0.8, "2": 0.9, "3": 0.4},
              "cos": {"1-2": 0.3, "1-3": 0.1, "2-3": -0.2}, "removed": {"2": 0.0, "3": 0.2},
              "rejections": [], "accepted": True, "eps": {"1": 0.0, "2": 0.1, "3": 0.0}}]
    tr = tier_rows(steps)
    assert tr[2]["finite_step_failures"] == 0.5 and tr[1]["finite_step_failures"] == 0.0
    assert tr[2]["conflict_share"] == 0.5 and tr[3]["conflict_share"] == 0.25
    assert abs(tr["rejections"]["rate"] - 1 / 3) < 1e-12

    # shift: in-domain reliability and out-of-domain share
    sh = shift_rows([{"split": "sys", "family": "A", "source": s, "s": v, "rating": v, "accept": v > 0.5,
                      "mu": float(v > 0.5), "in_domain": ind}
                     for s in ("unguided", "guided") for v, ind in ((0.1, True), (0.4, True), (0.7, True), (0.9, False))])
    assert sh[("A", "guided")]["spearman"] == 1.0 and abs(sh[("A", "guided")]["out_of_domain"] - 0.25) < 1e-12

    # interval comparison
    iv = interval_rows([{"variant": "interval", "prompt": "p", "verdict": "ok", "unguided_verdict": "ok",
                         "dy_ok": 0.0, "pair_correct": True, "nfe": 50, "seconds": 10},
                        {"variant": "point", "prompt": "p", "verdict": "strong", "unguided_verdict": "ok",
                         "dy_ok": 0.3, "pair_correct": False, "nfe": 50, "seconds": 10}])
    assert iv["interval"]["ofa"] == 0.0 and iv["point"]["ofa"] == 1.0 and iv["point"]["over"] == 1.0

    # selection refuses test data
    for bad in (logs("test", "A", 2, 4),):
        try:
            decide_support({}, {}, bad, [], 3, 0.8, 0.8, 0.5)
            raise AssertionError("leakage not detected")
        except LeakageError:
            pass
    try:
        choose_sigma_lo({0.5: logs("test", "A", 2, 4)}, ["A"], 3, 0.8)
        raise AssertionError("leakage not detected")
    except LeakageError:
        pass
    win = choose_sigma_lo({0.75: logs("sys", "A", 4, 4) + logs("sys", "B", 4, 1),
                           0.5: logs("sys", "A", 4, 4) + logs("sys", "B", 4, 1),
                           0.0: logs("sys", "A", 4, 5) + logs("sys", "B", 4, 4) + logs("sys", "C", 4, 1)},
                          ["A", "B", "C"], 3, 0.8)
    assert win == {"A": 0.75, "B": 0.0, "C": None}, win   # per-family windows; C never activates

    # support decision and frozen manifest
    validity = {"A": {"final_bin_valid": True}, "B": {"final_bin_valid": True}, "C": {"final_bin_valid": False}}
    single = {"A": {"dir_lb": 0.9, "slope_lb": 0.7}, "B": {"dir_lb": 0.9, "slope_lb": 0.7},
              "C": {"dir_lb": 0.9, "slope_lb": 0.7}}
    sysrec = logs("sys", "A", 10, 4) + logs("sys", "B", 10, 4) + logs("sys", "C", 10, 4)
    syspairs = pairs("sys", "A", 0.9) + pairs("sys", "B", 0.05) + pairs("sys", "C", 0.9)
    st = decide_support(validity, single, sysrec, syspairs, 3, 0.8, 0.8, 0.5)
    assert st["A"]["status"] == "degree-guided"
    assert st["B"]["status"] == "presence/binding only" and st["B"]["failed"] == ["full-system transfer"]
    assert st["C"]["failed"] == ["estimator not valid at final bin"]
    cfg = {"cfgNact": "3", "cfgActShare": "0.8", "cfgDirAcc": "0.8", "cfgTransfer": "0.5"}
    with tempfile.TemporaryDirectory() as d:
        m1 = freeze(validity, single, sysrec, syspairs, cfg, d)
        m2 = freeze(validity, single, sysrec, syspairs, cfg, d)
        assert m1["sha256"] == m2["sha256"]
        tex = open(os.path.join(d, "support_rows.tex")).read()
        assert "fdstatus@A\\endcsname{degree}" in tex and "fdstatus@B\\endcsname{pres./bind.}" in tex
    print(json.dumps({"reporting": "pass", "leakage_guard": "pass", "manifest": "pass"}))


if __name__ == "__main__":
    main()
