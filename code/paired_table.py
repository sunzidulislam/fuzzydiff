"""Tab. paired differences with 95% bootstrap CIs over the test prompts.

  python -I code/paired_table.py pages --out outputs/paired          # rating pages for the three settings
  python -I code/paired_table.py table --out outputs/paired          # after saving the ratings next to the pages

Arms: unguided SD3, FuzzyDiff Stage 1, full FuzzyDiff (stage1_sar), and the published baselines made by
baselines/generate_test.py (spo, ae). Pages, built blind by code/annotate.py: (i) soft attributes, FMAS and
success; (ii) colour mixtures, missing objects, leakage and Fidelity; (iii) size, missing objects, size relation
and Fidelity. Save each page's "Show ratings" text as {page}.ratings.json next to it (fmas, success, mixture,
size).

Upper block: full FuzzyDiff minus the strongest baseline on that metric (the baseline with the best mean on the
same prompts, chosen after the runs, as the caption says); lower block: Stage 1 minus unguided SD3. Each cell is
the mean paired difference over prompts with a 95% percentile bootstrap CI (annotate._paired_comparison).
Rates are in percent, FMAS on [0, 1], Fidelity 0-100.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import (FMAS_POINTS, _paired_comparison, build_fmas_page, build_mixture_page, build_size_page,  # noqa: E402
                      build_success_page)

ARMS = ("unguided", "stage1", "stage1_sar", "spo", "ae")
BASELINES = {"spo": "SPO", "ae": "A\\&E"}
PAGES = {"fmas": build_fmas_page, "success": build_success_page, "mixture": build_mixture_page, "size": build_size_page}
SUITE_OF_PAGE = {"fmas": "soft", "success": "soft", "mixture": "mixture", "size": "size"}
# (experiment, metric label, page, metric key, higher is better)
UPPER = (("(i)", "Avg FMAS", "fmas", "fmas", True), ("(i)", "Success rate", "success", "success", True),
         ("(ii)", "Missing (strict)", "mixture", "missing_strict", False), ("(ii)", "Leakage", "mixture", "leakage", False),
         ("(iii)", "Missing (strict)", "size", "missing_strict", False), ("(iii)", "Size violation", "size", "size", False),
         ("(iii)", "Fidelity", "size", "fidelity", True))
LOWER = (("(i)", "Avg FMAS", "fmas", "fmas", True), ("(ii)", "Leakage", "mixture", "leakage", False),
         ("(iii)", "Size violation", "size", "size", False))


def per_prompt(page, rated, groups):
    """metric -> prompt -> arm -> value, for one page's ratings."""
    out = defaultdict(lambda: defaultdict(dict))
    if page == "fmas":
        for r in rated:
            if r.get("rating") is not None:
                e = groups[r["group"]]
                out["fmas"][e["prompt"]][e["panels"][r["panel"]]] = (r["rating"] - 1) / (FMAS_POINTS - 1)
        return out
    if page == "success":
        for r in rated:
            if r["present"] is None or (r["present"] and r["acceptable"] is None):
                continue
            e = groups[r["group"]]
            out["success"][e["prompt"]][e["panels"][r["panel"]]] = 100.0 if r["present"] and r["acceptable"] else 0.0
        return out
    field = "leakage" if page == "mixture" else "size"
    answers = defaultdict(dict)
    for r in rated:
        if "object" in r:
            answers[(r["group"], r["panel"])][r["object"]] = r
            continue
        e = groups[r["group"]]
        arm = e["panels"][r["panel"]]
        if r.get(field) is not None:
            failed = r[field] if field == "leakage" else not r[field]       # size: "no" is the violation
            out[field][e["prompt"]][arm] = 100.0 if failed else 0.0
        if r.get("fidelity") is not None:
            out["fidelity"][e["prompt"]][arm] = float(r["fidelity"])
    for (group, panel), objs in answers.items():
        e = groups[group]
        rows = [objs.get(q) for q in e["objects"]]
        coloured = set(e.get("coloured", e["objects"]))
        if any(r is None or r["present"] is None or (r["present"] and q in coloured and r.get("colour") is None)
               for q, r in zip(e["objects"], rows)):
            continue
        wrong = any(not r["present"] or (r["present"] and r.get("colour") is False) for r in rows)
        out["missing_strict"][e["prompt"]][e["panels"][panel]] = 100.0 if wrong else 0.0
    return out


def load_metrics(out: Path):
    metrics = {}
    for page in PAGES:
        rp, kp = out / f"{page}.ratings.json", out / f"{page}.key.json"
        if rp.exists() and kp.exists():
            for name, table in per_prompt(page, json.load(open(rp)), json.load(open(kp))).items():
                metrics[(page, name)] = table
    return metrics


def _mean(table, arm):
    vals = [r[arm] for r in table.values() if arm in r]
    return sum(vals) / len(vals) if vals else None


def strongest(table, higher):
    found = [(b, _mean(table, b)) for b in BASELINES if _mean(table, b) is not None]
    if not found:
        return None
    return (max if higher else min)(found, key=lambda t: t[1])[0]


def table_rows(metrics) -> str:
    def cell(cmp):
        if not cmp or not cmp["n"]:
            return "--"
        lo, hi = cmp["ci95"]
        d = 3 if abs(cmp["difference"]) < 1 and max(abs(lo), abs(hi)) < 1 else 1
        return f"{cmp['difference']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"
    lines = []
    for exp, label, page, key, higher in UPPER:
        table = metrics.get((page, key))
        base = None if table is None else strongest(table, higher)
        cmp = None if base is None else _paired_comparison(table, "stage1_sar", base, lower_is_better=not higher)
        lines.append(f"{exp} & {label} & {BASELINES.get(base, '--')} & {cell(cmp)}\\\\")
    lines.append("\\midrule")
    for exp, label, page, key, higher in LOWER:
        table = metrics.get((page, key))
        cmp = None if table is None else _paired_comparison(table, "stage1", "unguided", lower_is_better=not higher)
        lines.append(f"{exp} & {label} & SD3 & {cell(cmp)}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("pages", "table"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "pages":
        for page, build in PAGES.items():
            sub = out / SUITE_OF_PAGE[page]
            build(sub, out / f"{page}.html", args.seed, arms=ARMS)
            n = len(json.load(open(out / f"{page}.key.json")))
            print(f"{page}.html: {n} prompts x {len(ARMS)} arms")
        return
    metrics = load_metrics(out)
    print(table_rows(metrics))
    summary = {f"{p}|{k}": {arm: _mean(t, arm) for arm in ARMS} for (p, k), t in metrics.items()}
    (out / "paired_scores.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
