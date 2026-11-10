"""Pick prompts SD3 gets wrong, so an ablation has failures to fix.

Reads the unguided images of a run and scores each prompt by its worst constraint: the lowest membership
mu among the final records (degree families, presence, leakage) that run_experiment.py already stores in
meta.json. A prompt fails when that minimum is below the threshold. The failing prompts, worst first, are
written as a JSON list that run_experiment.py --prompts-file reads.

With --objects-present a prompt is kept only when SD3 drew every object (all presence records pass) and
another constraint fails. Without it the worst prompts are mostly those with an object missing, where a
size or colour constraint is gated off and has nothing to act on.

The selection uses the pipeline's own check, so say so when reporting: "prompts on which SD3's image
violates at least one constraint by FuzzyDiff's scorer (mu < 0.5)", and with --objects-present "and
shows every object by the same scorer". The ratings stay independent.

  python -I code/screen.py --run outputs/screen_mix --out outputs/screen_mix.json --keep 5 --objects-present
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def screen(run_dir, threshold: float = 0.5, arm: str = "unguided"):
    """Every prompt of the arm, worst first: prompt, its lowest mu, and the constraints below threshold."""
    rows = []
    for path in sorted((Path(run_dir) / arm).glob("*/meta.json")):
        with open(path) as f:
            meta = json.load(f)
        if "failed" in meta or not meta.get("finals"):
            continue
        low = [f for f in meta["finals"] if f["mu"] < threshold]
        rows.append({"prompt": meta["prompt"], "mu_min": min(f["mu"] for f in meta["finals"]),
                     "failing": [f"{f['family']}/{f['component']} mu={f['mu']:.2f}" for f in low],
                     "objects_present": all(f["family"] != "presence" for f in low)})
    return sorted(rows, key=lambda r: (r["mu_min"], r["prompt"]))


def keep(rows, n: int, objects_present: bool = False):
    """The n worst failing prompts; with objects_present, only those whose objects SD3 all drew."""
    failing = [r for r in rows if r["failing"] and (r["objects_present"] or not objects_present)]
    return [r["prompt"] for r in failing[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="JSON list of the kept prompts")
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--objects-present", action="store_true", help="keep only prompts where SD3 drew every object")
    args = ap.parse_args()
    rows = screen(args.run, args.threshold)
    kept = keep(rows, args.keep, args.objects_present)
    Path(args.out).write_text(json.dumps(kept, indent=1))
    for r in rows:
        print(f"{r['mu_min']:.2f}  {'kept' if r['prompt'] in kept else '    '}  {r['prompt']}  {'; '.join(r['failing'])}")
    failing = [r for r in rows if r["failing"]]
    print(f"{len(failing)} of {len(rows)} prompts fail, {sum(r['objects_present'] for r in failing)} of them with"
          f" every object present; kept {len(kept)} in {args.out}")
    if len(kept) < args.keep:
        print("fewer failing prompts than asked for: screen more candidates (--n) before running the ablation")


if __name__ == "__main__":
    main()
