"""Pick prompts SD3 gets wrong, so an ablation has failures to fix.

Reads the unguided images of a run and scores each prompt by its worst constraint: the lowest membership
mu among the final records (degree families, presence, leakage) that run_experiment.py already stores in
meta.json. A prompt fails when that minimum is below the threshold. The failing prompts, worst first, are
written as a JSON list that run_experiment.py --prompts-file reads.

The selection uses the pipeline's own check, so say so when reporting: "prompts on which SD3's image
violates at least one constraint by FuzzyDiff's scorer (mu < 0.5)". The ratings stay independent.

  python -I code/screen.py --run outputs/screen_mix --out outputs/screen_mix.json --keep 5
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
                     "failing": [f"{f['family']}/{f['component']} mu={f['mu']:.2f}" for f in low]})
    return sorted(rows, key=lambda r: (r["mu_min"], r["prompt"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="JSON list of the kept prompts")
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    rows = screen(args.run, args.threshold)
    failing = [r for r in rows if r["failing"]]
    kept = [r["prompt"] for r in failing[:args.keep]]
    Path(args.out).write_text(json.dumps(kept, indent=1))
    for r in rows:
        print(f"{r['mu_min']:.2f}  {r['prompt']}  {'; '.join(r['failing'])}")
    print(f"{len(failing)} of {len(rows)} prompts fail; kept {len(kept)} in {args.out}")
    if len(kept) < args.keep:
        print("fewer failing prompts than asked for: screen more candidates (--n) before running the ablation")


if __name__ == "__main__":
    main()
