"""Guidance-strength sweep on the size hedge ladders (dev prompts only).

The Table 2 pilot showed Stage 1 leaving relative size unchanged: 76% of guided steps were capped by
cfgMaxPert (median step 0.006), and guidance starts at sigma 0.75, after SD3 has laid out the scene.
This sweep is a 2x2 design over those two causes, run on the size ladders:

  current   provisional values (dmax 0.2, eta0 0.05, sigma_hi 0.75)
  stronger  dmax 1.0, eta0 0.2
  earlier   sigma_hi 1.0 (guidance from the first step)
  both      stronger and earlier

Each setting is scored without the pipeline's own scorer, by OWLv2 (ladder.py):
  rho    Spearman of hedge rank vs. log box-area ratio, per ladder, averaged (ladder.py's rho)
  moved  mean |change in log box-area ratio| against SD3 on the same prompt and seed: does guidance move size at all
plus rho of FuzzyDiff's own evidence, which shows whether a setting fools its own scorer without moving
the measured size. Pick a setting on these dev ladders; report Table 2 on other prompts.

  python -I code/sweep.py --root outputs/sweep --out outputs/sweep.json --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import ladder_images, measure_run, summarize_auto  # noqa: E402

SETTINGS = {
    "current": {},
    "stronger": {"dmax": 1.0, "eta0": 0.2},
    "earlier": {"sigma_hi": 1.0},
    "both": {"dmax": 1.0, "eta0": 0.2, "sigma_hi": 1.0},
}
FLAGS = {"dmax": "--dmax", "eta0": "--eta0", "sigma_hi": "--sigma-hi"}


def sweep_flags(name: str) -> str:
    """run_experiment.py flags of one setting."""
    return " ".join(f"{FLAGS[k]} {v}" for k, v in SETTINGS[name].items())


def sweep_images(root):
    """Rows of ladder.py for SD3 (root/unguided) and each setting (root/<setting>, arm stage1), arm = setting."""
    root = Path(root)
    rows = ladder_images([root / "unguided"], arms=("unguided",))
    for name in SETTINGS:
        if (root / name).is_dir():
            rows += [{**r, "arm": name} for r in ladder_images([root / name], arms=("stage1",))]
    return rows


def _times(root, name):
    times = []
    for path in (Path(root) / name).glob("*/*/meta.json"):
        meta = json.loads(path.read_text())
        if "failed" not in meta and "time_s" in meta:
            times.append(meta["time_s"])
    return sum(times) / len(times) if times else None


def summarize_sweep(rows, root=None):
    """Per arm (unguided and each setting): measured rho, evidence rho, how far size moved, images, time."""
    measured = summarize_auto(rows)["rho"].get("size", {})
    evidence = summarize_auto([{**r, "value": r["evidence"]} for r in rows])["rho"].get("size", {})
    sd3 = {r["prompt"]: r["value"] for r in rows if r["arm"] == "unguided" and r.get("value") is not None}
    out = {}
    for arm in ["unguided", *SETTINGS]:
        mine = [r for r in rows if r["arm"] == arm]
        if not mine:
            continue
        moves = [abs(r["value"] - sd3[r["prompt"]]) for r in mine if r.get("value") is not None and r["prompt"] in sd3]
        out[arm] = {"rho": measured.get(arm), "evidence_rho": evidence.get(arm),
                    "moved": sum(moves) / len(moves) if moves else None, "images": len(mine),
                    "measured": sum(r.get("value") is not None for r in mine),
                    "time_s": _times(root, "unguided" if arm == "unguided" else arm) if root else None,
                    "flags": "" if arm == "unguided" else sweep_flags(arm)}
    return out


def sweep_table(summary) -> str:
    def num(x, digits=2):
        return "--" if x is None else f"{x:.{digits}f}"
    lines = [f"{'setting':10} {'rho':>6} {'95% CI':>15} {'ev. rho':>8} {'moved':>6} {'n':>4} {'s/img':>6}  flags"]
    for arm, s in summary.items():
        rho = s["rho"] or {}
        ci = rho.get("ci95")
        lines.append(f"{arm:10} {num(rho.get('value')):>6} {('[' + num(ci[0]) + ', ' + num(ci[1]) + ']') if ci else '--':>15} "
                     f"{num((s['evidence_rho'] or {}).get('value')):>8} {num(s['moved']):>6} {s['measured']:>4} "
                     f"{num(s['time_s'], 0):>6}  {s['flags']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with unguided/ and one folder per setting")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from calibration import ProvisionalCalibration
    from ladder import Owl
    from settings import load_settings
    rows = measure_run(sweep_images(args.root), Owl(args.device), ProvisionalCalibration(load_settings()).prototype)
    summary = summarize_sweep(rows, args.root)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    with open(Path(args.out).with_suffix(".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(sweep_table(summary))


if __name__ == "__main__":
    main()
