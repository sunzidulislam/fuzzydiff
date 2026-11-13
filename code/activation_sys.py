"""Tab. activation of degree guidance, bottom panel: realized activation per family on G_sys (pilot).

  (section 28's runs: unguided, stage1 and stage1_sar on G_sys, in outputs/sys)
  python -I code/run_experiment.py $(python -I code/activation_sys.py prompts) --group sys \
      --arms unguided,stage1,stage1_sar --log-all-steps --out outputs/sys          # F1 and F4 prompts
  python -I code/activation_sys.py table --out outputs/sys

Constraint records are the main-component lines of the Stage 1 step logs (stage1/*/log.jsonl); the final
membership comes from the unguided and the FuzzyDiff (stage1_sar) images of the same prompt and seed. The
statistics are reporting.activation_stats (activation_rows without its fitted-calibration guard): b0 = first bin
whose validity gate is open, Active = median active steps per constraint, >= n_act = share of constraints with at
least cfgNact active steps, Block = most frequent blocking gate on inactive steps, Acc. = accepted share of active
steps, Delta L(3) = mean tier-3 loss change per accepted step, mu = mean final membership unguided -> guided.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import assign_group, base_prompt, vocab  # noqa: E402
from reporting import activation_stats  # noqa: E402

FAMILIES = (("F1", "F1 Visibility"), ("F2a", "F2a Photometric attribute"), ("F2b", "F2b Open-vocab.\\ attribute"),
            ("F4", "F4 Attribute blending"), ("F6", "F6 Color mixture"), ("F8", "F8 Relative size"))


def extra_prompts(n=4):
    """F1 and F4 prompts whose base prompt falls in G_sys (no suite has them)."""
    out = []
    for make in (lambda o: f"a barely visible {o}", lambda o: f"a slightly white {o}"):
        out += [p for p in (make(o) for o in vocab()["objects"]) if assign_group(base_prompt(p)) == "sys"][:n]
    return out


def _main_finals(meta):
    return {f["cid"]: f for f in meta.get("finals", []) if f["component"] == "main" and f.get("mu") is not None}


def build_records(out: Path):
    records = []
    for log_path in sorted(Path(out, "stage1").glob("*/log.jsonl")):
        iid = log_path.parent.name
        for line in open(log_path):
            r = json.loads(line)
            if r.get("type") == "constraint" and r.get("component") == "main":
                records.append(r)
        metas = {}
        for arm in ("unguided", "stage1_sar"):
            p = Path(out, arm, iid, "meta.json")
            metas[arm] = json.loads(p.read_text()) if p.exists() else {}
        u, g = _main_finals(metas["unguided"]), _main_finals(metas["stage1_sar"])
        for cid in sorted(set(u) & set(g)):
            records.append({"type": "final", "family": g[cid]["family"], "cid": cid, "image": iid,
                            "mu_unguided": u[cid]["mu"], "mu_guided": g[cid]["mu"]})
    return records


def table_rows(stats) -> str:
    def f(x, d=2):
        return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"
    lines = []
    for fam, label in FAMILIES:
        r = stats.get(fam)
        if r is None:
            lines.append(f"{label} & -- & -- & -- & -- & -- & -- & --\\\\")
            continue
        lines.append(f"{label} & {r['b0'] if r['b0'] is not None else '--'} & {f(r['active_median'], 0)} & "
                     f"{100 * r['share_ge_nact']:.0f} & {r['block']} & {f(100 * r['accepted_share'], 0)} & "
                     f"{f(r['dL3_per_accepted'], 3)} & {f(r['mu_unguided'])}$\\to${f(r['mu_guided'])}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prompts", "table"))
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.action == "prompts":
        print(" ".join(f'--prompt "{p}"' for p in extra_prompts()))
        return
    from settings import load_settings
    n_act = int(float(load_settings().get("cfgNact")))
    stats = activation_stats(build_records(Path(args.out)), n_act)
    Path(args.out, "activation_scores.json").write_text(json.dumps(stats, indent=1, default=str))
    print(table_rows(stats))


if __name__ == "__main__":
    main()
