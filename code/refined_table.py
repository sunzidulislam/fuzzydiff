"""Tab. refinement-matched comparison (Exp. ii colour mixture, Exp. iii fuzzy comparison).

  python -I code/refined_table.py auto  --out outputs/paired --device cuda:1   # VQAScore (ii) and OWLv2 direction (iii)
  python -I code/refined_table.py pages --out outputs/paired                   # refined_mixture.html, refined_size.html
  python -I code/refined_table.py table --out outputs/paired                   # after saving the ratings

Rows and their arms (folders under {out}/mixture and {out}/size):
  SD3 (no refinement) unguided | SD3 + R unguided_r | SD 1.x + R sd1x_r | A&E + R ae_r | PD + R pd_r |
  SPO (native SDXL refiner) spo_refiner | SPO + R spo_r | FuzzyDiff S1 + R stage1_plain | S1 + SAR stage1_sar.
"+R" arms come from code/refine_arms.py; stage1_plain is the same pass inside run_experiment.py. A row whose
images are missing (PD: no validated implementation here) prints dashes and stays off the rating pages.

Human columns come from blind pages built by code/annotate.py: Missing lenient / strict (an object absent; absent or
wrong colour), Leakage (ii) or Size violation (iii) in percent of images, Fidelity 0-100. VQA: VQAScore (Lin et
al. 2024, t2v_metrics, CLIP-FlanT5-XL) of the image with its prompt, mean over images. OWL: share of images whose
OWLv2 box-area ratio m_{q,r} has the requested sign (an object without a box counts as a failure). The block below
the table: S1 + SAR minus the best +R baseline per metric, paired over prompts with 95% bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import _paired_comparison, build_mixture_page, build_size_page  # noqa: E402
from paired_table import per_prompt  # noqa: E402

ROWS = (("unguided", "SD3 (no refinement)"), ("unguided_r", "SD3 + R"), ("sd1x_r", "SD 1.x + R"), ("ae_r", "A\\&E + R"),
        ("pd_r", "PD + R"), ("spo_refiner", "SPO (native SDXL refiner)"), ("spo_r", "SPO + R"),
        ("stage1_plain", "FuzzyDiff S1 + R"), ("stage1_sar", "FuzzyDiff S1 + SAR (full)"))
R_BASELINES = ("unguided_r", "sd1x_r", "ae_r", "pd_r", "spo_r")
PAGES = {"mixture": ("refined_mixture", build_mixture_page, "leakage"), "size": ("refined_size", build_size_page, "size")}
VQA_MODEL = "clip-flant5-xl"


def present_arms(out: Path, setting):
    return [a for a, _ in ROWS if any(Path(out, setting, a).glob("*/image.png"))]


# ------------------------------------------------------------------------------------------ automatic metrics
def auto_metrics(out: Path, device):
    import t2v_metrics
    from PIL import Image
    from calibration import ProvisionalCalibration
    from ladder import Owl, _tensor, measure
    from predicates import translate
    from settings import load_settings
    rows = []
    vqa = t2v_metrics.VQAScore(model=VQA_MODEL, device=device)
    for arm in present_arms(out, "mixture"):
        for meta_path in sorted(Path(out, "mixture", arm).glob("*/meta.json")):
            prompt = json.loads(meta_path.read_text())["prompt"]
            score = float(vqa(images=[str(meta_path.parent / "image.png")], texts=[prompt]).flatten()[0])
            rows.append({"setting": "mixture", "arm": arm, "image": meta_path.parent.name, "prompt": prompt, "vqa": score})
    del vqa
    owl, prototype = Owl(device), ProvisionalCalibration(load_settings()).prototype
    for arm in present_arms(out, "size"):
        for meta_path in sorted(Path(out, "size", arm).glob("*/meta.json")):
            prompt = json.loads(meta_path.read_text())["prompt"]
            c = next((c for c in translate(prompt).constraints if c.family == "F8"), None)
            if c is None:
                continue
            img = Image.open(meta_path.parent / "image.png").convert("RGB")
            m = measure(_tensor(img), c, owl.boxes(img, list(c.args)), prototype)
            rows.append({"setting": "size", "arm": arm, "image": meta_path.parent.name, "prompt": prompt,
                         "owl_m": m, "owl": 100.0 if m is not None and m > 0 else 0.0})
    with open(out / "refined_auto.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


# ------------------------------------------------------------------------------------------ table
def lenient(rated, groups):
    """prompt -> arm -> 100 if any object is absent (the lenient missing rate)."""
    answers, out = defaultdict(dict), defaultdict(dict)
    for r in rated:
        if "object" in r:
            answers[(r["group"], r["panel"])][r["object"]] = r
    for (group, panel), objs in answers.items():
        e = groups[group]
        rows = [objs.get(q) for q in e["objects"]]
        if any(r is None or r["present"] is None for r in rows):
            continue
        out[e["prompt"]][e["panels"][panel]] = 100.0 if any(not r["present"] for r in rows) else 0.0
    return out


def collect(out: Path):
    """(setting, metric) -> prompt -> arm -> value."""
    metrics = {}
    for setting, (name, _, field) in PAGES.items():
        rp, kp = out / f"{name}.ratings.json", out / f"{name}.key.json"
        if rp.exists() and kp.exists():
            rated, groups = json.load(open(rp)), json.load(open(kp))
            page = "mixture" if setting == "mixture" else "size"
            for metric, table in per_prompt(page, rated, groups).items():
                metrics[(setting, metric)] = table
            metrics[(setting, "missing_lenient")] = lenient(rated, groups)
    auto = out / "refined_auto.jsonl"
    if auto.exists():
        for r in map(json.loads, open(auto)):
            key = (r["setting"], "vqa" if r["setting"] == "mixture" else "owl")
            metrics.setdefault(key, defaultdict(dict))[r["prompt"]][r["arm"]] = r["vqa" if r["setting"] == "mixture" else "owl"]
    return metrics


def _mean(table, arm):
    vals = [r[arm] for r in (table or {}).values() if arm in r]
    return sum(vals) / len(vals) if vals else None


COLUMNS = (("mixture", "missing", False), ("mixture", "leakage", False), ("mixture", "fidelity", True),
           ("mixture", "vqa", True), ("size", "missing", False), ("size", "size", False), ("size", "fidelity", True),
           ("size", "owl", True))


def table_rows(metrics) -> str:
    def f(x, d=1):
        return "--" if x is None else f"{x:.{d}f}"
    lines = []
    for arm, label in ROWS:
        cells = []
        for setting, metric, _ in COLUMNS:
            if metric == "missing":
                cells.append(f"{f(_mean(metrics.get((setting, 'missing_lenient')), arm))} / "
                             f"{f(_mean(metrics.get((setting, 'missing_strict')), arm))}")
            else:
                cells.append(f(_mean(metrics.get((setting, metric)), arm), 3 if metric == "vqa" else (2 if metric == "fidelity" else 1)))
        if arm == "stage1_plain":
            lines.append("\\midrule")
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


def differences(metrics):
    """S1 + SAR minus the best +R baseline per metric, paired over prompts."""
    out = {}
    for setting, metric, higher in COLUMNS:
        key = (setting, "missing_strict" if metric == "missing" else metric)
        table = metrics.get(key)
        found = [(b, _mean(table, b)) for b in R_BASELINES if _mean(table, b) is not None]
        if not found:
            continue
        best = (max if higher else min)(found, key=lambda t: t[1])[0]
        out[f"{setting} {key[1]}"] = {"baseline": best, **_paired_comparison(table, "stage1_sar", best, lower_is_better=not higher)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("auto", "pages", "table"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "auto":
        print(len(auto_metrics(out, args.device)), "automatic scores")
        return
    if args.action == "pages":
        for setting, (name, build, _) in PAGES.items():
            arms = tuple(present_arms(out, setting))
            build(out / setting, out / f"{name}.html", args.seed, arms=arms)
            print(f"{name}.html: {len(json.load(open(out / f'{name}.key.json')))} prompts x {len(arms)} arms ({', '.join(arms)})")
        return
    metrics = collect(out)
    print(table_rows(metrics))
    diffs = differences(metrics)
    (out / "refined_differences.json").write_text(json.dumps(diffs, indent=1))
    for name, d in diffs.items():
        ci = "--" if not d.get("ci95") else f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        print(f"% {name}: S1+SAR - {d['baseline']} = {d['difference'] if d['difference'] is None else round(d['difference'], 2)} {ci} (n={d['n']})")


if __name__ == "__main__":
    main()
