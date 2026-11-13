"""Tab. Stage-2 consistency check against human judgments, per object-size stratum, on G_cons.

  python -I code/run_experiment.py --suite mixture --group cons --n 20 --arms stage1,stage1_sar_nocheck --out outputs/s2c
  python -I code/run_experiment.py --suite size    --group cons --n 20 --arms stage1,stage1_sar_nocheck --out outputs/s2c
  python -I code/stage2_consistency.py drift --out outputs/s2c --device cuda:1     # CLIPSeg supports, all variants
  python -I code/stage2_consistency.py page  --out outputs/s2c                     # side-by-side label page
  python -I code/stage2_consistency.py score --out outputs/s2c --labels labels.json

Stage 2 runs with SAR on and the consistency check off, so the bias acts on every object and an object can
move. For every object, drift compares its CLIPSeg support on the Stage-2 image with the one on its Stage-1
image: soft IoU and the centroid shift relative to the reference radius sqrt(A/pi), the two numbers drift_ok
tests. The stratum is the Stage-1 support's area (sar.stratum). A person says, from the two images side by side,
whether the object moved or changed.

score fits the drift thresholds out of fold (calibration_fit.select_drift: per stratum the least miss rate with
the false-disable upper bound at most cfgFDmax; folds by base prompt) and reports, per stratum, the share of
objects disabled, the false-disable rate (disabled among unmoved objects), the miss rate (kept among moved
objects) and the fallback rate (no valid threshold, so no SAR), with 95% cluster-bootstrap intervals. Rows:
the default support; CLIPSeg on a 512^2 image; CLIPSeg temperature 0.5 and 2; and the fitted thresholds moved
by theta +- 0.1 and delta x {0.5, 2}.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_fit import FOLDS, fold_of, select_drift  # noqa: E402
from estimators import area, centroid  # noqa: E402
from predicates import translate  # noqa: E402
from prompts import base_prompt  # noqa: E402
from sar import soft_iou, stratum  # noqa: E402

STRATA = ("small", "medium", "large")
VARIANTS = {"default": {}, "512": {"resize": 512}, "temp 0.5": {"temperature": 0.5}, "temp 2": {"temperature": 2.0}}
STAGE2_ARM = "stage1_sar_nocheck"


def _load(path):
    from PIL import Image
    x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(x).permute(2, 0, 1)


def _supports(seg, x, objects, resize=None, temperature=1.0):
    if resize:
        x = F.interpolate(x[None], size=(resize, resize), mode="bilinear", align_corners=False, antialias=True)[0]
    old = getattr(seg, "temperature", None)
    if old is not None:
        seg.temperature = temperature
    try:
        with torch.no_grad():
            return seg(x, objects)
    finally:
        if old is not None:
            seg.temperature = old


def drift_records(seg, out: Path):
    """One record per (image, object, variant): stratum, soft IoU, relative centroid shift."""
    rows = []
    for meta_path in sorted(Path(out, STAGE2_ARM).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        ref_dir = Path(out, "stage1", meta_path.parent.name)
        if "failed" in meta or not (meta_path.parent / "image.png").exists() or not (ref_dir / "image.png").exists():
            continue
        objects = list(dict.fromkeys(translate(meta["prompt"]).objects))
        x1, x2 = _load(ref_dir / "image.png"), _load(meta_path.parent / "image.png")
        for name, kw in VARIANTS.items():
            s1, s2 = _supports(seg, x1, objects, **kw), _supports(seg, x2, objects, **kw)
            for q in objects:
                ref, now = s1[q].double(), s2[q].double()
                a = float(area(ref))
                shift = float(torch.linalg.norm(centroid(now) - centroid(ref)))
                rows.append({"image": meta_path.parent.name, "object": q, "variant": name, "prompt": meta["prompt"],
                             "base": base_prompt(meta["prompt"]), "stratum": stratum(a), "area": a,
                             "iou": float(soft_iou(now, ref)), "shift_rel": shift / max(math.sqrt(a / math.pi), 1e-6)})
    return rows


# ------------------------------------------------------------------------------------------ label page
PAGE = """<!doctype html><meta charset="utf-8"><title>Stage-2 movement labels</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:1000px;padding:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
img{width:100%%;display:block}section{margin-bottom:34px}label{margin-right:10px}p.q{margin:6px 0}textarea{width:100%%;height:140px}</style>
<h1>Did the object move?</h1><p>Each row shows the same prompt before (left) and after (right) a refinement pass.
For each object, say whether it moved, changed its shape or size, or disappeared between the two images (Yes),
or stayed where it was (No). Small changes of texture or colour do not count.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){var out={},missing=0;
  document.querySelectorAll("[data-object]").forEach(function(p){var c=p.querySelector("input:checked");if(!c)missing++;
    var id=p.dataset.image;out[id]=out[id]||{};out[id][p.dataset.object]=c?c.value==="yes":null;});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _jpeg(path, size=448):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_page(out: Path, page: Path, seed=0):
    folders = [p.parent for p in sorted(Path(out, STAGE2_ARM).glob("*/image.png"))
               if Path(out, "stage1", p.parent.name, "image.png").exists()]
    random.Random(seed).shuffle(folders)
    sections = []
    for folder in folders:
        prompt = json.loads((folder / "meta.json").read_text())["prompt"]
        iid = folder.name
        qs = "".join(f'<p class="q" data-image="{iid}" data-object="{html.escape(q)}">Did the {html.escape(q)} move or change? '
                     f'<label><input type="radio" name="{iid}-{html.escape(q)}" value="yes">Yes</label>'
                     f'<label><input type="radio" name="{iid}-{html.escape(q)}" value="no">No</label></p>'
                     for q in dict.fromkeys(translate(prompt).objects))
        sections.append(f'<section><h2>{html.escape(prompt)}</h2><div class="pair">'
                        f'<img src="data:image/jpeg;base64,{_jpeg(Path(out, "stage1", iid, "image.png"))}">'
                        f'<img src="data:image/jpeg;base64,{_jpeg(folder / "image.png")}"></div>{qs}</section>')
    page.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(sections)


# ------------------------------------------------------------------------------------------ scoring
def _decide(r, choice, theta_shift=0.0, delta_scale=1.0):
    """disabled / kept / fallback under the stratum's fitted thresholds (None = no valid threshold)."""
    c = choice.get(r["stratum"])
    if c is None:
        return "fallback"
    theta, delta = c.theta + theta_shift, c.delta_rel * delta_scale
    return "disabled" if (r["iou"] < theta or r["shift_rel"] > delta) else "kept"


def oof_decisions(rows, fd_max, k=FOLDS, **perturb):
    """Each object decided with thresholds fitted on the other folds (grouped by base prompt)."""
    out = []
    for i in range(k):
        train = [r for r in rows if fold_of(r["base"], k) != i]
        test = [r for r in rows if fold_of(r["base"], k) == i]
        if not test:
            continue
        choice = select_drift(train, fd_max) if train else {}
        out += [{**r, "decision": _decide(r, choice, **perturb)} for r in test]
    return out


def _rates(rows):
    if not rows:
        return None
    n = len(rows)
    dis = [r for r in rows if r["decision"] == "disabled"]
    fixed = [r for r in rows if not r["moved"] and r["decision"] != "fallback"]
    moved = [r for r in rows if r["moved"] and r["decision"] != "fallback"]
    return {"disabled": len(dis) / n,
            "false_dis": sum(r["decision"] == "disabled" for r in fixed) / len(fixed) if fixed else None,
            "miss": sum(r["decision"] == "kept" for r in moved) / len(moved) if moved else None,
            "fallback": sum(r["decision"] == "fallback" for r in rows) / n, "n": n}


def _ci(rows, key, n=1000, seed=0):
    groups = defaultdict(list)
    for r in rows:
        groups[r["base"]].append(r)
    keys, rng, vals = list(groups), random.Random(seed), []
    for _ in range(n):
        v = _rates([r for g in (rng.choice(keys) for _ in keys) for r in groups[g]])
        if v and v[key] is not None:
            vals.append(v[key])
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]) if vals else (None, None)


def score(out: Path, labels, fd_max, ci=True):
    drift = [json.loads(line) for line in open(Path(out) / "drift.jsonl")]
    rows = [{**r, "moved": bool(labels[r["image"]][r["object"]])} for r in drift
            if labels.get(r["image"], {}).get(r["object"]) is not None]
    settings = [("default", "small / medium / large", "default", {}),
                ("512", "$512^2$", "512", {}), ("temp 0.5", "CLIPSeg temp.\\ 0.5", "temp 0.5", {}),
                ("temp 2", "CLIPSeg temp.\\ 2", "temp 2", {}),
                ("theta+0.1", "$\\theta_{\\rm drift}+0.1$", "default", {"theta_shift": 0.1}),
                ("theta-0.1", "$\\theta_{\\rm drift}-0.1$", "default", {"theta_shift": -0.1}),
                ("delta x0.5", "$\\delta_{\\rm rel}\\times0.5$", "default", {"delta_scale": 0.5}),
                ("delta x2", "$\\delta_{\\rm rel}\\times2$", "default", {"delta_scale": 2.0})]
    result = {}
    for key, label, variant, perturb in settings:
        decided = oof_decisions([r for r in rows if r["variant"] == variant], fd_max, **perturb)
        per = {}
        for s in STRATA:
            cell = [r for r in decided if r["stratum"] == s]
            rates = _rates(cell)
            if rates and ci and key == "default":
                rates["ci"] = {m: _ci(cell, m) for m in ("disabled", "false_dis", "miss", "fallback")}
            per[s] = rates
        result[key] = {"label": label, "strata": per}
    return result


def table_rows(result) -> str:
    def pct(x):
        return "--" if x is None else f"{100 * x:.0f}"

    def cell(per, m, with_ci):
        parts = []
        for s in STRATA:
            r = per.get(s)
            if not r:
                parts.append("--")
                continue
            v = pct(r[m])
            if with_ci and r.get("ci") and r["ci"][m][0] is not None:
                v += f" [{pct(r['ci'][m][0])}, {pct(r['ci'][m][1])}]"
            parts.append(v)
        return " / ".join(parts)
    lines = []
    for key, r in result.items():
        per = r["strata"]
        ns = "/".join(str(per[s]["n"]) if per.get(s) else "0" for s in STRATA)
        lines.append(f"Out-of-fold & {r['label']} (n {ns}) & " +
                     " & ".join(cell(per, m, key == "default") for m in ("disabled", "false_dis", "miss", "fallback")) + "\\\\")
    lines.append("Test & small / medium / large & -- & -- & -- & --\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("drift", "page", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "drift":
        from estimators import ClipSegSupport
        from settings import load_settings
        seg = ClipSegSupport.from_pretrained(load_settings().get("cfgCLIPSeg", str), args.device)
        rows = drift_records(seg, out)
        with open(out / "drift.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(len(rows), "drift records")
        return
    if args.action == "page":
        print("label page:", out / "stage2_labels.html", build_page(out, out / "stage2_labels.html"), "images")
        return
    from settings import load_settings
    result = score(out, json.load(open(args.labels)), load_settings().get("cfgFDmax"))
    (out / "stage2_scores.json").write_text(json.dumps(result, indent=1, default=str))
    print(table_rows(result))


if __name__ == "__main__":
    main()
