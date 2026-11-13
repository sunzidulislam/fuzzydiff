"""Tab. calibration under the guidance shift (G_sys): does the calibration fitted on unguided images still hold on
guided ones?

  (section 28's runs: unguided and stage1_sar on G_sys, in outputs/sys)
  python -I code/run_experiment.py --suite soft --group sys --n 10 --arms abl_point,abl_one_sided --out outputs/sys
  python -I code/run_experiment.py --suite size --group sys --n 10 --arms abl_point,abl_one_sided --out outputs/sys
  python -I code/run_experiment.py --suite mixture --group sys --n 10 --arms abl_point,abl_one_sided --out outputs/sys
  python -I code/shift_sys.py records --out outputs/sys
  python -I code/shift_sys.py page    --out outputs/sys          # rating page
  python -I code/shift_sys.py score   --out outputs/sys --labels labels.json

Source: unguided = plain SD3; guided = FuzzyDiff (stage1_sar); variants = every other guided arm found in --out
(by default the point-target and one-sided-target Stage 1 ablations). Per F2a / F6 / F8 constraint of each image
(its main component, scored under the last bin as at the end of sampling): s = calibrated score, mu =
membership, raw = estimator value, in_domain = raw inside the calibrated domain E_{r,b}. A person rates how
strongly the image shows the constraint (0-100) and whether it is acceptable as worded. Columns, as in
reporting.shift_rows: n; rho = Spearman of s with the ratings; ECE = predicted membership mu vs. observed
acceptance (10 equal-mass bins); OOD = share of raw values outside E_{r,b}; rho and ECE use in-domain records.
reporting.shift_rows refuses provisional records, so the numbers here are a pilot and the caption must say so.
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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predicates import translate  # noqa: E402
from reporting import ece, spearman  # noqa: E402

FAMILIES = (("F2a", "F2a Photometric"), ("F6", "F6 Color mixture"), ("F8", "F8 Relative size"))
SOURCES = ("unguided", "guided", "variants")


def source_of(arm):
    return {"unguided": "unguided", "stage1_sar": "guided"}.get(arm, "variants")


def build_records(out: Path, settings):
    from calibration import ProvisionalCalibration
    from constraints import FAMILY_SCORERS
    from guidance import BINS
    calib = ProvisionalCalibration(settings)
    records = []
    for meta_path in sorted(Path(out).glob("*/*/meta.json")):
        arm, iid = meta_path.parent.parent.name, meta_path.parent.name
        meta = json.loads(meta_path.read_text())
        if "failed" in meta or "finals" not in meta or not (meta_path.parent / "image.png").exists():
            continue
        if arm not in ("unguided", "stage1_sar") and not (arm.startswith("abl_") or arm == "stage1"):
            continue
        program = translate(meta["prompt"])
        for f in meta["finals"]:
            if f["family"] not in dict(FAMILIES) or f["component"] != "main" or f.get("s") is None:
                continue
            c = program.constraints[f["cid"]]
            lo, hi = FAMILY_SCORERS[c.family].domain(c, calib, BINS, "main")
            raw = f.get("raw")
            records.append({"arm": arm, "source": source_of(arm), "image": iid, "prompt": meta["prompt"],
                            "cid": f["cid"], "family": f["family"], "args": list(c.args), "hedge": c.hedge,
                            "s": f["s"], "mu": f["mu"], "raw": raw,
                            "in_domain": raw is not None and math.isfinite(raw) and lo <= raw <= hi,
                            "calibration": meta.get("calibration")})
    with open(Path(out) / "shift.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return records


# ------------------------------------------------------------------------------------------ rating page
PAGE = """<!doctype html><meta charset="utf-8"><title>Constraint ratings</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:760px;padding:16px}img{width:100%%;max-width:512px;display:block}
section{margin-bottom:30px}label{margin-right:10px}p.q{margin:6px 0}input[type=range]{width:60%%}textarea{width:100%%;height:140px}</style>
<h1>How strongly does the image show it?</h1><p>For each part of the prompt: move the slider to how strongly the
image shows it (0 not at all, 100 as strongly as it gets), then say whether the image is acceptable for the prompt
as worded, including words like "slightly" or "very". The images come from different methods in random order.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.querySelectorAll("input[type=range]").forEach(function(r){r.oninput=function(){r.nextElementSibling.textContent=r.value;r.dataset.set=1;};});
document.getElementById("export").onclick=function(){var out={},missing=0;
  document.querySelectorAll("[data-key]").forEach(function(p){var c=p.querySelector("input:checked"),r=p.querySelector("input[type=range]");
    if(!c||!r.dataset.set)missing++;out[p.dataset.key]={accept:c?c.value==="yes":null,rating:r.dataset.set?+r.value:null};});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _what(r):
    """The quantity rated, in plain words: 'the mug's darkness', 'how much larger the kettle is than the apple'."""
    c = translate(r["prompt"]).constraints[r["cid"]]
    q = c.args[0]
    if r["family"] == "F8":
        word = "larger" if c.orientation > 0 else "smaller"
        return f"how much {word} the {q} is than the {c.args[1]}"
    if r["family"] == "F6":
        base, tint = c.quantity
        return f"how far the {q}'s colour is from {base} toward {tint}"
    word = {"lightness": "lightness" if c.orientation > 0 else "darkness",
            "chroma": "colourfulness" if c.orientation > 0 else "paleness",
            "scene_lightness": "brightness of the scene" if c.orientation > 0 else "dimness of the scene"}.get(c.quantity, c.quantity)
    return f"the {q}'s {word}"


def _jpeg(path, size=512):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def key(r):
    return f"{r['arm']}/{r['image']}|{r['cid']}"


def build_page(out: Path, page: Path, seed=0):
    records = [json.loads(line) for line in open(Path(out) / "shift.jsonl")]
    by_image = {}
    for r in records:
        by_image.setdefault((r["arm"], r["image"]), []).append(r)
    order = sorted(by_image)
    random.Random(seed).shuffle(order)
    sections = []
    for arm, iid in order:
        rs = by_image[(arm, iid)]
        qs = "".join(f'<p class="q" data-key="{html.escape(key(r))}">Rate {html.escape(_what(r))}:<br>'
                     f'<input type="range" min="0" max="100" value="50"> <b>not set</b><br>Acceptable as worded? '
                     f'<label><input type="radio" name="{html.escape(key(r))}" value="yes">Yes</label>'
                     f'<label><input type="radio" name="{html.escape(key(r))}" value="no">No</label></p>' for r in rs)
        sections.append(f'<section><h2>{html.escape(rs[0]["prompt"])}</h2>'
                        f'<img src="data:image/jpeg;base64,{_jpeg(Path(out, arm, iid, "image.png"))}">{qs}</section>')
    page.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(sections)


# ------------------------------------------------------------------------------------------ scoring
def shift_stats(records):
    """reporting.shift_rows' statistics without its fitted-calibration guard (pilot on provisional runs)."""
    out = {}
    for fam, _ in FAMILIES:
        for src in SOURCES:
            rs = [r for r in records if r["family"] == fam and r["source"] == src]
            ind = [r for r in rs if r["in_domain"]]
            rated = [r for r in ind if r.get("rating") is not None]
            judged = [r for r in ind if r.get("accept") is not None]
            out[(fam, src)] = {"n": len(rs),
                               "spearman": spearman([r["s"] for r in rated], [r["rating"] for r in rated]) if len(rated) > 2 else None,
                               "ece": ece([r["mu"] for r in judged], [float(r["accept"]) for r in judged]) if judged else None,
                               "out_of_domain": (1 - len(ind) / len(rs)) if rs else None}
    return out


def table_rows(stats) -> str:
    def f(x, d=2):
        return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"

    def pct(x):
        return "--" if x is None else f"{100 * x:.0f}"
    lines = []
    for fam, label in FAMILIES:
        cells = [stats[(fam, s)] for s in SOURCES]
        lines.append(f"{label} & unguided / guided / variants & {' / '.join(str(c['n']) for c in cells)} & "
                     f"{' / '.join(f(c['spearman']) for c in cells)} & {' / '.join(f(c['ece'], 3) for c in cells)} & "
                     f"{' / '.join(pct(c['out_of_domain']) for c in cells)}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("records", "page", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "records":
        from settings import load_settings
        rs = build_records(out, load_settings())
        print(len(rs), "constraint records from", len({(r["arm"], r["image"]) for r in rs}), "images")
        return
    if args.action == "page":
        print("rating page:", out / "shift.html", build_page(out, out / "shift.html"), "images")
        return
    records = [json.loads(line) for line in open(out / "shift.jsonl")]
    labels = json.load(open(args.labels)) if args.labels else {}
    for r in records:
        r.update({k: v for k, v in labels.get(key(r), {}).items() if k in ("accept", "rating")})
    stats = shift_stats(records)
    (out / "shift_scores.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in stats.items()}, indent=1))
    print(table_rows(stats))


if __name__ == "__main__":
    main()
