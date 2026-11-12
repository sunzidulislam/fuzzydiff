"""Tab. estimator validity per noise bin, from a calibration run on G_cal: label page, fit, and table rows.

  python -I code/run_experiment.py --suite soft --group cal --n 50 --arms unguided --log-all-steps --out outputs/calib
  python -I code/run_experiment.py --suite size --group cal --n 30 --arms unguided --log-all-steps --out outputs/calib
  python -I code/calibration_pilot.py page  --run outputs/calib --out outputs/calib/labels.html
  python -I code/calibration_pilot.py fit   --run outputs/calib --labels labels.json --out outputs/calib
  python -I code/calibration_pilot.py table --out outputs/calib

The label page asks, for every unguided image, the degree of its graded constraint (0 to 100), whether each
named object is there as an object of its own, and a box around each object that is. Boxes become 64 x 64
masks (the attention grid at 1024^2): their area fraction is the human size, and they decide whether an
attention peak lands on its object. fit joins the labels with the per-step raw values in the run logs
(fit_calibration.rows_from_run) and fits with fit_calibration.fit_main, which cross-fits by base prompt.

A pilot has far fewer than cfgNmin images per cell: every such cell prints its value with "ins.", and the
validity gate stays closed. The last column, out of the domain, counts the guided evaluations (arms other
than unguided, e.g. stage1 in the same run folder) whose domain gate failed.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_calibration import BINS, fit_main, rows_from_run  # noqa: E402
from predicates import translate  # noqa: E402
from prompts import base_prompt  # noqa: E402
from settings import load_settings  # noqa: E402

GRID = 64
DEGREE = ("F1", "F2a", "F2b", "F4", "F6", "F8")
ROWS = (("presence", "Presence (min.\\ AUROC / max.\\ ECE)"), ("F1", "F1 visibility"), ("F2a", "F2a lightness / chroma"),
        ("F2b", "F2b open vocabulary"), ("F4", "F4 whiteness"), ("F6", "F6 color mixture"),
        ("F8", "F8 size ($\\Delta$ vs.\\ human masks)"), ("support", "Support, Stage 2 (mIoU)"))


def degree_question(prompt):
    """The graded constraint of a prompt and its question, or (None, None)."""
    program = translate(prompt)
    c = next((c for c in program.constraints if c.family in DEGREE), None)
    if c is None:
        return None, None
    obj = c.args[0]
    if c.family == "F6":
        base, tint = c.quantity
        return c, f"How far is the {obj}'s colour from {base} toward {tint}? (0 pure {base}, 100 fully {tint})"
    if c.family == "F8":
        word = next(w for w in ("larger", "bigger", "smaller") if w in prompt.split())
        return c, f"How much {word} is the {c.args[0]} than the {c.args[1]}? (0 not {word} at all, 100 far {word})"
    words = [w for w in base_prompt(prompt).split() if w not in ("a", "an", "the") and w != obj]
    word = " ".join(words) or c.family
    return c, f"How {word} is the {obj}? (0 not {word} at all, 100 as {word} as it gets)"


def _items(run_dir, arms=("unguided",)):
    out = []
    for arm in arms:
        for meta_path in sorted(Path(run_dir, arm).glob("*/meta.json")):
            meta = json.load(open(meta_path))
            if "failed" in meta or not (meta_path.parent / "image.png").exists():
                continue
            out.append((f"{arm}/{meta_path.parent.name}", meta["prompt"], meta_path.parent / "image.png"))
    return out


def _jpeg(path, size=512):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html><meta charset="utf-8"><title>Calibration labels</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:900px;padding:16px}section{margin-bottom:36px}
.wrap{position:relative;width:512px;max-width:100%%}.wrap img,.wrap canvas{width:100%%;display:block}
.wrap canvas{position:absolute;left:0;top:0;cursor:crosshair}p.q{margin:6px 0}label{margin-right:10px}
input[type=range]{width:100%%}button.obj.on{outline:3px solid #c33}textarea{width:100%%;height:140px}</style>
<h1>Calibration labels</h1><p>For each image: move the slider to answer the question (0 to 100); say whether each
object is in the picture as an object of its own (printed on or merged into another counts as No); for each
object that is there, press its button and drag a box around it. Answer from what you see.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
var boxes={};
document.querySelectorAll("section").forEach(function(sec){
  var item=sec.dataset.item,cv=sec.querySelector("canvas"),ctx=cv.getContext("2d"),cur=null,start=null;
  boxes[item]={};
  function draw(extra){ctx.clearRect(0,0,cv.width,cv.height);ctx.lineWidth=3;ctx.font="16px system-ui";
    var all=Object.assign({},boxes[item]);if(extra)all[cur]=extra;
    Object.keys(all).forEach(function(o){var b=all[o];ctx.strokeStyle="#e33";ctx.fillStyle="#e33";
      ctx.strokeRect(b[0]*cv.width,b[1]*cv.height,(b[2]-b[0])*cv.width,(b[3]-b[1])*cv.height);
      ctx.fillText(o,b[0]*cv.width+4,b[1]*cv.height+16);});}
  sec.querySelectorAll("button.obj").forEach(function(btn){btn.onclick=function(){
    sec.querySelectorAll("button.obj").forEach(function(b){b.classList.remove("on");});btn.classList.add("on");cur=btn.dataset.obj;};});
  function pos(e){var r=cv.getBoundingClientRect();return [(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height];}
  cv.onmousedown=function(e){if(cur)start=pos(e);};
  cv.onmousemove=function(e){if(start){var p=pos(e);draw([Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])]);}};
  cv.onmouseup=function(e){if(!start)return;var p=pos(e);
    boxes[item][cur]=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;draw();};
});
document.addEventListener("input",function(e){if(e.target.type==="range"){e.target.dataset.touched="1";e.target.nextElementSibling.textContent=e.target.value;}});
document.getElementById("export").onclick=function(){
  var out={},missing=0;
  document.querySelectorAll("section").forEach(function(sec){
    var item=sec.dataset.item,s=sec.querySelector("input[type=range]"),rec={present:{},boxes:boxes[item]};
    rec.rating=s?(s.dataset.touched?+s.value:null):null;if(s&&rec.rating===null)missing++;
    sec.querySelectorAll("[data-present]").forEach(function(p){var c=p.querySelector("input:checked");
      rec.present[p.dataset.present]=c?c.value==="yes":null;if(!c)missing++;
      else if(c.value==="yes"&&!boxes[item][p.dataset.present])missing++;});
    out[item]=rec;});
  document.getElementById("missing").textContent=missing?missing+" answers or boxes missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def build_page(run_dir, out: Path, seed=0):
    items = _items(run_dir)
    random.Random(seed).shuffle(items)
    sections = []
    for n, (item, prompt, image) in enumerate(items):
        c, question = degree_question(prompt)
        objects = list(dict.fromkeys(translate(prompt).objects))
        slider = (f'<p class="q"><b>{html.escape(question)}</b><br><input type="range" min="0" max="100" value="50">'
                  f' <b>not set</b></p>') if c is not None else ""
        present = "".join(
            f'<p class="q" data-present="{html.escape(o)}">Is there a {html.escape(o)}? '
            f'<label><input type="radio" name="{n}-{html.escape(o)}" value="yes">Yes</label>'
            f'<label><input type="radio" name="{n}-{html.escape(o)}" value="no">No</label>'
            f' <button class="obj" data-obj="{html.escape(o)}">box: {html.escape(o)}</button></p>' for o in objects)
        sections.append(f'<section data-item="{html.escape(item)}"><h2>Image {n + 1}: {html.escape(prompt)}</h2>'
                        f'<div class="wrap"><img src="data:image/jpeg;base64,{_jpeg(image)}">'
                        f'<canvas width="512" height="512"></canvas></div>{slider}{present}</section>')
    out.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(items)


def box_mask(box, grid=GRID):
    m = np.zeros((grid, grid))
    x0, y0, x1, y1 = (int(round(v * grid)) for v in box)
    m[max(y0, 0):max(min(y1, grid), y0 + 1), max(x0, 0):max(min(x1, grid), x0 + 1)] = 1.0
    return m


def labels_to_annotations(labels):
    """ratings item -> degree in [0, 1]; presence item -> {object: 0/1}; masks item -> {object: grid mask}."""
    ratings, presence, masks = {}, {}, {}
    for item, rec in labels.items():
        if rec.get("rating") is not None:
            ratings[item] = rec["rating"] / 100.0
        presence[item] = {o: int(v) for o, v in rec.get("present", {}).items() if v is not None}
        masks[item] = {o: box_mask(b) for o, b in rec.get("boxes", {}).items() if rec.get("present", {}).get(o)}
    return ratings, presence, masks


def out_of_domain(run_dir):
    """Per family, the share of guided evaluations (non-unguided arms) whose domain gate failed."""
    total, outside = defaultdict(int), defaultdict(int)
    for log_path in Path(run_dir).glob("*/*/log.jsonl"):
        if log_path.parts[-3] == "unguided":
            continue
        for line in open(log_path):
            r = json.loads(line)
            if r.get("type") == "constraint" and "domain" in r.get("gates", {}):
                total[r["family"]] += 1
                outside[r["family"]] += not r["gates"]["domain"]
    return {f: {"n": total[f], "share": outside[f] / total[f]} for f in total}


def fit(run_dir, labels_path, out_dir):
    labels = json.load(open(labels_path))
    ratings, presence, masks = labels_to_annotations(labels)
    data = rows_from_run(run_dir, ratings, presence, masks)
    out_dir = Path(out_dir)
    (out_dir / "calibration_input.json").write_text(json.dumps(data))
    calib = fit_main(out_dir / "calibration_input.json", out_dir / "calibration.json")
    (out_dir / "out_of_domain.json").write_text(json.dumps(out_of_domain(run_dir), indent=1))
    return {k: len(v) for k, v in data.items()}, calib


def _cell(stats, b, n_min, key="rho_lb"):
    st = stats.get(b)
    if not st:
        return "--"
    value = st.get(key)
    if value is None:
        return "--"
    flag = "" if st["n"] >= n_min else ", ins."
    return f"{value:.2f} ({st['n']}{flag})"


def table_rows(out_dir, settings=None) -> str:
    s = settings or load_settings()
    n_min = s.get("cfgNmin")
    parts = json.load(open(Path(out_dir) / "calibration.json"))
    ood_path = Path(out_dir) / "out_of_domain.json"
    ood = json.load(open(ood_path)) if ood_path.exists() else {}
    lines = []
    for family, label in ROWS:
        if family == "presence":
            stats = parts.get("presence_cells", {})
            cells = []
            for b in BINS:
                st = stats.get(b)
                if not st:
                    cells.append("--")
                    continue
                flag = "" if st["n"] >= n_min and min(st["npos"], st["nneg"]) >= s.get("cfgNposMin") else ", ins."
                cells.append(f"{st['auc_lb']:.2f} / {st['ece_ub']:.2f} ({st['n']}{flag})")
        elif family == "F8":
            cells = [_cell(parts.get("size_cells", {}), b, n_min) for b in BINS]
        elif family == "support":
            cells = ["--"] * len(BINS)          # needs a Stage-2 run on G_cons with masks; not part of the pilot
        else:
            per_quantity = parts.get("cells", {}).get(family, {})
            cells = []
            for b in BINS:
                options = [q[b] for q in per_quantity.values() if b in q]
                worst = min(options, key=lambda st: st["rho_lb"]) if options else None
                cells.append(_cell({b: worst}, b, n_min) if worst else "--")
        o = ood.get(family if family != "support" else "", {})
        last = f"{100 * o['share']:.1f}\\,\\% ({o['n']})" if o else "--"
        lines.append(f"{label} & " + " & ".join(cells) + f" & {last}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("page", "fit", "table"))
    ap.add_argument("--run")
    ap.add_argument("--labels")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.action == "page":
        n = build_page(args.run, Path(args.out))
        print(f"label page: {args.out} ({n} images)")
    elif args.action == "fit":
        counts, calib = fit(args.run, args.labels, args.out)
        print("rows per section:", counts)
        print("valid:", json.dumps(calib.parts["valid"]))
        print(table_rows(args.out))
    else:
        print(table_rows(args.out))


if __name__ == "__main__":
    main()
