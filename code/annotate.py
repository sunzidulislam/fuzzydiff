"""Blind rating pages and annotation exchange.

A rating page is a single self-contained HTML file: images in shuffled order under anonymous ids, with
no prompt, arm, or file name. The key linking ids to files is written next to it and is not opened
until the ratings are exported.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import List

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reporting import slope_through_origin  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Blind rating</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:640px;padding:16px}figure{margin:0 0 28px}
img{width:100%%;display:block}label{display:block;margin:8px 0}input[type=range]{width:100%%}
textarea{width:100%%;height:140px}</style>
<h1>Blind rating</h1><p>%(question)s Rate from what you see, without guessing how the image was made.</p>
%(items)s
<button id="export">Show ratings</button><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){
  var out=[];document.querySelectorAll("figure").forEach(function(f){
    out.push({id:f.dataset.id,rating:+f.querySelector("input[type=range]").value,
              broken:f.querySelector("input[type=checkbox]").checked});});
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""
ITEM = """<figure data-id="%(id)s"><img alt="image %(n)d" src="data:image/jpeg;base64,%(data)s">
<label>Rating (0 to 100)<input type="range" min="0" max="100" value="50"></label>
<label><input type="checkbox"> Image is visibly broken</label></figure>"""


def _encode(path, size=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def build_rating_page(images: List[Path], out: Path, seed: int,
                      question: str = "How bright is each scene? 0 is very dark, 100 is very bright.") -> None:
    order = list(images)
    random.Random(seed).shuffle(order)
    key, items = {}, []
    for n, path in enumerate(order):
        anon = hashlib.sha256(f"{seed}|{n}".encode()).hexdigest()[:10]
        key[anon] = str(path)
        items.append(ITEM % {"id": anon, "n": n + 1, "data": _encode(path)})
    out = Path(out)
    out.write_text(PAGE % {"question": html.escape(question), "items": "\n".join(items)}, encoding="utf-8")
    with open(out.with_suffix(".key.json"), "w") as f:
        json.dump(key, f, indent=1)


COMPARE_PAGE = """<!doctype html><meta charset="utf-8"><title>Blind comparison</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:980px;padding:16px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
img{width:100%%;display:block}label{display:block;margin:6px 0}textarea{width:100%%;height:140px}
section{margin-bottom:36px}</style>
<h1>Blind comparison</h1><p>Each row shows one prompt in shuffled order. Mark colour that spread onto the wrong
object, and rate how well each image fits the prompt.</p>
%(groups)s
<button id="export">Show ratings</button><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){
  var out=[];document.querySelectorAll("[data-panel]").forEach(function(p){
    out.push({group:p.dataset.group,panel:p.dataset.panel,fidelity:+p.querySelector("input[type=range]").value,
              leakage:p.querySelector("input[type=checkbox]").checked});});
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def build_comparison_page(out_dir, out: Path, seed: int, arms=("stage1", "stage1_plain", "stage1_sar", "stage1_placebo")) -> None:
    """Same prompt and seed across arms, in shuffled panels. The key is written next to the page."""
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    per_image = {}
    for arm in arms:
        for path in sorted((out_dir / arm).glob("*/image.png")):
            per_image.setdefault(path.parent.name, {})[arm] = path
    groups, key = [], {}
    ids = [i for i, found in sorted(per_image.items()) if len(found) == len(arms)]
    rng.shuffle(ids)
    for n, image in enumerate(ids):
        group = hashlib.sha256(f"{seed}|group|{n}".encode()).hexdigest()[:10]
        found = per_image[image]
        with open(found[arms[0]].parent / "meta.json") as f:
            prompt = json.load(f)["prompt"]
        order = list(arms)
        rng.shuffle(order)
        panels = []
        key[group] = {}
        for letter, arm in zip("ABCDEFGH", order):
            key[group][letter] = arm
            panels.append(f'<figure data-panel="{letter}" data-group="{group}"><figcaption>{letter}</figcaption>'
                          f'<img alt="panel {letter}" src="data:image/jpeg;base64,{_encode(found[arm])}">'
                          '<label>Fits the prompt (0 to 100)<input type="range" min="0" max="100" value="50"></label>'
                          '<label><input type="checkbox"> Colour on the wrong object</label></figure>')
        groups.append(f"<section><h2>{html.escape(prompt)}</h2><div class=panels>{''.join(panels)}</div></section>")
    out = Path(out)
    out.write_text(COMPARE_PAGE % {"groups": "\n".join(groups)}, encoding="utf-8")
    with open(out.with_suffix(".key.json"), "w") as f:
        json.dump(key, f, indent=1)


def score_m1(ratings: Path, key: Path, out_dir: Path, settings) -> dict:
    """M1 pass bar of the spec, from the run's metadata and the blind ratings.

    Author ratings make this a pilot, not paper evidence."""
    with open(ratings) as f:
        rated = {r["id"]: r for r in json.load(f)}
    with open(key) as f:
        paths = json.load(f)
    by_image = {}
    for anon, path in paths.items():
        if anon in rated:
            by_image[str(Path(path).parent)] = rated[anon]

    metas = {}
    for meta_path in Path(out_dir).glob("*/*/meta.json"):
        with open(meta_path) as f:
            metas[meta_path.parent] = json.load(f)
    ds, dy, skipped, failed, human = [], [], 0, 0, []
    unguided = {(m["prompt"], m["seed"]): d for d, m in metas.items() if m.get("arm") == "unguided"}
    for d, m in metas.items():
        arm = str(m.get("arm", "")).split("@")[0]
        if arm not in ("darker", "brighter"):
            continue
        if "skipped" in m:
            skipped += 1
        elif "failed" in m:
            failed += 1
        else:
            ds.append(m["ds"])
            dy.append(m["dy"])
            base = unguided.get((m["prompt"], m["seed"]))
            if base is not None and str(d) in by_image and str(base) in by_image:
                diff = by_image[str(d)]["rating"] - by_image[str(base)]["rating"]
                human.append(diff > 0 if arm == "brighter" else diff < 0)
    n = len(ds)
    accuracy = sum((a > 0) == (b > 0) for a, b in zip(ds, dy)) / n if n else float("nan")
    slope = slope_through_origin(ds, dy) if n else float("nan")
    broken = [r["broken"] for r in by_image.values()]
    agreement = sum(human) / len(human) if human else float("nan")
    broken_share = sum(broken) / len(broken) if broken else float("nan")
    passed = (accuracy >= settings.get("cfgDirAcc") and slope >= settings.get("cfgTransfer")
              and agreement >= settings.get("cfgDirAcc") and broken_share <= 0.1 and failed == 0)
    return {"pairs": n, "skipped": skipped, "failed": failed, "dir_accuracy": accuracy, "transfer_slope": slope,
            "human_direction_agreement": agreement, "human_pairs": len(human), "broken_share": broken_share,
            "passed": bool(passed), "note": "pilot only: author ratings and provisional calibration"}
