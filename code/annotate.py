"""Blind rating pages and annotation exchange.

A rating page is a single self-contained HTML file: images in shuffled order under anonymous ids, with
no prompt, arm, or file name. The key linking ids to files is written next to it and is not opened
until the ratings are exported.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import re
import os
import random
import string
import sys
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predicates import translate  # noqa: E402
from reporting import cluster_bootstrap, slope_through_origin  # noqa: E402

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


FMAS_ARMS = ("unguided", "stage1", "stage1_plain", "stage1_sar")
FMAS_POINTS = 5
FMAS_COMPARISONS = (("stage1", "unguided"), ("stage1_sar", "stage1"), ("stage1_sar", "stage1_plain"))
FMAS_PAGE = """<!doctype html><meta charset="utf-8"><title>FMAS rating</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:980px;padding:16px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
figure{margin:0}img{width:100%%;display:block}label{margin-right:10px}textarea{width:100%%;height:140px}
section{margin-bottom:36px}</style>
<h1>FMAS rating</h1><p>Each row shows one prompt, with its images in shuffled order. Rate how well the degree in
each image matches the degree the prompt asks for (words like "slightly" or "very"), from 1 (not at all) to 5
(exactly). Rate from what you see, without guessing how the image was made.</p>
%(groups)s
<button id="export">Show ratings</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){
  var out=[],missing=0;document.querySelectorAll("[data-panel]").forEach(function(p){
    var c=p.querySelector("input:checked");if(!c)missing++;
    out.push({group:p.dataset.group,panel:p.dataset.panel,rating:c?+c.value:null});});
  document.getElementById("missing").textContent=missing?missing+" images not rated yet":"All images rated";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _blind_groups(out_dir, out: Path, seed: int, arms, tag, panel_controls, extra=None):
    """One section per prompt, arms in shuffled panels, the prompt shown and the arm hidden. Writes the key."""
    if len(arms) > len(string.ascii_uppercase):
        raise ValueError(f"{len(arms)} arms, but panels are lettered A to Z")
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    per_image = {}
    for arm in arms:
        for path in sorted((out_dir / arm).glob("*/image.png")):
            per_image.setdefault(path.parent.name, {})[arm] = path
    ids = [i for i, found in sorted(per_image.items()) if len(found) == len(arms)]
    rng.shuffle(ids)
    groups, key = [], {}
    for n, image in enumerate(ids):
        group = hashlib.sha256(f"{seed}|{tag}|{n}".encode()).hexdigest()[:10]
        found = per_image[image]
        with open(found[arms[0]].parent / "meta.json") as f:
            prompt = json.load(f)["prompt"]
        order = list(arms)
        rng.shuffle(order)
        key[group] = {"prompt": prompt, "panels": {}, **(extra(prompt) if extra else {})}
        panels = []
        for letter, arm in zip(string.ascii_uppercase, order):
            key[group]["panels"][letter] = arm
            panels.append(f'<figure data-panel="{letter}" data-group="{group}"><figcaption>{letter}</figcaption>'
                          f'<img alt="panel {letter}" src="data:image/jpeg;base64,{_encode(found[arm])}">'
                          f'{panel_controls(group, letter, prompt)}</figure>')
        groups.append(f"<section><h2>{html.escape(prompt)}</h2><div class=panels>{''.join(panels)}</div></section>")
    with open(Path(out).with_suffix(".key.json"), "w") as f:
        json.dump(key, f, indent=1)
    return "\n".join(groups)


def build_fmas_page(out_dir, out: Path, seed: int, arms=FMAS_ARMS) -> None:
    def controls(group, letter, prompt):
        return "".join(f'<label><input type="radio" name="{group}-{letter}" value="{v}">{v}</label>'
                       for v in range(1, FMAS_POINTS + 1))
    Path(out).write_text(FMAS_PAGE % {"groups": _blind_groups(out_dir, out, seed, arms, "fmas", controls)}, encoding="utf-8")


SUCCESS_PAGE = """<!doctype html><meta charset="utf-8"><title>Success rating</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:980px;padding:16px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
figure{margin:0}img{width:100%%;display:block}label{margin-right:10px}p.q{margin:6px 0}textarea{width:100%%;height:140px}
section{margin-bottom:36px}</style>
<h1>Success rating</h1><p>Each row shows one prompt, with its images in shuffled order. For each image, answer both
questions. If the object is missing, you can skip the second one. Answer from what you see, without guessing how the
image was made.</p>
%(groups)s
<button id="export">Show ratings</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){
  var out=[],missing=0;
  function answer(p,q){var c=p.querySelector('input[data-q="'+q+'"]:checked');return c?c.value==="yes":null;}
  document.querySelectorAll("[data-panel]").forEach(function(p){
    var present=answer(p,"present"),acceptable=answer(p,"acceptable");
    if(present===null||(present&&acceptable===null))missing++;
    out.push({group:p.dataset.group,panel:p.dataset.panel,present:present,acceptable:acceptable});});
  document.getElementById("missing").textContent=missing?missing+" images not finished yet":"All images rated";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def build_success_page(out_dir, out: Path, seed: int, arms=FMAS_ARMS) -> None:
    def controls(group, letter, prompt):
        def question(name, text):
            return (f'<p class="q">{text} '
                    f'<label><input type="radio" data-q="{name}" name="{group}-{letter}-{name}" value="yes">Yes</label>'
                    f'<label><input type="radio" data-q="{name}" name="{group}-{letter}-{name}" value="no">No</label></p>')
        return question("present", "Is the object there?") + question("acceptable", "Is the degree acceptable?")
    Path(out).write_text(SUCCESS_PAGE % {"groups": _blind_groups(out_dir, out, seed, arms, "success", controls)}, encoding="utf-8")


def _paired_comparison(by_prompt, a, b, lower_is_better=False):
    rows = [{"prompt": p, "d": r[a] - r[b]} for p, r in sorted(by_prompt.items()) if a in r and b in r]
    if not rows:
        return {"n": 0, "difference": None, "ci95": None, "better": 0, "worse": 0, "same": 0}
    low, high = cluster_bootstrap(rows, lambda rs: sum(r["d"] for r in rs) / len(rs))
    diffs = [r["d"] for r in rows]
    up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": len(rows), "difference": sum(diffs) / len(diffs), "ci95": [low, high],
            "better": down if lower_is_better else up, "worse": up if lower_is_better else down,
            "same": sum(d == 0 for d in diffs)}


MISSING_PAGE = SUCCESS_PAGE.replace("<title>Success rating</title>", "<title>Missing objects</title>").replace(
    "<h1>Success rating</h1><p>Each row shows one prompt, with its images in shuffled order. For each image, answer both\n"
    "questions. If the object is missing, you can skip the second one.",
    "<h1>Missing objects</h1><p>Each row shows one prompt, with its images in shuffled order. For each object in each\n"
    "image, say whether it is there and whether it has the colour the prompt gives it. If an object is missing, skip its\n"
    "colour question.").replace('images not finished yet', 'answers missing').replace(
    """  document.querySelectorAll("[data-panel]").forEach(function(p){
    var present=answer(p,"present"),acceptable=answer(p,"acceptable");
    if(present===null||(present&&acceptable===null))missing++;
    out.push({group:p.dataset.group,panel:p.dataset.panel,present:present,acceptable:acceptable});});""",
    """  document.querySelectorAll("[data-panel]").forEach(function(p){
    p.querySelectorAll("[data-object]").forEach(function(o){
      var present=answer(o,"present"),colour=answer(o,"colour");
      if(present===null||(present&&colour===null&&o.querySelector('[data-q="colour"]')))missing++;
      out.push({group:p.dataset.group,panel:p.dataset.panel,object:o.dataset.object,present:present,colour:colour});});});""")


def _coloured_objects(prompt):
    """Objects in prompt order, each with the colour term bound to it, if any."""
    program = translate(prompt)
    colour = {c.args[1]: c.args[0] for c in program.constraints if c.family == "binding"}
    return [(q, colour.get(q)) for q in dict.fromkeys(program.objects)]


MIXTURE_PAGE = MISSING_PAGE.replace("<title>Missing objects</title>", "<title>Colour mixture rating</title>").replace(
    "<h1>Missing objects</h1>", "<h1>Colour mixture rating</h1>").replace(
    "colour question.", "colour question. Then say whether a colour meant for one object appears on the other, and move\n"
    "the Fidelity slider to say how well the whole image fits the prompt (0 to 100).").replace(
    """      out.push({group:p.dataset.group,panel:p.dataset.panel,object:o.dataset.object,present:present,colour:colour});});});""",
    """      out.push({group:p.dataset.group,panel:p.dataset.panel,object:o.dataset.object,present:present,colour:colour});});
    var leak=answer(p,"leakage"),slider=p.querySelector("input[type=range]");
    var fidelity=slider.dataset.touched?+slider.value:null;
    if(leak===null||fidelity===null)missing++;
    out.push({group:p.dataset.group,panel:p.dataset.panel,leakage:leak,fidelity:fidelity});});
  document.querySelectorAll("input[type=range]").forEach(function(s){s.dataset.touched="";});""").replace(
    "<script>", """<script>
document.addEventListener("input",function(e){if(e.target.type==="range"){e.target.dataset.touched="1";
  e.target.nextElementSibling.textContent=e.target.value;}});""")
# the line that clears "touched" above must not run on export; remove it again
MIXTURE_PAGE = MIXTURE_PAGE.replace("""
  document.querySelectorAll("input[type=range]").forEach(function(s){s.dataset.touched="";});""", "")


def _object_questions(group, letter, prompt):
    def question(obj, name, text):
        return (f'<p class="q">{html.escape(text)} '
                f'<label><input type="radio" data-q="{name}" name="{group}-{letter}-{obj}-{name}" value="yes">Yes</label>'
                f'<label><input type="radio" data-q="{name}" name="{group}-{letter}-{obj}-{name}" value="no">No</label></p>')
    blocks = []
    for obj, colour in _coloured_objects(prompt):
        ask = question(obj, "present", f"Is there a {obj}?")
        ask += question(obj, "colour", f"Is the {obj} {colour}?") if colour else ""
        blocks.append(f'<div data-object="{html.escape(obj)}">{ask}</div>')
    return "".join(blocks)


def _key_objects(prompt):
    pairs = _coloured_objects(prompt)
    return {"objects": [q for q, _ in pairs], "coloured": [q for q, c in pairs if c]}


def build_mixture_page(out_dir, out: Path, seed: int, arms=FMAS_ARMS) -> None:
    """Exp. ii in one pass: missing objects (per object), attribute leakage and Fidelity (per image)."""
    def controls(group, letter, prompt):
        leak = (f'<p class="q">Is a colour meant for one object on the wrong object? '
                f'<label><input type="radio" data-q="leakage" name="{group}-{letter}-leakage" value="yes">Yes</label>'
                f'<label><input type="radio" data-q="leakage" name="{group}-{letter}-leakage" value="no">No</label></p>')
        fidelity = ('<p class="q">Fidelity: how well does the whole image fit the prompt? '
                    '<input type="range" min="0" max="100" value="50"> <b>not set</b></p>')
        return _object_questions(group, letter, prompt) + leak + fidelity
    Path(out).write_text(MIXTURE_PAGE % {"groups": _blind_groups(out_dir, out, seed, arms, "mixture", controls, _key_objects)},
                         encoding="utf-8")


def build_missing_page(out_dir, out: Path, seed: int, arms=FMAS_ARMS) -> None:
    def controls(group, letter, prompt):
        def question(obj, name, text):
            return (f'<p class="q">{html.escape(text)} '
                    f'<label><input type="radio" data-q="{name}" name="{group}-{letter}-{obj}-{name}" value="yes">Yes</label>'
                    f'<label><input type="radio" data-q="{name}" name="{group}-{letter}-{obj}-{name}" value="no">No</label></p>')
        blocks = []
        for obj, colour in _coloured_objects(prompt):
            ask = question(obj, "present", f"Is there a {obj}?")
            ask += question(obj, "colour", f"Is the {obj} {colour}?") if colour else ""
            blocks.append(f'<div data-object="{html.escape(obj)}">{ask}</div>')
        return "".join(blocks)

    def objects(prompt):
        pairs = _coloured_objects(prompt)
        return {"objects": [q for q, _ in pairs], "coloured": [q for q, c in pairs if c]}
    Path(out).write_text(MISSING_PAGE % {"groups": _blind_groups(out_dir, out, seed, arms, "missing", controls, objects)},
                         encoding="utf-8")


def score_missing(ratings: Path, key: Path) -> dict:
    """Missing objects in percent of images, lenient (absent) and strict (absent or wrong colour); lower is better."""
    with open(ratings) as f:
        rated = json.load(f)
    with open(key) as f:
        groups = json.load(f)
    return _score_missing_rows([r for r in rated if "object" in r], groups)


def _score_missing_rows(rated, groups, comparisons=FMAS_COMPARISONS) -> dict:
    answers = defaultdict(dict)
    for r in rated:
        answers[(r["group"], r["panel"])][r["object"]] = r
    by_prompt = {"lenient": {}, "strict": {}}
    unrated = 0
    for (group, panel), objs in answers.items():
        entry = groups[group]
        expected = entry["objects"]
        rows = [objs.get(q) for q in expected]
        coloured = set(entry.get("coloured", expected))
        if any(r is None or r["present"] is None or (r["present"] and q in coloured and r["colour"] is None)
               for q, r in zip(expected, rows)):
            unrated += 1
            continue
        arm = entry["panels"][panel]
        absent = any(not r["present"] for r in rows)
        wrong = any(r["present"] and r["colour"] is False for r in rows)
        by_prompt["lenient"].setdefault(entry["prompt"], {})[arm] = 100.0 if absent else 0.0
        by_prompt["strict"].setdefault(entry["prompt"], {})[arm] = 100.0 if absent or wrong else 0.0
    out = {"n_prompts": len(by_prompt["lenient"]), "unrated": unrated, "comparisons": {},
           "note": "lower is better; pilot only: author ratings and provisional calibration"}
    for kind, table in by_prompt.items():
        per_arm = defaultdict(list)
        for scores in table.values():
            for arm, value in scores.items():
                per_arm[arm].append(value)
        out[kind] = {arm: sum(v) / len(v) for arm, v in per_arm.items()}
        out["comparisons"][kind] = {f"{a} - {b}": _paired_comparison(table, a, b, lower_is_better=True)
                                    for a, b in comparisons}
    return out


def score_fmas(ratings: Path, key: Path, points: int = FMAS_POINTS) -> dict:
    """Avg FMAS per arm, rescaled to [0, 1], and paired differences on the same prompts."""
    with open(ratings) as f:
        rated = json.load(f)
    with open(key) as f:
        groups = json.load(f)
    by_prompt, unrated = {}, 0
    for r in rated:
        if r["rating"] is None:
            unrated += 1
            continue
        if not 1 <= r["rating"] <= points:
            raise ValueError(f"rating {r['rating']} is outside the scale 1 to {points}")
        entry = groups[r["group"]]
        by_prompt.setdefault(entry["prompt"], {})[entry["panels"][r["panel"]]] = (r["rating"] - 1) / (points - 1)
    per_arm = defaultdict(list)
    for scores in by_prompt.values():
        for arm, value in scores.items():
            per_arm[arm].append(value)
    return {"avg_fmas": {arm: sum(v) / len(v) for arm, v in per_arm.items()}, "n_prompts": len(by_prompt), "unrated": unrated,
            "comparisons": {f"{a} - {b}": _paired_comparison(by_prompt, a, b) for a, b in FMAS_COMPARISONS},
            "note": "pilot only: author ratings and provisional calibration"}


def score_success(ratings: Path, key: Path) -> dict:
    """Success Rate per arm in percent: the object is present and its degree acceptable."""
    with open(ratings) as f:
        rated = json.load(f)
    with open(key) as f:
        groups = json.load(f)
    by_prompt, present_by_arm, unrated = {}, defaultdict(list), 0
    for r in rated:
        if r["present"] is None or (r["present"] and r["acceptable"] is None):
            unrated += 1
            continue
        entry = groups[r["group"]]
        arm = entry["panels"][r["panel"]]
        present_by_arm[arm].append(100.0 if r["present"] else 0.0)
        by_prompt.setdefault(entry["prompt"], {})[arm] = 100.0 if r["present"] and r["acceptable"] else 0.0
    per_arm = defaultdict(list)
    for scores in by_prompt.values():
        for arm, value in scores.items():
            per_arm[arm].append(value)
    return {"success_rate": {arm: sum(v) / len(v) for arm, v in per_arm.items()},
            "object_present": {arm: sum(v) / len(v) for arm, v in present_by_arm.items()},
            "n_prompts": len(by_prompt), "unrated": unrated,
            "comparisons": {f"{a} - {b}": _paired_comparison(by_prompt, a, b) for a, b in FMAS_COMPARISONS},
            "note": "pilot only: author ratings and provisional calibration"}


def score_mixture(ratings: Path, key: Path, field: str = "leakage", comparisons=FMAS_COMPARISONS) -> dict:
    """Exp. ii: missing objects (lenient, strict), leakage in percent of images, and mean Fidelity (0 to 100).
    With field="size" it scores Exp. iii instead: size violations in place of leakage."""
    with open(ratings) as f:
        rated = json.load(f)
    with open(key) as f:
        groups = json.load(f)
    missing = _score_missing_rows([r for r in rated if "object" in r], groups, comparisons)
    leakage, fidelity, unrated = {}, {}, missing["unrated"]
    for r in (r for r in rated if "object" not in r):
        entry = groups[r["group"]]
        arm = entry["panels"][r["panel"]]
        if r[field] is None or r["fidelity"] is None:
            unrated += 1
            continue
        if not 0 <= r["fidelity"] <= 100:
            raise ValueError(f"fidelity {r['fidelity']} is outside 0 to 100")
        # leakage: "yes" is the failure; size: "no" (the relation does not match) is the failure
        failed = r[field] if field == "leakage" else not r[field]
        leakage.setdefault(entry["prompt"], {})[arm] = 100.0 if failed else 0.0
        fidelity.setdefault(entry["prompt"], {})[arm] = float(r["fidelity"])

    def per_arm(table):
        values = defaultdict(list)
        for scores in table.values():
            for arm, v in scores.items():
                values[arm].append(v)
        return {arm: sum(v) / len(v) for arm, v in values.items()}
    name = "leakage" if field == "leakage" else "size_violation"
    paired = {"missing_lenient": missing["comparisons"]["lenient"], "missing_strict": missing["comparisons"]["strict"],
              name: {f"{a} - {b}": _paired_comparison(leakage, a, b, lower_is_better=True) for a, b in comparisons},
              "fidelity": {f"{a} - {b}": _paired_comparison(fidelity, a, b) for a, b in comparisons}}
    return {"missing": {"lenient": missing["lenient"], "strict": missing["strict"]}, name: per_arm(leakage),
            "fidelity": per_arm(fidelity), "n_prompts": len({e["prompt"] for e in groups.values()}), "unrated": unrated,
            "comparisons": paired,
            "note": f"missing and {name.replace('_', ' ')}: lower is better; fidelity: higher is better; "
                    "pilot only: author ratings and provisional calibration"}


# Tab. ablation. Keep in step with run_experiment.ABLATION_TABLE_ARMS (test_ablation checks it).
ABLATION_ROWS = (("unguided", "SD3, no guidance"), ("stage1", "Stage~1 (full)"),
                 ("abl_point", "\\quad point target at interval center"),
                 ("abl_one_sided", "\\quad one-sided (``more is better'')"),
                 ("abl_attention_only", "\\quad attention-only evidence"),
                 ("abl_overlap", "\\quad overlap $b$ instead of contrastive $\\beta$"),
                 (None, "\\quad w/o gradient-validity gate"),
                 ("abl_cond_x0", "\\quad conditional $\\hat x_0$ (not CFG)"),
                 (None, "\\quad w/o noise-bin validity gates"),
                 ("abl_floored", "\\quad floored log loss (saturating)"),
                 ("abl_weighted_sum", "\\quad weighted sum (no priority)"),
                 ("abl_no_acceptance", "\\quad w/o acceptance test"),
                 ("stage1_plain", "S1 + plain SD3 img2img"), ("stage1_sar", "S1 + SAR (logit bias, ours)"),
                 ("stage1_placebo", "S1 + SAR, placebo priors"), ("stage1_sar_nocheck", "S1 + SAR w/o consistency check"))
ABLATION_ARM_NAMES = tuple(a for a, _ in ABLATION_ROWS if a)
ABLATION_COMPARISONS = (("stage1", "unguided"),
                        *((a, "stage1") for a in ABLATION_ARM_NAMES if a.startswith("abl_")),
                        ("stage1_plain", "stage1"), ("stage1_sar", "stage1"), ("stage1_sar", "stage1_plain"),
                        ("stage1_placebo", "stage1_sar"), ("stage1_sar_nocheck", "stage1_sar"))


def _size_question(prompt):
    """The size relation in the prompt's own words, e.g. 'Is the suitcase slightly larger than the mug?'"""
    program = translate(prompt)
    relation = next((c for c in program.constraints if c.family == "F8"), None)
    if relation is None:
        return "Does the size relation match the prompt?"
    q, r = relation.args
    match = re.search(rf"\b{re.escape(q)}\b(.*?)\bthan\b\s+(?:a|an|the)?\s*{re.escape(r)}\b", prompt)
    words = match.group(1).strip() if match else None
    if not words:
        return "Does the size relation match the prompt?"
    return f"Is the {q} {words} than the {r}?"


SIZE_PAGE = MIXTURE_PAGE.replace("<title>Colour mixture rating</title>", "<title>Size relation rating</title>").replace(
    "<h1>Colour mixture rating</h1>", "<h1>Size relation rating</h1>").replace(
    MIXTURE_PAGE[MIXTURE_PAGE.index("For each object in each"):MIXTURE_PAGE.index("Answer from what you see")],
    "For each object in each\nimage, say whether it is there. Then say whether the size relation matches the prompt, "
    "including how much\nlarger or smaller it should be, and move the Fidelity slider to say how well the whole image "
    "fits the prompt (0 to 100). ").replace(
    'answer(p,"leakage")', 'answer(p,"size")').replace("leakage:leak", "size:leak")


def build_size_page(out_dir, out: Path, seed: int, arms=FMAS_ARMS) -> None:
    """Exp. iii in one pass: missing objects (per object), size relation and Fidelity (per image)."""
    def controls(group, letter, prompt):
        size = (f'<p class="q">{html.escape(_size_question(prompt))} '
                f'<label><input type="radio" data-q="size" name="{group}-{letter}-size" value="yes">Yes</label>'
                f'<label><input type="radio" data-q="size" name="{group}-{letter}-size" value="no">No</label></p>')
        fidelity = ('<p class="q">Fidelity: how well does the whole image fit the prompt? '
                    '<input type="range" min="0" max="100" value="50"> <b>not set</b></p>')
        return _object_questions(group, letter, prompt) + size + fidelity
    Path(out).write_text(SIZE_PAGE % {"groups": _blind_groups(out_dir, out, seed, arms, "size", controls, _key_objects)},
                         encoding="utf-8")


def _cell(value, digits=1):
    return "\\tbd" if value is None else f"{value:.{digits}f}"


def ablation_table_rows(mixture=None, size=None) -> str:
    """LaTeX rows of Tab. ablation from score_mixture (Exp. ii) and score_mixture(field="size") (Exp. iii).
    Missing objects are the strict rate; a missing experiment stays TBD."""
    lines = []
    for arm, label in ABLATION_ROWS:
        if arm == "stage1_plain":
            lines.append("\\midrule")
        if arm is None:
            lines.append(f"{label} & \\multicolumn{{6}}{{c}}{{= Stage~1 (no such gate in provisional calibration)}}\\\\")
            continue
        cells = []
        for scores, name in ((mixture, "leakage"), (size, "size_violation")):
            scores = scores or {}
            cells += [_cell(scores.get("missing", {}).get("strict", {}).get(arm)), _cell(scores.get(name, {}).get(arm)),
                      _cell(scores.get("fidelity", {}).get(arm))]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


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


MIN_RATINGS = 3
GRID = 64


def _anonymize(out_dir, seed):
    """(anonymous id, image path) in shuffled order; the key maps ids back to arm/image."""
    images = sorted(Path(out_dir).glob("*/*/image.png"))
    random.Random(seed).shuffle(images)
    return [(hashlib.sha256(f"{seed}|{n}".encode()).hexdigest()[:10], path) for n, path in enumerate(images)]


def _write_key(pairs, path):
    with open(path, "w") as f:
        json.dump({anon: "/".join(p.parts[-3:-1]) for anon, p in pairs}, f, indent=1)


def export_rating_tasks(out_dir, task, out_csv, ratings_per_item=MIN_RATINGS, attention_checks=0.05, seed=0) -> None:
    """Task list for a rating platform. Attention checks repeat an item with a stated answer."""
    pairs = _anonymize(out_dir, seed)
    rng = random.Random(seed)
    out_csv = Path(out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "image", "task", "ratings_needed", "check"])
        for anon, path in pairs:
            w.writerow([anon, f"{anon}.jpg", task, ratings_per_item, ""])
        for n in range(max(1, round(attention_checks * len(pairs)))):
            anon, _ = rng.choice(pairs)
            w.writerow([f"check{n}", f"{anon}.jpg", "attention check: move the slider to 90", ratings_per_item, "check"])
    _write_key(pairs, out_csv.with_suffix(".key.json"))
    images = out_csv.parent / (out_csv.stem + "_images")
    images.mkdir(exist_ok=True)
    for anon, path in pairs:
        Image.open(path).convert("RGB").save(images / f"{anon}.jpg", quality=95)


def export_mask_tasks(out_dir, out_json, seed=0) -> None:
    """Label Studio task list over anonymized copies of the images."""
    pairs = _anonymize(out_dir, seed)
    out_json = Path(out_json)
    images = out_json.parent / (out_json.stem + "_images")
    images.mkdir(exist_ok=True)
    tasks = []
    for anon, path in pairs:
        Image.open(path).convert("RGB").save(images / f"{anon}.png")
        with open(path.parent / "meta.json") as f:
            objects = sorted({f_["family"] for f_ in json.load(f).get("finals", [])})
        tasks.append({"data": {"image": f"{images.name}/{anon}.png", "item": anon}, "meta": {"families": objects}})
    with open(out_json, "w") as f:
        json.dump(tasks, f, indent=1)
    _write_key(pairs, out_json.with_suffix(".key.json"))


def import_ratings(path) -> List[dict]:
    """Rows of workers who passed every attention check; each item needs at least 3 such ratings."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    failed = {r["worker"] for r in rows if r.get("check") == "fail"}
    kept = [r for r in rows if r["worker"] not in failed and not r.get("check")]
    counts = defaultdict(int)
    for r in kept:
        counts[r["item"]] += 1
    short = sorted(i for i, n in counts.items() if n < MIN_RATINGS)
    if short:
        raise ValueError(f"items with fewer than {MIN_RATINGS} valid ratings: {short[:10]}")
    return [{"item": r["item"], "worker": r["worker"], "rating": float(r["rating"]), "n": counts[r["item"]]} for r in kept]


def import_masks(path, grid=GRID) -> dict:
    """Label Studio polygon export -> {item: {label: grid x grid binary mask}}. Points are percentages."""
    with open(path) as f:
        tasks = json.load(f)
    out = {}
    for task in tasks:
        masks = {}
        for annotation in task.get("annotations", []):
            for shape in annotation.get("result", []):
                value = shape["value"]
                label = value["polygonlabels"][0]
                canvas = Image.new("L", (grid * 4, grid * 4), 0)
                points = [(x / 100 * grid * 4, y / 100 * grid * 4) for x, y in value["points"]]
                ImageDraw.Draw(canvas).polygon(points, fill=255)
                small = np.asarray(canvas.resize((grid, grid), Image.BILINEAR), dtype=float) / 255
                masks[label] = np.maximum(masks.get(label, 0), (small > 0.5).astype(float))
        out[task["data"]["item"]] = masks
    return out
