"""Tab. hedge ladder: does a method order barely < slightly < moderately < very < extremely?

Per family (attribute, colour mixture, size) and method:
  Pair  human pairwise ordering accuracy (%, chance 50), from one 0-100 degree rating per image: the five
        images of a ladder are shown together, shuffled, under the base prompt without its hedge, and
        each of the ten hedge pairs counts 1 when the stronger hedge is rated higher, 0.5 on a tie.
  rho   Spearman between hedge rank and an independent measurement, per ladder, averaged over ladders:
        attribute  L* (lightness) or C*ab (chroma) of the object,
        colour     OKLab mixture fraction on the prototype axis of the colour term,
        size       log box-area ratio of the two objects,
        all inside OWLv2 boxes, never CLIPSeg or SigLIP masks. A ladder counts only when every image has
        its boxes ("size uses only prompts with measurable magnitude").
  Evidence row: Spearman between FuzzyDiff's own evidence s_k on each final image (meta.json finals,
        oriented like the prompt) and the human degree rating, over the rated images of every method.

  python -I code/ladder.py --runs outputs/ladder_attribute,outputs/ladder_mixture,outputs/ladder_size \
      --out outputs/ladder_auto.json --device cuda:1
"""
from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import os
import random
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import _encode  # noqa: E402
from color_stats import mixture_stats, srgb_to_oklab  # noqa: E402
from estimators import srgb_to_cielab  # noqa: E402
from predicates import translate  # noqa: E402
from prompts import LADDER_HEDGES, base_prompt, degree_word, hedge_of  # noqa: E402
from reporting import cluster_bootstrap, spearman  # noqa: E402

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OWL_THRESHOLD = 0.1
BOX_CORE = 0.6           # attribute and colour are read from the central 60% of the box, away from the background
FAMILIES = {"F2a": "attribute", "F6": "mixture", "F8": "size"}
# Keep in step with run_experiment.LADDER_ARMS (test_ladder checks it). None: no SD3-Medium release.
LADDER_ROWS = (("unguided", "SD3"), ("prompt_weight", "SD3 + prompt weighting"), (None, "Concept Sliders"),
               (None, "SPO"), ("stage1", "FuzzyDiff Stage 1"), ("stage1_sar", "FuzzyDiff"))
LADDER_ARM_NAMES = tuple(a for a, _ in LADDER_ROWS if a)


def degree_constraint(prompt):
    """The constraint the hedge grades (F2a, F6 or F8)."""
    return next(c for c in translate(prompt).constraints if c.family in FAMILIES)


def ladder_images(run_dirs, arms=LADDER_ARM_NAMES):
    """Every finished image of the ladder runs: arm, prompt, base, hedge rank, family, path, evidence."""
    rows = []
    for run in run_dirs:
        for arm in arms:
            for path in sorted((Path(run) / arm).glob("*/image.png")):
                with open(path.parent / "meta.json") as f:
                    meta = json.load(f)
                if "failed" in meta:
                    continue
                c = degree_constraint(meta["prompt"])
                rows.append({"arm": arm, "prompt": meta["prompt"], "base": base_prompt(meta["prompt"]),
                             "rank": LADDER_HEDGES.index(hedge_of(meta["prompt"])), "family": FAMILIES[c.family],
                             "image": str(path), "evidence": _evidence(meta, c)})
    return rows


def _evidence(meta, c):
    """FuzzyDiff's own degree score s of the final image, oriented so larger means more of the prompt's word."""
    for f in meta.get("finals", []):
        if f["family"] == c.family and f["component"] == "main" and f["s"] is not None:
            return c.orientation * f["s"]
    return None


# ------------------------------------------------------------------------------ independent measurement
class Owl:
    """OWLv2 boxes: the best box per object name above the threshold, in pixels, or None."""

    def __init__(self, device, name=OWL_MODEL):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to(device).eval()
        self.processor = Owlv2Processor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def boxes(self, img, names):
        x = self.processor(text=[[f"a photo of a {n}" for n in names]], images=img.convert("RGB"),
                           return_tensors="pt").to(self.device)
        out = self.model(**x)
        scores = out.logits[0].sigmoid().float().cpu()                 # [boxes, names]
        pred = out.pred_boxes[0].float().cpu()                        # cx, cy, w, h of the padded square
        side = max(img.size)
        found = {}
        for j, name in enumerate(names):
            best = int(scores[:, j].argmax())
            if scores[best, j] < OWL_THRESHOLD:
                found[name] = None
                continue
            cx, cy, w, h = (pred[best] * side).tolist()
            found[name] = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        return found


def _core(x, box):
    x0, y0, x1, y1 = box
    mx, my = (1 - BOX_CORE) / 2 * (x1 - x0), (1 - BOX_CORE) / 2 * (y1 - y0)
    _, h, w = x.shape
    a, b = max(0, int(y0 + my)), min(h, int(math.ceil(y1 - my)))
    c, d = max(0, int(x0 + mx)), min(w, int(math.ceil(x1 - mx)))
    return x[:, a:b, c:d] if b > a and d > c else None


def measure(x, c, boxes, prototype):
    """The degree the prompt's word asks for, measured in the OWLv2 boxes, or None without a box.
    x is [3, H, W] in [0, 1]; prototype(term) gives the term's OKLab colour."""
    if c.family == "F8":
        q, r = boxes.get(c.args[0]), boxes.get(c.args[1])
        if q is None or r is None:
            return None
        area = [max(b[2] - b[0], 1e-6) * max(b[3] - b[1], 1e-6) for b in (q, r)]
        return c.orientation * math.log(area[0] / area[1])
    box = boxes.get(c.args[0])
    core = None if box is None else _core(x, box)
    if core is None:
        return None
    if c.family == "F2a":
        lab = srgb_to_cielab(core)
        value = lab[0].mean() if c.quantity == "lightness" else torch.sqrt(lab[1] ** 2 + lab[2] ** 2).mean()
        return c.orientation * float(value)
    base, tint = c.quantity
    c2 = prototype(tint)
    c1 = torch.stack([c2[0], torch.zeros_like(c2[0]), torch.zeros_like(c2[0])]) if base == "achromatic" else prototype(base)
    small = F.interpolate(core[None], size=(128, 128), mode="area")[0]
    lab = srgb_to_oklab(small)
    return float(mixture_stats(lab, torch.ones(128, 128, dtype=lab.dtype), c1, c2)["rho"])


def _tensor(img):
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float64) / 255).permute(2, 0, 1)


def measure_run(rows, detector, prototype):
    """Adds "value" (the independent measurement, or None) to each image row."""
    for r in rows:
        c = degree_constraint(r["prompt"])
        img = Image.open(r["image"])
        names = list(dict.fromkeys(c.args))
        r["value"] = measure(_tensor(img), c, detector.boxes(img, names), prototype)
    return rows


def _ladders(rows, field):
    """(family, arm, base) -> {rank: value} over rows that have the field."""
    out = {}
    for r in rows:
        if r.get(field) is not None:
            out.setdefault((r["family"], r["arm"], r["base"]), {})[r["rank"]] = r[field]
    return out


def _mean_ci(per_ladder):
    """Mean over ladders with a cluster-bootstrap CI over base prompts."""
    if not per_ladder:
        return {"value": None, "ci95": None, "n": 0}
    rows = [{"prompt": b, "v": v} for b, v in per_ladder]
    low, high = cluster_bootstrap(rows, lambda rs: sum(r["v"] for r in rs) / len(rs))
    return {"value": sum(v for _, v in per_ladder) / len(per_ladder), "ci95": [low, high], "n": len(per_ladder)}


def _rank_rho(ladder):
    ranks = sorted(ladder)
    rho = spearman(ranks, [ladder[k] for k in ranks])
    return 0.0 if math.isnan(rho) else rho        # a flat ladder orders nothing


def summarize_auto(rows):
    """rho per family and arm: per-ladder Spearman of hedge rank vs. measurement, over complete ladders."""
    ladders = _ladders(rows, "value")
    out, excluded = {}, {}
    for family, arm in sorted({(r["family"], r["arm"]) for r in rows}):
        bases = sorted({r["base"] for r in rows if r["family"] == family and r["arm"] == arm})
        complete = [(b, ladders[(family, arm, b)]) for b in bases
                    if len(ladders.get((family, arm, b), {})) == len(LADDER_HEDGES)]
        out.setdefault(family, {})[arm] = _mean_ci([(b, _rank_rho(v)) for b, v in complete])
        excluded.setdefault(family, {})[arm] = len(bases) - len(complete)
    return {"rho": out, "excluded_ladders": excluded,
            "measure": f"OWLv2 ({OWL_MODEL}, threshold {OWL_THRESHOLD}); attribute L* or C*ab and colour OKLab "
                       f"mixture fraction in the central {BOX_CORE:.0%} of the box; size log box-area ratio"}


# ------------------------------------------------------------------------------ human ratings
LADDER_PAGE = """<!doctype html><meta charset="utf-8"><title>Hedge ladder rating</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:1100px;padding:16px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
figure{margin:0}img{width:100%%;display:block}input[type=range]{width:100%%}p.q{margin:6px 0}
textarea{width:100%%;height:140px}section{margin-bottom:36px}</style>
<h1>Hedge ladder rating</h1><p>Each row shows one prompt five times, at different strengths, in shuffled order.
Answer the question under the row for every image by moving its slider (0 to 100). Compare the images in a row
with each other: the slider says how strong the quality is, not how good the image is. Answer from what you see,
without guessing how the image was made.</p>
%(groups)s
<button id="export">Show ratings</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.addEventListener("input",function(e){if(e.target.type==="range"){e.target.dataset.touched="1";
  e.target.nextElementSibling.textContent=e.target.value;}});
document.getElementById("export").onclick=function(){
  var out=[],missing=0;
  document.querySelectorAll("[data-panel]").forEach(function(p){
    var s=p.querySelector("input[type=range]"),v=s.dataset.touched?+s.value:null;
    if(v===null)missing++;
    out.push({group:p.dataset.group,panel:p.dataset.panel,degree:v});});
  document.getElementById("missing").textContent=missing?missing+" images not rated yet":"All images rated";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def ladder_question(prompt):
    """The degree question of a ladder, in the prompt's words, with what 0 and 100 mean."""
    c = degree_constraint(prompt)
    word = degree_word(prompt)
    if c.family == "F2a":
        return f"How {word} is the {c.args[0]}? (0 not {word} at all, 100 as {word} as it gets)"
    if c.family == "F6":
        base, tint = c.quantity
        return f"How far is the {c.args[0]}'s colour from {base} toward {tint}? (0 pure {base}, 100 fully {tint})"
    q, r = c.args
    other = "smaller" if word == "larger" else "larger"
    return f"How much {word} is the {q} than the {r}? (0 the same size or {other}, 100 far {word})"


def build_ladder_page(run_dirs, out: Path, seed: int, arms=LADDER_ARM_NAMES) -> None:
    """One section per complete ladder (base prompt and method), its five images shuffled; the hedge and the
    method are hidden. Writes the key next to the page."""
    rng = random.Random(seed)
    ladders = {}
    for r in ladder_images(run_dirs, arms):
        ladders.setdefault((r["family"], r["arm"], r["base"]), {})[r["rank"]] = r
    order = sorted(k for k, v in ladders.items() if len(v) == len(LADDER_HEDGES))
    rng.shuffle(order)
    groups, key = [], {}
    for n, (family, arm, base) in enumerate(order):
        group = hashlib.sha256(f"{seed}|ladder|{n}".encode()).hexdigest()[:10]
        images = list(ladders[(family, arm, base)].values())
        rng.shuffle(images)
        key[group] = {"base": base, "arm": arm, "family": family, "panels": {}}
        panels = []
        for letter, r in zip(string.ascii_uppercase, images):
            key[group]["panels"][letter] = {"prompt": r["prompt"], "rank": r["rank"], "evidence": r["evidence"]}
            panels.append(f'<figure data-panel="{letter}" data-group="{group}"><figcaption>{letter}</figcaption>'
                          f'<img alt="panel {letter}" src="data:image/jpeg;base64,{_encode(r["image"])}">'
                          '<p class="q"><input type="range" min="0" max="100" value="50"> <b>not set</b></p></figure>')
        question = html.escape(ladder_question(images[0]["prompt"]))
        groups.append(f"<section><h2>{html.escape(base)}</h2><p class=q><b>{question}</b></p>"
                      f"<div class=panels>{''.join(panels)}</div></section>")
    Path(out).write_text(LADDER_PAGE % {"groups": "\n".join(groups)}, encoding="utf-8")
    with open(Path(out).with_suffix(".key.json"), "w") as f:
        json.dump(key, f, indent=1)


def _pair_accuracy(ladder):
    """Share of hedge pairs, both rated, that the ratings put in hedge order; a tie counts half."""
    pairs = [(i, j) for i, j in itertools.combinations(sorted(ladder), 2)]
    if not pairs:
        return None
    return sum(1.0 if ladder[j] > ladder[i] else 0.5 if ladder[j] == ladder[i] else 0.0 for i, j in pairs) / len(pairs)


def score_ladder(ratings: Path, key: Path) -> dict:
    """Pair accuracy (%) per family and method, and the evidence-vs-human Spearman per family."""
    with open(ratings) as f:
        rated = json.load(f)
    with open(key) as f:
        groups = json.load(f)
    images, unrated = [], 0
    for r in rated:
        entry = groups[r["group"]]
        panel = entry["panels"][r["panel"]]
        if r["degree"] is None:
            unrated += 1
            continue
        if not 0 <= r["degree"] <= 100:
            raise ValueError(f"degree {r['degree']} is outside 0 to 100")
        images.append({"family": entry["family"], "arm": entry["arm"], "base": entry["base"], "rank": panel["rank"],
                       "degree": float(r["degree"]), "evidence": panel["evidence"]})
    ladders = _ladders(images, "degree")
    pair = {}
    for family, arm in sorted({(e["family"], e["arm"]) for e in groups.values()}):
        per = [(b, _pair_accuracy(v) * 100) for (f, a, b), v in sorted(ladders.items())
               if f == family and a == arm and len(v) >= 2]
        pair.setdefault(family, {})[arm] = _mean_ci(per)
    evidence = {}
    for family in sorted({e["family"] for e in groups.values()}):
        rows = [{"prompt": i["base"], "x": i["evidence"], "y": i["degree"]} for i in images
                if i["family"] == family and i["evidence"] is not None]
        if len(rows) < 3:
            evidence[family] = {"value": None, "ci95": None, "n": len(rows)}
            continue

        def rho(rs):
            return spearman([r["x"] for r in rs], [r["y"] for r in rs]) if len(rs) > 2 else float("nan")
        evidence[family] = {"value": rho(rows), "ci95": list(cluster_bootstrap(rows, rho)), "n": len(rows)}
    return {"pair": pair, "evidence_vs_human": evidence, "unrated": unrated,
            "note": "pair: % of hedge pairs in order (chance 50); evidence: Spearman of s_k with the human degree "
                    "over rated images of every method; pilot only: author ratings and provisional calibration"}


# ------------------------------------------------------------------------------ table
def _cell(entry, scale, digits):
    value = (entry or {}).get("value")
    return "\\tbd" if value is None else f"{value * scale:.{digits}f}"


def ladder_table_rows(human=None, auto=None) -> str:
    """LaTeX rows of Tab. hedge ladder from score_ladder and summarize_auto; anything not scored stays TBD.
    Concept Sliders and SPO have no SD3-Medium release, so their rows stay TBD."""
    human, auto = human or {}, auto or {}
    lines = []
    for arm, label in LADDER_ROWS:
        cells = []
        for family in ("attribute", "mixture", "size"):
            if arm is None:
                cells += ["\\tbd", "\\tbd"]
                continue
            cells += [_cell(human.get("pair", {}).get(family, {}).get(arm), 1, 1),
                      _cell(auto.get("rho", {}).get(family, {}).get(arm), 1, 2)]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    lines.append("\\midrule")
    evidence = [c for family in ("attribute", "mixture", "size")
                for c in ("--", _cell(human.get("evidence_vs_human", {}).get(family), 1, 2))]
    lines.append("Evidence $s_k$ vs.\\ human degree & " + " & ".join(evidence) + "\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated ladder run folders")
    ap.add_argument("--out", required=True, help="summary JSON; per-image values go next to it as .jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from calibration import ProvisionalCalibration
    from settings import load_settings
    calib = ProvisionalCalibration(load_settings())
    rows = measure_run(ladder_images(args.runs.split(",")), Owl(args.device), calib.prototype)
    out = Path(args.out)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize_auto(rows)
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
