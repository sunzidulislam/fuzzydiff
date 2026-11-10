"""Tab. presence evidence: SigLIP global, region and combined presence evidence on the final image, overall and on
stress subsets (small, crowded, context-correlated absent). Each cell: AUROC / Brier / ECE (10 equal-mass bins).

  python -I code/presence_eval.py generate --out outputs/presence --vae-device cuda:1   # plain SD3, about 30 min
  python -I code/presence_eval.py page --out outputs/presence                         # labels.html for the rater
  python -I code/presence_eval.py score --out outputs/presence --labels labels.json   # the table rows

Labels come from a person, never from the prompt: an object named in the prompt can be missing or merged into
another (threshold_probe.py assumes them, which is why its numbers are not paper evidence). Global and region
evidence are mapped to probabilities by a logistic fit on the other folds, grouped by prompt; the combined row is
the model the method uses (presence_models in the settings), which never saw these images.

Subsets. small: present objects whose CLIPSeg support covers < 2% of the image, with the absent objects queried in
the small-object images; crowded: every pair from images that name >= 4 objects; context: every pair from the
context images, whose absent object is one the scene suggests (a lake without a boat).
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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from calibration_fit import _auroc, cross_fit  # noqa: E402
from constraints import Evidence  # noqa: E402
from estimators import ClipSegSupport, SigLIPEvidence  # noqa: E402
from guidance import _bin, _euler  # noqa: E402
from measure import EstimatorDeps, Measure  # noqa: E402
from prompts import vocab  # noqa: E402
from reporting import ece  # noqa: E402
from run_experiment import _save_image, _tiny_estimators, image_id  # noqa: E402
from sd3_backend import load_backend, tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

SMALL_AREA = 0.02
CROWDED = 4
EVIDENCE = (("global", "SigLIP global $h^g_q$"), ("region", "SigLIP region $h^r_q$"),
            ("combined", "Combined $\\tilde h_q$ (used)"))
SUBSETS = ("all", "small", "crowded", "context")


def _p(subset, prompt, present, context=()):
    return {"subset": subset, "prompt": prompt, "present": list(present), "context": list(context)}


# Ten prompts per subset. "present" are the objects the prompt names (the rater decides whether they are there);
# "context" are absent objects the scene makes likely. Every image also gets one random absent vocabulary object.
PROMPTS = [
    _p("ordinary", "a mug and a teapot on a kitchen table", ["mug", "teapot"]),
    _p("ordinary", "a lamp next to a sofa in a living room", ["lamp", "sofa"]),
    _p("ordinary", "a kitten sitting beside a pillow", ["kitten", "pillow"]),
    _p("ordinary", "a guitar leaning against a chair", ["guitar", "chair"]),
    _p("ordinary", "a vase and a candle on a shelf", ["vase", "candle"]),
    _p("ordinary", "a backpack and an umbrella by the door", ["backpack", "umbrella"]),
    _p("ordinary", "a bowl of apples next to a plate", ["bowl", "apple", "plate"]),
    _p("ordinary", "a puppy lying on a blanket", ["puppy", "blanket"]),
    _p("ordinary", "a bicycle parked beside a bench", ["bicycle", "bench"]),
    _p("ordinary", "a clock above a basket on a wall shelf", ["clock", "basket"]),
    _p("small", "a tiny mug at the far end of a long wooden table", ["mug"]),
    _p("small", "a small candle in the corner of a large empty room", ["candle"]),
    _p("small", "a distant kitten in a wide grassy field", ["kitten"]),
    _p("small", "a tiny apple on a huge white floor", ["apple"]),
    _p("small", "a small clock high on a tall brick wall", ["clock"]),
    _p("small", "a tiny shoe in the middle of an empty parking lot", ["shoe"]),
    _p("small", "a distant bench in a vast snowy park", ["bench"]),
    _p("small", "a small bottle on a long empty beach", ["bottle"]),
    _p("small", "a tiny cactus on a wide desert plain", ["cactus"]),
    _p("small", "a far away umbrella on a large sandy beach", ["umbrella"]),
    _p("crowded", "a kitchen counter with a mug, a teapot, a bowl, a plate and a kettle", ["mug", "teapot", "bowl", "plate", "kettle"]),
    _p("crowded", "a living room with a sofa, a lamp, a pillow, a clock and a vase", ["sofa", "lamp", "pillow", "clock", "vase"]),
    _p("crowded", "a desk with a book, a lamp, a mug, a clock and a radio", ["book", "lamp", "mug", "clock", "radio"]),
    _p("crowded", "a hallway with a backpack, an umbrella, a shoe, a hat and a bench", ["backpack", "umbrella", "shoe", "hat", "bench"]),
    _p("crowded", "a picnic blanket with an apple, a basket, a bottle, a bowl and a plate", ["blanket", "apple", "basket", "bottle", "bowl"]),
    _p("crowded", "a music corner with a guitar, a drum, a stool, a lamp and a radio", ["guitar", "drum", "stool", "lamp", "radio"]),
    _p("crowded", "a bedroom with a pillow, a blanket, a lamp, a clock and a book", ["pillow", "blanket", "lamp", "clock", "book"]),
    _p("crowded", "a porch with a pumpkin, a bench, a lamp, a basket and a cushion", ["pumpkin", "bench", "lamp", "basket", "cushion"]),
    _p("crowded", "a shelf with a jar, a vase, a candle, a clock and a book", ["jar", "vase", "candle", "clock", "book"]),
    _p("crowded", "a garden with a puppy, a kitten, a rabbit, a bench and a basket", ["puppy", "kitten", "rabbit", "bench", "basket"]),
    _p("context", "an empty lake at sunset", [], ["boat"]),
    _p("context", "an empty office desk with a lamp", ["lamp"], ["laptop"]),
    _p("context", "a breakfast table with a plate", ["plate"], ["cup"]),
    _p("context", "an empty dog bed in a living room", [], ["puppy"]),
    _p("context", "an empty birdcage on a stand", [], ["bird"]),
    _p("context", "a bare christmas living room with a sofa", ["sofa"], ["tree"]),
    _p("context", "an empty road through a forest", [], ["car"]),
    _p("context", "a fireplace with no fire next to a chair", ["chair"], ["fire"]),
    _p("context", "an empty fruit bowl on a counter", ["bowl"], ["apple"]),
    _p("context", "a tennis court with no players", [], ["racket"]),
]


def queries(entry, seed=0):
    """Objects asked about in one image: the named ones, the context ones, and one random absent object."""
    words = set(entry["prompt"].replace(",", " ").split()) | set(entry["present"]) | set(entry["context"])
    pool = [o for o in vocab()["objects"] if o not in words]
    extra = random.Random(f"{entry['prompt']}|{seed}").choice(pool)
    return list(dict.fromkeys(entry["present"] + entry["context"] + [extra]))


def _sample(backend, cond, seed, steps, cfg):
    sigmas = backend.sigmas(steps)
    z = backend.noise_like_latent(seed)
    with torch.no_grad():
        for i in range(steps):
            z = _euler(z, sigmas[i], sigmas[i + 1], backend.velocity(z, sigmas[i], cond, cfg=cfg).v)
        return backend.decode(z), _bin(sigmas[-1])


def score_image(measure, image, objects, bin):
    """Presence evidence and support area of every queried object on the finished image."""
    x = image[0] if image.dim() == 4 else image
    ev = Evidence(x0=image)
    with torch.no_grad():
        measure._objects(x, objects, ev, bin)
    return {q: {"global": float(o["g"]), "region": float(o["r"]), "h": float(o["h"]), "area": float(o["A"])}
            for q, o in ev.objects.items()}


def generate(backend, deps, out: Path, prompts, seed, steps, cfg):
    (out / "images").mkdir(parents=True, exist_ok=True)
    path = out / "presence.jsonl"
    done = {json.loads(line)["image"] for line in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for n, entry in enumerate(prompts):
            iid = image_id(entry["prompt"], seed)
            if iid in done:
                continue
            cond = backend.encode_prompt(entry["prompt"])
            image, b = _sample(backend, cond, seed, steps, cfg)
            _save_image(image, out / "images" / f"{iid}.png")
            for q, v in score_image(Measure(deps, entry["prompt"], cond), image, queries(entry, seed), b).items():
                f.write(json.dumps({"image": iid, "subset": entry["subset"], "prompt": entry["prompt"], "object": q,
                                    "named": q in entry["present"], "context_absent": q in entry["context"],
                                    "n_named": len(entry["present"]), "bin": b, **v}) + "\n")
            f.flush()
            print(f"{n + 1}/{len(prompts)} {entry['prompt']}", flush=True)


LABEL_PAGE = """<!doctype html><meta charset="utf-8"><title>Presence labels</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:900px;padding:16px}img{width:100%%;max-width:512px;display:block}
section{margin-bottom:32px}label{margin-right:10px}p.q{margin:6px 0}textarea{width:100%%;height:140px}</style>
<h1>Presence labels</h1><p>For each image, say whether each listed object is in the picture as an object of its own.
An object printed on another, merged into another, or only suggested by the scene counts as No. Answer from what you
see, not from the caption.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){
  var out=[],missing=0;
  document.querySelectorAll("[data-object]").forEach(function(o){
    var c=o.querySelector("input:checked");if(!c)missing++;
    out.push({image:o.dataset.image,object:o.dataset.object,present:c?c.value==="yes":null});});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _jpeg(path, size=512):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_page(out: Path, page: Path, seed=0):
    """Images in shuffled order; the caption and the subset are hidden, only the objects asked about are listed."""
    rows = [json.loads(line) for line in open(out / "presence.jsonl")]
    by_image = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r["object"])
    order = sorted(by_image)
    random.Random(seed).shuffle(order)
    sections = []
    for n, iid in enumerate(order):
        objects = sorted(by_image[iid])
        qs = "".join(f'<p class="q" data-image="{iid}" data-object="{html.escape(q)}">Is there a {html.escape(q)}? '
                     f'<label><input type="radio" name="{iid}-{html.escape(q)}" value="yes">Yes</label>'
                     f'<label><input type="radio" name="{iid}-{html.escape(q)}" value="no">No</label></p>' for q in objects)
        sections.append(f'<section><h2>Image {n + 1}</h2><img alt="image {n + 1}" '
                        f'src="data:image/jpeg;base64,{_jpeg(out / "images" / f"{iid}.png")}">{qs}</section>')
    page.write_text(LABEL_PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")


class _Constant:
    """A fold whose training rows hold one class only predicts that class's rate."""

    def __init__(self, rate):
        self.rate = rate

    def predict_proba(self, x):
        return np.tile([1 - self.rate, self.rate], (len(x), 1))


def _logistic(x, y):
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return _Constant(float(y.mean()) if len(y) else 0.5)
    return LogisticRegression(C=1e3, max_iter=1000).fit(np.asarray(x, float).reshape(-1, 1), y)


def held_out(rows):
    """Adds p_global and p_region (logistic maps fitted on the other folds, grouped by prompt) and p_combined."""
    for key in ("global", "region"):
        def fit(train, key=key):
            return _logistic([r[key] for r in train], [r["label"] for r in train])

        def predict(model, test, key=key):
            p = model.predict_proba(np.asarray([r[key] for r in test], float).reshape(-1, 1))[:, 1]
            return [{**r, f"p_{key}": float(v)} for r, v in zip(test, p)]

        rows = cross_fit(rows, fit, predict, group="prompt")
    return [{**r, "p_combined": 1 / (1 + math.exp(-r["h"]))} for r in rows]


def subset_rows(rows, subset):
    if subset == "all":
        return rows
    if subset == "small":
        return [r for r in rows if (r["label"] and r["area"] < SMALL_AREA) or (not r["label"] and r["subset"] == "small")]
    if subset == "crowded":
        return [r for r in rows if r["n_named"] >= CROWDED]
    return [r for r in rows if r["subset"] == "context" and (r["named"] or r["context_absent"])]


def metrics(rows, key):
    labels = [int(r["label"]) for r in rows]
    if len(set(labels)) < 2:
        return {"auroc": None, "brier": None, "ece": None, "n": len(rows), "n_present": sum(labels)}
    p = [r[f"p_{key}"] for r in rows]
    return {"auroc": _auroc(p, labels), "brier": float(np.mean([(a - b) ** 2 for a, b in zip(p, labels)])),
            "ece": ece(p, labels, bins=10), "n": len(rows), "n_present": sum(labels)}


def score(out: Path, labels_path: Path):
    labels = {(x["image"], x["object"]): x["present"] for x in json.load(open(labels_path))}
    rows = [{**r, "label": labels[(r["image"], r["object"])]} for r in map(json.loads, open(out / "presence.jsonl"))
            if labels.get((r["image"], r["object"])) is not None]
    rows = held_out(rows)
    table = {key: {s: metrics(subset_rows(rows, s), key) for s in SUBSETS} for key, _ in EVIDENCE}
    named_missing = sum(1 for r in rows if r["named"] and not r["label"])
    return {"table": table, "n_pairs": len(rows), "named_but_absent": named_missing}


def table_rows(result) -> str:
    def cell(m):
        if m["auroc"] is None:
            return "--"
        return f"{m['auroc']:.2f}/{m['brier']:.2f}/{m['ece']:.2f}"
    return "\n".join(f"{label} & " + " & ".join(cell(result["table"][key][s]) for s in SUBSETS) + "\\\\"
                     for key, label in EVIDENCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("generate", "page", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "page":
        build_page(out, out / "labels.html")
        print("labelling page:", out / "labels.html")
        return
    if args.action == "score":
        result = score(out, Path(args.labels))
        (out / "presence_scores.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(table_rows(result))
        return
    s = load_settings()
    calib = ProvisionalCalibration(s)
    if args.tiny:
        backend, (seg, siglip), steps = tiny_backend(), _tiny_estimators(), 4
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), args.resolution, torch.float16)
        device = args.vae_device or args.tf_device
        seg = ClipSegSupport.from_pretrained(s.get("cfgCLIPSeg", str), device)
        siglip = SigLIPEvidence.from_pretrained(s.get("cfgSigLIP", str), device)
        steps = s.get("cfgSteps", int)
    for entry in PROMPTS:
        backend.encode_prompt(entry["prompt"])
    if not args.tiny:
        backend.unload_text_encoders()
    generate(backend, EstimatorDeps(seg, siglip, calib, s), out, PROMPTS, args.seed, steps, s.get("cfgCFG"))


if __name__ == "__main__":
    main()
