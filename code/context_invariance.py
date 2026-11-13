"""Tab. context invariance of hedge intervals: does a hedge mean the same interval for different object
categories? Acceptance judgments on G_cal images, a pooled interval against one per category, out of fold.

  python -I code/context_invariance.py generate --out outputs/context --vae-device cuda:1   # 360 images, ~4 h
  python -I code/context_invariance.py page     --out outputs/context                        # context.html
  python -I code/context_invariance.py fit      --out outputs/context --labels labels.json

Per family, one degree word at two hedges over four categories (objects, or object pairs for size), 15 seeds
each, plain SD3. F8 images are measured with OWLv2 (the oriented log box-area ratio m_{q,r}, as in Tab. fitted
intervals); F2a and F6 images get a perceived degree (0 to 100) on the label page. Every image gets an
acceptance judgment: does it match the prompt as worded.

fit, per family and hedge: tau/w = between-category SD of the interval centre over the pooled half-width;
Delta LL = held-out log-likelihood per judgment of the per-category curves minus the pooled curve (5 folds by
seed, so a category is in every fold); p = likelihood-ratio test of per-category vs. pooled curves (4 parameters
per curve), the two hedges of a row combined, Holm-corrected over the rows.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acceptance import PAGE  # noqa: E402
from calibration_fit import _log_likelihood, fit_hedge  # noqa: E402
from predicates import translate  # noqa: E402
from prompts import assign_group, base_prompt  # noqa: E402

SEEDS = 15
FAMILIES = {   # family: (degree word, hedges, preferred categories in order; the first four in G_cal are used)
    "F2a": ("dark", ("slightly", "very"), ["towel", "umbrella", "hat", "pillow", "bench", "teapot", "candle", "shoe"]),
    "F6": ("bluish-green", ("slightly", "moderately"), ["vase", "sofa", "backpack", "mug", "towel", "lamp", "bottle"]),
    "F8": ("larger", ("slightly", "considerably"),
           [("kettle", "apple"), ("backpack", "hat"), ("suitcase", "shoe"), ("basket", "bottle"), ("sofa", "chair"),
            ("pumpkin", "mug"), ("bench", "backpack"), ("guitar", "book")]),
}
LABELS = {"F2a": "F2a lightness / chroma", "F6": "F6 color mixture", "F8": "F8 apparent size"}


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def prompt_of(family, hedge, category):
    word = FAMILIES[family][0]
    if family == "F8":
        q, r = category
        return f"{_art(q)} {q} {hedge} {word} than {_art(r)} {r}"
    return f"{_art(hedge)} {hedge} {word} {category}"


def categories(family, k=4):
    _, hedges, options = FAMILIES[family]
    chosen = [c for c in options if assign_group(base_prompt(prompt_of(family, hedges[0], c))) == "cal"]
    if len(chosen) < k:
        raise ValueError(f"{family}: only {len(chosen)} categories fall in G_cal")
    return chosen[:k]


def jobs(seeds=SEEDS):
    out = []
    for family, (_, hedges, _) in FAMILIES.items():
        for category in categories(family):
            for hedge in hedges:
                for seed in range(seeds):
                    name = category if isinstance(category, str) else "-".join(category)
                    out.append({"family": family, "hedge": hedge, "category": name, "seed": seed,
                                "prompt": prompt_of(family, hedge, category)})
    return out


def _id(job):
    return f"{job['family']}_{job['category']}_{job['hedge']}_s{job['seed']}"


def generate(backend, out: Path, todo, steps, cfg):
    from presence_eval import _sample
    from run_experiment import _save_image
    for n, job in enumerate(todo):
        folder = out / _id(job)
        if (folder / "meta.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        image, _ = _sample(backend, backend.encode_prompt(job["prompt"]), job["seed"], steps, cfg)
        _save_image(image, folder / "image.png")
        (folder / "meta.json").write_text(json.dumps(job))
        print(f"{n + 1}/{len(todo)} {job['prompt']} (seed {job['seed']})", flush=True)


def measure_size(out: Path, device):
    """Adds m (the oriented log box-area ratio, OWLv2) to each F8 image's meta, or null without both boxes."""
    from PIL import Image
    from ladder import Owl, _tensor, measure
    owl = None
    for meta_path in sorted(out.glob("F8_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if "m" in meta:
            continue
        owl = owl or Owl(device)
        img = Image.open(meta_path.parent / "image.png").convert("RGB")
        c = next(c for c in translate(meta["prompt"]).constraints if c.family == "F8")
        meta["m"] = measure(_tensor(img), c, owl.boxes(img, list(c.args)), None)
        meta_path.write_text(json.dumps(meta))


def _degree_question(meta):
    c = next(c for c in translate(meta["prompt"]).constraints if c.family in ("F2a", "F6"))
    if c.family == "F6":
        base, tint = c.quantity
        return f"How far is the {c.args[0]}'s colour from {base} toward {tint}? (0 pure {base}, 100 fully {tint})"
    word = FAMILIES["F2a"][0]
    return f"How {word} is the {c.args[0]}? (0 not {word} at all, 100 as {word} as it gets)"


def build_page(out: Path, page: Path, seed=0):
    from presence_eval import _jpeg
    metas = [json.loads(p.read_text()) | {"_dir": p.parent} for p in sorted(out.glob("*/meta.json"))]
    random.Random(seed).shuffle(metas)
    sections = []
    for n, meta in enumerate(metas):
        slider = "" if meta["family"] == "F8" else (
            f'<p class="q">{html.escape(_degree_question(meta))}<br><input type="range" min="0" max="100" value="50">'
            f' <b>not set</b></p>')
        sections.append(f'<section data-id="{meta["_dir"].name}"><h2>{n + 1}. {html.escape(meta["prompt"])}</h2>'
                        f'<img src="data:image/jpeg;base64,{_jpeg(meta["_dir"] / "image.png")}">'
                        f'<p class="q"><b>Matches the prompt as worded?</b> '
                        f'<label><input type="radio" name="c{n}" value="yes">Yes</label>'
                        f'<label><input type="radio" name="c{n}" value="no">No</label></p>{slider}</section>')
    page.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(sections)


# ------------------------------------------------------------------------------------------ fitting
def judgments(out: Path, labels):
    """(family, hedge) -> rows of s, accept, category, seed."""
    cells = defaultdict(list)
    for meta_path in sorted(out.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        lab = labels.get(meta_path.parent.name)
        if not lab or lab.get("accept") is None:
            continue
        s = meta.get("m") if meta["family"] == "F8" else (None if lab.get("degree") is None else lab["degree"] / 100)
        if s is None:
            continue
        cells[(meta["family"], meta["hedge"])].append({"s": float(s), "accept": float(bool(lab["accept"])),
                                                       "category": meta["category"], "seed": meta["seed"]})
    return cells


def _fit(rows, scale):
    s, a = np.array([r["s"] for r in rows]), np.array([r["accept"] for r in rows])
    if len(rows) < 5 or len(set(a.tolist())) < 2:
        return None
    return fit_hedge(s, a, scale=scale)


def _ll(fit, rows, train=None):
    if fit is None:              # too few or one-sided training answers: the training acceptance rate, constant
        base = train if train else rows
        rate = np.clip(np.mean([r["accept"] for r in base]) if base else 0.5, 1e-3, 1 - 1e-3)
        return float(sum(math.log(rate if r["accept"] else 1 - rate) for r in rows))
    return float(_log_likelihood(np.array([r["s"] for r in rows]), np.array([r["accept"] for r in rows]),
                                 fit.l, fit.u, fit.kappa, fit.plateau))


def cell_stats(rows, scale, folds=5):
    pooled = _fit(rows, scale)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cat_fits = {c: _fit(rs, scale) for c, rs in by_cat.items()}
    centres = [(f.l + f.u) / 2 for f in cat_fits.values() if f is not None]
    half = None if pooled is None else max((pooled.u - pooled.l) / 2, 1e-6)
    tau_w = float(np.std(centres, ddof=1) / half) if half and len(centres) >= 2 else None
    oof_pooled, oof_cat, n_test = 0.0, 0.0, 0
    for k in range(folds):
        test = [r for r in rows if r["seed"] % folds == k]
        train = [r for r in rows if r["seed"] % folds != k]
        if not test or not train:
            continue
        oof_pooled += _ll(_fit(train, scale), test, train)
        for c in by_cat:
            tr, te = [r for r in train if r["category"] == c], [r for r in test if r["category"] == c]
            if te:
                oof_cat += _ll(_fit(tr, scale), te, tr)
        n_test += len(test)
    in_pooled = _ll(pooled, rows)
    in_cat = sum(_ll(cat_fits[c], rs) for c, rs in by_cat.items())
    return {"n": len(rows), "categories": len(by_cat), "tau_w": tau_w,
            "dll": (oof_cat - oof_pooled) / n_test if n_test else None,
            "lr": max(2 * (in_cat - in_pooled), 0.0), "df": 4 * (len(by_cat) - 1),
            "pooled": None if pooled is None else [pooled.l, pooled.u],
            "per_category": {c: (None if f is None else [f.l, f.u]) for c, f in cat_fits.items()}}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def fit_all(out: Path, labels):
    cells = judgments(out, labels)
    rows = []
    for family, (_, hedges, _) in FAMILIES.items():
        stats = {}
        for hedge in hedges:
            data = cells.get((family, hedge), [])
            scale = (0.0, max([r["s"] for r in data] + [1e-6])) if family == "F8" else (0.0, 1.0)
            stats[hedge] = cell_stats(data, scale) if data else None
        valid = [st for st in stats.values() if st]
        lr, df = sum(st["lr"] for st in valid), sum(st["df"] for st in valid)
        taus = [st["tau_w"] for st in valid if st["tau_w"] is not None]
        dlls = [st["dll"] for st in valid if st["dll"] is not None]
        rows.append({"family": family, "hedges": "/".join(hedges), "stats": stats,
                     "tau_w": float(np.mean(taus)) if taus else None, "dll": float(np.mean(dlls)) if dlls else None,
                     "p_raw": float(chi2.sf(lr, df)) if df else None, "n": sum(st["n"] for st in valid)})
    raw = [r["p_raw"] for r in rows if r["p_raw"] is not None]
    adjusted = iter(holm(raw))
    for r in rows:
        r["p"] = next(adjusted) if r["p_raw"] is not None else None
    return rows


def table_rows(rows) -> str:
    def f(x, d=2):
        return "--" if x is None else f"{x:.{d}f}"
    return "\n".join(f"{LABELS[r['family']]} & {r['hedges'].replace('/', ' / ')} & {f(r['tau_w'])} & {f(r['dll'], 3)} & "
                     f"{f(r['p'], 3)}\\\\" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("generate", "page", "fit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "page":
        print("label page:", out / "context.html", build_page(out, out / "context.html"), "images")
        return
    if args.action == "fit":
        rows = fit_all(out, json.load(open(args.labels)))
        (out / "context_fit.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps(rows, indent=1))
        print(table_rows(rows))
        return
    from settings import load_settings
    from sd3_backend import load_backend, tiny_backend
    s = load_settings()
    todo = jobs(args.seeds)
    if args.tiny:
        backend, steps = tiny_backend(), 4
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), 1024, torch.float16)
        steps = s.get("cfgSteps", int)
    for p in dict.fromkeys(j["prompt"] for j in todo):
        backend.encode_prompt(p)
    if not args.tiny:
        backend.unload_text_encoders()
    out.mkdir(parents=True, exist_ok=True)
    generate(backend, out, todo, steps, s.get("cfgCFG"))
    if not args.tiny:
        measure_size(out, args.vae_device or args.tf_device)


if __name__ == "__main__":
    main()
