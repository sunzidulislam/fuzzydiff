"""Tab. fitted intervals: acceptance judgments on ladder images, the interval [l, u] of each hedge and quantity
(Eq. acceptance_curve, fit_hedge), and the held-out log-likelihood per judgment against a two-flank logistic model.

  python -I code/acceptance.py page --source ladder_sdxl --source ladder_spo --source ladder_size_sd3 --out accept.html
  python -I code/acceptance.py fit  --source ... --labels accept_labels.json

A source is a ladder rating page: NAME.html, NAME.key.json and the OWLv2 measurements NAME_auto.jsonl (or the
file given after a colon, NAME:measurements.jsonl). The page shows each ladder image alone, in shuffled order,
with its own prompt (hedge included), and asks whether the image matches the prompt as worded; for lightness,
chroma and colour mixture also the perceived degree (0 to 100), which is s on the perceived-degree scale. For
size, s is the measured oriented magnitude m_{q,r} (log box-area ratio, oriented like the prompt), as the
fitted size intervals use.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_fit import _log_likelihood, fit_hedge, pool_orientations  # noqa: E402
from ladder import degree_constraint  # noqa: E402
from prompts import LADDER_HEDGES, hedge_of  # noqa: E402

QUANTITIES = ("lightness", "chroma", "color mix.", "size")


def _source(spec):
    name, _, measured = spec.partition(":")
    return Path(name + ".html"), Path(name + ".key.json"), Path(measured or name + "_auto.jsonl")


def quantity_of(prompt):
    c = degree_constraint(prompt)
    if c.family == "F2a":
        return c, c.quantity
    return c, "color mix." if c.family == "F6" else "size"


def items(sources):
    """Every ladder panel: id, prompt, image (base64 JPEG), quantity, measured size magnitude."""
    out = []
    for spec in sources:
        page, key, measured = _source(spec)
        text = page.read_text(encoding="utf-8")
        images = dict(((g, p), b) for p, g, b in
                      re.findall(r'data-panel="(\w)" data-group="(\w+)"><figcaption>\w</figcaption><img alt="[^"]*" '
                                 r'src="data:image/jpeg;base64,([^"]+)"', text))
        keys = json.loads(key.read_text())
        m = {}
        if measured.exists():
            for line in open(measured):
                r = json.loads(line)
                m[(r["arm"], r["prompt"])] = r.get("value")
        for group, entry in keys.items():
            for panel, info in entry["panels"].items():
                prompt = info["prompt"]
                _, quantity = quantity_of(prompt)
                out.append({"id": f"{page.stem}|{group}|{panel}", "prompt": prompt, "arm": entry["arm"],
                            "base": entry["base"], "image": images.get((group, panel)), "quantity": quantity,
                            "m": m.get((entry["arm"], prompt))})
    return [i for i in out if i["image"]]


PAGE = """<!doctype html><meta charset="utf-8"><title>Acceptance judgments</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:760px;padding:16px}img{width:100%%;max-width:448px;display:block}
section{margin-bottom:30px}label{margin-right:10px}p.q{margin:6px 0}input[type=range]{width:100%%;max-width:448px}
textarea{width:100%%;height:140px}</style>
<h1>Acceptance judgments</h1><p>Each image is shown with the prompt it was made from. Answer from what you see.
<b>Matches?</b> Yes if the image fits the prompt as worded, including how strong the word is ("slightly",
"extremely"); No otherwise. Where there is a slider, also say how strong the quality is (0 to 100), comparing
with what the word means, not with the other images.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.addEventListener("input",function(e){if(e.target.type==="range"){e.target.dataset.touched="1";e.target.nextElementSibling.textContent=e.target.value;}});
document.getElementById("export").onclick=function(){var out={},missing=0;
  document.querySelectorAll("section").forEach(function(s){var c=s.querySelector("input[type=radio]:checked"),
    r=s.querySelector("input[type=range]"),rec={accept:c?c.value==="yes":null};
    if(!c)missing++;if(r){rec.degree=r.dataset.touched?+r.value:null;if(rec.degree===null)missing++;}
    out[s.dataset.id]=rec;});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _degree_question(prompt):
    c, quantity = quantity_of(prompt)
    if quantity == "color mix.":
        base, tint = c.quantity
        return f"How far is the {c.args[0]}'s colour from {base} toward {tint}? (0 pure {base}, 100 fully {tint})"
    words = prompt.split()
    word = words[words.index(hedge_of(prompt)) + 1]
    return f"How {word} is the {c.args[0]}? (0 not {word} at all, 100 as {word} as it gets)"


def build_page(sources, out: Path, seed=0):
    rows = items(sources)
    random.Random(seed).shuffle(rows)
    sections = []
    for n, r in enumerate(rows):
        slider = "" if r["quantity"] == "size" else (
            f'<p class="q">{html.escape(_degree_question(r["prompt"]))}<br><input type="range" min="0" max="100" value="50">'
            f' <b>not set</b></p>')
        name = f"a{n}"
        sections.append(f'<section data-id="{html.escape(r["id"])}"><h2>{n + 1}. {html.escape(r["prompt"])}</h2>'
                        f'<img src="data:image/jpeg;base64,{r["image"]}">'
                        f'<p class="q"><b>Matches the prompt as worded?</b> '
                        f'<label><input type="radio" name="{name}" value="yes">Yes</label>'
                        f'<label><input type="radio" name="{name}" value="no">No</label></p>{slider}</section>')
    out.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(rows)


# ------------------------------------------------------------------------------------------ fitting
def judgments(sources, labels):
    """(quantity, hedge) -> rows with s, accept, orientation, base."""
    cells = defaultdict(list)
    for r in items(sources):
        lab = labels.get(r["id"])
        if not lab or lab.get("accept") is None:
            continue
        c, quantity = quantity_of(r["prompt"])
        s = r["m"] if quantity == "size" else (None if lab.get("degree") is None else lab["degree"] / 100)
        if s is None:
            continue
        cells[(quantity, hedge_of(r["prompt"]))].append({"s": float(s), "accept": bool(lab["accept"]),
                                                          "orientation": c.orientation, "base": r["base"]})
    return cells


def _two_flank(s, a):
    """Two-flank logistic p = sigmoid(a1 (s - c1)) sigmoid(a2 (c2 - s)), fitted by maximum likelihood."""
    def nll(w):
        p = 1 / (1 + np.exp(-w[0] * (s - w[1]))) / (1 + np.exp(-w[2] * (w[3] - s)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.sum(a * np.log(p) + (1 - a) * np.log(1 - p))
    lo, hi = float(np.min(s)), float(np.max(s))
    best = None
    for c1, c2 in ((lo, hi), (np.quantile(s, 0.25), np.quantile(s, 0.75))):
        res = minimize(nll, x0=[5.0, c1, 5.0, c2], method="Nelder-Mead", options={"maxiter": 4000})
        best = res if best is None or res.fun < best.fun else best
    return best.x


def _ll_two_flank(w, s, a):
    p = np.clip(1 / (1 + np.exp(-w[0] * (s - w[1]))) / (1 + np.exp(-w[2] * (w[3] - s))), 1e-6, 1 - 1e-6)
    return float(np.sum(a * np.log(p) + (1 - a) * np.log(1 - p)))


def fit_cell(rows, scale):
    """Interval and held-out log-likelihood per judgment, ours and the two-flank logistic, on one 80/20 split."""
    s, a = np.array([r["s"] for r in rows]), np.array([r["accept"] for r in rows], float)
    if len(rows) < 5 or len(set(a.tolist())) < 2:
        return None
    fit = fit_hedge(s, a, scale=scale)
    order = np.random.default_rng(0).permutation(len(s))          # the split fit_hedge uses
    cut = int(0.8 * len(s))
    train, test = order[:cut], order[cut:]
    w = _two_flank(s[train], a[train])
    # the curve may put a flank beyond the scale (a one-sided hedge); the reported interval stays on the scale
    return {"l": max(fit.l, scale[0]), "u": min(fit.u, scale[1]), "n": len(rows), "one_sided": fit.one_sided,
            "ll_ours": fit.heldout_ll,
            "ll_logistic": _ll_two_flank(w, s[test], a[test]) / max(len(test), 1)}


def fit_all(sources, labels):
    cells = judgments(sources, labels)
    out = {}
    for (quantity, hedge), rows in cells.items():
        values = [r["s"] for r in rows]
        scale = (0.0, max(values)) if quantity == "size" else (0.0, 1.0)
        if quantity == "size":                                  # pool "larger" and "smaller" unless the LR test refuses
            pos, neg = [r for r in rows if r["orientation"] > 0], [r for r in rows if r["orientation"] < 0]
            fp, fn = fit_cell(pos, scale), fit_cell(neg, scale)
            if fp and fn:
                data = tuple((np.array([r["s"] for r in g]), np.array([r["accept"] for r in g], float)) for g in (pos, neg))
                if not pool_orientations(fit_hedge(*data[0], scale=scale), fit_hedge(*data[1], scale=scale), data):
                    out[(quantity, hedge)] = {**fp, "note": "larger only (pooling refused)"}
                    continue
        res = fit_cell(rows, scale)
        if res:
            out[(quantity, hedge)] = res
        else:
            out[(quantity, hedge)] = {"n": len(rows), "l": None}
    return out


def table_rows(fits) -> str:
    lines = []
    for hedge in LADDER_HEDGES:
        cells = []
        for q in QUANTITIES:
            f = fits.get((q, hedge))
            if not f or f.get("l") is None:
                cells.append("--" if not f else f"ins. ({f['n']})")
                continue
            cells.append(f"[{f['l']:.2f}, {f['u']:.2f}] ({f['n']})")
        lines.append(f"{hedge} & " + " & ".join(cells) + "\\\\")
    ll = []
    for q in QUANTITIES:
        fs = [f for (qq, _), f in fits.items() if qq == q and f.get("ll_ours") is not None]
        if not fs:
            ll.append("--")
            continue
        w = np.array([f["n"] for f in fs], float)
        ours = float(np.average([f["ll_ours"] for f in fs], weights=w))
        logi = float(np.average([f["ll_logistic"] for f in fs], weights=w))
        ll.append(f"{ours:.2f} / {logi:.2f}")
    lines.append("\\midrule\nLL: ours / logistic & " + " & ".join(ll) + "\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("page", "fit"))
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--labels")
    args = ap.parse_args()
    if args.action == "page":
        print("acceptance page:", args.out, build_page(args.source, Path(args.out)), "images")
        return
    fits = fit_all(args.source, json.load(open(args.labels)))
    print(json.dumps({f"{q}|{h}": v for (q, h), v in fits.items()}, indent=1))
    print(table_rows(fits))


if __name__ == "__main__":
    main()
