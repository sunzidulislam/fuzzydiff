"""Tab. estimator-to-perception transfer in full multi-constraint sampling (G_sys): FuzzyDiff against the
unguided backbone at the same seed, per constraint.

  python -I code/run_experiment.py --suite soft    --group sys --n 10 --arms unguided,stage1,stage1_sar --log-all-steps --out outputs/sys
  python -I code/run_experiment.py --suite size    --group sys --n 10 --arms unguided,stage1,stage1_sar --log-all-steps --out outputs/sys
  python -I code/run_experiment.py --suite mixture --group sys --n 10 --arms unguided,stage1,stage1_sar --log-all-steps --out outputs/sys
  python -I code/run_experiment.py --prompt ... (the F2b prompts of f2b_prompts()) --group sys ...      (same arms)
  python -I code/transfer_sys.py pairs --out outputs/sys --device cuda:1   # Delta s and the OWLv2 Delta y
  python -I code/transfer_sys.py page  --out outputs/sys                   # acceptance of the guided images
  python -I code/transfer_sys.py score --out outputs/sys --labels labels.json

Per constraint of a graded family (F2a, F2b, F6, F8) and per binding: Delta s = the estimator's score on the
FuzzyDiff image minus on the unguided one (binding: the last logged step's beta, Stage 1 against the unguided
run), direction = the way the unguided score has to move to reach its interval (constraints already inside it
are left out), Delta y = the same change of the independent OWLv2 measurement (gradient_validity.independent;
binding: the colour's mixture fraction on its object minus on the other object). n_active = the program's
non-presence constraints. Exploitation = the share of constraints the estimator calls satisfied (membership at
least 0.5) that a person rejects. The statistics are reporting.transfer_rows' own, applied here to records from
the provisional calibration: transfer_rows refuses those (tables read fitted calibration only), so the numbers
are a pilot and the caption must say so.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predicates import Constraint, translate  # noqa: E402
from prompts import assign_group, base_prompt, vocab  # noqa: E402
from reporting import cluster_bootstrap, slope_through_origin  # noqa: E402

FAMILIES = (("F2a", "F2a Photometric attribute"), ("F2b", "F2b Open-vocab.\\ attribute"),
            ("F6", "F6 Color mixture / hedged color"), ("F8", "F8 Relative size"),
            ("binding", "Binding (attribute--object)"))
GRADED = ("F2a", "F2b", "F6", "F8", "F1", "F4", "F9")


def f2b_prompts(n=6):
    """F2b prompts whose base prompt falls in G_sys (the suites have none)."""
    out = []
    for attr, hedge in (("folded", "somewhat"), ("curved", "slightly"), ("twisted", "slightly")):
        found = [f"a {hedge} {attr} {o}" for o in vocab()["objects"]
                 if assign_group(base_prompt(f"a {hedge} {attr} {o}")) == "sys"]
        out += found[: n // 3]
    return out


def _meta(out, arm, iid):
    p = Path(out, arm, iid, "meta.json")
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return None if "failed" in m else m


def _last_raw(log_path, family, cid):
    """The constraint's raw value at the last logged step (binding has no score on a finished image)."""
    if not Path(log_path).exists():
        return None
    last = None
    for line in open(log_path):
        r = json.loads(line)
        if (r.get("type") == "constraint" and r.get("family") == family and r.get("cid") == cid
                and r.get("component", "main") == "main" and r.get("raw") is not None):
            last = r["raw"]
    return None if last is None else float(last)


def _final(meta, cid):
    return next((f for f in meta["finals"] if f["cid"] == cid and f["component"] == "main"), None)


def _direction(s, low, high):
    if s is None:
        return 0
    if low is not None and not math.isinf(low) and s < low:
        return +1
    if high is not None and not math.isinf(high) and s > high:
        return -1
    return 0


def _binding_y(owl, img, c, objects, prototype):
    """How much closer the attribute's colour sits to its own object than to the other one: OKLab distance of the
    other object's core mean colour to the attribute colour minus that of its own object (OWLv2 box cores; the
    attribute colour of a blend is the midpoint of its two prototypes). Positive: the colour is on the right object."""
    import torch
    from color_stats import srgb_to_oklab
    from ladder import _core, _tensor
    from predicates import _color_term
    attr, q = c.args
    term = _color_term(attr)
    others = [o for o in objects if o != q]
    if term is None or not others:
        return None
    base, tint, _ = term
    target = prototype(tint) if base == "achromatic" else (prototype(base) + prototype(tint)) / 2
    boxes, x = owl.boxes(img, [q, others[0]]), _tensor(img)

    def dist(name):
        core = None if boxes.get(name) is None else _core(x, boxes[name])
        return None if core is None else float(torch.linalg.norm(srgb_to_oklab(core).reshape(3, -1).mean(1) - target))
    own, other = dist(q), dist(others[0])
    return None if own is None or other is None else other - own


def build_pairs(out: Path, device):
    from PIL import Image
    from calibration import ProvisionalCalibration
    from gradient_validity import independent
    from ladder import Owl
    from settings import load_settings
    owl, prototype = Owl(device), ProvisionalCalibration(load_settings()).prototype
    pairs = []
    for meta_path in sorted(Path(out, "stage1_sar").glob("*/meta.json")):
        iid = meta_path.parent.name
        g, u, s1 = _meta(out, "stage1_sar", iid), _meta(out, "unguided", iid), _meta(out, "stage1", iid)
        if g is None or u is None:
            continue
        program = translate(g["prompt"])
        objects = list(dict.fromkeys(program.objects))
        n_active = sum(c.family != "presence" for c in program.constraints)
        img_g = Image.open(Path(out, "stage1_sar", iid, "image.png")).convert("RGB")
        img_u = Image.open(Path(out, "unguided", iid, "image.png")).convert("RGB")
        for cid, c in enumerate(program.constraints):
            if c.family not in ("F2a", "F2b", "F6", "F8", "binding"):
                continue
            if c.family == "binding":
                s_u = _last_raw(Path(out, "unguided", iid, "log.jsonl"), "binding", cid)
                s_g = None if s1 is None else _last_raw(Path(out, "stage1", iid, "log.jsonl"), "binding", cid)
                low, high, mu = 0.0, math.inf, None                 # beta > margin; the gate's margin is ~0
                y_u, y_g = _binding_y(owl, img_u, c, objects, prototype), _binding_y(owl, img_g, c, objects, prototype)
            else:
                fu, fg = _final(u, cid), _final(g, cid)
                if fu is None or fg is None:
                    continue
                s_u, s_g, low, high, mu = fu["s"], fg["s"], fg["low"], fg["high"], fg["mu"]
                con = [c.family, list(c.args), c.hedge, list(c.quantity) if isinstance(c.quantity, tuple) else c.quantity,
                       c.orientation]
                y_u, y_g = independent(owl, img_u, con, prototype), independent(owl, img_g, con, prototype)
            direction = _direction(s_u, low, high)
            if s_u is None or s_g is None or y_u is None or y_g is None or direction == 0:
                continue
            pairs.append({"image": iid, "prompt": g["prompt"], "base": base_prompt(g["prompt"]), "cid": cid,
                          "family": c.family, "args": list(c.args), "ds": s_g - s_u, "dy": y_g - y_u,
                          "direction": direction, "n_active": n_active, "mu_est": mu,
                          "calibration": g.get("calibration")})
    with open(Path(out) / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return pairs


# ------------------------------------------------------------------------------------------ acceptance page
PAGE = """<!doctype html><meta charset="utf-8"><title>Constraint acceptance</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:760px;padding:16px}img{width:100%%;max-width:512px;display:block}
section{margin-bottom:30px}label{margin-right:10px}p.q{margin:6px 0}textarea{width:100%%;height:140px}</style>
<h1>Does the image satisfy each part of its prompt?</h1><p>Answer each question from what you see, including how
strong the words are ("slightly", "very").</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){var out={},missing=0;
  document.querySelectorAll("[data-key]").forEach(function(p){var c=p.querySelector("input:checked");if(!c)missing++;
    out[p.dataset.key]=c?c.value==="yes":null;});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _question(pair):
    """The constraint in the prompt's own words: 'Is the mug slightly dark?', 'Is the puppy moderately smaller than
    the pillow?', 'Is the sofa yellowish-green (and that colour not on another object)?'."""
    fam, args = pair["family"], pair["args"]
    if fam == "binding":
        return f"Is the {args[1]} {args[0]} (and that colour not on another object)?"
    c = translate(pair["prompt"]).constraints[pair["cid"]]
    words = pair["prompt"].split()
    hedge = next((w for w in words if w == c.hedge), None)
    if fam == "F8":
        i = words.index(hedge) if hedge else next(i for i, w in enumerate(words) if w in ("larger", "bigger", "smaller"))
        return f"Is the {args[0]} {' '.join(words[i:i + 2])} than the {args[1]}?"
    if fam == "F2b":
        return f"Is the {args[0]} {hedge + ' ' if hedge else ''}{args[1]}?"
    if hedge:
        i = words.index(hedge)
        return f"Is the {args[0]} {' '.join(words[i:i + 2])}?"
    if fam == "F6" and args[0] in words and words.index(args[0]) > 0:
        return f"Is the {args[0]}'s colour {words[words.index(args[0]) - 1]}?"
    return f"Does the {args[0]} match “{pair['prompt']}”?"


def _jpeg(path, size=512):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_page(out: Path, page: Path, seed=0):
    pairs = [json.loads(line) for line in open(Path(out) / "pairs.jsonl")]
    by_image = {}
    for p in pairs:
        by_image.setdefault(p["image"], []).append(p)
    order = sorted(by_image)
    random.Random(seed).shuffle(order)
    sections = []
    for iid in order:
        ps = by_image[iid]
        qs = "".join(f'<p class="q" data-key="{iid}|{p["cid"]}">{html.escape(_question(p))} '
                     f'<label><input type="radio" name="{iid}-{p["cid"]}" value="yes">Yes</label>'
                     f'<label><input type="radio" name="{iid}-{p["cid"]}" value="no">No</label></p>' for p in ps)
        sections.append(f'<section><h2>{html.escape(ps[0]["prompt"])}</h2>'
                        f'<img src="data:image/jpeg;base64,{_jpeg(Path(out, "stage1_sar", iid, "image.png"))}">{qs}</section>')
    page.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(sections)


# ------------------------------------------------------------------------------------------ scoring
def transfer_stats(pairs, s_min=0.05, n_boot=2000):
    """reporting.transfer_rows' statistics without its fitted-calibration guard (pilot on provisional runs)."""
    out = {}
    for fam, _ in FAMILIES:
        ps = [p for p in pairs if p["family"] == fam]
        if not ps:
            out[fam] = {"n": 0}
            continue
        moved = [p for p in ps if abs(p["ds"]) >= s_min]

        def st(rows):
            keep = [r for r in rows if abs(r["ds"]) >= s_min]
            return slope_through_origin([r["ds"] * r["direction"] for r in keep], [r["dy"] * r["direction"] for r in keep])
        strata = {name: st([p for p in ps if cond(p)]) for name, cond in
                  (("1", lambda p: p["n_active"] == 1), ("2", lambda p: p["n_active"] == 2), (">=3", lambda p: p["n_active"] >= 3))}
        expl = [p for p in ps if p.get("mu_est") is not None and p["mu_est"] >= 0.5 and p.get("human_accept") is not None]
        out[fam] = {"n": len(ps), "dir_acc": (sum(p["dy"] * p["direction"] > 0 for p in moved) / len(moved)) if moved else None,
                    "slope": st(ps), "slope_ci": cluster_bootstrap([{**p, "prompt": p["base"]} for p in ps], st, n=n_boot),
                    "slope_by_active": strata,
                    "exploitation": (sum(not p["human_accept"] for p in expl) / len(expl)) if expl else None,
                    "n_moved": len(moved)}
    return out


def table_rows(stats) -> str:
    def f(x):
        return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"

    def pct(x):
        return "--" if x is None else f"{100 * x:.0f}"
    lines = []
    for fam, label in FAMILIES:
        r = stats.get(fam, {"n": 0})
        if not r["n"]:
            lines.append(f"{label} & 0 & -- & -- & --/--/-- & --\\\\")
            continue
        lo, hi = r["slope_ci"]
        sb = r["slope_by_active"]
        lines.append(f"{label} & {r['n']} & {pct(r['dir_acc'])} & {f(r['slope'])} [{f(lo)}, {f(hi)}] & "
                     f"{f(sb['1'])}/{f(sb['2'])}/{f(sb['>=3'])} & {pct(r['exploitation'])}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prompts", "pairs", "page", "score"))
    ap.add_argument("--out")
    ap.add_argument("--labels")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    if args.action == "prompts":
        print("\n".join(f2b_prompts()))
        return
    out = Path(args.out)
    if args.action == "pairs":
        print(len(build_pairs(out, args.device)), "pairs")
        return
    if args.action == "page":
        print("acceptance page:", out / "transfer.html", build_page(out, out / "transfer.html"), "images")
        return
    pairs = [json.loads(line) for line in open(out / "pairs.jsonl")]
    labels = json.load(open(args.labels)) if args.labels else {}
    for p in pairs:
        p["human_accept"] = labels.get(f"{p['image']}|{p['cid']}")
    stats = transfer_stats(pairs)
    (out / "transfer_scores.json").write_text(json.dumps(stats, indent=1, default=str))
    print(table_rows(stats))


if __name__ == "__main__":
    main()
