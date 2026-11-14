"""Tab. gradient-validity test: single-family interventions on G_int, measured independently of the estimators.

  python -I code/gradient_validity.py run   --out outputs/gv --vae-device cuda:1    # 12 rows x 2 prompts, ~5-6 h
  python -I code/gradient_validity.py page  --out outputs/gv                        # before/after judgment page
  python -I code/gradient_validity.py score --out outputs/gv --labels labels.json

For every prompt, the program is cut to one constraint of the row's family. A plain image gives the
unguided score s. Stage 1 then runs twice with a commanded target: s moved up and down by cfgGVShift on a
[0, 1] scale (by fixed steps on the other scales), kept cfgGVMargin inside the scale; a direction with less
than cfgGVHeadroom of room is skipped (run_experiment.commanded_targets). The independent measurement y is
OWLv2, a model none of the estimators use: box colour (F2a, F4, F6), box-area ratio (F8), box-centre offset
(F9), detection score (presence, F1) and the score of 'attribute object' over 'object' (F2b). A person judges,
from the unguided and the intervened image side by side, whether the quantity went up, stayed, or went down.

Columns: Dir. acc. (human / automatic) = share of non-skipped interventions whose measurement moved in the
commanded direction; Transfer = through-origin slope of Delta y on Delta s over interventions with |Delta s| at
least s_min; Exploit = share whose estimate moved as commanded while the measurement did not; Skip = share of
up / down interventions skipped for lack of headroom; Pass = Dir. acc. >= cfgDirAcc and Transfer >= cfgTransfer.
Binding: fix rate (an unguided image with the colour on the wrong object, fixed by the up intervention) / break
rate (a correct one broken by the down intervention), from the person's answers.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
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
from predicates import Program, translate  # noqa: E402
from prompts import assign_group, base_prompt, select, vocab  # noqa: E402

SEED = 0


def _pick(suite, n, keep):
    rows = [r.prompt for r in select(suite, "int", 200) if keep(r.program)]
    return rows[:n]


def _art(word):
    return "an" if word[0] in "aeiou" else "a"


def _int_prompts(make, n=2):
    """The first n prompts over the vocabulary (single objects, then ordered pairs) whose base prompt falls in
    G_int, so no hand-written prompt of another group (least of all Test) is used."""
    objects = vocab()["objects"]
    candidates = [make(o) for o in objects] if make.__code__.co_argcount == 1 else         [make(a, b) for a in objects for b in objects if a != b]
    chosen = [p for p in candidates if p and assign_group(base_prompt(p)) == "int"]
    return chosen[:n]


def _has(family, quantity=None):
    def keep(program):
        return any(c.family == family and (quantity is None or c.quantity == quantity) for c in program.constraints)
    return keep


def rows():
    """(row key, label, family, quantity or None, attribute word or None, two prompts)."""
    return [
        ("presence", "Presence", "presence", None, None,
         _int_prompts(lambda a, b: f"{_art(a)} {a} and {_art(b)} {b} on a table")),
        ("binding", "Binding $\\beta$", "binding", None, None, _pick("mixture", 2, _has("binding"))),
        ("F1", "F1 visibility", "F1", None, None, _int_prompts(lambda o: f"a barely visible {o}")),
        ("F2a_l", "F2a lightness", "F2a", "lightness", None, _pick("soft", 2, _has("F2a", "lightness"))),
        ("F2a_c", "F2a chroma", "F2a", "chroma", None, _pick("soft", 2, _has("F2a", "chroma"))),
        ("F2b_curved", "F2b ``curved''", "F2b", None, "curved", _int_prompts(lambda o: f"a slightly curved {o}")),
        ("F2b_folded", "F2b ``folded''", "F2b", None, "folded", _int_prompts(lambda o: f"a somewhat folded {o}")),
        ("F2b_twisted", "F2b ``twisted''", "F2b", None, "twisted", _int_prompts(lambda o: f"a slightly twisted {o}")),
        ("F4", "F4 whiteness", "F4", None, None, _int_prompts(lambda o: f"a slightly white {o}")),
        ("F6", "F6 color mixture", "F6", None, None, _pick("soft", 2, _has("F6"))),
        ("F8", "F8 size", "F8", None, None, _pick("size", 2, _has("F8"))),
        ("F9", "F9 spatial", "F9", None, None, _int_prompts(lambda a, b: f"{_art(a)} {a} left of {_art(b)} {b}")),
    ]


def target_constraint(program, family, quantity=None, attribute=None):
    """The constraint the row intervenes on; for presence the last object's (the one most often lost)."""
    cands = [c for c in program.constraints if c.family == family and (quantity is None or c.quantity == quantity)
             and (attribute is None or attribute in c.args)]
    if not cands:
        raise ValueError(f"no {family} constraint in {program}")
    return cands[-1] if family == "presence" else cands[0]


# commanded shifts on each family's own scale: (shift, lower, upper bound of the scale or None)
SCALES = {"presence": (2.0, None, None), "binding": (0.3, -1.0, 1.0), "F8": (0.7, None, None), "F9": (0.3, -1.0, 1.0)}
S_MIN = {"presence": 0.5, "F8": 0.1}


def commanded_band(family, s, sign, settings):
    """Target interval for a commanded move of s in direction sign, or None without headroom."""
    from run_experiment import TARGET_BAND, commanded_targets
    if s is None:
        return None
    if family not in SCALES:                                      # [0, 1] scales: the run's own rule
        return commanded_targets(s, sign * settings.get("cfgGVShift"), settings.get("cfgGVMargin"),
                                 settings.get("cfgGVHeadroom"))
    shift, lo, hi = SCALES[family]
    target = s + sign * shift
    if lo is not None:
        margin = settings.get("cfgGVMargin") * (hi - lo)
        target = min(max(target, lo + margin), hi - margin)
        if abs(target - s) < settings.get("cfgGVHeadroom") * (hi - lo):
            return None
    band = TARGET_BAND * (1 if lo is None else (hi - lo))
    return (target - band, target + band)


def _s_from(finals, log_path, family):
    """The family's final score; binding has none on a finished image, so its last logged step's raw value."""
    for f in finals:
        if f["family"] == family and f["component"] == "main" and f.get("s") is not None:
            return float(f["s"])
    if log_path and Path(log_path).exists():
        last = None
        for line in open(log_path):
            r = json.loads(line)
            if r.get("type") == "constraint" and r.get("family") == family and r.get("raw") is not None:
                last = r["raw"]
        return None if last is None else float(last)
    return None


def run_one(backend, deps, settings, prompt, family, quantity, attribute, folder: Path, steps_cfg):
    """Unguided, up and down runs of one prompt; returns its record."""
    from calibration import ProvisionalCalibration
    from grounding import phrase_spans
    from guidance import run_stage1
    from measure import Measure
    from reporting import StepLog
    from run_experiment import _save_image
    program = translate(prompt)
    c = target_constraint(program, family, quantity, attribute)
    single = Program(objects=program.objects, constraints=[c])
    cond = backend.encode_prompt(prompt, phrase_spans(prompt, program))
    calib = ProvisionalCalibration(settings)
    record = {"prompt": prompt, "family": family, "constraint": [c.family, list(c.args), c.hedge,
                                                                    c.quantity if not isinstance(c.quantity, tuple) else list(c.quantity),
                                                                    c.orientation], "runs": {}}
    for name, sign in (("base", 0), ("up", +1), ("down", -1)):
        sub = folder / name
        if (sub / "meta.json").exists():
            record["runs"][name] = json.loads((sub / "meta.json").read_text())
            continue
        sub.mkdir(parents=True, exist_ok=True)
        targets = None
        if sign:
            band = commanded_band(family, record["runs"]["base"]["s"], sign, settings)
            if band is None:
                meta = {"skipped": "headroom", "sign": sign}
                (sub / "meta.json").write_text(json.dumps(meta))
                record["runs"][name] = meta
                continue
            targets = {0: band}
        cfg = steps_cfg if sign else dataclasses.replace(steps_cfg, guided=False, log_all=True)
        log = StepLog(str(sub / "log.jsonl"), "int", calib.tag)
        result = run_stage1(backend, single, cond, SEED, calib, Measure(deps, prompt, cond), cfg, log,
                            targets=targets, prompt=prompt)
        log.close()
        _save_image(result.image, sub / "image.png")
        meta = {"sign": sign, "s": _s_from(result.finals, sub / "log.jsonl", family),
                "band": None if targets is None else list(targets[0])}
        (sub / "meta.json").write_text(json.dumps(meta))
        record["runs"][name] = meta
    return record


# ------------------------------------------------------------------------------------------ independent measure
def _owl_best(owl, img, texts):
    """Best OWLv2 score and box (pixels) per text query."""
    x = owl.processor(text=[[t for t in texts]], images=img, return_tensors="pt").to(owl.device)
    with torch.no_grad():
        out = owl.model(**x)
    scores = out.logits[0].sigmoid().float().cpu()
    boxes = out.pred_boxes[0].float().cpu()
    side = max(img.size)
    best = {}
    for j, t in enumerate(texts):
        i = int(scores[:, j].argmax())
        cx, cy, w, h = (boxes[i] * side).tolist()
        best[t] = (float(scores[i, j]), [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return best


def independent(owl, img, constraint, prototype):
    """The row's measurement on one image (oriented like the constraint), or None."""
    from ladder import _core, _tensor, measure
    family, args, _, quantity, orientation = constraint
    c = translate_constraint(constraint)
    if family in ("F2a", "F6", "F8"):
        return measure(_tensor(img), c, owl.boxes(img, list(dict.fromkeys(args))), prototype)
    if family in ("presence", "F1"):
        return _owl_best(owl, img, [f"a photo of a {args[0]}"])[f"a photo of a {args[0]}"][0]
    if family == "F2b":
        q, attr = args
        best = _owl_best(owl, img, [f"a photo of a {attr} {q}", f"a photo of a {q}"])
        return best[f"a photo of a {attr} {q}"][0] - best[f"a photo of a {q}"][0]
    if family == "F4":
        box = owl.boxes(img, [args[0]]).get(args[0])
        core = None if box is None else _core(_tensor(img), box)
        if core is None:
            return None
        from color_stats import srgb_to_oklab
        from measure import WHITENESS_CHROMA
        mean = srgb_to_oklab(core).reshape(3, -1).mean(dim=1)         # the scorer's definition, in the OWLv2 box core
        return float(mean[0].clamp(0, 1) * torch.exp(-torch.sqrt(mean[1] ** 2 + mean[2] ** 2) / WHITENESS_CHROMA))
    if family == "F9":
        from constraints import SPATIAL_AXES
        boxes = owl.boxes(img, list(args))
        if any(boxes.get(q) is None for q in args):
            return None
        centre = [((b[0] + b[2]) / 2 / img.size[0], (b[1] + b[3]) / 2 / img.size[1]) for b in (boxes[args[0]], boxes[args[1]])]
        axis, sign = SPATIAL_AXES[quantity]
        return sign * (centre[1][axis] - centre[0][axis])
    return None


def translate_constraint(constraint):
    from predicates import Constraint
    family, args, hedge, quantity, orientation = constraint
    return Constraint(family, tuple(args), hedge, tuple(quantity) if isinstance(quantity, list) else quantity, orientation)


def measure_all(out: Path, device):
    from PIL import Image
    from calibration import ProvisionalCalibration
    from ladder import Owl
    from settings import load_settings
    owl, prototype = Owl(device), ProvisionalCalibration(load_settings()).prototype
    for rec_path in sorted(out.glob("*/*/record.json")):
        rec = json.loads(rec_path.read_text())
        for name, run in rec["runs"].items():
            img_path = rec_path.parent / name / "image.png"
            if "y" in run or not img_path.exists():
                continue
            run["y"] = independent(owl, Image.open(img_path).convert("RGB"), rec["constraint"], prototype)
        rec_path.write_text(json.dumps(rec))


# ------------------------------------------------------------------------------------------ judgment page
PAGE = """<!doctype html><meta charset="utf-8"><title>Intervention judgments</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:1000px;padding:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
img{width:100%%;display:block}section{margin-bottom:34px}label{margin-right:10px}p.q{margin:6px 0}textarea{width:100%%;height:140px}</style>
<h1>Before and after</h1><p>Each row shows one prompt twice. Answer the question about the right image compared
with the left one, from what you see.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){var out={},missing=0;
  document.querySelectorAll("[data-q]").forEach(function(p){var c=p.querySelector("input:checked");if(!c)missing++;
    var id=p.dataset.id;out[id]=out[id]||{};out[id][p.dataset.q]=c?c.value:null;});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _question(constraint):
    family, args, _, quantity, orientation = constraint
    q = args[0]
    word = {"presence": f"is the {q} more clearly there", "F1": f"is the {q} more visible",
            "F4": f"is the {q} whiter", "F9": f"is the {q} further {quantity} of the {args[-1]}"}.get(family)
    if family == "F2a":
        word = f"is the {q} {'lighter' if quantity == 'lightness' else 'more colourful'}"
    elif family == "F2b":
        word = f"is the {q} more {args[1]}"
    elif family == "F6":
        word = f"is the {q}'s colour further toward {quantity[1]}"
    elif family == "F8":
        word = f"is the {q} {'larger' if orientation > 0 else 'smaller'} relative to the {args[1]}"
    return f"In the right image, {word}?"


def _radios(name, options):
    return "".join(f'<label><input type="radio" name="{html.escape(name)}" value="{html.escape(v)}">{html.escape(v)}</label>'
                   for v in options)


def _jpeg(path, size=448):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_page(out: Path, page: Path, seed=0):
    items = []
    for rec_path in sorted(out.glob("*/*/record.json")):
        rec = json.loads(rec_path.read_text())
        base = rec_path.parent / "base" / "image.png"
        for name in ("up", "down"):
            img = rec_path.parent / name / "image.png"
            if base.exists() and img.exists():
                items.append((rec_path.parent, name, rec))
    random.Random(seed).shuffle(items)
    sections = []
    for folder, name, rec in items:
        iid = f"{folder.parent.name}/{folder.name}/{name}"
        family, args = rec["constraint"][0], rec["constraint"][1]
        if family == "binding":
            prog = translate(rec["prompt"])
            objects = list(dict.fromkeys(prog.objects))
            qs = "".join(f'<p class="q" data-id="{iid}" data-q="{side}">Which object is {html.escape(args[0])} in the '
                         f'{side} image? {_radios(f"{iid}-{side}", objects + ["both", "neither"])}</p>' for side in ("left", "right"))
        else:
            qs = (f'<p class="q" data-id="{iid}" data-q="change">{html.escape(_question(rec["constraint"]))} '
                  f'{_radios(f"{iid}-change", ["more", "same", "less"])}</p>')
        sections.append(f'<section><h2>{html.escape(rec["prompt"])}</h2><div class="pair">'
                        f'<img src="data:image/jpeg;base64,{_jpeg(folder / "base" / "image.png")}">'
                        f'<img src="data:image/jpeg;base64,{_jpeg(folder / name / "image.png")}"></div>{qs}</section>')
    page.write_text(PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(sections)


# ------------------------------------------------------------------------------------------ scoring
def _row_stats(records, labels, settings, family):
    s_min = S_MIN.get(family, 0.05)
    interventions, skips = [], {+1: [], -1: []}
    for folder, rec in records:
        base = rec["runs"].get("base", {})
        for name in ("up", "down"):
            run = rec["runs"].get(name)
            if run is None:
                continue
            sign = run.get("sign", +1 if name == "up" else -1)
            skips[sign].append(run.get("skipped") is not None)
            if run.get("skipped"):
                continue
            lab = labels.get(f"{folder.parent.name}/{folder.name}/{name}", {})
            ds = None if run.get("s") is None or base.get("s") is None else run["s"] - base["s"]
            dy = None if run.get("y") is None or base.get("y") is None else run["y"] - base["y"]
            interventions.append({"sign": sign, "ds": ds, "dy": dy, "label": lab, "objects": rec})
    if family == "binding":
        fix, brk = [], []
        for it in interventions:
            q = it["objects"]["constraint"][1][1]
            left, right = it["label"].get("left"), it["label"].get("right")
            if left is None or right is None:
                continue
            ok_left, ok_right = left in (q, "both"), right in (q, "both")
            if it["sign"] > 0 and not ok_left:
                fix.append(ok_right)
            if it["sign"] < 0 and ok_left:
                brk.append(not ok_right)
        return {"fix": float(np.mean(fix)) if fix else None, "n_fix": len(fix),
                "break": float(np.mean(brk)) if brk else None, "n_break": len(brk),
                "skip_up": float(np.mean(skips[+1])) if skips[+1] else None,
                "skip_down": float(np.mean(skips[-1])) if skips[-1] else None, "n": len(interventions)}
    human = [{"more": 1, "same": 0, "less": -1}[it["label"]["change"]] * it["sign"] > 0
             for it in interventions if it["label"].get("change")]
    auto = [it["dy"] * it["sign"] > 0 for it in interventions if it["dy"] is not None]
    moved = [it for it in interventions if it["ds"] is not None and it["dy"] is not None and abs(it["ds"]) >= s_min]
    transfer = (sum(it["dy"] * it["ds"] for it in moved) / sum(it["ds"] ** 2 for it in moved)) if moved else None
    exploit = [it["ds"] * it["sign"] >= s_min and it["dy"] * it["sign"] <= 0
               for it in interventions if it["ds"] is not None and it["dy"] is not None]
    dir_h = float(np.mean(human)) if human else None
    dir_a = float(np.mean(auto)) if auto else None
    dir_used = dir_h if dir_h is not None else dir_a
    passed = None if dir_used is None or transfer is None else (
        dir_used >= settings.get("cfgDirAcc") and transfer >= settings.get("cfgTransfer"))
    return {"dir_h": dir_h, "n_h": len(human), "dir_a": dir_a, "n_a": len(auto), "transfer": transfer,
            "n_transfer": len(moved), "exploit": float(np.mean(exploit)) if exploit else None,
            "skip_up": float(np.mean(skips[+1])) if skips[+1] else None,
            "skip_down": float(np.mean(skips[-1])) if skips[-1] else None, "pass": passed, "n": len(interventions)}


def score(out: Path, labels, settings):
    result = {}
    for key, label, family, *_ in rows():
        records = [(p.parent, json.loads(p.read_text())) for p in sorted(Path(out, key).glob("*/record.json"))]
        result[key] = {"label": label, "family": family, **_row_stats(records, labels, settings, family)}
    return result


def table_rows(result) -> str:
    def pct(x):
        return "--" if x is None else f"{100 * x:.0f}"

    def num(x):
        return "--" if x is None else f"{x:.2f}"
    lines = []
    for key, r in result.items():
        skip = f"{pct(r['skip_up'])} / {pct(r['skip_down'])}"
        if r["family"] == "binding":
            lines.append(f"{r['label']} & {pct(r['fix'])} / {pct(r['break'])} & -- & -- & {skip} & --\\\\")
            continue
        passed = "--" if r["pass"] is None else ("yes" if r["pass"] else "no")
        lines.append(f"{r['label']} & {pct(r['dir_h'])} / {pct(r['dir_a'])} & {num(r['transfer'])} & "
                     f"{pct(r['exploit'])} & {skip} & {passed}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "page", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--only", help="comma-separated row keys")
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    from settings import load_settings
    s = load_settings()
    if args.action == "page":
        print("judgment page:", out / "gv.html", build_page(out, out / "gv.html"), "interventions")
        return
    if args.action == "score":
        result = score(out, json.load(open(args.labels)) if args.labels else {}, s)
        (out / "gv_scores.json").write_text(json.dumps(result, indent=1, default=str))
        print(table_rows(result))
        return
    from calibration import ProvisionalCalibration
    from estimators import ClipSegSupport, SigLIPEvidence
    from grounding import phrase_spans
    from measure import EstimatorDeps
    from run_experiment import _tiny_estimators, guide_cfg
    from sd3_backend import load_backend, tiny_backend
    todo = [r for r in rows() if not args.only or r[0] in args.only.split(",")]
    cfg = dataclasses.replace(guide_cfg(s), sigma_hi=0.9, clip="rms", dmax=0.02, eta0=0.04)   # the size setting
    if args.tiny:
        backend, (seg, siglip) = tiny_backend(), _tiny_estimators()
        cfg = dataclasses.replace(cfg, steps=2)
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), 1024, torch.float16)
        device = args.vae_device or args.tf_device
        seg = ClipSegSupport.from_pretrained(s.get("cfgCLIPSeg", str), device)
        siglip = SigLIPEvidence.from_pretrained(s.get("cfgSigLIP", str), device)
    for _, _, _, _, _, prompts in todo:
        for p in prompts:
            backend.encode_prompt(p, phrase_spans(p, translate(p)))
    if not args.tiny:
        backend.unload_text_encoders()
    deps = EstimatorDeps(seg, siglip, ProvisionalCalibration(s), s)
    for key, _, family, quantity, attribute, prompts in todo:
        for i, prompt in enumerate(prompts):
            folder = out / key / f"p{i}"
            rec = run_one(backend, deps, s, prompt, family, quantity, attribute, folder, cfg)
            (folder / "record.json").write_text(json.dumps(rec))
            print(f"{key} {i + 1}/{len(prompts)}: {prompt} | s " +
                  " ".join(f"{n}={r.get('s')}" for n, r in rec["runs"].items()), flush=True)
    if not args.tiny:
        measure_all(out, args.vae_device or args.tf_device)


if __name__ == "__main__":
    main()
