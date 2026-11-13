"""Tab. joint-attention validation: pointing, pixel AUROC, binding AUROC (b and beta) and the share of
uninformative maps, per block group of SD3-Medium and for the selected readout blocks at early and late steps.

  python -I code/attention_probe.py run   --out outputs/attn --vae-device cuda:1          # about 1.5 h on 2 x T4
  python -I code/attention_probe.py page  --out outputs/attn                              # binding label page
  python -I code/attention_probe.py score --out outputs/attn --boxes labels.json --binding binding.json

run samples plain SD3 (seed 0, the sampler of run_experiment.py) for the calibration prompts of G_cal (the same
50 soft-attribute and 30 size prompts as calibration_pilot.py, so their object boxes apply: unguided sampling at a
fixed seed reproduces the image) and 20 two-colour mixture prompts for binding. At every step of the guided window
it reads all 24 blocks and keeps, per block group and phrase, the localization field P (64 x 64), the confidence
statistics, and for each attribute and object the overlap b(a, q) and the contrastive score beta(a, q).

score: pointing = the peak of P lies inside the object's box; pixel AUROC = P against the box mask; binding AUROC =
b or beta of the true (attribute, object) pairs against the swapped ones, labelled by a person; uninformative =
the map fails the confidence test of grounding.aggregate_maps.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from calibration_fit import _auroc  # noqa: E402
from calibration_pilot import box_mask  # noqa: E402
from grounding import aggregate_maps, binding, binding_keys, overlap, phrase_spans  # noqa: E402
from guidance import _euler  # noqa: E402
from prompts import select  # noqa: E402
from run_experiment import _save_image, image_id  # noqa: E402
from sd3_backend import load_backend, tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

GROUPS = {"blocks 0-7": range(0, 8), "blocks 8-15": range(8, 16), "blocks 16-23": range(16, 24)}
WINDOW = (0.25, 0.90)          # the guided window of the runs (sigma_lo, sigma_hi of the size setting)


def probe_prompts(n_soft=50, n_size=30, n_binding=20):
    return ([(r.prompt, "calib") for r in select("soft", "cal", n_soft)] +
            [(r.prompt, "calib") for r in select("size", "cal", n_size)] +
            [(r.prompt, "binding") for r in select("mixture", "cal", n_binding)])


def groups(settings, n_blocks=24):
    out = {k: [b for b in v if b < n_blocks] for k, v in GROUPS.items()}
    out["selected L"] = [b for b in settings.get("cfgReadBlocks", list) if b < n_blocks]
    return {k: v for k, v in out.items() if v}


def probe_image(backend, prompt, program, cond, settings, calib, steps, seed=0):
    """Plain sampling; at each step of the window, every group's fields and statistics."""
    tau = settings.get("cfgSmaxTemp")
    thresholds = calib.attention_thresholds("objects")
    keys = binding_keys(prompt, program)
    objects = list(dict.fromkeys(program.objects))
    sigmas = backend.sigmas(steps)
    z = backend.noise_like_latent(seed)
    fields, records = {}, []
    blocks_of = groups(settings, len(backend.procs))
    with torch.no_grad():
        for i in range(steps):
            sigma = sigmas[i]
            if WINDOW[0] <= float(sigma) <= WINDOW[1]:
                out = backend.velocity(z, sigma, cond, readout=True)
                for g, blocks in blocks_of.items():
                    grounded = aggregate_maps(out, cond, blocks, thresholds)
                    for q, gp in grounded.items():
                        fields[f"{g}|{i}|{q}"] = gp.P.detach().float().cpu().numpy().astype(np.float16)
                        records.append({"type": "map", "group": g, "step": i, "sigma": float(sigma), "phrase": q,
                                        "object": q.split("#")[0] in objects, "peak": gp.peak, "share": gp.share,
                                        "mass": gp.mass, "informative": bool(gp.informative)})
                    for (a, q), (ka, kq) in keys.items():
                        if ka in grounded and kq in grounded:
                            others = [grounded[k].P for k in grounded if k != kq and k.split("#")[0] in objects]
                            for cand in [k for k in grounded if k.split("#")[0] in objects]:
                                rival = [grounded[k].P for k in grounded if k != cand and k.split("#")[0] in objects]
                                records.append({"type": "pair", "group": g, "step": i, "sigma": float(sigma),
                                                "attribute": a, "object": cand.split("#")[0], "bound_to": q,
                                                "b": float(overlap(grounded[ka].P, grounded[cand].P)),
                                                "beta": float(binding(grounded[ka].P, grounded[cand].P, rival or others, tau))})
                v = out.v
            else:
                v = backend.velocity(z, sigma, cond).v
            z = _euler(z, sigma, sigmas[i + 1], v)
        image = backend.decode(z)
    return image, fields, records


def run(backend, settings, out: Path, prompts, steps, seed=0):
    from predicates import translate
    calib = ProvisionalCalibration(settings)
    out.mkdir(parents=True, exist_ok=True)
    for n, (prompt, kind) in enumerate(prompts):
        folder = out / image_id(prompt, seed)
        if (folder / "records.jsonl").exists():
            continue
        program = translate(prompt)
        cond = backend.encode_prompt(prompt, phrase_spans(prompt, program))
        image, fields, records = probe_image(backend, prompt, program, cond, settings, calib, steps, seed)
        folder.mkdir(parents=True, exist_ok=True)
        _save_image(image, folder / "image.png")
        np.savez_compressed(folder / "fields.npz", **fields)
        (folder / "meta.json").write_text(json.dumps({"prompt": prompt, "kind": kind, "seed": seed}))
        with open(folder / "records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"{n + 1}/{len(prompts)} {prompt}", flush=True)


# ------------------------------------------------------------------------------------------ binding labels
BINDING_PAGE = """<!doctype html><meta charset="utf-8"><title>Binding labels</title>
<style>body{font:15px system-ui;margin:0 auto;max-width:900px;padding:16px}img{width:100%%;max-width:512px;display:block}
section{margin-bottom:32px}label{margin-right:10px}p.q{margin:6px 0}textarea{width:100%%;height:140px}</style>
<h1>Binding labels</h1><p>For each image and each colour, say which object has that colour. "both" if both objects
have it, "neither" if none does or the object is missing.</p>
%(sections)s
<button id="export">Show labels</button> <span id="missing"></span><textarea id="result" readonly></textarea>
<script>
document.getElementById("export").onclick=function(){var out={},missing=0;
  document.querySelectorAll("[data-attr]").forEach(function(p){var c=p.querySelector("input:checked");if(!c)missing++;
    var id=p.dataset.image;out[id]=out[id]||{};out[id][p.dataset.attr]=c?c.value:null;});
  document.getElementById("missing").textContent=missing?missing+" answers missing":"All answered";
  document.getElementById("result").value=JSON.stringify(out);};
</script>"""


def _jpeg(path, size=512):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_page(out: Path, page: Path):
    from predicates import translate
    sections = []
    for meta_path in sorted(out.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["kind"] != "binding":
            continue
        program = translate(meta["prompt"])
        objects = list(dict.fromkeys(program.objects))
        attrs = [c.args[0] for c in program.constraints if c.family == "binding"]
        iid = meta_path.parent.name
        qs = "".join(f'<p class="q" data-image="{iid}" data-attr="{html.escape(a)}">Which object is {html.escape(a)}? '
                     + "".join(f'<label><input type="radio" name="{iid}-{html.escape(a)}" value="{html.escape(v)}">{html.escape(v)}</label>'
                               for v in objects + ["both", "neither"]) + "</p>" for a in attrs)
        sections.append(f'<section><h2>{html.escape(meta["prompt"])}</h2>'
                        f'<img src="data:image/jpeg;base64,{_jpeg(meta_path.parent / "image.png")}">{qs}</section>')
    page.write_text(BINDING_PAGE % {"sections": "\n".join(sections)}, encoding="utf-8")
    return len(sections)


# ------------------------------------------------------------------------------------------ scoring
def _phase(records):
    """early / late split at the median sigma of the window steps."""
    sig = sorted({r["sigma"] for r in records})
    return float(np.median(sig)) if sig else 0.0


def _row_key(r, mid):
    if r["group"] != "selected L":
        return r["group"]
    return "selected L, early" if r["sigma"] >= mid else "selected L, late"


def score(out: Path, boxes_path=None, binding_path=None):
    boxes = json.load(open(boxes_path)) if boxes_path else {}
    boxes = {k.split("/")[-1]: v for k, v in boxes.items()}          # labels are keyed unguided/<image id>
    bind = json.load(open(binding_path)) if binding_path else {}
    rows = defaultdict(lambda: {"point": [], "auroc": [], "uninf": [], "pairs": []})
    for folder in sorted(p.parent for p in out.glob("*/records.jsonl")):
        iid = folder.name
        records = [json.loads(line) for line in open(folder / "records.jsonl")]
        mid = _phase(records)
        fields = np.load(folder / "fields.npz")
        lab = boxes.get(iid, {})
        for r in records:
            key = _row_key(r, mid)
            if r["type"] == "map":
                if r["object"]:
                    rows[key]["uninf"].append(not r["informative"])
                head = r["phrase"].split("#")[0]
                box = lab.get("boxes", {}).get(head) if lab.get("present", {}).get(head) else None
                if r["object"] and box is not None:
                    P = fields[f"{r['group']}|{r['step']}|{r['phrase']}"].astype(np.float32)
                    m = box_mask(box, P.shape[-1]).astype(bool)
                    y, x = divmod(int(P.argmax()), P.shape[-1])
                    rows[key]["point"].append(bool(m[y, x]))
                    if 0 < m.sum() < m.size:
                        rows[key]["auroc"].append(_auroc(P.ravel(), m.ravel()))
            elif r["type"] == "pair" and iid in bind and bind[iid].get(r["attribute"]) is not None:
                answer = bind[iid][r["attribute"]]
                label = answer == r["object"] or answer == "both"
                rows[key]["pairs"].append({"b": r["b"], "beta": r["beta"], "label": label})
    table = {}
    for key, v in rows.items():
        pairs = v["pairs"]
        labels = [p["label"] for p in pairs]
        two = len(set(labels)) == 2
        table[key] = {"pointing": float(np.mean(v["point"])) if v["point"] else None, "n_point": len(v["point"]),
                      "pixel_auroc": float(np.mean(v["auroc"])) if v["auroc"] else None,
                      "binding_b": _auroc([p["b"] for p in pairs], labels) if two else None,
                      "binding_beta": _auroc([p["beta"] for p in pairs], labels) if two else None, "n_pairs": len(pairs),
                      "uninformative": float(np.mean(v["uninf"])) if v["uninf"] else None, "n_maps": len(v["uninf"])}
    return table


ORDER = (("blocks 0-7", "Blocks 0--7"), ("blocks 8-15", "Blocks 8--15"), ("blocks 16-23", "Blocks 16--23"),
         ("selected L, early", "Selected $\\mathcal L$, early guided steps"),
         ("selected L, late", "Selected $\\mathcal L$, late guided steps"))


def table_rows(table) -> str:
    def f(x, pct=False):
        if x is None:
            return "--"
        return f"{100 * x:.1f}" if pct else f"{x:.2f}"
    lines = []
    for key, label in ORDER:
        t = table.get(key)
        if t is None:
            lines.append(f"{label} & -- & -- & -- / -- & --\\\\")
            continue
        lines.append(f"{label} & {f(t['pointing'], True)} & {f(t['pixel_auroc'])} & {f(t['binding_b'])} / "
                     f"{f(t['binding_beta'])} & {f(t['uninformative'], True)}\\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("run", "page", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes")
    ap.add_argument("--binding")
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--n", type=int, nargs=3, default=(50, 30, 20), metavar=("SOFT", "SIZE", "BINDING"))
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    if args.action == "page":
        print("binding page:", out / "binding.html", build_page(out, out / "binding.html"), "images")
        return
    if args.action == "score":
        table = score(out, args.boxes, args.binding)
        (out / "attention_scores.json").write_text(json.dumps(table, indent=1))
        print(json.dumps(table, indent=1))
        print(table_rows(table))
        return
    s = load_settings()
    prompts = probe_prompts(*args.n)
    if args.tiny:
        backend, steps = tiny_backend(read_blocks=(0, 1, 2)), 4
    else:
        backend = load_backend(args.model, list(range(24)), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), 1024, torch.float16)
        steps = s.get("cfgSteps", int)
    from predicates import translate
    for prompt, _ in prompts:
        backend.encode_prompt(prompt, phrase_spans(prompt, translate(prompt)))
    if not args.tiny:
        backend.unload_text_encoders()
    run(backend, s, out, prompts, steps)


if __name__ == "__main__":
    main()
