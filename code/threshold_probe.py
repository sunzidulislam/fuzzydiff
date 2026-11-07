"""Measure the presence and attention values the models really produce, to set provisional thresholds.

Samples development prompts without guidance. At noisy steps and on the final image it scores the
objects in the prompt and objects that are not, and checks whether each phrase's attention peak lands
on its object in the final image. Labels are assumed, not annotated: an object in the prompt counts as
present, and the object mask comes from CLIPSeg. Results replace guesses and are never paper evidence.

  python -I code/threshold_probe.py --out outputs/probe --vae-device cuda:1
  python -I code/threshold_probe.py --out outputs/probe --summarize
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from calibration_fit import _auroc, cross_fit, fit_presence, select_attention  # noqa: E402
from estimators import ClipSegSupport, SigLIPEvidence  # noqa: E402
from grounding import binding_keys, phrase_spans  # noqa: E402
from guidance import _bin, _euler  # noqa: E402
from measure import EstimatorDeps, Measure  # noqa: E402
from prompts import base_prompt, select, vocab  # noqa: E402
from run_experiment import _save_image, _tiny_estimators, image_id  # noqa: E402
from sd3_backend import load_backend, tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

SHARE_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
ABSENT_PER_PROMPT = 3


def probe_prompts(n):
    half = n // 2
    return select("plain_color", "dev", half) + select("mixture", "dev", n - half)


def absent_objects(prompt, program, n, seed):
    words = set(re.findall(r"[a-z-]+", prompt.lower())) | set(program.objects)
    pool = [o for o in vocab()["objects"] if o not in words]
    return random.Random(f"{prompt}|{seed}").sample(pool, n)


def attention_rows(attention, supports, attribute_of, base):
    """One row per phrase: does its attention peak land on its object in the final image?"""
    rows = []
    for key, a in attention.items():
        head = attribute_of.get(key, key).split("#")[0]
        if head not in supports:
            continue
        side = a["P"].shape[-1]
        y, x = divmod(int(a["P"].argmax()), side)
        S = supports[head]
        h = S.shape[-1]
        rows.append({"type": "attention", "kind": "attributes" if key in attribute_of else "objects",
                     "config": "read_blocks", "peak": a["peak"], "share": a["share"], "mass": a["mass"],
                     "correct": bool(S[y * h // side, x * h // side] > 0.5), "base": base})
    return rows


def _presence_rows(evidence, labels, bin, base):
    return [{"type": "presence", "bin": bin, "object": q, "label": labels[q], "base": base,
             "global": float(o["g"]), "region": float(o["r"])}
            for q, o in evidence.objects.items() if q in labels]


def probe_image(backend, measure, cond, prompt, program, absent, seed, steps, every, sigma_lo, sigma_hi, cfg=7.0):
    """Sample one image without guidance. Returns the image and its rows."""
    base = base_prompt(prompt)
    labels = {**{q: 1 for q in program.objects}, **{q: 0 for q in absent}}
    keys = binding_keys(prompt, program)
    attribute_of = {a_key: q_key for a_key, q_key in keys.values()}
    sigmas = backend.sigmas(steps)
    z = backend.noise_like_latent(seed)
    rows, attention = [], []
    seen = 0
    with torch.no_grad():
        for i in range(steps):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            if sigma_lo <= float(sigma) <= sigma_hi and seen % every == 0:
                b = _bin(sigma)
                out = backend.velocity(z, sigma, cond, readout=True, cfg=cfg)
                x0 = backend.decode(backend.clean(z, sigma, out.v))
                evidence = measure(backend, x0, out, program, b)
                measure._objects(x0[0] if x0.dim() == 4 else x0, absent, evidence, b)
                rows += _presence_rows(evidence, labels, b, base)
                attention.append(evidence.attention)
                v = out.v
                seen += 1
            else:
                if sigma_lo <= float(sigma) <= sigma_hi:
                    seen += 1
                v = backend.velocity(z, sigma, cond, cfg=cfg).v
            z = _euler(z, sigma, sigma_next, v)
        image = backend.decode(z)
        final = measure(backend, image, None, program, _bin(sigmas[-1]))
        measure._objects(image[0] if image.dim() == 4 else image, absent, final, _bin(sigmas[-1]))
    rows += _presence_rows(final, labels, _bin(sigmas[-1]), base)
    supports = {q: final.objects[q]["S"].detach().cpu() for q in program.objects if q in final.objects}
    for step_attention in attention:
        rows += attention_rows(step_attention, supports, attribute_of, base)
    return image, rows


def _presence_summary(rows):
    """Per noise bin: one logistic model, judged only on prompts it was not fitted to."""
    by_bin = defaultdict(list)
    for r in rows:
        by_bin[str(r["bin"])].append(r)

    def fit(train):
        return fit_presence([r["global"] for r in train], [r["region"] for r in train], [r["label"] for r in train],
                            [r["base"] for r in train])

    def predict(model, test):
        return [{"logit": model.logit(r["global"], r["region"]), "label": r["label"]} for r in test]

    out = {}
    for b, cell in sorted(by_bin.items()):
        if len({r["label"] for r in cell}) < 2:
            continue
        held = cross_fit(cell, fit, predict)
        logits = np.array([r["logit"] for r in held])
        labels = np.array([bool(r["label"]) for r in held])
        pos, neg = logits[labels], logits[~labels]
        model = fit(cell).to_dict()
        out[b] = {"n_present": int(labels.sum()), "n_absent": int((~labels).sum()), "auc": _auroc(logits, labels),
                  "present_pass": float((pos >= 0).mean()), "absent_pass": float((neg >= 0).mean()),
                  "present_reach_target": float((pos >= np.log(0.9 / 0.1)).mean()),
                  "model": {k: model[k] for k in ("w_global", "w_region", "bias")}}
    return out


def _attention_summary(rows, share_grid, required):
    candidates = [{"config": "read_blocks", "kappa_pk": 2.0, "m_min": m, "tau_min": 0.0} for m in share_grid]
    chosen = select_attention(rows, candidates, required)
    out = {}
    for kind, c in chosen.items():
        cell = [r for r in rows if r["kind"] == kind]
        shares = [r["share"] for r in cell] or [float("nan")]
        out[kind] = {"feasible": c.feasible, "m_min": c.m_min, "accuracy_at_m_min": c.accuracy, "folds_passed": c.folds_passed,
                     "n": len(cell), "accuracy_all": float(np.mean([r["correct"] for r in cell])) if cell else None,
                     "share_q10_q50_q90": [float(q) for q in np.percentile(shares, [10, 50, 90])],
                     "kept": float(np.mean([r["share"] >= c.m_min for r in cell])) if c.feasible else 0.0}
    return out


def summarize(rows, share_grid=SHARE_GRID, required=0.8):
    return {"presence": _presence_summary([r for r in rows if r["type"] == "presence"]),
            "attention": _attention_summary([r for r in rows if r["type"] == "attention"], share_grid, required)}


def settings_yaml(summary):
    """The values to place in configs/provisional.yaml."""
    values = {"presence_models": {int(b): v["model"] for b, v in summary["presence"].items()}}
    feasible = [a["m_min"] for a in summary["attention"].values() if a["feasible"]]
    if feasible:
        values["attention_m_min"] = max(feasible)   # one threshold serves objects and attributes alike
    return yaml.safe_dump(values, sort_keys=False)


def _report(rows, out):
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("--- values for configs/provisional.yaml ---")
    print(settings_yaml(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--every", type=int, default=2, help="measure every Nth step in the window")
    ap.add_argument("--summarize", action="store_true", help="only summarize an earlier run")
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "probe.jsonl"
    if args.summarize:
        _report([json.loads(line) for line in open(path)], out)
        return

    s = load_settings()
    calib = ProvisionalCalibration(s)
    steps = 4 if args.tiny else s.get("cfgSteps", int)
    if args.tiny:
        backend = tiny_backend()
        seg, siglip = _tiny_estimators()
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), args.resolution, torch.float16)
        device = args.vae_device or args.tf_device
        seg = ClipSegSupport.from_pretrained(s.get("cfgCLIPSeg", str), device)
        siglip = SigLIPEvidence.from_pretrained(s.get("cfgSigLIP", str), device)
    deps = EstimatorDeps(seg, siglip, calib, s)
    prompts = probe_prompts(args.n)
    conds = {r.prompt: backend.encode_prompt(r.prompt, phrase_spans(r.prompt, r.program)) for r in prompts}
    if not args.tiny:
        backend.unload_text_encoders()
    rows = []
    (out / "images").mkdir(exist_ok=True)
    with open(path, "w") as f:
        for n, r in enumerate(prompts):
            absent = absent_objects(r.prompt, r.program, ABSENT_PER_PROMPT, seed=0)
            image, new = probe_image(backend, Measure(deps, r.prompt, conds[r.prompt]), conds[r.prompt], r.prompt, r.program,
                                     absent, 0, steps, args.every, s.get("cfgSigmaLo"), s.get("cfgSigmaHi"), s.get("cfgCFG"))
            _save_image(image, out / "images" / f"{image_id(r.prompt, 0)}.png")
            for row in new:
                f.write(json.dumps(row) + "\n")
            f.flush()
            rows += new
            print(f"{n + 1}/{len(prompts)} {r.prompt}", flush=True)
    _report(rows, out)


if __name__ == "__main__":
    main()
