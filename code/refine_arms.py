"""Tab. refinement-matched comparison: "+R" passes another method's image through the same unmodified SD3
image-to-image pass that FuzzyDiff's Stage 2 uses.

  python -I code/refine_arms.py --out outputs/paired --arms unguided,sd1x,ae,spo --vae-device cuda:1

For each setting folder in --out (mixture, size) and each source arm, every image is resized to the SD3 resolution
(1024 px, bicubic), re-encoded with the SD3 VAE and refined by sar.run_stage2(mode="plain"): the same checkpoint,
starting noise sigma_s (cfgStrength), cfgKSteps steps, CFG scale, prompt and seed (noise seed 1_000_003 + seed) as
the stage1_plain arm, i.e. "FuzzyDiff S1 + R". The result goes to {setting}/{arm}_r/{image_id}/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SETTINGS = ("mixture", "size")


def to_tensor(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def refine(backend, deps, prompt, seed, image, log_path):
    """One plain SD3 img2img pass from an image (no Stage-1 traces: plain mode uses none)."""
    from grounding import phrase_spans
    from guidance import Stage1Out
    from predicates import translate
    from reporting import StepLog
    from sar import run_stage2
    program = translate(prompt)
    cond = backend.encode_prompt(prompt, phrase_spans(prompt, program))
    stub = Stage1Out(latent=None, image=image.to(backend.tf_device, backend.latent_dtype), traces={}, supports={},
                     finals=[])
    log = StepLog(str(log_path), "refine", deps.calib.tag)
    with torch.no_grad():
        result = run_stage2(backend, stub, cond, program, prompt, deps.calib, deps.seg, deps.sar_cfg, seed, log,
                            mode="plain", steps=deps.steps)
    log.close()
    return result.image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated source arms, e.g. unguided,sd1x,ae,spo")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    args = ap.parse_args()
    from types import SimpleNamespace
    from calibration import ProvisionalCalibration
    from run_experiment import _save_image, _tiny_estimators, sar_cfg
    from sd3_backend import load_backend, tiny_backend
    from settings import load_settings
    s = load_settings()
    if args.tiny:
        backend, (seg, _) = tiny_backend(), _tiny_estimators()
        steps, size = 2, 128                                    # the tiny VAE decodes 16 latents to 128 px
    else:
        from estimators import ClipSegSupport
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, torch.float16, s.get("cfgCFG"), 1024, torch.float16)
        seg = ClipSegSupport.from_pretrained(s.get("cfgCLIPSeg", str), args.vae_device or args.tf_device)
        steps, size = s.get("cfgSteps", int), 1024
    cfg = sar_cfg(s)
    if args.tiny:
        import dataclasses
        cfg = dataclasses.replace(cfg, k=1)
    deps = SimpleNamespace(calib=ProvisionalCalibration(s), seg=seg, sar_cfg=cfg, steps=steps)
    for setting in args.settings.split(","):
        for arm in args.arms.split(","):
            for meta_path in sorted(Path(args.out, setting, arm).glob("*/meta.json")):
                iid = meta_path.parent.name
                dst = Path(args.out, setting, f"{arm}_r", iid)
                if (dst / "image.png").exists() or not (meta_path.parent / "image.png").exists():
                    continue
                meta = json.loads(meta_path.read_text())
                dst.mkdir(parents=True, exist_ok=True)
                image = refine(backend, deps, meta["prompt"], int(meta.get("seed", 0)),
                               to_tensor(meta_path.parent / "image.png", size), dst / "log.jsonl")
                _save_image(image, dst / "image.png")
                (dst / "meta.json").write_text(json.dumps({"prompt": meta["prompt"], "arm": f"{arm}_r", "seed": meta.get("seed", 0),
                                                          "source": arm, "refinement": {"k": cfg.k, "strength": cfg.strength}}))
                print(f"{setting} {arm}_r: {meta['prompt']}", flush=True)


if __name__ == "__main__":
    main()
