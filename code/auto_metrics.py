"""Automatic metrics of Table 1: Similarity (text-image and text-text) and CLIP-IQA.

Follows the protocol of Attend-and-Excite, which the manuscript cites for these metrics:
  text-image  cosine between CLIP ViT-B/16 embeddings of the image and of the prompt;
  text-text   a BLIP caption of the image, then the cosine between CLIP ViT-B/16 text embeddings of the
              caption and of the prompt.
ViT-B/16 rather than ViT-L/14, because SD3 encodes the prompt with CLIP-L/14. CLIPSeg, which guides
Stage 1, is built on CLIP ViT-B/16, so these scores are not independent of guidance (the manuscript says so).

CLIP-IQA (Wang et al. 2023) is piq's implementation of the original: CLIP ResNet-50 at the image's own size,
softmax over "Good photo." and "Bad photo.", reporting the probability of "Good photo.".

  python -I code/auto_metrics.py --run outputs/fmas --device cuda:1
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import FMAS_ARMS, FMAS_COMPARISONS, _paired_comparison  # noqa: E402

CLIP_MODEL = "openai/clip-vit-base-patch16"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
MODELS = "text-image: CLIP ViT-B/16 (openai/clip-vit-base-patch16); text-text: BLIP base captions, CLIP ViT-B/16 text"
IQA_MODEL = "CLIP-IQA: piq 0.8.0, CLIP ResNet-50, 'Good photo.' vs 'Bad photo.'"


def cosine(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-12))


class Clip:
    def __init__(self, device, name=CLIP_MODEL):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def image(self, img):
        x = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        return self.model.get_image_features(**x)[0].cpu()

    @torch.no_grad()
    def text(self, s):
        x = self.processor(text=[s], return_tensors="pt", padding=True, truncation=True).to(self.device)
        return self.model.get_text_features(**x)[0].cpu()


class Captioner:
    def __init__(self, device, name=CAPTION_MODEL):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.model = BlipForConditionalGeneration.from_pretrained(name).to(device).eval()
        self.processor = BlipProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def caption(self, img):
        x = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        out = self.model.generate(**x, max_new_tokens=30, num_beams=1, do_sample=False)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


def piq_clip_iqa():
    """piq's CLIPIQA class without running piq/__init__.py, which imports torchvision for other metrics."""
    if "piq.clip_iqa" not in sys.modules and "piq" not in sys.modules:
        root = Path(importlib.util.find_spec("piq").origin).parent
        for name, path in (("piq", root), ("piq.feature_extractors", root / "feature_extractors")):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    return importlib.import_module("piq.clip_iqa").CLIPIQA


class ClipIQA:
    def __init__(self, device):
        self.model = piq_clip_iqa()(data_range=1.0).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, img):
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
        return float(self.model(x.to(self.device)))


def score_run(out_dir, arms, clip, captioner, iqa=None):
    """One row per finished image of each arm; CLIP-IQA only when iqa is given."""
    rows = []
    for arm in arms:
        for path in sorted((Path(out_dir) / arm).glob("*/image.png")):
            with open(path.parent / "meta.json") as f:
                prompt = json.load(f)["prompt"]
            img = Image.open(path)
            caption = captioner.caption(img)
            prompt_text = clip.text(prompt)
            rows.append({"arm": arm, "prompt": prompt, "image": str(path), "caption": caption,
                         "text_image": cosine(clip.image(img), prompt_text),
                         "text_text": cosine(clip.text(caption), prompt_text)})
            if iqa is not None:
                rows[-1]["clip_iqa"] = iqa.score(img)
    return rows


def summarize(rows, comparisons=FMAS_COMPARISONS):
    metrics = [m for m in ("text_image", "text_text", "clip_iqa") if rows and all(m in r for r in rows)]
    models = MODELS + ("; " + IQA_MODEL if "clip_iqa" in metrics else "")
    out = {"n_prompts": len({r["prompt"] for r in rows}), "models": models, "comparisons": {}}
    for metric in metrics:
        per_arm, by_prompt = defaultdict(list), {}
        for r in rows:
            per_arm[r["arm"]].append(r[metric])
            by_prompt.setdefault(r["prompt"], {})[r["arm"]] = r[metric]
        out[metric] = {arm: sum(v) / len(v) for arm, v in per_arm.items()}
        out["comparisons"][metric] = {f"{a} - {b}": _paired_comparison(by_prompt, a, b) for a, b in comparisons}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run folder with one subfolder per arm")
    ap.add_argument("--arms", default=",".join(FMAS_ARMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-iqa", action="store_true", help="similarity only")
    args = ap.parse_args()
    iqa = None if args.no_iqa else ClipIQA(args.device)
    rows = score_run(args.run, args.arms.split(","), Clip(args.device), Captioner(args.device), iqa)
    run = Path(args.run)
    with open(run / "similarity.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    (run / "similarity.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
