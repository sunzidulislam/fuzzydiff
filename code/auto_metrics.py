"""Automatic similarity (Table 1, "Similarity"): text-image and text-text.

Follows the protocol of Attend-and-Excite, which the manuscript cites for these metrics:
  text-image  cosine between CLIP ViT-B/16 embeddings of the image and of the prompt;
  text-text   a BLIP caption of the image, then the cosine between CLIP ViT-B/16 text embeddings of the
              caption and of the prompt.
ViT-B/16 rather than ViT-L/14, because SD3 encodes the prompt with CLIP-L/14. CLIPSeg, which guides
Stage 1, is built on CLIP ViT-B/16, so these scores are not independent of guidance (the manuscript says so).

  python -I code/auto_metrics.py --run outputs/fmas --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import FMAS_ARMS, FMAS_COMPARISONS, _paired_comparison  # noqa: E402

CLIP_MODEL = "openai/clip-vit-base-patch16"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
MODELS = "text-image: CLIP ViT-B/16 (openai/clip-vit-base-patch16); text-text: BLIP base captions, CLIP ViT-B/16 text"


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


def score_run(out_dir, arms, clip, captioner):
    """One row per finished image of each arm."""
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
    return rows


def summarize(rows, comparisons=FMAS_COMPARISONS):
    out = {"n_prompts": len({r["prompt"] for r in rows}), "models": MODELS, "comparisons": {}}
    for metric in ("text_image", "text_text"):
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
    args = ap.parse_args()
    rows = score_run(args.run, args.arms.split(","), Clip(args.device), Captioner(args.device))
    run = Path(args.run)
    with open(run / "similarity.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    (run / "similarity.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
