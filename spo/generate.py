"""Table 3 hedge ladders with SPO (SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep), loaded as its model card shows.

SPO is a preference-tuned SDXL checkpoint: StableDiffusionXLPipeline in fp16 with the madebyollin/sdxl-vae-fp16-fix
VAE, guidance 5 (the card's example), the pipeline's default 50 steps, 1024 px. The prompt is the full ladder
prompt with its hedge word, at seed 0; ladder_plan.json is the same plan as concept_sliders/ (10 bases per family
x 5 hedges). Images go to {out}/{run}/spo/{image_id}/ next to the SD3 ladder runs, with a meta.json, so
code/ladder.py measures them like the other rows (--arms spo).

  python -I spo/generate.py --out /kaggle/working/outputs --device cuda:0 --part 0 --parts 2
"""
import argparse
import json
import os
from pathlib import Path

# Kaggle images ship JAX and TensorFlow; diffusers and transformers import them when present, and their LLVM
# clashes with PyTorch's ("Option 'bitcode-mdindex-threshold' registered more than once", abort).
for name, value in (("USE_TF", "0"), ("USE_FLAX", "0"), ("USE_JAX", "0"), ("USE_TORCH", "1")):
    os.environ.setdefault(name, value)

import torch  # noqa: E402

PLAN = Path(__file__).with_name("ladder_plan.json")
MODEL = "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep"
VAE = "madebyollin/sdxl-vae-fp16-fix"
SETTINGS = {"num_inference_steps": 50, "guidance_scale": 5.0, "height": 1024, "width": 1024}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--part", type=int, default=0, help="this process takes every parts-th image from part")
    ap.add_argument("--parts", type=int, default=1)
    args = ap.parse_args()
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    plan = json.load(open(PLAN))
    todo = [im for im in plan["images"][args.part::args.parts]
            if not (Path(args.out) / im["run"] / "spo" / im["image_id"] / "image.png").exists()]
    print(f"part {args.part}: {len(todo)} images to make", flush=True)
    if not todo:
        return
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    for n, im in enumerate(todo):
        folder = Path(args.out) / im["run"] / "spo" / im["image_id"]
        folder.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(args.device).manual_seed(plan["seed"])
        image = pipe(im["prompt"], generator=generator, **SETTINGS).images[0]
        image.save(folder / "image.png")
        (folder / "meta.json").write_text(json.dumps({"prompt": im["prompt"], "arm": "spo", "seed": plan["seed"],
                                                      "model": MODEL, "vae": VAE, **SETTINGS}, indent=1))
        print(f"part {args.part}: {n + 1}/{len(todo)} {im['prompt']}", flush=True)


if __name__ == "__main__":
    main()
