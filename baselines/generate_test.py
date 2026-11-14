"""Published baselines on the test prompts, for the paired comparison (Tab. paired differences).

  python -I baselines/generate_test.py --method spo --n 20 --out /kaggle/working/outputs/paired --device cuda:0
  python -I baselines/generate_test.py --method ae  --n 20 --out /kaggle/working/outputs/paired --device cuda:1

The prompts are the first n prompts of each setting (soft, mixture, size) in --group, the ones run_experiment.py
--group G --n n uses, at seed 0. The paired table is meant for the test group, which run_experiment.py only opens
after the calibration is fitted and the manifest frozen; until then use a development group (dev) and call the
table a pilot. Each image goes to {out}/{setting}/{method}/{image_id}/ with a meta.json, under the same
image id as the SD3 arms, so the rating pages of code/annotate.py show it as one more blind panel; each setting has
its own folder ({out}/soft, {out}/mixture, {out}/size), like the SD3 runs of notebook section 31.

spo: SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep as its model card shows (SDXL, fp16-fix VAE, guidance 5, 50 steps,
1024 px), as in spo/generate.py. ae: Attend-and-Excite with the authors' published settings on its native backbone,
Stable Diffusion 1.4 at 512 px (50 steps, guidance 7.5, max_iter_to_alter 25); the excited tokens are the first
token of every object noun of the prompt.
"""
import argparse
import json
import os
import sys
from pathlib import Path

for name, value in (("USE_TF", "0"), ("USE_FLAX", "0"), ("USE_JAX", "0"), ("USE_TORCH", "1")):
    os.environ.setdefault(name, value)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from predicates import translate  # noqa: E402
from prompts import select  # noqa: E402
from run_experiment import image_id  # noqa: E402

SUITES = ("soft", "mixture", "size")
SEED = 0
SPO = {"model": "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep", "vae": "madebyollin/sdxl-vae-fp16-fix",
       "settings": {"num_inference_steps": 50, "guidance_scale": 5.0, "height": 1024, "width": 1024}}
AE = {"model": "CompVis/stable-diffusion-v1-4",
      "settings": {"num_inference_steps": 50, "guidance_scale": 7.5, "max_iter_to_alter": 25, "height": 512, "width": 512}}


def test_prompts(n, group="test"):
    """(setting, prompt) for the first n prompts of each setting in the group."""
    return [(suite, r.prompt) for suite in SUITES for r in select(suite, group, n)]


def object_token_indices(tokenizer, prompt):
    """CLIP token positions (BOS = 0) of the first sub-token of every object noun in the prompt."""
    objects = set(translate(prompt).objects)
    ids = tokenizer(prompt).input_ids
    pieces = [tokenizer.decode([i]).strip().lower() for i in ids]
    out = []
    for pos, piece in enumerate(pieces):
        for obj in objects:
            joined = ""
            for nxt in pieces[pos:pos + 4]:                      # a noun split into up to 4 sub-tokens
                joined += nxt
                if joined == obj:
                    break
            if joined == obj:
                out.append(pos)
                break
    return sorted(set(out))


def load(method, device):
    if method == "spo":
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(SPO["model"], torch_dtype=torch.float16)
        pipe.vae = AutoencoderKL.from_pretrained(SPO["vae"], torch_dtype=torch.float16)
    else:
        from diffusers import StableDiffusionAttendAndExcitePipeline
        pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(AE["model"], torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, method, prompt, device):
    generator = torch.Generator(device).manual_seed(SEED)
    if method == "spo":
        return pipe(prompt, generator=generator, **SPO["settings"]).images[0], {}
    tokens = object_token_indices(pipe.tokenizer, prompt)
    return pipe(prompt, token_indices=tokens, generator=generator, **AE["settings"]).images[0], {"token_indices": tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("spo", "ae"), required=True)
    ap.add_argument("--n", type=int, default=20, help="prompts per setting")
    ap.add_argument("--group", default="test", help="prompt group; test only after the manifest is frozen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    todo = [(s, p) for s, p in test_prompts(args.n, args.group)
            if not (Path(args.out) / s / args.method / image_id(p, SEED) / "image.png").exists()]
    print(f"{args.method}: {len(todo)} images to make", flush=True)
    if not todo:
        return
    pipe = load(args.method, args.device)
    spec = SPO if args.method == "spo" else AE
    for n, (suite, prompt) in enumerate(todo):
        folder = Path(args.out) / suite / args.method / image_id(prompt, SEED)
        folder.mkdir(parents=True, exist_ok=True)
        image, extra = generate(pipe, args.method, prompt, args.device)
        image.save(folder / "image.png")
        (folder / "meta.json").write_text(json.dumps({"prompt": prompt, "arm": args.method, "seed": SEED, "split": args.group,
                                                      "model": spec["model"], **spec["settings"], **extra}, indent=1))
        print(f"{args.method}: {n + 1}/{len(todo)} {prompt}", flush=True)


if __name__ == "__main__":
    main()
