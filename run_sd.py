"""Plain Stable Diffusion 1.x generation: a prompt, a seed, the object words to watch.

No fuzzy guidance. This is the backbone on its own, for seeing what the base model
produces before any FuzzyDiff method is applied.

Stability never released open weights for Stable Diffusion 1.6; it exists only behind
their hosted API. The open 1.x line stops at 1.5, so that is the default. Pass --model
to point at any other SD 1.x checkpoint, local path or Hub id.

The scheduler matches the FuzzyDiff notebook (DPMSolver++ with Karras sigmas) so images
from here are directly comparable to the method's output at the same seed and steps.

    python run_sd.py --prompt "a red book and a yellow clock" --words "red book,yellow clock" --seed 42
"""
import argparse
import json
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

DEFAULT_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'


def build_pipeline(model_id, device):
    # fp16 has no CPU kernels, so a CPU run must stay in fp32.
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, algorithm_type='dpmsolver++', use_karras_sigmas=True)
    if device.type == 'cuda':
        pipeline.enable_model_cpu_offload(gpu_id=0)
    else:
        pipeline.to(device)
    return pipeline


def generate(pipeline, prompt, seed, steps=50, guidance=7.5, size=512, negative_prompt=''):
    return pipeline(prompt=prompt, negative_prompt=negative_prompt, height=size, width=size,
                    num_inference_steps=steps, guidance_scale=guidance,
                    generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]


def missing_words(tokenizer, prompt, words):
    """A phrase the tokenizer cannot find is almost always a typo, not a model failure."""
    ids = tokenizer.encode(prompt)
    missing = []
    for word in words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        spans = (ids[start:start + len(word_ids)] for start in range(len(ids) - len(word_ids) + 1))
        if not word_ids or word_ids not in spans:
            missing.append(word)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt', required=True, help='Text prompt.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--words', default='',
                        help='Comma-separated object words to record and check against the prompt.')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='SD 1.x checkpoint (local path or Hub id).')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.5, help='Classifier-free guidance scale.')
    parser.add_argument('--size', type=int, default=512, help='Square image size; SD 1.x is trained at 512.')
    parser.add_argument('--negative', default='', help='Negative prompt.')
    parser.add_argument('--output', default=None, help='Output directory.')
    options = parser.parse_args()

    words = [word.strip() for word in options.words.split(',') if word.strip()]
    output = Path(options.output) if options.output else (
        (Path('/kaggle/working/outputs') if Path('/kaggle/working').exists()
         else Path('./outputs')) / 'sd_baseline')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('No GPU found; running on CPU in float32. This is slow.', flush=True)

    print(f'Loading {options.model} ...', flush=True)
    pipeline = build_pipeline(options.model, device)
    absent = missing_words(pipeline.tokenizer, options.prompt, words)
    if absent:
        print('Warning: these object words are not in the prompt:', absent, flush=True)

    image = generate(pipeline, options.prompt, options.seed, options.steps,
                     options.guidance, options.size, options.negative)
    stem = f'seed_{options.seed}'
    image_path = output / f'{stem}.png'
    image.save(image_path)
    (output / f'{stem}_metadata.json').write_text(json.dumps({
        'backbone': options.model, 'method': 'plain Stable Diffusion, no fuzzy guidance',
        'prompt': options.prompt, 'object_words': words, 'words_not_in_prompt': absent,
        'seed': options.seed, 'negative_prompt': options.negative,
        'settings': {'steps': options.steps, 'guidance_scale': options.guidance,
                     'height': options.size, 'width': options.size,
                     'scheduler': 'DPMSolverMultistep dpmsolver++ karras'},
        'image': str(image_path),
        'versions': {'torch': torch.__version__, 'diffusers': __import__('diffusers').__version__},
    }, indent=2), encoding='utf-8')
    print('Saved:', image_path.resolve(), flush=True)


if __name__ == '__main__':
    main()
