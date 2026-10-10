"""Matched-seed stage-one diagnosis, using models loaded by the notebook."""
import gc
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

NEGATIVE = ('cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, '
            'toy-like, plastic texture, low quality, blurry, distorted, text, watermark')


def run_comparison(namespace, prompt='A very fast car', words=('fast', 'car'), seed=142,
                   learning_rates=(0.2,), sharpness=100.0, t_norm='min', negative=None, bindings=(),
                   binding_weight=1.0, membership_mode='share', output=None):
    pipeline = namespace['model']
    words = list(words)
    negative_prompt = NEGATIVE if negative is None else negative
    settings = dict(prompt=prompt, negative_prompt=negative_prompt, height=768, width=768,
                    num_inference_steps=50, guidance_scale=9.5)
    directory = Path(output) if output else (
        (Path('/kaggle/working/outputs') if Path('/kaggle/working').exists()
         else Path('./outputs')) / f'comparison_seed_{seed}')
    directory.mkdir(parents=True, exist_ok=True)
    report = {'seed': seed, 'settings': settings, 'words_to_track': words,
              'learning_rates': list(learning_rates),
              'membership_sharpness': sharpness, 't_norm': t_norm,
              'membership_mode': membership_mode, 'bindings': [list(pair) for pair in bindings],
              'binding_weight': binding_weight,
              'stage': 'base only, no refinement',
              'precision': {'unet': str(pipeline.unet.dtype),
                            'native_latents': str(pipeline.unet.dtype), 'custom_latents': 'torch.float32'},
              'comparison_note': 'Native/custom also differ in latent precision; custom-off/custom-on isolates guidance.',
              'truth_note': 'phrase_truth is an attention-derived objective, not visual accuracy or attribute intensity.',
              'versions': {'torch': torch.__version__, 'diffusers': __import__('diffusers').__version__},
              'runs': {}}
    original_processors = dict(pipeline.unet.attn_processors)

    def save(label, image, store=None):
        file = directory / f'{label}.png'
        image.save(file)
        scores = namespace['clip_prompt_similarities'](image, prompt)
        truths = (namespace['phrase_truth_scores'](store, 10.0, sharpness)
                  if store is not None else None)
        diagnostics = list(store.guidance_diagnostics) if store is not None else []
        report['runs'][label] = {'image': str(file), 'clip': scores, 'phrase_truth': truths,
                                 'guidance_diagnostics': diagnostics,
                                 'token_groups': store.token_groups if store is not None else None}
        (directory / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(label, 'CLIP:', scores['full_text'], 'truth:', truths, 'saved:', file, flush=True)

    native = None
    try:
        # Share weights rather than load a second SDXL model. Reset offload hooks per owner.
        pipeline.remove_all_hooks()
        native = StableDiffusionXLPipeline(**pipeline.components, add_watermarker=False)
        native.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        native.enable_model_cpu_offload(gpu_id=0)
        native.enable_vae_tiling()
        print('Generating native SDXL baseline...', flush=True)
        image = native(**settings, generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
        save('01_native_sdxl', image)
        native.remove_all_hooks()
        del native
        native = None
        gc.collect()
        torch.cuda.empty_cache()

        pipeline.unet.set_attn_processor(dict(original_processors))
        pipeline.enable_model_cpu_offload(gpu_id=0)
        # Guidance off first: its phrase truths are the baseline each rate is measured against.
        runs = [('02_custom_guidance_off', 0, 0.2)]
        runs += [(f'03_guidance_lr_{rate:g}', 30, rate) for rate in learning_rates]
        for label, updates, rate in runs:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            print('Generating', label, flush=True)
            image, _, store = namespace['generate'](
                prompt, words, seed=seed, num_steps=50, guidance=9.5, height=768, width=768,
                max_iter_to_alter=updates, attend_excite_lr=rate, negative_prompt=negative_prompt,
                relations=[(attribute, 'bound_to', obj) for attribute, obj in bindings],
                membership_sharpness=sharpness, t_norm=t_norm, pipeline=pipeline,
                binding_loss_weight=binding_weight, membership_mode=membership_mode)
            save(label, image, store)
        print('Compare all images in:', directory.resolve(), flush=True)
    finally:
        if native is not None:
            native.remove_all_hooks()
        pipeline.remove_all_hooks()
        pipeline.unet.set_attn_processor(dict(original_processors))
        pipeline.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
