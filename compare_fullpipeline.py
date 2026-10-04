"""Matched-seed stage-one diagnosis, using models loaded by the notebook."""
import gc
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def run_comparison(namespace):
    pipeline = namespace['model']
    prompt = 'A very fast car'
    words = ['fast', 'car']
    negative_prompt = ('cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, '
                       'toy-like, plastic texture, low quality, blurry, distorted, text, watermark')
    settings = dict(prompt=prompt, negative_prompt=negative_prompt, height=768, width=768,
                    num_inference_steps=50, guidance_scale=9.5)
    seed = 142
    directory = (Path('/kaggle/working/outputs') if Path('/kaggle/working').exists()
                 else Path('./outputs')) / f'comparison_seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    report = {'seed': seed, 'settings': settings, 'words_to_track': words,
              'stage': 'base only, no refinement',
              'precision': {'unet': str(pipeline.unet.dtype),
                            'native_latents': str(pipeline.unet.dtype), 'custom_latents': 'torch.float32'},
              'comparison_note': 'Native/custom also differ in latent precision; custom-off/custom-on isolates guidance.',
              'versions': {'torch': torch.__version__, 'diffusers': __import__('diffusers').__version__},
              'runs': {}}
    original_processors = dict(pipeline.unet.attn_processors)

    def save(label, image, diagnostics=(), token_groups=None):
        file = directory / f'{label}.png'
        image.save(file)
        scores = namespace['clip_prompt_similarities'](image, prompt)
        report['runs'][label] = {'image': str(file), 'clip': scores,
                                 'guidance_diagnostics': list(diagnostics), 'token_groups': token_groups}
        (directory / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(label, 'CLIP:', scores['full_text'], 'saved:', file, flush=True)

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
        for label, updates in [('02_custom_guidance_off', 0), ('03_raw_gradient_guidance', 30)]:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            print('Generating', label, flush=True)
            image, _, store = namespace['generate'](
                prompt, words, seed=seed, num_steps=50, guidance=9.5, height=768, width=768,
                max_iter_to_alter=updates, attend_excite_lr=0.2, negative_prompt=negative_prompt,
                relations=[], pipeline=pipeline)
            save(label, image, store.guidance_diagnostics, store.token_groups)
        print('Compare all three images in:', directory.resolve(), flush=True)
    finally:
        if native is not None:
            native.remove_all_hooks()
        pipeline.remove_all_hooks()
        pipeline.unet.set_attn_processor(dict(original_processors))
        pipeline.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
