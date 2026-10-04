"""One FuzzyDiff source generation, then CPU text-conditioned terrain segmentation."""
import gc
import platform

import numpy as np


def generate_source(design, root):
    from snow_experiment import make_design
    from snow_gpu import SnowBackend
    base_design = make_design()
    settings = base_design['settings']
    settings.update(height=design['height'], width=design['width'],
                    base_steps=design['base_steps'], guidance=design['guidance'],
                    words_to_track=['mountains'])
    settings['negative_prompt'] += ', ' + design['extra_negative']
    job = {'id': 'clean_mountain_reference', 'method': 'fuzzy', 'seed': design['seed'],
           'prompt': design['prompt'], 'attention': 'attention/source.pt'}
    backend = SnowBackend(base_design)
    try:
        backend.prepare('fuzzy')
        image, metadata = backend.generate(job, root)
        metadata.update(origin='FuzzyDiff bare-rock scene prompt', prompt=design['prompt'],
                        negative_prompt=settings['negative_prompt'], snow_free_verified=False,
                        generation_settings=settings)
        return image, metadata
    finally:
        backend.close()


def estimate_terrain(image, design):
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import binary_closing, label
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    checkpoint = design['mask_model']
    prompts = ['rocky mountain terrain', 'sky', 'snow']
    print('Estimating mountain/sky/snow masks on CPU. Inspect mask_overlay.png afterward.', flush=True)
    processor = CLIPSegProcessor.from_pretrained(checkpoint)
    model = CLIPSegForImageSegmentation.from_pretrained(checkpoint).eval().requires_grad_(False).to('cpu')
    try:
        inputs = processor(text=prompts, images=[image] * len(prompts), padding=True, return_tensors='pt')
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = F.interpolate(logits[:, None], size=(image.height, image.width),
                                          mode='bilinear', align_corners=False).sigmoid()[:, 0].numpy()
        mountains, sky, snow = probabilities
        terrain = (mountains >= design['mask_threshold']) & (mountains > sky)
        terrain = binary_closing(terrain, iterations=1)
        components, count = label(terrain)
        areas = np.bincount(components.ravel())
        if count:
            keep = np.flatnonzero(areas >= max(16, terrain.size // 1000))
            keep = keep[keep != 0]
            terrain = np.isin(components, keep)
        if not terrain.any():
            raise ValueError('CLIPSeg found no usable mountain terrain; inspect source and supply --mask.')
        snow_fraction = float(((snow >= 0.5) & terrain).sum() / terrain.sum())
        metadata = {'origin': 'CLIPSeg CPU estimate', 'checkpoint': checkpoint,
                    'checkpoint_commit': getattr(model.config, '_commit_hash', None),
                    'queries': prompts, 'threshold': design['mask_threshold'],
                    'terrain_fraction_of_image': float(terrain.mean()),
                    'estimated_existing_snow_fraction_of_terrain': snow_fraction,
                    'accuracy_verified': False, 'python': platform.python_version(),
                    'versions': {'torch': str(torch.__version__), 'transformers': __import__('transformers').__version__}}
        if snow_fraction > 0.1:
            print(f'WARNING: estimated existing snow covers {snow_fraction:.1%} of the terrain. '
                  'Check source.png; a genuinely bare-rock --source gives a clearer sweep.', flush=True)
        return terrain, metadata
    finally:
        del model, processor
        gc.collect()
