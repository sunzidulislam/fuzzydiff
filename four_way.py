"""Four-way comparison: {SD 1.5, SDXL} x {no fuzzy, FuzzyDiff}, matched prompt and seed.

Conditions, in the order the ablation reads:

  sd15_plain  stock Stable Diffusion 1.5, no FuzzyDiff at all
  sd15_fuzzy  SD 1.5 with the FuzzyDiff guidance pipeline
  sdxl_plain  the FuzzyDiff pipeline on SDXL with guidance switched off
  sdxl_fuzzy  the FuzzyDiff pipeline on SDXL with guidance on

Base stage only; refinement is deliberately excluded so the comparison isolates fuzzy
guidance. Each backbone runs at its own native resolution, which is standard practice
but means sd15_* and sdxl_* are not pixel-comparable; the two within-backbone pairs are.
"""
import gc
import json
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler

LABELS = {'sd15_plain': 'SD 1.5\n(no fuzzy)', 'sd15_fuzzy': 'SD 1.5\n+ FuzzyDiff',
          'sdxl_plain': 'Ours on SDXL\n(fuzzy off)', 'sdxl_fuzzy': 'Ours on SDXL\n+ FuzzyDiff'}
GRID = (('sd15_plain', 'sd15_fuzzy'), ('sdxl_plain', 'sdxl_fuzzy'))


def clip_scores(namespace, image, prompt, phrases):
    """Full-prompt similarity plus one score per tracked phrase.

    min_part is the Attend-and-Excite style metric: the weakest prompt component is
    what a missing object or an ignored attribute shows up in, where full-prompt
    similarity averages it away.
    """
    model, preprocess, device = namespace['clip_model'], namespace['clip_preprocess'], namespace['clip_device']
    templates = namespace['imagenet_templates']
    with torch.no_grad():
        features = model.encode_image(preprocess(image).unsqueeze(0).to(device))
        features = (features / features.norm(dim=-1, keepdim=True)).float()

        def similarity(text):
            return (features @ namespace['get_embedding_for_prompt'](model, text, templates)).item()

        parts = {phrase: similarity(phrase) for phrase in phrases}
    return {'full': similarity(prompt), 'parts': parts,
            'min_part': min(parts.values()) if parts else None}


def save_grid(directory, images, scores, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(8, 8.8))
    for row, names in enumerate(GRID):
        for column, name in enumerate(names):
            axis = axes[row][column]
            axis.axis('off')
            if name not in images:
                axis.set_title(f'{LABELS[name]}\n(not run)', fontsize=10)
                continue
            axis.imshow(images[name])
            minimum = scores[name]['clip']['min_part']
            axis.set_title(f'{LABELS[name]}\nmin-part CLIP {minimum:.4f}', fontsize=10)
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    path = directory / 'four_way.png'
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def unload(pipeline):
    pipeline.remove_all_hooks()
    pipeline.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()


def run_four_way(namespace, prompt, words, seed=42, steps=50, learning_rate=0.2, updates=30,
                 sd_model='stable-diffusion-v1-5/stable-diffusion-v1-5', negative='',
                 sdxl_guidance=9.5, sd_guidance=7.5, sharpness=100.0, t_norm='min', output=None):
    words = list(words)
    directory = Path(output) if output else (
        (Path('/kaggle/working/outputs') if Path('/kaggle/working').exists()
         else Path('./outputs')) / f'four_way_seed_{seed}')
    directory.mkdir(parents=True, exist_ok=True)
    report = {'prompt': prompt, 'words_to_track': words, 'seed': seed,
              'stage': 'base only, no refinement',
              'settings': {'steps': steps, 'fuzzy_lr': learning_rate, 'guided_steps': updates,
                           'membership_sharpness': sharpness, 't_norm': t_norm,
                           'sdxl_guidance': sdxl_guidance, 'sd_guidance': sd_guidance,
                           'sdxl_size': 768, 'sd_size': 512, 'sd_model': sd_model,
                           'negative_prompt': negative},
              'metric_note': 'min_part is the weakest tracked phrase; phrase_truth is the '
                             'method\'s own objective and is not an independent metric for it.',
              'runs': {}}
    images = {}

    def record(name, image, store=None, truths=None):
        image.save(directory / f'{name}.png')
        images[name] = image
        entry = {'image': str(directory / f'{name}.png'),
                 'clip': clip_scores(namespace, image, prompt, words)}
        if store is not None:
            entry['phrase_truth'] = truths
            entry['guidance_diagnostics'] = list(store.guidance_diagnostics)
        report['runs'][name] = entry
        (directory / 'four_way.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"{name}: full {entry['clip']['full']:.4f}  min-part {entry['clip']['min_part']:.4f}"
              f"  truth {entry.get('phrase_truth')}", flush=True)

    # --- SDXL arms, using the pipeline the notebook already loaded ---
    pipeline = namespace['model']
    original_processors = dict(pipeline.unet.attn_processors)
    try:
        for name, guided in (('sdxl_plain', 0), ('sdxl_fuzzy', updates)):
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            print('Generating', name, flush=True)
            image, _, store = namespace['generate'](
                prompt, words, seed=seed, num_steps=steps, guidance=sdxl_guidance,
                height=768, width=768, max_iter_to_alter=guided, attend_excite_lr=learning_rate,
                negative_prompt=negative, relations=[], membership_sharpness=sharpness,
                t_norm=t_norm, pipeline=pipeline)
            record(name, image, store, namespace['phrase_truth_scores'](store))
    finally:
        pipeline.unet.set_attn_processor(original_processors)
        unload(pipeline)
    # Free SDXL before SD 1.5 comes up; a T4 cannot hold both.
    namespace.pop('model', None)
    namespace.pop('vae', None)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # --- SD 1.5 arms ---
    import run_sd
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fuzzy = run_sd.load_fuzzy_definitions()
    print('Loading', sd_model, flush=True)
    # One checkpoint serves both arms: the inherited __call__ is stock diffusers
    # sampling, while .sample() is the fuzzy loop. Plain runs first, before the
    # custom attention processors are registered.
    sd_pipeline = run_sd.build_pipeline(sd_model, device, fuzzy)
    try:
        absent = run_sd.missing_words(sd_pipeline.tokenizer, prompt, words)
        if absent:
            print('Warning: tracked phrases absent from the prompt:', absent, flush=True)
        report['words_not_in_prompt'] = absent
        print('Generating sd15_plain', flush=True)
        record('sd15_plain', run_sd.generate_plain(sd_pipeline, prompt, seed, steps,
                                                   sd_guidance, 512, negative))
        print('Generating sd15_fuzzy', flush=True)
        cfg, groups = run_sd.build_config(fuzzy, sd_pipeline.tokenizer, prompt, words, seed,
                                          directory, steps, sd_guidance, 512,
                                          learning_rate, updates, sharpness, t_norm)
        image, store = run_sd.generate_fuzzy(sd_pipeline, fuzzy, cfg, groups, negative)
        record('sd15_fuzzy', image, store,
               fuzzy['phrase_truth_scores'](store, cfg.alpha, cfg.membership_sharpness))
    finally:
        unload(sd_pipeline)
        del sd_pipeline
        gc.collect()
        torch.cuda.empty_cache()

    ranking = sorted(report['runs'], key=lambda name: report['runs'][name]['clip']['min_part'],
                     reverse=True)
    report['ranking_by_min_part'] = ranking
    report['within_backbone_delta'] = {
        'sd15': report['runs']['sd15_fuzzy']['clip']['min_part'] - report['runs']['sd15_plain']['clip']['min_part'],
        'sdxl': report['runs']['sdxl_fuzzy']['clip']['min_part'] - report['runs']['sdxl_plain']['clip']['min_part']}
    grid = save_grid(directory, images, report['runs'], prompt)
    report['grid'] = str(grid)
    (directory / 'four_way.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Ranking by min-part CLIP:', ranking, flush=True)
    print('Fuzzy minus no-fuzzy, per backbone:', report['within_backbone_delta'], flush=True)
    print('Grid and metrics in:', directory.resolve(), flush=True)
    return report
