"""Stable Diffusion 1.x generation, with or without the FuzzyDiff guidance pipeline.

Stability never released open weights for Stable Diffusion 1.6; it exists only behind
their hosted API. The open 1.x line stops at 1.5, which is also the backbone the
Attend-and-Excite and Predicated Diffusion baselines use, so that is the default.
Pass --model to point at any other SD 1.x checkpoint, local path or Hub id.

Without --fuzzy this is the plain backbone. With --fuzzy the same fuzzy logic the SDXL
notebook uses is loaded from that notebook rather than reimplemented, so the two
backbones cannot drift apart; only the sampling loop differs, because SD 1.x conditions
on one text encoder and needs no added time/text embeddings.

The scheduler matches the FuzzyDiff notebook (DPMSolver++ with Karras sigmas) so images
are comparable across backbones at the same seed and step count.

    python run_sd.py --prompt "a red book and a yellow clock" --words "red book,yellow clock"
    python run_sd.py --prompt "a red book and a yellow clock" --words "red book,yellow clock" --fuzzy
    python run_sd.py --prompt "a red book and a yellow clock" --words "red,book,yellow,clock" \
        --fuzzy --bind "red>book" --bind "yellow>clock"
"""
import argparse
import json
from pathlib import Path

DEFAULT_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
from run_fullpipeline import RELATION_VERBS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt', required=True, help='Text prompt.')
    parser.add_argument('--words', default='',
                        help='Comma-separated object words/phrases, each appearing verbatim in the prompt.')
    parser.add_argument('--fuzzy', action='store_true', help='Apply FuzzyDiff guidance; requires --words.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='SD 1.x checkpoint (local path or Hub id).')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.5, help='Classifier-free guidance scale.')
    parser.add_argument('--size', type=int, default=512, help='Square image size; SD 1.x is trained at 512.')
    parser.add_argument('--lr', type=float, default=0.2, help='Fuzzy guidance step size.')
    parser.add_argument('--updates', type=int, default=25, help='Number of guided denoising steps.')
    parser.add_argument('--sharpness', type=float, default=100.0,
                        help='Membership sharpness; lower keeps truths graded instead of near-binary.')
    parser.add_argument('--membership-mode', choices=('share', 'relative'), default='share',
                        help='Experimental relative mode normalizes each phrase by its own spatial peak.')
    parser.add_argument('--hedge-guidance', action='store_true',
                        help='Experimental bound-attribute attention targets; requires share membership.')
    parser.add_argument('--tnorm', choices=('min', 'product'), default='min',
                        help="Fuzzy conjunction: 'min' gradients only the weakest phrase, 'product' all of them.")
    parser.add_argument('--bind', action='append', metavar='ATTR>OBJECT',
                        help='Require an attribute to hold where an object is, e.g. '
                             '--bind "yellow>clock". Both sides must be tracked phrases.')
    parser.add_argument('--binding-weight', type=float, default=1.0,
                        help='Weight of each binding conjunct in the loss.')
    parser.add_argument('--relate', action='append', metavar='PHRASE:VERB:PHRASE',
                        help='Add a relation between two tracked phrases, e.g. '
                             '--relate "green apple:larger_than:red apple". Verbs: '
                             + ', '.join(RELATION_VERBS) + '.')
    parser.add_argument('--negative', default='', help='Negative prompt.')
    parser.add_argument('--output', default=None, help='Output directory.')
    options = parser.parse_args(argv)
    if options.hedge_guidance and (not options.fuzzy or options.membership_mode != 'share'):
        parser.error('--hedge-guidance requires --fuzzy and --membership-mode share.')

    from fuzzydiff.models.sd15 import (
        torch, asdict, load_fuzzy_definitions, build_pipeline, missing_words,
        build_config, resolve_relations, parse_relations, parse_bindings,
        generate_fuzzy, generate_plain)

    words = [word.strip() for word in options.words.split(',') if word.strip()]
    if options.fuzzy and not words:
        parser.error('--fuzzy needs --words: there is nothing to ground without tracked phrases.')
    output = Path(options.output) if options.output else (
        (Path('/kaggle/working/outputs') if Path('/kaggle/working').exists()
         else Path('./outputs')) / ('sd_fuzzy' if options.fuzzy else 'sd_baseline'))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('No GPU found; running on CPU in float32. This is slow.', flush=True)

    fuzzy = load_fuzzy_definitions() if options.fuzzy else None
    print(f'Loading {options.model} ...', flush=True)
    pipeline = build_pipeline(options.model, device, fuzzy)
    absent = missing_words(pipeline.tokenizer, options.prompt, words)
    if absent:
        print('Warning: these object words are not in the prompt:', absent, flush=True)

    truths, diagnostics, settings = None, [], {}
    if options.fuzzy:
        cfg, groups = build_config(fuzzy, pipeline.tokenizer, options.prompt, words, options.seed,
                                   output, options.steps, options.guidance, options.size,
                                   options.lr, options.updates, options.sharpness, options.tnorm,
                                   options.membership_mode, options.hedge_guidance)
        cfg.binding_loss_weight = options.binding_weight
        relations = resolve_relations(groups, parse_relations(options.relate)
                                      + [(a, 'bound_to', b) for a, b in parse_bindings(options.bind)])
        print('Tracked phrase tokens:', groups, flush=True)
        if relations:
            print('Bindings:', parse_bindings(options.bind), flush=True)
        image, store = generate_fuzzy(pipeline, fuzzy, cfg, groups, options.negative, relations)
        truths = fuzzy['phrase_truth_scores'](store, cfg.alpha, cfg.membership_sharpness)
        diagnostics = store.guidance_diagnostics
        settings = {**asdict(cfg), 'output_path': str(cfg.output_path)}
    else:
        image = generate_plain(pipeline, options.prompt, options.seed, options.steps,
                               options.guidance, options.size, options.negative)
        settings = {'steps': options.steps, 'guidance_scale': options.guidance,
                    'height': options.size, 'width': options.size}

    stem = f'seed_{options.seed}'
    image_path = output / f'{stem}.png'
    image.save(image_path)
    (output / f'{stem}_metadata.json').write_text(json.dumps({
        'backbone': options.model,
        'method': 'FuzzyDiff guidance' if options.fuzzy else 'plain Stable Diffusion, no fuzzy guidance',
        'prompt': options.prompt, 'object_words': words, 'words_not_in_prompt': absent,
        'seed': options.seed, 'negative_prompt': options.negative,
        'scheduler': 'DPMSolverMultistep dpmsolver++ karras', 'settings': settings,
        'hedge_guidance_applied': bool(settings.get('hedge_targets') and diagnostics),
        'phrase_truth': truths, 'guidance_diagnostics': diagnostics, 'image': str(image_path),
        'truth_note': 'Attention-derived objective, not visual accuracy or attribute intensity.',
        'versions': {'torch': torch.__version__, 'diffusers': __import__('diffusers').__version__},
    }, indent=2), encoding='utf-8')
    if truths:
        print('Phrase presence truth:', truths, flush=True)
    if diagnostics:
        print('Guidance:', fuzzy['summarize_guidance'](diagnostics), flush=True)
    print('Saved:', image_path.resolve(), flush=True)


if __name__ == '__main__':
    main()
