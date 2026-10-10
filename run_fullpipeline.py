"""Execute the canonical notebook after installing requirements.txt."""
import argparse
import importlib
import json
from pathlib import Path

# Shared by --full, --compare and --four-way so the three paths agree; mirrored in
# compare_fullpipeline.NEGATIVE, which a test asserts stays in step.
DEFAULT_NEGATIVE = ('cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, '
                    'toy-like, plastic texture, low quality, blurry, distorted, text, watermark')

# Mirror of run_sd.RELATION_VERBS, duplicated so --check stays free of the torch
# import chain. tests/test_sd_fuzzy.py asserts the two lists stay in step.
RELATION_VERBS = ('left_of', 'right_of', 'above', 'below', 'bound_to',
                  'larger_than', 'smaller_than')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Use 768px / 50 steps instead of the smoke test.')
    parser.add_argument('--check', action='store_true', help='Compile every code cell without loading models.')
    parser.add_argument('--compare', action='store_true', help='Compare native SDXL, guidance-off and guidance-on; no refinement.')
    parser.add_argument('--four-way', action='store_true',
                        help='SD 1.5 and SDXL, each with and without fuzzy guidance; no refinement.')
    parser.add_argument('--prompt', default='A very fast car', help='Generation prompt.')
    parser.add_argument('--words', default='fast,car', help='Comma-separated tracked phrases.')
    parser.add_argument('--seed', type=int, default=142, help='Generation seed.')
    parser.add_argument('--output', help='Output directory; use a new directory for each experiment.')
    parser.add_argument('--membership-mode', choices=('share', 'relative'), default='share',
                        help='Experimental relative mode normalizes each phrase by its own spatial peak.')
    parser.add_argument('--sharpness', type=float, default=100.0,
                        help='Membership sharpness; lower keeps truths graded instead of near-binary.')
    parser.add_argument('--tnorm', choices=('min', 'product'), default='min',
                        help="Fuzzy conjunction: 'min' gradients only the weakest phrase, 'product' all of them.")
    parser.add_argument('--bind', action='append', metavar='ATTR>OBJECT',
                        help='Require an attribute to hold where an object is, e.g. '
                             '--bind "yellow>clock". Both sides must be tracked phrases.')
    parser.add_argument('--binding-weight', type=float, default=1.0,
                        help='Weight of each binding conjunct in the loss.')
    parser.add_argument('--refine', action='store_true',
                        help='--four-way only: add sdxl_full, the complete method with refinement.')
    parser.add_argument('--negative', default=DEFAULT_NEGATIVE,
                        help='Negative prompt; pass an empty string to disable.')
    parser.add_argument('--relate', action='append', metavar='PHRASE:VERB:PHRASE',
                        help='Add a relation between two tracked phrases, e.g. '
                             '--relate "green apple:larger_than:red apple".')
    parser.add_argument('--lr', default='0.2',
                        help='Comma-separated guidance step sizes; --compare renders one image per value.')
    options = parser.parse_args()
    words = [word.strip() for word in options.words.split(',') if word.strip()]
    try:
        rates = [float(rate) for rate in options.lr.split(',') if rate.strip()]
        if not rates or any(not 0 <= rate < float('inf') for rate in rates):
            raise ValueError('--lr must contain finite, nonnegative values.')
        if not options.compare and len(rates) != 1:
            raise ValueError('Multiple --lr values require --compare.')
        bindings = []
        for value in options.bind or ():
            attribute, separator, obj = value.partition('>')
            if not separator or not attribute.strip() or not obj.strip():
                raise ValueError('--bind expects "attribute>object".')
            pair = (attribute.strip(), obj.strip())
            if any(phrase not in words for phrase in pair):
                raise ValueError('Both binding operands must appear in --words.')
            bindings.append(pair)
        relations = []
        for value in options.relate or ():
            parts = [part.strip() for part in value.split(':')]
            if len(parts) != 3 or not all(parts):
                raise ValueError('--relate expects "phrase:verb:phrase".')
            subject, verb, obj = parts
            if verb not in RELATION_VERBS:
                raise ValueError(f'Unsupported relation {verb!r}; use one of {", ".join(RELATION_VERBS)}.')
            if any(phrase not in words for phrase in (subject, obj)):
                raise ValueError('Both relation operands must appear in --words.')
            relations.append((subject, verb, obj))
    except ValueError as error:
        parser.error(str(error))
    path = Path(__file__).parent / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__', 'FULL_RUN': options.full,
                 'RUN_OPTIONS': dict(prompt=options.prompt, words=words, seed=options.seed,
                                     attend_excite_lr=rates[0], membership_sharpness=options.sharpness,
                                     t_norm=options.tnorm, membership_mode=options.membership_mode,
                                     relations=[(a, 'bound_to', b) for a, b in bindings] + relations,
                                     binding_loss_weight=options.binding_weight,
                                     negative_prompt=options.negative, output=options.output)}
    count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        # Dependencies are installed once by the caller, not inside Python exec.
        source = '\n'.join(line for line in source.splitlines()
                           if not line.lstrip().startswith(('%pip ', '!pip ')))
        compiled = compile(source, f'{path}:cell-{index}', 'exec')
        count += 1
        last_cell = index == len(notebook['cells']) - 1
        if not options.check and not ((options.compare or options.four_way) and last_cell):
            print(f'Running notebook cell {index}', flush=True)
            exec(compiled, namespace)
    if options.check:
        print(f'Compiled {count} code cells successfully.')
    elif options.four_way:
        import four_way
        import run_sd
        importlib.reload(run_sd)
        importlib.reload(four_way)
        four_way.run_four_way(namespace, prompt=options.prompt,
                              words=words,
                              seed=options.seed,
                              learning_rate=rates[0],
                              sharpness=options.sharpness, t_norm=options.tnorm,
                              bindings=bindings, binding_weight=options.binding_weight,
                              relations=relations, negative=options.negative,
                              refine=options.refine,
                              membership_mode=options.membership_mode, output=options.output)
    elif options.compare:
        # %run reuses the kernel, so a module imported before a git pull would be
        # served from sys.modules; reload so the file on disk is what runs.
        import compare_fullpipeline
        importlib.reload(compare_fullpipeline)
        run_comparison = compare_fullpipeline.run_comparison
        run_comparison(namespace, prompt=options.prompt,
                       words=words,
                       seed=options.seed,
                       learning_rates=rates, sharpness=options.sharpness, t_norm=options.tnorm,
                       bindings=bindings, binding_weight=options.binding_weight,
                       membership_mode=options.membership_mode, output=options.output)


if __name__ == '__main__':
    main()
