"""Generate one final image with a registered model, optionally save or display it."""
import argparse
from pathlib import Path

from fuzzydiff.config import load_request_config, merge_request_config
from fuzzydiff.generation.runner import generate_image, save_result


def display_image(image):
    from IPython.display import display
    display(image)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    for name in ('model', 'prompt', 'negative', 'words', 'config', 'output'):
        parser.add_argument('--' + name)
    for name in ('seed', 'size', 'steps', 'updates', 'refine-steps'):
        parser.add_argument('--' + name, type=int)
    for name in ('guidance', 'lr', 'sharpness', 'binding-weight', 'predicate-strength', 'refine-strength'):
        parser.add_argument('--' + name, type=float)
    parser.add_argument('--tnorm', choices=('min', 'product'))
    parser.add_argument('--membership-mode', choices=('share', 'relative'))
    parser.add_argument('--hedge-guidance', action='store_true')
    parser.add_argument('--bind', action='append', metavar='ATTR>OBJECT')
    parser.add_argument('--relate', action='append', metavar='PHRASE:VERB:PHRASE')
    parser.add_argument('--display', action='store_true')
    flags = vars(parser.parse_args(argv))
    output = flags.pop('output', None)
    display = flags.pop('display', False)
    config_path = flags.pop('config', None)
    try:
        config = load_request_config(Path(config_path)) if config_path else {}
        if 'words' in flags:
            flags['words'] = [w.strip() for w in flags['words'].split(',') if w.strip()]
        method_options = {}
        for name in ('lr', 'sharpness', 'binding_weight', 'predicate_strength', 'refine_strength',
                     'updates', 'refine_steps', 'tnorm', 'membership_mode', 'hedge_guidance'):
            if name in flags:
                method_options[name] = flags.pop(name)
        if 'bind' in flags:
            pairs = [value.split('>') for value in flags.pop('bind')]
            method_options['bindings'] = [[part.strip() for part in pair] for pair in pairs]
        if 'relate' in flags:
            method_options['relations'] = [[part.strip() for part in value.split(':')]
                                           for value in flags.pop('relate')]
        flags['method_options'] = method_options
        request = merge_request_config(config, flags)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    result = generate_image(request)
    if output:
        files = save_result(result, Path(output), request.seed)
        print('Saved:', files['image'].resolve())
    if display:
        display_image(result.image)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
