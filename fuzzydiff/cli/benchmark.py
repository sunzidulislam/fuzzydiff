"""Benchmark archived notebook implementations and full FuzzyDiff on one prompt.

No notebook is modified. Use --dry-run before GPU execution. --prepare-cache performs
downloads in separate processes before offline, cached-loading measurements.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

from fuzzydiff.models.registry import LABELS, LIMITATIONS, NOTEBOOKS


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--words', required=True, help='Tracked phrases appearing verbatim in prompt.')
    parser.add_argument('--methods', default=','.join(NOTEBOOKS))
    parser.add_argument('--seeds', default='1024,999,42,142')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--negative', default='', help='Default empty avoids inconsistent archived negative handling.')
    parser.add_argument('--lr', type=float, default=5.0)
    parser.add_argument('--sharpness', type=float, default=20.0)
    parser.add_argument('--tnorm', choices=['min', 'product'], default='product')
    parser.add_argument('--membership-mode', choices=['share', 'relative'], default='share')
    parser.add_argument('--updates', type=int, default=30)
    parser.add_argument('--predicate-strength', type=float, default=0.5)
    parser.add_argument('--refine-steps', type=int, default=30)
    parser.add_argument('--refine-strength', type=float, default=0.4)
    parser.add_argument('--bind', action='append', default=[])
    parser.add_argument('--output', default='/kaggle/working/outputs/prompt_benchmark')
    parser.add_argument('--prepare-cache', action='store_true', help='Populate model caches before measurement.')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print plan; no imports of Torch or downloads.')
    options = vars(parser.parse_args(argv))
    try:
        options['words'] = [w.strip() for w in options['words'].split(',') if w.strip()]
        options['methods'] = [m.strip() for m in options['methods'].split(',') if m.strip()]
        options['seeds'] = [int(s.strip()) for s in options['seeds'].split(',')]
        if not options['words'] or any(w not in options['prompt'] for w in options['words']):
            raise ValueError('Every --words phrase must appear verbatim in --prompt.')
        if not options['methods'] or set(options['methods']) - set(NOTEBOOKS):
            raise ValueError('Unknown method; choose from ' + ','.join(NOTEBOOKS))
        if len(set(options['methods'])) != len(options['methods']):
            raise ValueError('Methods must be unique.')
        if len(set(options['seeds'])) != len(options['seeds']) or any(
                not 0 <= s < 2**63 for s in options['seeds']):
            raise ValueError('Seeds must be unique nonnegative integers below 2**63.')
        if options['repeats'] < 1 or options['warmup'] < 1:
            raise ValueError('--repeats and --warmup must be at least 1.')
        if options['size'] < 128 or options['size'] % 8:
            raise ValueError('--size must be at least 128 and divisible by 8.')
        if set(options['methods']) & {'attend', 'predicate'} and options['size'] not in (
                128, 256, 512, 1024):
            raise ValueError('Archived attend/predicate need 16x16 attention maps; '
                             'use --size 128, 256, 512 or 1024 (512 recommended).')
        if options['steps'] < 4 or options['refine_steps'] < 1 or options['updates'] < 0:
            raise ValueError('Use at least 4 steps, positive refine steps, and nonnegative updates.')
        for key in ('guidance', 'lr', 'sharpness', 'predicate_strength', 'refine_strength'):
            if not math.isfinite(options[key]) or options[key] < 0:
                raise ValueError(f'--{key.replace("_", "-")} must be finite and nonnegative.')
        if not 0 < options['refine_strength'] <= 1 or int(
                options['refine_steps'] * options['refine_strength']) < 1:
            raise ValueError('Refinement must produce at least one denoising step.')
        if options['guidance'] <= 1 and any(m != 'enhancing' for m in options['methods']):
            raise ValueError('Archived predicate implementation requires CFG > 1.')
        options['bindings'] = []
        for binding in options.pop('bind'):
            a, sep, b = binding.partition('>')
            a, b = a.strip(), b.strip()
            if not sep or a not in options['words'] or b not in options['words']:
                raise ValueError('--bind operands must both appear in --words.')
            options['bindings'].append([a, b])
        for method in options['methods']:
            if not NOTEBOOKS[method].is_file():
                raise ValueError(f'Missing notebook: {NOTEBOOKS[method]}')
    except ValueError as error:
        parser.error(str(error))
    options['output'] = str(Path(options['output']).resolve())
    return options


def main(argv=None):
    from fuzzydiff.experiments.benchmark import main as run, run_worker
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--worker':
        if len(argv) != 4 or argv[3] not in ('prepare', 'measure'):
            raise ValueError('--worker expects METHOD CONFIG prepare|measure.')
        settings = json.loads(Path(argv[2]).read_text(encoding='utf-8'))
        settings['preparing'] = argv[3] == 'prepare'
        return run_worker(argv[1], settings)
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
