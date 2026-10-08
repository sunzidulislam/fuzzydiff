"""Execute the canonical notebook after installing requirements.txt."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Use 768px / 50 steps instead of the smoke test.')
    parser.add_argument('--check', action='store_true', help='Compile every code cell without loading models.')
    parser.add_argument('--compare', action='store_true', help='Compare native SDXL, guidance-off and guidance-on; no refinement.')
    parser.add_argument('--prompt', default='A very fast car', help='Comparison prompt.')
    parser.add_argument('--words', default='fast,car', help='Comma-separated tracked phrases for --compare.')
    parser.add_argument('--seed', type=int, default=142, help='Comparison seed.')
    parser.add_argument('--lr', default='0.2',
                        help='Comma-separated guidance step sizes; --compare renders one image per value.')
    options = parser.parse_args()
    path = Path(__file__).parent / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__', 'FULL_RUN': options.full}
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
        if not options.check and not (options.compare and index == len(notebook['cells']) - 1):
            print(f'Running notebook cell {index}', flush=True)
            exec(compiled, namespace)
    if options.check:
        print(f'Compiled {count} code cells successfully.')
    elif options.compare:
        from compare_fullpipeline import run_comparison
        run_comparison(namespace, prompt=options.prompt,
                       words=[word.strip() for word in options.words.split(',') if word.strip()],
                       seed=options.seed,
                       learning_rates=[float(rate) for rate in options.lr.split(',') if rate.strip()])


if __name__ == '__main__':
    main()
