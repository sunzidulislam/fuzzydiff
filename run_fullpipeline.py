"""Execute the canonical notebook after installing requirements.txt."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Use 768px / 50 steps instead of the smoke test.')
    parser.add_argument('--check', action='store_true', help='Compile every code cell without loading models.')
    parser.add_argument('--compare', action='store_true', help='Compare native SDXL, guidance-off and guidance-on; no refinement.')
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
        run_comparison(namespace)


if __name__ == '__main__':
    main()
