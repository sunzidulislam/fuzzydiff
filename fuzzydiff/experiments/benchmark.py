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


from fuzzydiff.reporting.tables import write_json, append_record, summarize, save_reports
from fuzzydiff.paths import repository_root
from fuzzydiff.notebook_runtime import source_hashes


def parse_options(argv=None):
    from fuzzydiff.cli.benchmark import parse_options as parse
    return parse(argv)


def run_worker(method, options):
    # Imports happen only in isolated workers, after cache policy is configured.
    import gc
    import importlib.metadata
    import torch
    from contextlib import ExitStack
    from fuzzydiff.models.registry import NotebookModel
    from fuzzydiff.reporting.timing import TimeLedger, instrument_pipelines

    directory = Path(options['output']) / method
    directory.mkdir(exist_ok=True)
    rows, metadata = [], {}
    model = None
    phase = 'load'
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('This benchmark requires a GPU. Enable a Kaggle GPU accelerator.')
        torch.cuda.set_device(0)
        torch.cuda.init()
        metadata['environment'] = {
            'python': sys.version, 'platform': platform.platform(),
            'gpu': torch.cuda.get_device_name(0),
            'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'cuda': torch.version.cuda, 'cpu': platform.processor(), 'cpu_threads': os.cpu_count(),
            'torch_threads': torch.get_num_threads(),
            'versions': {p: importlib.metadata.version(p)
                         for p in ('torch', 'diffusers', 'transformers', 'accelerate')},
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        model = NotebookModel(method, options)
        metadata.update(model.metadata())
        write_json(directory / ('preparation.json' if options.get('preparing') else 'model.json'), metadata)
        if options.get('preparing'):
            return 0
        phase = 'warmup'
        for i in range(options['warmup']):
            model.reset()
            image = model.generate(options['seeds'][0])
            del image
        # Model-offload reset also reinstalls hooks and may empty the device cache.
        model.reset()
        gc.collect()
        torch.cuda.empty_cache()
        phase = 'sampling'
        for seed in options['seeds']:
            for repeat in range(options['repeats']):
                row = {'method': method, 'seed': seed, 'repeat': repeat, 'status': 'error'}
                try:
                    model.reset()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = perf_counter()
                    image = model.generate(seed)
                    torch.cuda.synchronize()
                    row.update(inference_s=perf_counter() - start,
                               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                               status='ok', details=model.last_details)
                    path = directory / f'seed_{seed}_repeat_{repeat}.png'
                    start = perf_counter()
                    image.save(path)
                    row.update(image=str(path), save_s=perf_counter() - start)
                    # Profiling is a separate inference, never part of the native timer.
                    model.reset()
                    ledger = TimeLedger(synchronize=torch.cuda.synchronize)
                    counts = {'unet_forward_calls': 0}

                    def count_forward(*args):
                        counts['unet_forward_calls'] += 1

                    try:
                        with ExitStack() as stack:
                            stack.enter_context(instrument_pipelines(model.pipelines, ledger))
                            stack.enter_context(model.profile_preprocessing(ledger))
                            for pipeline in model.pipelines:
                                handle = pipeline.unet.register_forward_pre_hook(count_forward)
                                stack.callback(handle.remove)
                            with ledger.measure('generation'):
                                profiled_image = model.generate(seed, ledger)
                        seconds = dict(ledger.seconds)
                        row['profile'] = {
                            'preprocess_s': seconds.get('preprocess', 0),
                            'tokenize_s': seconds.get('tokenize', 0),
                            'text_encode_s': seconds.get('text_encode', 0),
                            'generation_s': seconds.get('generation', 0) + seconds.get('vae', 0),
                            'profile_total_s': sum(seconds.values()),
                            'exclusive_seconds': seconds, 'calls': dict(ledger.calls), **counts,
                            'same_pixels_as_native': image.tobytes() == profiled_image.tobytes(),
                        }
                        row['profile_status'] = 'ok'
                        del profiled_image
                    except Exception as error:
                        row.update(profile_status='error', profile_error=f'{type(error).__name__}: {error}')
                        traceback.print_exc()
                    del image
                except Exception as error:
                    row.update(status='error', error=f'{type(error).__name__}: {error}')
                    traceback.print_exc()
                row = append_record(rows, row, directory / 'runs.json')
                gc.collect()
                print(f'{method} seed={seed} repeat={repeat}: {row["status"]}', flush=True)
        return int(any(r['status'] != 'ok' or r.get('profile_status') != 'ok' for r in rows))
    except Exception as error:
        traceback.print_exc()
        row = {'method': method, 'status': 'error', 'error': f'{type(error).__name__}: {error}',
               'phase': 'preparation' if options.get('preparing') else phase}
        append_record(rows, row, directory / (
            'preparation_error.json' if options.get('preparing') else 'runs.json'))
        return 1
    finally:
        if model is not None:
            model.close()


def launch_worker(method, config_path, directory, preparing=False):
    environment = dict(os.environ)
    environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    # Keep measured loading offline. A preparation process may populate the cache.
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        environment[key] = '0' if preparing else '1'
    directory = Path(directory).resolve()
    log = directory / method / ('prepare.log' if preparing else 'benchmark.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        completed = subprocess.run(
            [sys.executable, '-m', 'fuzzydiff.cli.benchmark', '--worker', method,
             str(Path(config_path).resolve()), 'prepare' if preparing else 'measure'],
            stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=repository_root())
    return completed.returncode


def main(argv=None):
    options = parse_options(argv)
    from fuzzydiff.models.registry import definition_code
    for method in options['methods']:
        definition_code(NOTEBOOKS[method])
    print('Whole-model comparison:', ', '.join(LABELS[m] for m in options['methods']), flush=True)
    print('Prompt:', options['prompt'], flush=True)
    print(f"{len(options['seeds'])} seeds x {options['repeats']} repeats, "
          'plus warmup and one separate profiling pass per measured image.', flush=True)
    for method in options['methods']:
        for note in LIMITATIONS[method]:
            print(f'[{method}] {note}', flush=True)
    if options['dry_run']:
        print('Dry run passed; no model loaded, no files written.')
        return 0
    directory = Path(options['output'])
    # Never silently mix or overwrite different experimental runs.
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Output directory is not empty; select a new --output directory.')
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.json'
    write_json(config_path, options)
    root = repository_root()
    files = list(NOTEBOOKS.values()) + list((root / 'fuzzydiff').rglob('*.py'))
    write_json(directory / 'source_hashes.json', source_hashes(files))
    records, metadata, statuses = [], {}, {}
    for method in options['methods']:
        if options['prepare_cache']:
            print(f'Preparing cached checkpoints: {method} (see {method}/prepare.log)', flush=True)
            started = perf_counter()
            code = launch_worker(method, config_path, directory, preparing=True)
            statuses[method + '_preparation'] = {'exit_code': code,
                                                'elapsed_s': perf_counter() - started}
            # Measurement still runs: it records its own cache/load error if preparation failed.
        print(f'Benchmarking {method} (see {method}/benchmark.log)', flush=True)
        code = launch_worker(method, config_path, directory)
        statuses[method] = {'exit_code': code}
        run_path = directory / method / 'runs.json'
        method_rows = json.loads(run_path.read_text()) if run_path.exists() else []
        expected = len(options['seeds']) * options['repeats']
        if len(method_rows) < expected and not any(r['status'] == 'error' for r in method_rows):
            method_rows.append({'method': method, 'status': 'error',
                                'error': 'Worker did not complete all requested samples; inspect log.'})
        records.extend(method_rows)
        meta_path = directory / method / 'model.json'
        if meta_path.exists():
            metadata[method] = json.loads(meta_path.read_text())
        save_reports(directory, records, metadata)
        write_json(directory / 'worker_status.json', statuses)
        print(f'{method}: exit {code}; reports saved.', flush=True)
    print('Report:', directory / 'comparison.md', flush=True)
    return int(any(v['exit_code'] for v in statuses.values()))


