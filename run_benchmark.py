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

from benchmark_models import LABELS, LIMITATIONS, NOTEBOOKS


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


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def append_record(rows, row, path):
    try:
        json.dumps(row, allow_nan=False)
    except (ValueError, TypeError) as error:
        # A numerical failure in an archived algorithm must not erase earlier runs.
        row = {**{key: row[key] for key in ('method', 'seed', 'repeat', 'image') if key in row},
               'status': 'error', 'phase': 'recording_diagnostics',
               'error': f'Non-JSON or non-finite diagnostic: {error}'}
    rows.append(row)
    write_json(path, rows)
    return row


def summarize(records):
    summaries = {}
    for method in dict.fromkeys(row['method'] for row in records):
        rows = [row for row in records if row['method'] == method]
        valid = [r for r in rows if r['status'] == 'ok']
        values = [r['inference_s'] for r in valid]
        entry = {'successful_runs': len(valid), 'failed_runs': len(rows) - len(valid),
                 'inference_mean_s': statistics.mean(values) if values else None,
                 'inference_std_s': statistics.stdev(values) if len(values) > 1 else None,
                 'inference_median_s': statistics.median(values) if values else None}
        for name in ('peak_allocated_gib', 'peak_reserved_gib'):
            measurements = [r[name] for r in valid if name in r]
            entry[name] = max(measurements) if measurements else None
        profiles = [r['profile'] for r in valid if r.get('profile_status') == 'ok']
        entry['profiled_runs'] = len(profiles)
        for name in ('preprocess_s', 'tokenize_s', 'text_encode_s',
                     'generation_s', 'profile_total_s'):
            measurements = [r[name] for r in profiles]
            entry[name + '_mean'] = statistics.mean(measurements) if measurements else None
        summaries[method] = entry
    return summaries


def save_reports(directory, records, metadata):
    summary = summarize(records)
    write_json(directory / 'results.json', {'methods': metadata, 'runs': records, 'summary': summary})
    flat = []
    for row in records:
        flat.append({**{k: v for k, v in row.items() if not isinstance(v, (dict, list))},
                     **{f'profile_{k}': v for k, v in row.get('profile', {}).items()
                        if not isinstance(v, (dict, list))}})
    if flat:
        with (directory / 'timings.csv').open('w', encoding='utf-8', newline='') as stream:
            fields = list(dict.fromkeys(k for row in flat for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
    lines = ['# Whole-model timing comparison', '',
             'Exploratory implementations; see limitations below. No quality ranking is inferred.', '',
             'Loading is one offline cached-checkpoint measurement per method. '
             'Preprocess/tokenize/encode/generate are exclusive components from a separate profiled pass. '
             'Their sum is profiled time, NOT the uninstrumented total. '
             'Generation includes both base and refinement where applicable. '
             'Enhancing uses native SDXL-Turbo settings (4 steps, CFG=0); '
             'its speed is not a matched-step comparison.', '',
             '| Method | Load s | Prep ms* | Token ms* | Encode ms* | Generate s* | Whole inference s, mean ± SD | Median s | Peak allocated / reserved GiB | Inference OK / failed | Profiles OK |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']

    def fmt(value, scale=1):
        return 'N/A' if value is None else f'{value * scale:.3f}'

    for method, row in summary.items():
        load_s = metadata.get(method, {}).get('load_s')
        lines.append(f"| {LABELS[method]} | {fmt(load_s)} | "
                     f"{fmt(row['preprocess_s_mean'], 1000)} | {fmt(row['tokenize_s_mean'], 1000)} | "
                     f"{fmt(row['text_encode_s_mean'], 1000)} | {fmt(row['generation_s_mean'])} | "
                     f"{fmt(row['inference_mean_s'])} ± {fmt(row['inference_std_s'])} | "
                     f"{fmt(row['inference_median_s'])} | {fmt(row['peak_allocated_gib'])} / "
                     f"{fmt(row['peak_reserved_gib'])} | {row['successful_runs']} / {row['failed_runs']} | "
                     f"{row['profiled_runs']} |")
    lines += ['', '*Instrumented pass; synchronization overhead can perturb timing. '
              'The whole-inference column is the primary latency measurement. '
              'Stage timings are not a second performance ranking.', '',
              'Four seeds on one prompt are a pilot, not a general quality or speed claim. '
              'Repeated images from one seed are timing repetitions, not independent quality samples.', '',
              '## Implementation limitations', '']
    for method in summary:
        for note in LIMITATIONS[method]:
            lines.append(f'- {LABELS[method]}: {note}')
    (directory / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_worker(method, options):
    # Imports happen only in isolated workers, after cache policy is configured.
    import gc
    import importlib.metadata
    import torch
    from contextlib import ExitStack
    from benchmark_models import NotebookModel
    from benchmark_timing import TimeLedger, instrument_pipelines

    directory = Path(options['output']) / method
    directory.mkdir(exist_ok=True)
    rows, metadata = [], {}
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


def launch_worker(method, config_path, directory, preparing=False):
    environment = dict(os.environ)
    environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    # Keep measured loading offline. A preparation process may populate the cache.
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        environment[key] = '0' if preparing else '1'
    log = directory / method / ('prepare.log' if preparing else 'benchmark.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('w', encoding='utf-8') as stream:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--worker', method,
             str(config_path), 'prepare' if preparing else 'measure'],
            stdout=stream, stderr=subprocess.STDOUT, env=environment)
    return completed.returncode


def main(argv=None):
    options = parse_options(argv)
    from benchmark_models import definition_code
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
    files = list(NOTEBOOKS.values()) + [Path(__file__), Path(__file__).with_name('benchmark_models.py'),
                                      Path(__file__).with_name('benchmark_timing.py')]
    write_json(directory / 'source_hashes.json', {
        str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files if p.exists()})
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


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        settings = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
        settings['preparing'] = sys.argv[4] == 'prepare'
        raise SystemExit(run_worker(sys.argv[2], settings))
    raise SystemExit(main())
