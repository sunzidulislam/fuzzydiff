"""Format measured experiment records without model imports."""
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


