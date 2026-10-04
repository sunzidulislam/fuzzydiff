"""Validate independent rating sheets and summarize the fixed study descriptively."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from snow_experiment import METHODS, atomic_json, jobs_for, valid_output


METRICS = ('snow_coverage', 'prompt_adherence', 'scene_preservation')


def _summary(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'range': max(values) - min(values) if values else None,
            'values': values}


def _spearman(values):
    # Spearman correlation with prompt level, including average ranks for ties.
    ranks = [1 + sum(other < value for other in values) +
             (sum(other == value for other in values) - 1) / 2 for value in values]
    if len(set(ranks)) == 1:
        return None  # Constant observed coverage: correlation is undefined.
    return statistics.correlation(list(range(len(values))), ranks)


def analyze(root, rating_paths):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    key = json.loads((root / 'review_key.json').read_text(encoding='utf-8'))
    if key['design'] != manifest['design']:
        raise ValueError('Review key and manifest have different designs.')
    jobs = jobs_for(manifest['design'], manifest.get('scope') == 'full')
    by_id = {j['id']: j for j in jobs}
    included = {anon: by_id[identifier] for anon, identifier in key['images'].items()
                if identifier in by_id and valid_output(root, by_id[identifier], manifest['runs'].get(identifier, {}))}
    all_jobs = jobs_for(manifest['design'], True)
    reference_available = {}
    reference_hashes = {}
    for job in jobs:
        block = (job['scene'], job['seed'])
        reference = next(j for j in all_jobs if (j['scene'], j['seed']) == block and
                         j['level'] == 2 and j['method'] == 'native')
        reference_available[block] = valid_output(root, reference, manifest['runs'].get(reference['id'], {}))
        reference_hashes[block] = manifest['runs'].get(reference['id'], {}).get('image_sha256')
    ratings = {anon: {} for anon in included}
    seen = set()
    for path in rating_paths:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            required = {'reviewer_id', 'image_id', 'image_sha256', 'reference_sha256', *METRICS}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'{path}: missing required rating columns.')
            for row_number, row in enumerate(reader, 2):
                anon, reviewer = row['image_id'].strip(), row['reviewer_id'].strip()
                location = f'{path}:{row_number}'
                if anon not in included or not reviewer:
                    raise ValueError(f'{location}: unknown/unavailable image or empty reviewer ID.')
                if row['image_sha256'] != manifest['runs'][included[anon]['id']]['image_sha256']:
                    raise ValueError(f'{location}: image changed after the review sheet was exported.')
                if (anon, reviewer) in seen:
                    raise ValueError(f'{location}: Duplicate image/reviewer rating.')
                seen.add((anon, reviewer))
                parsed = {'reviewer_id': reviewer, 'notes': row.get('notes', '')}
                for metric in METRICS:
                    raw = row[metric].strip()
                    if raw and raw not in ('1', '2', '3', '4', '5'):
                        raise ValueError(f'{location}: {metric} must be an integer from 1 to 5, or blank.')
                    parsed[metric] = int(raw) if raw else None
                job = included[anon]
                if parsed['scene_preservation'] is not None and not reference_available[(job['scene'], job['seed'])]:
                    raise ValueError(f'{location}: scene preservation cannot be rated without its reference.')
                if parsed['scene_preservation'] is not None and row['reference_sha256'] != reference_hashes[(job['scene'], job['seed'])]:
                    raise ValueError(f'{location}: scene reference changed after the review sheet was exported.')
                ratings[anon][reviewer] = parsed
    summaries = []
    for anon, job in included.items():
        entries = list(ratings[anon].values())
        summary = {'image_id': anon, 'job': job,
                   'complete_reviewers': sum(all(e[m] is not None for m in METRICS) for e in entries),
                   'reference_available': reference_available[(job['scene'], job['seed'])],
                   'ratings': entries}
        for metric in METRICS:
            summary[metric] = _summary([e[metric] for e in entries if e[metric] is not None])
        summaries.append(summary)
    missing = [j['id'] for j in jobs if j['id'] not in {item['id'] for item in included.values()}]
    trends = []
    for scene, seed in dict.fromkeys((j['scene'], j['seed']) for j in jobs):
        for method in METHODS:
            group = sorted([s for s in summaries if s['job']['scene'] == scene and
                            s['job']['seed'] == seed and s['job']['method'] == method],
                           key=lambda s: s['job']['level'])
            complete = len(group) == 5 and all(s['snow_coverage']['n'] >= 2 for s in group)
            means = [s['snow_coverage']['mean'] for s in group] if complete else None
            trends.append({'scene': scene, 'seed': seed, 'method': method,
                           'all_levels_have_two_coverage_ratings': complete,
                           'coverage_means_in_level_order': means,
                           'nondecreasing': all(a <= b for a, b in zip(means, means[1:])) if complete else None,
                           'spearman_with_prompt_level': _spearman(means) if complete else None})
    result = {'scope': manifest.get('scope', 'pilot'), 'expected_images': len(jobs),
              'available_images': len(included), 'failed_or_missing_images': missing,
              'all_images_have_two_complete_reviews': not missing and
                  all(s['complete_reviewers'] >= 2 for s in summaries),
              'image_summaries': summaries, 'within_block_coverage_trends': trends,
              'limitations': manifest['design']['limitations'] +
                  ['Descriptive summaries only; ordinal-score means are convenient summaries, not physical coverage.',
                   'Distinct reviewer IDs do not verify distinct people; the study owner must confirm independence.',
                   'Tied observed coverage can be nondecreasing without demonstrating useful control.']}
    atomic_json(root / 'ratings_summary.json', result)
    target = root / 'ratings_summary.csv'
    fields = ['image_id', 'scene', 'seed', 'level', 'method', 'complete_reviewers'] + \
             [f'{metric}_{suffix}' for metric in METRICS for suffix in ('n', 'mean', 'range')]
    with target.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {field: summary['job'][field] for field in ('scene', 'seed', 'level', 'method')}
            row.update(image_id=summary['image_id'], complete_reviewers=summary['complete_reviewers'])
            row.update({f'{metric}_{suffix}': summary[metric][suffix]
                        for metric in METRICS for suffix in ('n', 'mean', 'range')})
            writer.writerow(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('ratings', type=Path, nargs='+')
    args = parser.parse_args()
    result = analyze(args.output, args.ratings)
    print(f"Ratings summarized for {result['available_images']}/{result['expected_images']} images.")
    print('At least two complete reviewer IDs per image:', result['all_images_have_two_complete_reviews'])
    print('Reviewer independence must be confirmed by the study owner. Results are descriptive.')


if __name__ == '__main__':
    main()
