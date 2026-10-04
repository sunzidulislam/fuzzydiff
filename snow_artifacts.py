"""Labeled research figures and separately shareable, anonymous human-review exports."""
import csv
from html import escape
import json
from pathlib import Path
import random
import statistics
import uuid
import zipfile

from PIL import Image

from snow_experiment import METHODS, METHOD_LABELS, atomic_json, jobs_for, valid_output


RATING_FIELDS = ['reviewer_id', 'image_id', 'image_sha256', 'reference_sha256',
                 'snow_coverage', 'prompt_adherence', 'scene_preservation', 'notes']


def _clean_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert('RGB')
        # Creating from pixels discards text, EXIF, ICC, and source metadata.
        clean = Image.frombytes('RGB', rgb.size, rgb.tobytes())
        clean.save(target, format='PNG')


def _render_grid(root, manifest, group, scene, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    design = manifest['design']
    figure, axes = plt.subplots(5, 3, figsize=(8.8, 13.8), layout='constrained')
    for row, level in enumerate(reversed(range(5))):
        for column, method in enumerate(METHODS):
            axis = axes[row, column]
            job = next(j for j in group if j['level'] == level and j['method'] == method)
            record = manifest['runs'].get(job['id'], {})
            axis.set_xticks([])
            axis.set_yticks([])
            if valid_output(root, job, record):
                with Image.open(root / job['image']) as image:
                    axis.imshow(image.convert('RGB'))
            else:
                axis.set_facecolor('#f1f1f1')
                axis.set_aspect('equal')
                axis.text(0.5, 0.5, record.get('status', 'pending').upper() + '\nNo valid output',
                          ha='center', va='center', transform=axis.transAxes, color='#9d3029')
            if row == 0:
                axis.set_title(METHOD_LABELS[column], fontsize=12, pad=10)
            if column == 0:
                axis.set_ylabel(design['levels'][level].capitalize() + '\nsnow cover', fontsize=11)
    figure.suptitle(f"Snow coverage prompts | {scene.capitalize()} | seed {seed}\n"
                    'Matched prompts and seeds; independently sampled images', fontsize=13)
    figure.supxlabel('Rows are ordinal prompt targets. Native/custom latent precision differs.\n'
                     'Scene geometry is not guaranteed; assess every output, including failures.', fontsize=9)
    directory = root / 'grids'
    directory.mkdir(exist_ok=True)
    file = directory / f'{scene}_seed_{seed}'
    figure.savefig(file.with_suffix('.png'), dpi=300, facecolor='white')
    figure.savefig(file.with_suffix('.pdf'), facecolor='white')
    plt.close(figure)


def _page(cards, title):
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>' + escape(title) + '</title><style>'
            'body{font:17px system-ui;margin:24px auto;max-width:1000px;padding:0 20px;background:#fafafa}'
            'article{background:white;padding:20px;margin:30px 0;border:1px solid #ddd}'
            '.pair{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:250px}'
            'img{width:100%;max-width:768px;height:auto}figcaption{margin:8px 0}'
            '</style><h1>' + escape(title) + '</h1>' + ''.join(cards) + '</html>')


def _review_pack(root, manifest, jobs):
    key_path = root / 'review_key.json'
    design = manifest['design']
    all_jobs = jobs_for(design, True)
    if key_path.exists():
        key = json.loads(key_path.read_text(encoding='utf-8'))
        if key['design'] != design:
            raise ValueError('Private review key belongs to a different study.')
    else:
        images = {'IMG_' + uuid.uuid4().hex[:16]: job['id'] for job in all_jobs}
        references = {f'{scene}_{seed}': 'REF_' + uuid.uuid4().hex[:16]
                      for scene in design['scenes'] for seed in design['seeds']}
        orders = {}
        for reviewer in ('reviewer_1', 'reviewer_2'):
            ids = list(images)
            random.SystemRandom().shuffle(ids)
            orders[reviewer] = ids
        key = {'design': design, 'images': images, 'references': references, 'orders': orders}
        atomic_json(key_path, key)
    pack = root / 'review_pack'
    pack.mkdir(exist_ok=True)
    by_id = {j['id']: j for j in jobs}
    included = {anon: by_id[job_id] for anon, job_id in key['images'].items()
                if job_id in by_id and valid_output(root, by_id[job_id], manifest['runs'].get(job_id, {}))}
    shared_files = ['README.md']
    for anon, job in included.items():
        filename = f'images/{anon}.png'
        _clean_copy(root / job['image'], pack / filename)
        shared_files.append(filename)
    references = {}
    reference_hashes = {}
    for job in included.values():
        block = f"{job['scene']}_{job['seed']}"
        if block in references:
            continue
        reference = next(j for j in all_jobs if j['scene'] == job['scene'] and
                         j['seed'] == job['seed'] and j['level'] == 2 and j['method'] == 'native')
        if valid_output(root, reference, manifest['runs'].get(reference['id'], {})):
            filename = f"references/{key['references'][block]}.png"
            _clean_copy(root / reference['image'], pack / filename)
            shared_files.append(filename)
            references[block] = filename
            reference_hashes[block] = manifest['runs'][reference['id']]['image_sha256']
        else:
            references[block] = None
            reference_hashes[block] = ''
    for reviewer, order in key['orders'].items():
        ids = [anon for anon in order if anon in included]
        coverage, alignment = [], []
        for anon in ids:
            job = included[anon]
            image = f'<img src="images/{anon}.png" alt="Anonymous candidate {anon}">'
            heading = '<h2>' + escape(anon) + '</h2>'
            coverage.append('<article>' + heading + image + '</article>')
            reference = references[f"{job['scene']}_{job['seed']}"]
            ref_html = (f'<img src="{reference}" alt="Shared scene reference">'
                        if reference else '<p>Reference unavailable; leave scene_preservation blank.</p>')
            alignment.append('<article>' + heading + '<p>Prompt: ' + escape(job['prompt']) + '</p>'
                             '<div class="pair"><figure><figcaption>Candidate</figcaption>' + image +
                             '</figure><figure><figcaption>Shared scene reference</figcaption>' +
                             ref_html + '</figure></div></article>')
        for kind, cards in (('coverage', coverage), ('alignment', alignment)):
            filename = f'{kind}_{reviewer}.html'
            (pack / filename).write_text(_page(cards, f'{kind.capitalize()} review — {reviewer}'), encoding='utf-8')
            shared_files.append(filename)
        # A shared pack always contains blank sheets. Archive entered data privately.
        sheet = pack / f'{reviewer}.csv'
        if sheet.exists():
            with sheet.open(newline='', encoding='utf-8') as stream:
                prior = list(csv.DictReader(stream))
            if any(row.get(field) for row in prior for field in
                   ('snow_coverage', 'prompt_adherence', 'scene_preservation', 'notes')) or \
                    any(row.get('reviewer_id') != reviewer for row in prior):
                archive = root / 'saved_rating_sheets' / f'{reviewer}_{uuid.uuid4().hex}.csv'
                archive.parent.mkdir(exist_ok=True)
                archive.write_bytes(sheet.read_bytes())
        with sheet.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=RATING_FIELDS)
            writer.writeheader()
            for anon in ids:
                fingerprint = manifest['runs'][included[anon]['id']]['image_sha256']
                block = f"{included[anon]['scene']}_{included[anon]['seed']}"
                writer.writerow(dict.fromkeys(RATING_FIELDS, '') |
                                {'reviewer_id': reviewer, 'image_id': anon, 'image_sha256': fingerprint,
                                 'reference_sha256': reference_hashes[block]})
        shared_files.append(sheet.name)
    instructions = '''# Independent image review

Use two different human reviewers. Do not discuss ratings or inspect labeled
figures/private mappings until both reviews are finished. Reviewer IDs in the
sheets are aliases; the study owner must verify they are distinct people.

1. Open coverage_reviewer_1.html (or reviewer_2) in a browser. Work in displayed
   order. Enter **snow_coverage** in the matching CSV, without opening alignment:
   1 = very little/no visible snow on mountains; 2 = slight; 3 = moderate;
   4 = heavy; 5 = nearly complete coverage. Judge visible coverage, not brightness.
2. Save those ratings. Open alignment_reviewer_1.html (or reviewer_2).
   Keep coverage ratings unchanged. Rate **prompt_adherence** from 1 (does not
   match) to 5 (strongly matches the whole prompt, including requested snow).
3. Rate **scene_preservation** from 1 (substantially different composition or
   mountain/lake geometry) to 5 (closely matches the shared scene reference).
   Ignore the intended amount of snow when judging geometry. Leave this field
   blank if the reference is unavailable. References are comparison anchors,
   not ground-truth images. Add comments in **notes**, particularly for ambiguity
   or image defects. Poor images still need ratings; do not omit them.

Use integers 1–5. Blank means not rated, never zero. Save completed copies of
the CSVs separately and return them to the study owner. This pack may omit
failed/missing generations; the private study report retains those failures.
Keep image_sha256 and reference_sha256 unchanged: they identify the pixels
you reviewed. Regenerated packs contain blank sheets; entered sheets found in
the pack folder are archived privately by the owner before rebuilding.
'''
    (pack / 'README.md').write_text(instructions, encoding='utf-8')
    temporary = root / 'review_pack.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in sorted(set(shared_files)):
            archive.write(pack / filename, arcname=filename)
    temporary.replace(root / 'review_pack.zip')


def export_artifacts(root, full=False):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    jobs = jobs_for(manifest['design'], full)
    blocks = list(dict.fromkeys((j['scene'], j['seed']) for j in jobs))
    for scene, seed in blocks:
        _render_grid(root, manifest, [j for j in jobs if j['scene'] == scene and j['seed'] == seed], scene, seed)
    _review_pack(root, manifest, jobs)
    missing = [j['id'] for j in jobs if not valid_output(root, j, manifest['runs'].get(j['id'], {}))]
    times = {}
    for method in METHODS:
        values = [attempt['elapsed_seconds'] for record in manifest['runs'].values()
                  if record['job']['method'] == method for attempt in record['attempts']
                  if attempt.get('status') == 'complete']
        times[method] = {'successful_attempts': len(values),
                         'mean_seconds': statistics.mean(values) if values else None}
    status = {'expected': len(jobs), 'complete': len(jobs) - len(missing), 'missing': missing,
              'runtime_by_method': times, 'limitations': manifest['design']['limitations'],
              'runtime_note': 'Measured job time includes evaluation/output handling, excludes model loading. '
                              'Pilot times do not guarantee full-study runtime or memory safety.'}
    atomic_json(root / 'artifact_status.json', status)
    print(f"Exported {len(blocks)} grids and anonymous review pack ({status['complete']}/{len(jobs)} valid images).", flush=True)
    return status
