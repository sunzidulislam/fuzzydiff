"""Fixed snow study design and durable, resumable generation ledger (no GPU imports)."""
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone


METHODS = ('native', 'fuzzy', 'refined')
METHOD_LABELS = ('Native SDXL', 'FuzzyDiff', 'FuzzyDiff + refiner')
LEVELS = ('very slight', 'slight', 'moderate', 'heavy', 'very heavy')
SCENES = {
    'alpine': 'a wide view of alpine mountains, a fixed wide-angle camera viewpoint',
    'rocky': 'a close view of rocky mountains and jagged peaks, a fixed eye-level camera viewpoint',
    'lake': 'mountains beside a calm lake, a fixed wide-angle camera viewpoint from the shore',
}
NEGATIVE = ('cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, '
            'toy-like, plastic texture, low quality, blurry, distorted, text, watermark')


def code_fingerprint():
    root = Path(__file__).resolve().parent
    names = ('pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb', 'snow_experiment.py', 'snow_gpu.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names if (root / name).exists()}


def make_design():
    return {
        'schema': 1, 'levels': list(LEVELS), 'scenes': dict(SCENES),
        'seeds': [42, 84, 142, 202, 314], 'methods': list(METHODS),
        'pilot': {'scene': 'alpine', 'seed': 142},
        'settings': {'height': 768, 'width': 768, 'base_steps': 50, 'guidance': 9.5,
                     'max_iter_to_alter': 30, 'attend_excite_lr': 0.2,
                     'alpha': 10.0, 'spatial_weight': 0.5, 'attn_res': 16,
                     'refiner_steps': 30, 'refiner_strength': 0.4,
                     'predicate_strength': 2.5, 'words_to_track': ['mountains', 'snow'],
                     'relations': [], 'negative_prompt': NEGATIVE},
        'checkpoints': {'base': 'stabilityai/stable-diffusion-xl-base-1.0',
                        'refiner': 'stabilityai/stable-diffusion-xl-refiner-1.0',
                        'vae': 'madebyollin/sdxl-vae-fp16-fix', 'clip': 'ViT-B/32'},
        'precision': {'unet': 'float16', 'native_latents': 'float16', 'custom_latents': 'float32'},
        'limitations': ['Independent text-to-image sampling can change scene geometry.',
                        'Native/custom also differ in latent precision.',
                        'Native moderate-snow image is the scene-preservation reference; this may favor native.',
                        'Prompt levels are ordinal, not numeric snow coverage targets.',
                        'CLIP is supplemental; no human ratings or improvement claims are inferred.'],
        'code_sha256': code_fingerprint(),
    }


def jobs_for(design, full=False):
    jobs = []
    for scene, description in design['scenes'].items():
        for seed in design['seeds']:
            if not full and (scene != design['pilot']['scene'] or seed != design['pilot']['seed']):
                continue
            for level, degree in enumerate(design['levels']):
                stem = f'{scene}_seed_{seed}_level_{level + 1}'
                prompt = (f'A realistic landscape photograph of {description}, with {degree} '
                          'snow cover on the mountains, daylight, clear sky, natural colors.')
                for method in design['methods']:
                    job = {'id': f'{stem}_{method}', 'base_id': stem, 'scene': scene,
                           'seed': seed, 'level': level, 'method': method, 'prompt': prompt,
                           'image': f'images/{stem}_{method}.png'}
                    if method in ('fuzzy', 'refined'):
                        job['attention'] = f'attention/{stem}.pt'
                    if method == 'refined':
                        job['dependency'] = f'{stem}_fuzzy'
                    jobs.append(job)
    return jobs


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def valid_output(root, job, record):
    if record.get('status') != 'complete':
        return False
    path = root / job['image']
    if not path.is_file() or record.get('image_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
        return False
    if job['method'] == 'fuzzy':
        path = root / job['attention']
        return (path.is_file() and record.get('attention_sha256') ==
                hashlib.sha256(path.read_bytes()).hexdigest())
    return True


def upgrade_original_metadata_failure(root, manifest, design):
    """One narrowly allowlisted metadata repair; never mix changed generation settings."""
    old = manifest['design']
    old_hashes = old.get('code_sha256', {})
    new_hashes = design.get('code_sha256', {})
    notebooks = {'dc0be5b2c7b0905f4aa3aa82fd85501fc13c4ec154c76d50a4cd587adc0dac91',
                 '642457b70b7b65df05002bb31c51eb0fbb1e5fde4c026f6695036bdb1e839fe1'}
    same_settings = {k: v for k, v in old.items() if k != 'code_sha256'} == \
                    {k: v for k, v in design.items() if k != 'code_sha256'}
    if not (same_settings and
            old_hashes.get('snow_gpu.py') == '7d48b27b7f334bf5cf8011881cdc62b45c76073a7862c25ec608bcc9c0e99f95' and
            old_hashes.get('snow_experiment.py') == '14561514686917a6078271b2801afc12f3b68779cb529c9f5970ef0fc91840e7' and
            new_hashes.get('snow_gpu.py') == '7af010bfc6ab5404b7cfaffe46bd234d37a12f81274f692dd89662fb09cbe0ea' and
            old_hashes.get('pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb') in notebooks and
            new_hashes.get('pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb') in notebooks and
            not any(r.get('status') == 'complete' for r in manifest['runs'].values())):
        return False
    key_path = root / 'review_key.json'
    if key_path.exists():
        key = json.loads(key_path.read_text(encoding='utf-8'))
        if key['design'] not in (old, design):
            return False
        key['design'] = design
        atomic_json(key_path, key)
    manifest.setdefault('code_upgrades', []).append({
        'at': utc_now(), 'reason': 'Repair CLIPTextConfig metadata access; generation algorithm unchanged.',
        'from': old_hashes, 'to': new_hashes})
    manifest['design'] = design
    print('Upgraded the original failed metadata run; preserving attempts and study settings.', flush=True)
    return True


def run_study(root, design, backend, full=False):
    """Generate in weight-sharing phases; backend failures are recorded, never hidden."""
    root = Path(root)
    path = root / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['design'] != design and not upgrade_original_metadata_failure(root, manifest, design):
            raise ValueError('Output directory contains a different design or code version; use a new --output.')
        if manifest.get('scope') == 'full' and not full:
            raise ValueError('This directory is a full study; rerun with --full to preserve its scope.')
    else:
        manifest = {'design': design, 'created_at': utc_now(), 'runs': {}, 'phase_times': []}
    jobs = jobs_for(design, full)
    for job in jobs:
        record = manifest['runs'].setdefault(job['id'], {'job': job, 'status': 'pending', 'attempts': []})
        if not valid_output(root, job, record):
            record['status'] = 'pending'
    # A replaced base invalidates its dependent refinement, even if its PNG still exists.
    for job in jobs:
        if job['method'] == 'refined' and manifest['runs'][job['dependency']]['status'] != 'complete':
            manifest['runs'][job['id']]['status'] = 'pending'
    manifest['scope'] = 'full' if full else 'pilot'
    atomic_json(path, manifest)
    try:
        for method in METHODS:
            pending = [j for j in jobs if j['method'] == method and manifest['runs'][j['id']]['status'] != 'complete']
            if not pending:
                continue
            started = time.perf_counter()
            prepare_error = None
            try:
                backend.prepare(method)
            except Exception as error:
                prepare_error = f'{type(error).__name__}: {error}'
            manifest['phase_times'].append({'method': method, 'loaded_at': utc_now(),
                                             'loading_seconds': time.perf_counter() - started,
                                             'error': prepare_error})
            atomic_json(path, manifest)
            for job in pending:
                record = manifest['runs'][job['id']]
                if job.get('dependency') and manifest['runs'][job['dependency']]['status'] != 'complete':
                    record['status'] = 'blocked'
                    record['reason'] = 'FuzzyDiff base generation failed; refinement needs its image and attention.'
                    atomic_json(path, manifest)
                    continue
                attempt = {'started_at': utc_now()}
                record['attempts'].append(attempt)
                record['status'] = 'running'
                atomic_json(path, manifest)
                started = time.perf_counter()
                try:
                    if prepare_error:
                        raise RuntimeError(prepare_error)
                    print(f"Generating {job['id']}", flush=True)
                    image, metadata = backend.generate(job, root)
                    # Reject invalid metadata before it can poison the durable manifest.
                    json.dumps(metadata, allow_nan=False)
                    target = root / job['image']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.png.tmp')
                    image.save(temporary, format='PNG')
                    if job['method'] == 'fuzzy' and not (root / job['attention']).is_file():
                        raise RuntimeError('FuzzyDiff did not persist the attention required for refinement.')
                    temporary.replace(target)
                    record.update(status='complete', metadata=metadata,
                                  image_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                    if job['method'] == 'fuzzy':
                        record['attention_sha256'] = hashlib.sha256((root / job['attention']).read_bytes()).hexdigest()
                    record.pop('reason', None)
                    attempt['status'] = 'complete'
                except Exception as error:
                    record['status'] = attempt['status'] = 'failed'
                    attempt['error'] = f'{type(error).__name__}: {error}'
                    print(f"FAILED {job['id']}: {attempt['error']}", flush=True)
                finally:
                    attempt['elapsed_seconds'] = time.perf_counter() - started
                    attempt['finished_at'] = utc_now()
                    atomic_json(path, manifest)
    finally:
        backend.close()
    return manifest
