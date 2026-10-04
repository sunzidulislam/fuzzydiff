"""Explicit fixed-scene snow appearance controls; not latent snow guidance."""
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from snow_experiment import atomic_json, utc_now


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_design(seed=142, source=None, mask=None):
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Seed must be a nonnegative integer.')
    repo = Path(__file__).resolve().parent
    files = ('snow_control.py', 'snow_control_source.py', 'snow_gpu.py', 'snow_experiment.py',
             'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb')
    return {'schema': 1, 'seed': seed, 'coverage': [i / 10 for i in range(11)],
            'thickness': [round(float(x), 4) for x in np.linspace(0.1, 1.0, 9)],
            'height': 768, 'width': 768, 'base_steps': 50, 'guidance': 9.5,
            'prompt': 'A realistic summer landscape photograph of bare rocky alpine mountains, '
                      'exposed gray and brown rock faces, rugged stone ridges, a green valley, '
                      'clear blue sky, daylight, fixed wide-angle camera viewpoint, natural colors.',
            'extra_negative': 'snow, snow-covered mountains, white mountain peaks, winter, ice, glacier',
            'mask_model': 'CIDAS/clipseg-rd64-refined', 'mask_threshold': 0.35,
            'input_source_sha256': file_hash(source) if source is not None else None,
            'input_mask_sha256': file_hash(mask) if mask is not None else None,
            'code_sha256': {name: file_hash(repo / name) for name in files if (repo / name).exists()},
            'interpretation': 'FuzzyDiff scene plus procedural image-space snow rendering. '
                              'C is selected terrain-mask area; D is relative material opacity/texture, '
                              'not physical snow depth. No learned fuzzy-guidance improvement is inferred.'}


def save_png(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.png.tmp')
    image.save(temporary, format='PNG')
    temporary.replace(path)


class SnowRenderer:
    """Nested area masks and a monotonic snow material on a fixed RGB scene."""
    def __init__(self, image, terrain_mask, seed=142):
        self.source = image.convert('RGB')
        self.pixels = np.asarray(self.source).copy()
        terrain = np.asarray(terrain_mask)
        if terrain.shape != self.pixels.shape[:2] or not np.isfinite(terrain).all():
            raise ValueError('Terrain mask must be finite and match the source dimensions.')
        self.terrain = terrain > 0
        if not self.terrain.any():
            raise ValueError('Terrain mask is empty; inspect segmentation or supply --mask.')
        self.count = int(self.terrain.sum())
        self.rgb = self.pixels.astype(np.float32) / 255
        height, width = self.terrain.shape
        random = np.random.default_rng(seed)
        noise = random.normal(size=(height, width)).astype(np.float32)
        field = np.zeros((height, width), dtype=np.float32)
        for weight, sigma in ((0.6, max(height / 48, 1)), (0.4, max(height / 12, 1))):
            smooth = gaussian_filter(noise, sigma)
            field += weight * (smooth - smooth.min()) / max(float(np.ptp(smooth)), 1e-8)
        # Screen height is only a visual accumulation heuristic, not scene elevation.
        field = 0.5 * field + 0.5 * (1 - np.arange(height, dtype=np.float32)[:, None] / max(height - 1, 1))
        self.field = field
        locations = np.flatnonzero(self.terrain)
        self.order = locations[np.argsort(-field.ravel()[locations], kind='stable')]
        luma = self.rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        shade = gaussian_filter(luma, max(height / 150, 1))
        low, high = np.percentile(shade[self.terrain], [5, 95])
        illumination = 0.58 + 0.42 * np.clip((shade - low) / max(float(high - low), 0.05), 0, 1)
        detail = luma - gaussian_filter(luma, max(height / 90, 1))
        tint = np.array([0.94, 0.96, 0.99], dtype=np.float32)
        material = tint[None, None, :] * illumination[:, :, None] + detail[:, :, None] * 0.1
        self.snow = np.clip(np.maximum(self.rgb, material), 0, 1)

    def coverage_mask(self, coverage):
        if not math.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError('Coverage must be finite and between 0 and 1.')
        mask = np.zeros(self.terrain.shape, dtype=bool)
        mask.ravel()[self.order[:int(round(coverage * self.count))]] = True
        return mask

    def render(self, coverage, thickness):
        if not math.isfinite(thickness) or not 0 <= thickness <= 1:
            raise ValueError('Thickness must be finite and between 0 and 1.')
        selected = self.coverage_mask(coverage)
        feathered = gaussian_filter(selected.astype(np.float32), max(self.terrain.shape[0] / 500, 0.5))
        feathered *= self.terrain
        # Increasing D attenuates exposed rock texture by increasing material opacity.
        alpha = feathered * thickness
        result = np.rint(np.clip(self.rgb + alpha[:, :, None] * (self.snow - self.rgb), 0, 1) * 255).astype('uint8')
        result[~self.terrain] = self.pixels[~self.terrain]
        metrics = {'target_mask_coverage': coverage, 'selected_mask_coverage': float(selected.sum() / self.count),
                   'relative_thickness': thickness, 'terrain_pixels': self.count,
                   'selected_pixels': int(selected.sum()),
                   'mean_snow_layer_opacity_inside_terrain': float(alpha[self.terrain].mean()),
                   'changed_terrain_fraction': float(np.any(result != self.pixels, axis=-1)[self.terrain].mean()),
                   'outside_terrain_pixels_changed': int(np.any(result != self.pixels, axis=-1)[~self.terrain].sum())}
        return Image.fromarray(result), metrics


def _verified_asset(root, record, label):
    path = root / record['image']
    if not path.is_file() or file_hash(path) != record['sha256']:
        raise ValueError(f'Saved {label} is missing or changed. Restore it or use a new --output directory.')
    return path


def run_control(root, design, source=None, mask=None):
    """Persist one source/mask and resume the 99 deterministic CPU renderings."""
    root = Path(root)
    path = root / 'control_manifest.json'
    environment = {'python': platform.python_version(),
                   **{name: version(name) for name in ('numpy', 'scipy', 'Pillow')}}
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['design'] != design:
            raise ValueError('Directory contains a different control design/code version; use a new --output.')
        if manifest.get('renderer_environment') != environment:
            raise ValueError('Renderer environment changed; use the original package versions or a new --output.')
    else:
        manifest = {'design': design, 'created_at': utc_now(), 'source': None, 'mask': None,
                    'source_attempts': [], 'mask_attempts': [], 'runs': {}, 'renderer_environment': environment}
    atomic_json(path, manifest)
    if manifest['source']:
        source_file = _verified_asset(root, manifest['source'], 'source')
    else:
        attempt = {'started_at': utc_now(), 'status': 'running'}
        manifest['source_attempts'].append(attempt)
        atomic_json(path, manifest)
        started = time.perf_counter()
        try:
            if source is not None:
                if file_hash(source) != design['input_source_sha256']:
                    raise ValueError('Input source changed after planning.')
                with Image.open(source) as image:
                    image = image.convert('RGB')
                metadata = {'origin': 'user-supplied image', 'snow_free_verified': False}
            else:
                if design['input_source_sha256']:
                    raise ValueError('The original --source is needed to initialize this directory.')
                from snow_control_source import generate_source
                image, metadata = generate_source(design, root)
            json.dumps(metadata, allow_nan=False)
            source_file = root / 'source.png'
            save_png(source_file, image)
            manifest['source'] = {'image': 'source.png', 'sha256': file_hash(source_file), 'metadata': metadata}
            attempt['status'] = 'complete'
        except Exception as error:
            attempt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            attempt['elapsed_seconds'] = time.perf_counter() - started
            atomic_json(path, manifest)
    with Image.open(source_file) as image:
        image = image.convert('RGB')
    if manifest['mask']:
        mask_file = _verified_asset(root, manifest['mask'], 'mask')
        with Image.open(mask_file) as loaded:
            terrain = np.asarray(loaded.convert('L')) >= 128
    else:
        attempt = {'started_at': utc_now(), 'status': 'running'}
        manifest['mask_attempts'].append(attempt)
        atomic_json(path, manifest)
        started = time.perf_counter()
        try:
            if mask is not None:
                if file_hash(mask) != design['input_mask_sha256']:
                    raise ValueError('Input mask changed after planning.')
                with Image.open(mask) as loaded:
                    if loaded.size != image.size:
                        raise ValueError('Mask dimensions must match the source; no automatic resizing.')
                    terrain = np.asarray(loaded.convert('L')) >= 128
                metadata = {'origin': 'user-supplied terrain mask', 'accuracy_verified': False}
            else:
                if design['input_mask_sha256']:
                    raise ValueError('The original --mask is needed to initialize this directory.')
                from snow_control_source import estimate_terrain
                terrain, metadata = estimate_terrain(image, design)
            SnowRenderer(image, terrain, design['seed'])  # Validate before marking the mask complete.
            mask_file = root / 'mountain_mask.png'
            save_png(mask_file, Image.fromarray(terrain.astype('uint8') * 255))
            manifest['mask'] = {'image': 'mountain_mask.png', 'sha256': file_hash(mask_file), 'metadata': metadata}
            attempt['status'] = 'complete'
        except Exception as error:
            attempt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            attempt['elapsed_seconds'] = time.perf_counter() - started
            atomic_json(path, manifest)
    renderer = SnowRenderer(image, terrain, design['seed'])
    overlay = np.asarray(image).astype(np.float32)
    overlay[terrain] = 0.65 * overlay[terrain] + 0.35 * np.array([20, 240, 100])
    save_png(root / 'mask_overlay.png', Image.fromarray(np.rint(overlay).astype('uint8')))
    save_png(root / 'accumulation_field.png', Image.fromarray(np.rint(renderer.field * 255).astype('uint8')))
    for row, coverage in enumerate(design['coverage']):
        selected = renderer.coverage_mask(coverage)
        save_png(root / f'coverage_masks/coverage_{row:02d}.png', Image.fromarray(selected.astype('uint8') * 255))
        for column, thickness in enumerate(design['thickness']):
            identifier = f'coverage_{row:02d}_thickness_{column:02d}'
            record = manifest['runs'].setdefault(identifier, {
                'image': f'images/{identifier}.png', 'status': 'pending', 'attempts': []})
            target = root / record['image']
            if record['status'] == 'complete' and target.is_file() and file_hash(target) == record.get('sha256'):
                continue
            attempt = {'started_at': utc_now(), 'status': 'running'}
            record['status'] = 'running'
            record['attempts'].append(attempt)
            atomic_json(path, manifest)
            started = time.perf_counter()
            try:
                output, metrics = renderer.render(coverage, thickness)
                save_png(target, output)
                record.update(status='complete', sha256=file_hash(target), metrics=metrics)
                attempt['status'] = 'complete'
            except Exception as error:
                record['status'] = attempt['status'] = 'failed'
                attempt['error'] = f'{type(error).__name__}: {error}'
                print(f'FAILED {identifier}: {attempt["error"]}', flush=True)
            finally:
                attempt['elapsed_seconds'] = time.perf_counter() - started
                atomic_json(path, manifest)
    return manifest
