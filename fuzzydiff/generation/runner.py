"""Run one model once; explicit saving, no benchmark warmups or profiling."""
import json
from dataclasses import asdict
from pathlib import Path

from fuzzydiff.models.registry import create_session, resolve_model
from fuzzydiff.notebook_runtime import source_hashes
from fuzzydiff.paths import repository_root
from .requests import GenerationRequest, GenerationResult


def generate_image(request: GenerationRequest) -> GenerationResult:
    spec = resolve_model(request.model)
    options = dict(prompt=request.prompt, words=list(request.words), size=request.size,
                   steps=request.steps if request.steps is not None else 50,
                   guidance=request.guidance if request.guidance is not None else 7.5,
                   negative=request.negative, output='outputs', preparing=True,
                   lr=5.0, updates=30, sharpness=20.0, tnorm='product',
                   membership_mode='share', bindings=[], predicate_strength=.5,
                   refine_steps=30, refine_strength=.4)
    options.update(request.method_options)
    session = create_session(spec.id, options)
    try:
        image = session.generate(request.seed)
        metadata = {**session.metadata(), 'model': spec.id, 'seed': request.seed,
                    'request': asdict(request), 'limitations': list(spec.limitations),
                    'truth_note': 'Attention-derived objectives are not visual accuracy or attribute intensity.'}
        diagnostics = dict(getattr(session, 'last_details', {}))
        root = repository_root()
        paths = list((root / 'fuzzydiff').rglob('*.py'))
        if spec.notebook is not None:
            paths.append(spec.notebook)
        if spec.id in ('attend_archived', 'predicate_archived'):
            paths.append(root / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb')
        metadata['source_sha256'] = source_hashes(paths)
        return GenerationResult(image, metadata, diagnostics)
    finally:
        session.close()


def save_result(result: GenerationResult, directory: Path, seed: int) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'image': directory / f'seed_{seed}.png',
             'metadata': directory / f'seed_{seed}_metadata.json'}
    payload = {**result.metadata, 'diagnostics': result.diagnostics, 'image': str(paths['image'])}
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    result.image.save(paths['image'])
    paths['metadata'].write_text(encoded + '\n', encoding='utf-8')
    return paths
