"""Read-only notebook adapters. Generation algorithms stay in their notebooks.

The archived implementations are exploratory baselines, not validated reproductions.
All adapters use CPU model offload and explicit common size/step/CFG settings.
"""
import hashlib
from pathlib import Path


from fuzzydiff.paths import repository_root

ROOT = repository_root()
NOTEBOOKS = {
    'stable': ROOT / 'pipeline_fuzzy/analysis/stable-f3.ipynb',
    'attend': ROOT / 'pipeline_fuzzy/analysis/attend-f3.ipynb',
    'predicate': ROOT / 'pipeline_fuzzy/analysis/predicate-f3.ipynb',
    'aesthetic': ROOT / 'pipeline_fuzzy/analysis/aesthetic-f3.ipynb',
    'enhancing': ROOT / 'pipeline_fuzzy/analysis/enhancing-f3 (1).ipynb',
    'fuzzydiff': ROOT / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb',
}
LABELS = {'stable': 'SD 1.4 plain (archived)',
          'attend': 'Attend-and-Excite (archived, unvalidated)',
          'predicate': 'Predicated Diffusion (archived, unvalidated)',
          'aesthetic': 'SDXL base + refiner', 'enhancing': 'SDXL-Turbo (Enhancing notebook)',
          'fuzzydiff': 'FuzzyDiff full'}
LIMITATIONS = {
    'stable': ['This notebook uses SD 1.4, not the SD 1.5 CLI backbone.'],
    'attend': ['Archived attention storage detaches tensors; gradient guidance is disconnected. '
               'Do not present this as a validated Attend-and-Excite result.',
               'The archived pipeline ignores the negative prompt.'],
    'predicate': ['Archived implementation has not been validated against the published method.',
                  'Phrase-final tokens are used for existence; --bind adds correspondence pairs. '
                  'No automatic leakage or possession constraints are invented.',
                  'The archived run_on_prompt ignores the negative prompt.'],
    'aesthetic': ['This is SDXL base + refiner, not an aesthetic-guidance objective.',
                  'The archived refiner receives no negative prompt.'],
    'fuzzydiff': ['Attention truth is not a visual accuracy measurement.'],
    'enhancing': ['This notebook uses SDXL-Turbo with no fuzzy objective or refinement.',
                  'Uses its native 4-step, CFG=0 settings rather than common steps/CFG; '
                  'timings are not a matched-step comparison.',
                  'Tracked phrases are metadata only; CFG=0 ignores negative conditioning.',
                  'Uses standard attention with VAE slicing/tiling; optional xFormers is not enabled.'],
}


def effective_sampling_settings(method, options):
    """Turbo retains the notebook's native distilled sampling configuration."""
    return {'steps': 4 if method == 'enhancing' else options['steps'],
            'guidance': 0.0 if method == 'enhancing' else options['guidance'],
            'size': options['size']}


from dataclasses import dataclass
from fuzzydiff.notebook_runtime import definition_code, load_definitions


@dataclass(frozen=True)
class ModelSpec:
    id: str
    aliases: tuple[str, ...]
    label: str
    notebook: Path | None
    limitations: tuple[str, ...]


_IDENTITIES = {'stable': 'sd14', 'attend': 'attend_archived',
               'predicate': 'predicate_archived', 'aesthetic': 'sdxl_base_refiner',
               'enhancing': 'sdxl_turbo', 'fuzzydiff': 'fuzzydiff_full'}
MODELS = tuple(ModelSpec(identity, (alias,), LABELS[alias], NOTEBOOKS[alias],
                         tuple(LIMITATIONS[alias])) for alias, identity in _IDENTITIES.items()) + (
    ModelSpec('sd15', (), 'SD 1.5 plain', None, ()),)


def resolve_model(name: str) -> ModelSpec:
    for spec in MODELS:
        if name == spec.id or name in spec.aliases:
            return spec
    raise ValueError(f'Unknown model {name!r}; choose from ' + ', '.join(s.id for s in MODELS))


def adapter_class(model: str):
    spec = resolve_model(model)
    if spec.id == 'sd15':
        from .sd15 import PlainSD15Session
        return PlainSD15Session
    if spec.id in ('sd14', 'attend_archived', 'predicate_archived'):
        from .archived import ArchivedSession
        return ArchivedSession
    if spec.id == 'sdxl_turbo':
        from .sdxl_turbo import TurboSession
        return TurboSession
    from .sdxl import SDXLSession
    return SDXLSession


def create_session(model: str, options: dict):
    spec = resolve_model(model)
    method = spec.aliases[0] if spec.aliases else spec.id
    cls = adapter_class(spec.id)
    session = cls.__new__(cls)
    try:
        cls.__init__(session, method, options)
    except BaseException:
        session.close()
        raise
    return session


def __getattr__(name):
    # Legacy class access stays lazy so registry/help do not load model runtimes.
    if name == 'NotebookModel':
        from .session import NotebookModel
        return NotebookModel
    if name == 'component_commit_hash':
        from .session import component_commit_hash
        return component_commit_hash
    raise AttributeError(name)
