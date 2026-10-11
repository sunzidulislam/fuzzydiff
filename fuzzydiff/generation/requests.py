"""Data-only generation contracts and early validation."""
from dataclasses import dataclass, field
import math

from fuzzydiff.models.registry import resolve_model


FUZZY_OPTIONS = {'lr', 'updates', 'sharpness', 'tnorm', 'membership_mode',
                 'bindings', 'relations', 'binding_weight', 'hedge_guidance',
                 'predicate_strength', 'refine_steps', 'refine_strength'}


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    seed: int = 142
    size: int = 512
    steps: int | None = None
    guidance: float | None = None
    negative: str = ''
    words: tuple[str, ...] = ()
    method_options: dict = field(default_factory=dict)

    def __post_init__(self):
        spec = resolve_model(self.model)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError('Prompt must be nonempty text.')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('Seed must be an integer in [0, 2**63).')
        if type(self.size) is not int or self.size < 128 or self.size % 8:
            raise ValueError('Size must be at least 128 and divisible by 8.')
        if spec.id in ('attend_archived', 'predicate_archived') and self.size != 512:
            raise ValueError('Archived attention methods require size 512.')
        if self.steps is not None and (type(self.steps) is not int or self.steps < 1):
            raise ValueError('Steps must be a positive integer.')
        if self.guidance is not None and (not math.isfinite(self.guidance) or self.guidance < 0):
            raise ValueError('Guidance must be finite and nonnegative.')
        if spec.id in ('attend_archived', 'predicate_archived', 'fuzzydiff_full') and self.guidance is not None and self.guidance <= 1:
            raise ValueError('This adapter requires CFG > 1.')
        if not isinstance(self.negative, str):
            raise ValueError('Negative prompt must be text.')
        if not isinstance(self.words, (list, tuple)) or any(
                not isinstance(w, str) or not w or w not in self.prompt for w in self.words):
            raise ValueError('Tracked phrases must appear verbatim in the prompt.')
        if spec.id in ('attend_archived', 'predicate_archived', 'fuzzydiff_full') and not self.words:
            raise ValueError('This adapter requires tracked phrases (--words).')
        if not isinstance(self.method_options, dict):
            raise ValueError('method_options must be an object.')
        allowed = (FUZZY_OPTIONS if spec.id == 'fuzzydiff_full' else
                   {'bindings'} if spec.id == 'predicate_archived' else set())
        unknown = set(self.method_options) - allowed
        if unknown:
            raise ValueError('Unsupported method options: ' + ', '.join(sorted(unknown)))
        o = self.method_options
        for key in ('lr', 'sharpness', 'binding_weight', 'predicate_strength', 'refine_strength'):
            if key in o and (not isinstance(o[key], (int, float)) or not math.isfinite(o[key]) or o[key] < 0):
                raise ValueError(f'{key} must be finite and nonnegative.')
        for key in ('updates', 'refine_steps'):
            if key in o and (type(o[key]) is not int or o[key] < (1 if key == 'refine_steps' else 0)):
                raise ValueError(f'Invalid {key}.')
        if o.get('tnorm', 'product') not in ('min', 'product') or o.get('membership_mode', 'share') not in ('share', 'relative'):
            raise ValueError('Invalid tnorm or membership_mode.')
        if o.get('hedge_guidance') and o.get('membership_mode', 'share') != 'share':
            raise ValueError('Hedge guidance requires share membership.')
        if not 0 < o.get('refine_strength', .4) <= 1 or int(o.get('refine_strength', .4) * o.get('refine_steps', 30)) < 1:
            raise ValueError('Refinement must produce at least one step.')
        for pair in o.get('bindings', ()):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(w not in self.words for w in pair):
                raise ValueError('Binding operands must be tracked phrases.')
        # The CLI mirror is data-only; parsing does not import Torch.
        from run_fullpipeline import RELATION_VERBS
        for relation in o.get('relations', ()):
            if len(relation) != 3 or relation[1] not in RELATION_VERBS or any(w not in self.words for w in (relation[0], relation[2])):
                raise ValueError('Relation must use a supported verb and tracked operands.')
        object.__setattr__(self, 'words', tuple(self.words))
        object.__setattr__(self, 'method_options', dict(o))


@dataclass
class GenerationResult:
    image: object
    metadata: dict
    diagnostics: dict
