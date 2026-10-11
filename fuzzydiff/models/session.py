"""Shared model lifecycle and benchmark instrumentation; no scientific algorithms."""
import hashlib
from contextlib import nullcontext, contextmanager, ExitStack
from functools import wraps
from pathlib import Path
from time import perf_counter

from .registry import NOTEBOOKS, LIMITATIONS, effective_sampling_settings, adapter_class, create_session
from fuzzydiff.notebook_runtime import load_definitions


def component_commit_hash(component):
    config = getattr(component, 'config', None)
    if isinstance(config, dict):
        return config.get('_commit_hash')
    return getattr(config, '_commit_hash', None)


class ModelSession:
    def __init__(self, method, options):
        import torch
        self.torch = torch
        self.method, self.options = method, options
        self.device = torch.device('cuda:0')
        self.pipelines = []
        self.load_parts = {}
        self.last_details = {}
        start = perf_counter()
        self.ns = load_definitions(NOTEBOOKS[method])
        if method in ('attend', 'predicate'):
            from .sd15 import load_fuzzy_definitions
            self.token_groups = load_fuzzy_definitions()['get_token_groups']
        self.definition_setup_s = perf_counter() - start
        common = {'local_files_only': not options.get('preparing', False)}

        def load(name, factory):
            torch.cuda.synchronize()
            start = perf_counter()
            result = factory()
            torch.cuda.synchronize()
            self.load_parts[name] = perf_counter() - start
            return result

        def place(pipeline):
            pipeline.enable_model_cpu_offload(gpu_id=0)
            pipeline.set_progress_bar_config(disable=True)
            self.pipelines.append(pipeline)
            return pipeline

        self._load(load, place, common)
        self.ns['device'] = self.device
        self.ns['model'] = self.base
        # Evaluation is excluded, not changed into a fabricated metric.
        self.ns['clip_prompt_similarities'] = lambda *args, **kwargs: None
        self.original_processors = [dict(p.unet.attn_processors) for p in self.pipelines]
        self.load_s = sum(self.load_parts.values())

    def metadata(self):
        return {'checkpoints': self.checkpoints,
                'effective_sampling_settings': effective_sampling_settings(self.method, self.options),
                'dtype': str(self.base.unet.dtype),
                'schedulers': [type(p.scheduler).__name__ for p in self.pipelines],
                'placement': 'model_cpu_offload, cuda:0',
                'load_s': self.load_s, 'load_parts_s': self.load_parts,
                'definition_setup_s': self.definition_setup_s,
                'notebook_sha256': hashlib.sha256(NOTEBOOKS[self.method].read_bytes()).hexdigest(),
                'component_commit_hashes': {
                    f'{i}.{name}': component_commit_hash(getattr(p, name, None))
                    for i, p in enumerate(self.pipelines)
                    for name in ('unet', 'vae', 'text_encoder', 'text_encoder_2')},
                'adapter_notes': [
                    'Common explicit size, steps and CFG except native SDXL-Turbo 4 steps/CFG=0; '
                    'CPU model offload on every method.',
                    'No CLIP evaluation, display or image saving inside inference.',
                    'SDXL refiner has its own VAE to avoid shared offload hooks.',
                    'SD 1.5 uses the maintained stable-diffusion-v1-5 checkpoint repository.',
                    'Core archived algorithms are not repaired by this runner.'],
                'limitations': LIMITATIONS[self.method]}

    def reset(self):
        # Run outside the timer: each sample starts with the same processor setup.
        for pipeline, processors in zip(self.pipelines, self.original_processors):
            pipeline.unet.set_attn_processor(dict(processors))
            pipeline.maybe_free_model_hooks()

    @contextmanager
    def profile_preprocessing(self, ledger):
        from fuzzydiff.reporting.timing import patch_method

        def timed(original):
            @wraps(original)
            def wrapped(*args, **kwargs):
                with ledger.measure('preprocess'):
                    return original(*args, **kwargs)
            return wrapped

        with ExitStack() as stack:
            for name in ('get_token_groups', 'register_attention_control',
                         'build_phrase_membership_masks', 'attach_predicate_control'):
                if name in self.ns:
                    original = self.ns[name]
                    self.ns[name] = timed(original)
                    stack.callback(self.ns.__setitem__, name, original)
            for name in ('RunConfig', 'AttentionStore'):
                cls = self.ns.get(name)
                if cls is not None:
                    stack.enter_context(patch_method(cls, '__init__', timed(cls.__init__)))
            cls = self.ns.get('FuzzyAttendExciteSDXLPipeline')
            if cls is not None:
                stack.enter_context(patch_method(cls, 'register_attention_control',
                                                timed(cls.register_attention_control)))
            yield

    def generate(self, seed, ledger=None):
        o, ns, torch = self.options, self.ns, self.torch
        phase = ledger.measure if ledger else lambda name: nullcontext()
        with phase('preprocess'):
            prompt, words = o['prompt'], o['words']
            sampling = effective_sampling_settings(self.method, o)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            common = dict(prompt=prompt, height=o['size'], width=o['size'],
                          num_inference_steps=sampling['steps'], guidance_scale=sampling['guidance'],
                          generator=generator)
            groups, indices = {}, []
            if self.method in ('attend', 'predicate'):
                # Use the canonical phrase tokenizer, not hard-coded prompt indices.
                groups = self.token_groups(self.base.tokenizer, prompt, words)
                indices = sorted({group[-1] for group in groups.values()})
            self.last_details = {'phrase_token_groups': groups, 'target_indices': indices,
                                 'effective_sampling_settings': sampling}
        return self._generate(seed, common, prompt, words, generator, groups, indices, phase)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        pipelines = getattr(self, 'pipelines', [])
        originals = getattr(self, 'original_processors', [])
        try:
            for index, pipeline in enumerate(pipelines):
                try:
                    if index < len(originals):
                        pipeline.unet.set_attn_processor(dict(originals[index]))
                finally:
                    pipeline.maybe_free_model_hooks()
        finally:
            self.pipelines = []
            self.base = self.refiner = None
            self.ns = {}
            self.original_processors = []


class NotebookModel(ModelSession):
    """Legacy construction delegates to the same sessions as direct generation."""
    def __init__(self, method, options):
        session = create_session(method, options)
        self.__dict__.update(session.__dict__)

    def _generate(self, *args):
        return adapter_class(self.method)._generate(self, *args)
