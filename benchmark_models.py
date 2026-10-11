"""Read-only notebook adapters. Generation algorithms stay in their notebooks.

The archived implementations are exploratory baselines, not validated reproductions.
All adapters use CPU model offload and explicit common size/step/CFG settings.
"""
import ast
import hashlib
import json
import sys
import types
from contextlib import nullcontext, contextmanager, ExitStack
from functools import wraps
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parent
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


def component_commit_hash(component):
    config = getattr(component, 'config', None)
    if isinstance(config, dict):
        return config.get('_commit_hash')
    return getattr(config, '_commit_hash', None)


def definition_code(path):
    """Select definitions/imports, never notebook model loads, batches or shell cells."""
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    body = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        source = '\n'.join(line for line in source.splitlines()
                           if not line.lstrip().startswith(('!', '%')))
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                body.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                roots = ([name.name.split('.')[0] for name in node.names]
                         if isinstance(node, ast.Import) else [node.module.split('.')[0]])
                # Only used by notebook display, evaluation or unused helpers.
                if not set(roots) & {'clip', 'cv2', 'IPython', 'pandas'}:
                    body.append(node)
            elif isinstance(node, ast.Assign) and all(
                    isinstance(target, ast.Name) and target.id in
                    {'HEDGE_TARGETS', 'RELATION_VERBS'} for target in node.targets):
                body.append(node)
    return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                   str(path), 'exec')


def load_definitions(path):
    # dataclasses resolves annotations through the defining module.
    name = '_benchmark_' + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(definition_code(path), module.__dict__)
    return module.__dict__


class NotebookModel:
    def __init__(self, method, options):
        import torch
        from diffusers import (AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline,
                               StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DDIMScheduler)
        self.torch = torch
        self.method, self.options = method, options
        self.device = torch.device('cuda:0')
        self.pipelines = []
        self.load_parts = {}
        self.last_details = {}
        start = perf_counter()
        self.ns = load_definitions(NOTEBOOKS[method])
        if method in ('attend', 'predicate'):
            from run_sd import load_fuzzy_definitions
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
            return pipeline

        if method in ('stable', 'attend', 'predicate'):
            cls = (StableDiffusionPipeline if method == 'stable' else
                   self.ns['AttendAndExcitePipeline' if method == 'attend'
                           else 'PredicatedDiffPipeline'])
            checkpoint = ('stable-diffusion-v1-5/stable-diffusion-v1-5'
                          if method == 'attend' else 'CompVis/stable-diffusion-v1-4')
            dtype = torch.float32 if method == 'predicate' else torch.float16
            self.base = load('base', lambda: place(cls.from_pretrained(
                checkpoint, torch_dtype=dtype, safety_checker=None,
                requires_safety_checker=False, **common)))
            start = perf_counter()
            if method == 'attend':
                self.base.scheduler = DDIMScheduler.from_config(self.base.scheduler.config)
                self.base.enable_attention_slicing(1)
            elif method == 'stable':
                self.base.enable_attention_slicing()
            self.base.enable_vae_slicing()
            torch.cuda.synchronize()
            self.load_parts['configuration'] = perf_counter() - start
            self.pipelines = [self.base]
            self.checkpoints = [checkpoint]
        elif method == 'enhancing':
            self.base = load('base', lambda: place(StableDiffusionXLPipeline.from_pretrained(
                'stabilityai/sdxl-turbo', torch_dtype=torch.float16,
                variant='fp16', **common)))
            start = perf_counter()
            self.base.enable_vae_slicing()
            self.base.enable_vae_tiling()
            torch.cuda.synchronize()
            self.load_parts['configuration'] = perf_counter() - start
            self.pipelines = [self.base]
            self.checkpoints = ['stabilityai/sdxl-turbo']
        else:
            vae = load('vae', lambda: AutoencoderKL.from_pretrained(
                'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16, **common))
            cls = (self.ns['FuzzyAttendExciteSDXLPipeline'] if method == 'fuzzydiff'
                   else DiffusionPipeline)
            self.base = load('base', lambda: place(cls.from_pretrained(
                'stabilityai/stable-diffusion-xl-base-1.0', vae=vae,
                torch_dtype=torch.float16, variant='fp16', use_safetensors=True, **common)))
            # Separate VAE objects avoid conflicting Accelerate offload hooks.
            refiner_options = dict(common)
            if method == 'aesthetic':
                refiner_options['vae'] = load('refiner_vae', lambda: AutoencoderKL.from_pretrained(
                    'madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16, **common))
            self.refiner = load('refiner', lambda: place(DiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-xl-refiner-1.0', torch_dtype=torch.float16,
                variant='fp16', use_safetensors=True, **refiner_options)))
            start = perf_counter()
            if method == 'fuzzydiff':
                for component in (self.base.unet, self.base.vae, self.base.text_encoder,
                                  self.base.text_encoder_2, self.refiner.unet):
                    component.eval().requires_grad_(False)
                self.base.unet.enable_gradient_checkpointing()
                for pipeline in (self.base, self.refiner):
                    pipeline.enable_vae_tiling()
                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, algorithm_type='dpmsolver++',
                        use_karras_sigmas=True)
            else:
                self.base.enable_attention_slicing()
            torch.cuda.synchronize()
            self.load_parts['configuration'] = perf_counter() - start
            self.pipelines = [self.base, self.refiner]
            self.checkpoints = ['stabilityai/stable-diffusion-xl-base-1.0',
                                'stabilityai/stable-diffusion-xl-refiner-1.0',
                                'madebyollin/sdxl-vae-fp16-fix']
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
        from benchmark_timing import patch_method

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
        if self.method in ('stable', 'enhancing'):
            return self.base(**common, negative_prompt=o['negative']).images[0]
        if self.method == 'attend':
            with phase('preprocess'):
                store = ns['AttentionStore'](attention_res=16)
                ns['register_attention_control'](self.base, store)
            images = self.base(**common, attention_store=store, indices_to_alter=indices,
                               max_iter_to_alter=25, negative_prompt=o['negative'])
            return images[0]
        if self.method == 'predicate':
            with phase('preprocess'):
                cfg = ns['RunConfig'](
                    prompt=prompt, seeds=[seed], output_path=Path(o['output']) / 'predicate',
                    n_inference_steps=o['steps'], guidance_scale=o['guidance'],
                    attention_exist_indices=indices,
                    attention_corr_indices=[[groups[a][-1], groups[b][-1]]
                                            for a, b in o['bindings']])
                store = ns['AttentionStore']()
                ns['register_attention_control'](self.base, store)
            # Same algorithm as run_on_prompt; explicit size replaces hidden defaults.
            result, losses = self.base(
                **common, neg_prompt=cfg.neg_prompt, attention_store=store,
                indices_to_alter=indices, attention_res=cfg.attention_res,
                max_iter_to_alter=cfg.max_iter_to_alter, run_attention_sd=cfg.run_attention_sd,
                thresholds=cfg.thresholds, scale_factor=cfg.scale_factor,
                scale_range=cfg.scale_range, attention_corr_indices=cfg.attention_corr_indices,
                attention_leak_indices=cfg.attention_leak_indices,
                attention_exist_indices=cfg.attention_exist_indices,
                attention_possession_indices=cfg.attention_possession_indices)
            self.last_details['loss_values'] = losses
            self.base.maybe_free_model_hooks()
            return result.images[0]
        if self.method == 'aesthetic':
            latent = self.base(**common, negative_prompt=o['negative'],
                               denoising_end=0.4, output_type='latent').images
            self.base.maybe_free_model_hooks()
            return self.refiner(prompt=prompt, image=latent, denoising_start=0.4,
                                num_inference_steps=o['steps'], guidance_scale=o['guidance'],
                                generator=generator).images[0]
        # Instrument orchestration independently of the fuzzy objective in the notebook.
        image, _, store = ns['generate'](
            prompt, words, seed=seed, num_steps=o['steps'], guidance=o['guidance'],
            height=o['size'], width=o['size'], max_iter_to_alter=o['updates'],
            attend_excite_lr=o['lr'], membership_sharpness=o['sharpness'],
            t_norm=o['tnorm'], membership_mode=o['membership_mode'],
            negative_prompt=o['negative'], relations=[(a, 'bound_to', b) for a, b in o['bindings']],
            pipeline=self.base)
        self.base.maybe_free_model_hooks()
        self.last_details['guidance_diagnostics'] = list(store.guidance_diagnostics)
        return ns['refine_with_predicate'](
            image, prompt, store, seed=seed, negative_prompt=o['negative'],
            refiner=self.refiner, strength=o['refine_strength'], num_steps=o['refine_steps'],
            guidance=o['guidance'], predicate_strength=o['predicate_strength'],
            membership_sharpness=o['sharpness'], membership_mode=o['membership_mode'])
