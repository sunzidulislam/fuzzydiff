"""GPU adapter for the snow study; delegates guidance to the canonical notebook."""
import gc
import json
from pathlib import Path
import platform


def notebook_cells():
    path = Path(__file__).resolve().parent / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            source = '\n'.join(line for line in source.splitlines()
                               if not line.lstrip().startswith(('%pip ', '!pip ')))
            yield index, compile(source, f'{path}:cell-{index}', 'exec')


def save_attention(path, store):
    import torch
    payload = {'attn_res': store.attn_res, 'token_groups': store.token_groups,
               'averages': {k: v.detach().cpu() for k, v in store.get_average_attention().items()},
               'diagnostics': store.guidance_diagnostics}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_attention(path, store_type):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    store = store_type(attn_res=payload['attn_res'])
    store.token_groups = payload['token_groups']
    store.attention_store = payload['averages']
    store.counts = {key: 1 for key in store.attention_store}
    store.guidance_diagnostics = payload['diagnostics']
    store.active = False
    return store


class SnowBackend:
    def __init__(self, design):
        self.design = design
        self.namespace = None
        self.base = self.native = self.refiner = None
        self.original_processors = None
        self.phase = None

    def _definitions(self):
        if self.namespace is not None:
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Enable GPU T4 in Kaggle settings. --plan and --artifacts work on CPU.')
        namespace = {'__name__': 'snow_notebook', 'FULL_RUN': True}
        # These numbered cells are deliberately guarded by the notebook hash in the ledger.
        # 21-23 load/setup base weights; 30 is the notebook's car demo entry point.
        for index, compiled in notebook_cells():
            if index not in (21, 22, 23, 30):
                exec(compiled, namespace)
        self.namespace = namespace

    def _load_base(self):
        if self.base is not None:
            return
        for index, compiled in notebook_cells():
            if index in (21, 22, 23):
                exec(compiled, self.namespace)
        self.base = self.namespace['model']
        self.original_processors = dict(self.base.unet.attn_processors)

    def _drop_base(self):
        if self.native is not None:
            self.native.remove_all_hooks()
            self.native = None
        if self.base is not None:
            self.base.remove_all_hooks()
            self.base.unet.set_attn_processor(dict(self.original_processors))
            self.base = None
        if self.namespace is not None:
            self.namespace.pop('model', None)
            self.namespace.pop('vae', None)
        self.original_processors = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def prepare(self, method):
        self._definitions()
        from diffusers import StableDiffusionXLPipeline
        if method == 'refined':
            self._drop_base()
            if self.refiner is None:
                self.refiner = self.namespace['load_refiner'](self.namespace['device'])
        else:
            self._load_base()
            if self.native is not None:
                self.native.remove_all_hooks()
                self.native = None
            self.base.remove_all_hooks()
            self.base.unet.set_attn_processor(dict(self.original_processors))
            if method == 'native':
                self.native = StableDiffusionXLPipeline(**self.base.components, add_watermarker=False)
                self.native.enable_model_cpu_offload(gpu_id=0)
                self.native.enable_vae_tiling()
            else:
                self.base.enable_model_cpu_offload(gpu_id=0)
        self.phase = method

    def generate(self, job, root):
        import torch
        from diffusers import DPMSolverMultistepScheduler
        from PIL import Image
        s = self.design['settings']
        torch.cuda.reset_peak_memory_stats()
        common = dict(prompt=job['prompt'], negative_prompt=s['negative_prompt'])
        metadata = {'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(),
                    'versions': {name: str(__import__(name).__version__)
                                 for name in ('torch', 'diffusers', 'transformers', 'accelerate')},
                    'diagnostics': [], 'token_groups': None}
        if job['method'] == 'native':
            self.native.scheduler = DPMSolverMultistepScheduler.from_config(self.native.scheduler.config)
            image = self.native(**common, height=s['height'], width=s['width'],
                                num_inference_steps=s['base_steps'], guidance_scale=s['guidance'],
                                generator=torch.Generator(device='cpu').manual_seed(job['seed'])).images[0]
            pipeline = self.native
        elif job['method'] == 'fuzzy':
            self.base.scheduler = DPMSolverMultistepScheduler.from_config(self.base.scheduler.config)
            image, _, store = self.namespace['generate'](
                job['prompt'], s['words_to_track'], seed=job['seed'], num_steps=s['base_steps'],
                guidance=s['guidance'], height=s['height'], width=s['width'],
                max_iter_to_alter=s['max_iter_to_alter'], attend_excite_lr=s['attend_excite_lr'],
                alpha=s['alpha'], spatial_loss_weight=s['spatial_weight'], attn_res=s['attn_res'],
                negative_prompt=s['negative_prompt'], relations=s['relations'], pipeline=self.base)
            save_attention(root / job['attention'], store)
            metadata.update(diagnostics=store.guidance_diagnostics, token_groups=store.token_groups)
            pipeline = self.base
        else:
            store = load_attention(root / job['attention'], self.namespace['AttentionStore'])
            indices = sorted({idx for group in store.token_groups.values() for idx in group})
            source = root / f"images/{job['dependency']}.png"
            with Image.open(source) as base_image:
                image = self.namespace['refine_with_predicate'](
                    base_image.convert('RGB'), job['prompt'], store, indices, seed=job['seed'],
                    negative_prompt=s['negative_prompt'], refiner=self.refiner,
                    strength=s['refiner_strength'], num_steps=s['refiner_steps'],
                    guidance=s['guidance'], predicate_strength=s['predicate_strength'])
            metadata.update(token_groups=store.token_groups,
                            refiner_token_groups=self.namespace['get_token_groups'](
                                self.refiner.tokenizer_2, job['prompt'], s['words_to_track']))
            pipeline = self.refiner
        torch.cuda.synchronize()
        metadata['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        metadata['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
        metadata['clip'] = self.namespace['clip_prompt_similarities'](image, job['prompt'])
        metadata['scheduler'] = dict(pipeline.scheduler.config)
        metadata['checkpoint_commits'] = {
            name: getattr(getattr(pipeline, name, None), 'config', {}).get('_commit_hash')
            for name in ('unet', 'vae', 'text_encoder', 'text_encoder_2')}
        return image, metadata

    def close(self):
        if self.namespace is None:
            return
        self._drop_base()
        if self.refiner is not None:
            self.refiner.remove_all_hooks()
            self.refiner = None
        self.namespace = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
