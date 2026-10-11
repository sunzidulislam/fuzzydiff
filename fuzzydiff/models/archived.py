"""ArchivedSession adapter; delegates algorithms to its source notebook."""
from pathlib import Path
from time import perf_counter
from .session import ModelSession


class ArchivedSession(ModelSession):
    def _load(self, load, place, common):
        from diffusers import (AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline,
                               StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DDIMScheduler)
        method, options, torch = self.method, self.options, self.torch
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

    def _generate(self, seed, common, prompt, words, generator, groups, indices, phase):
        o, ns = self.options, self.ns
        if self.method == 'stable':
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
