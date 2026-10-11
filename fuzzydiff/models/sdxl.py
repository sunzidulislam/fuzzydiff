"""SDXLSession adapter; delegates algorithms to its source notebook."""
from pathlib import Path
from time import perf_counter
from .session import ModelSession


class SDXLSession(ModelSession):
    def _load(self, load, place, common):
        from diffusers import (AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline,
                               StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DDIMScheduler)
        method, options, torch = self.method, self.options, self.torch
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

    def _generate(self, seed, common, prompt, words, generator, groups, indices, phase):
        o, ns = self.options, self.ns
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
