"""TurboSession adapter; delegates algorithms to its source notebook."""
from pathlib import Path
from time import perf_counter
from .session import ModelSession


class TurboSession(ModelSession):
    def _load(self, load, place, common):
        from diffusers import (AutoencoderKL, DiffusionPipeline, StableDiffusionPipeline,
                               StableDiffusionXLPipeline, DPMSolverMultistepScheduler, DDIMScheduler)
        method, options, torch = self.method, self.options, self.torch
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

    def _generate(self, context):
        o, ns = self.options, self.ns
        return self.base(**context.common, negative_prompt=o['negative']).images[0]
