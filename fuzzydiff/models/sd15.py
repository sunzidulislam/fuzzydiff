"""Stable Diffusion 1.x generation, with or without the FuzzyDiff guidance pipeline.

Stability never released open weights for Stable Diffusion 1.6; it exists only behind
their hosted API. The open 1.x line stops at 1.5, which is also the backbone the
Attend-and-Excite and Predicated Diffusion baselines use, so that is the default.
Pass --model to point at any other SD 1.x checkpoint, local path or Hub id.

Without --fuzzy this is the plain backbone. With --fuzzy the same fuzzy logic the SDXL
notebook uses is loaded from that notebook rather than reimplemented, so the two
backbones cannot drift apart; only the sampling loop differs, because SD 1.x conditions
on one text encoder and needs no added time/text embeddings.

The scheduler matches the FuzzyDiff notebook (DPMSolver++ with Karras sigmas) so images
are comparable across backbones at the same seed and step count.

    python run_sd.py --prompt "a red book and a yellow clock" --words "red book,yellow clock"
    python run_sd.py --prompt "a red book and a yellow clock" --words "red book,yellow clock" --fuzzy
    python run_sd.py --prompt "a red book and a yellow clock" --words "red,book,yellow,clock" \
        --fuzzy --bind "red>book" --bind "yellow>clock"
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

DEFAULT_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
from fuzzydiff.paths import canonical_notebook
from fuzzydiff.notebook_runtime import load_prefix_definitions

NOTEBOOK = canonical_notebook()
# Cell prefixes holding backbone-independent fuzzy logic: config, membership and truth,
# the loss, attention collection and tracked-phrase tokenizer alignment.
FUZZY_DEFINITIONS = ('@dataclass', 'def soft_truth', 'def compute_fuzzy_loss',
                     'class AttentionStore', 'class AttendExciteAttnProcessor',
                     'def get_token_groups')


def load_fuzzy_definitions():
    namespace = {'torch': torch, 'F': F, 'math': math, 'Path': Path,
                 'dataclass': dataclass, 'field': field, 'asdict': asdict,
                 'List': List, 'Dict': Dict, 'Optional': Optional, 'Union': Union}
    return load_prefix_definitions(NOTEBOOK, FUZZY_DEFINITIONS, namespace)


class FuzzyStableDiffusionPipeline(StableDiffusionPipeline):
    """Single-prompt SD 1.x sampling with fuzzy latent guidance."""
    fuzzy = None

    def register_attention_control(self, controller):
        if not hasattr(self, '_original_attention_processors'):
            self._original_attention_processors = dict(self.unet.attn_processors)
        processors = {}
        for name, original in self._original_attention_processors.items():
            place = 'mid' if name.startswith('mid') else ('up' if name.startswith('up') else 'down')
            processors[name] = (self.fuzzy['AttendExciteAttnProcessor'](controller, place, original)
                                if '.attn2.' in name else original)
        self.unet.set_attn_processor(processors)

    def _update_latents_with_fuzzy_loss(self, latents, store, cfg, timestep, prompt_embeds,
                                        relations, step_index):
        with torch.enable_grad():
            # Float32 leaf prevents small fp16 latent updates from rounding away.
            leaf = latents.detach().float().requires_grad_(True)
            store.begin_forward(keep_grad=True)
            try:
                model_input = self.scheduler.scale_model_input(leaf.to(prompt_embeds.dtype), timestep)
                self.unet(model_input, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)
                loss = self.fuzzy['compute_fuzzy_loss'](store.get_current_step_attention(),
                                                        cfg.token_groups, cfg.alpha,
                                                        cfg.spatial_loss_weight, relations,
                                                        cfg.text_span, cfg.membership_sharpness,
                                                        getattr(cfg, 't_norm', 'min'),
                                                        getattr(cfg, 'binding_loss_weight', 1.0),
                                                        getattr(cfg, 'membership_mode', 'share'),
                                                        getattr(cfg, 'hedge_targets', {}))
                if not loss.requires_grad:
                    raise RuntimeError('Fuzzy loss is disconnected from the latent graph.')
                gradient = torch.autograd.grad(loss, leaf)[0]
                if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite fuzzy loss or gradient; lower --lr.')
                rms = gradient.square().mean().sqrt()
                if rms.item() == 0:
                    raise RuntimeError('Fuzzy latent gradient is zero.')
                rate = cfg.attend_excite_lr * math.sqrt(max(0.0, 1 - step_index / cfg.n_inference_steps))
                delta_tensor = rate * gradient
                delta_rms = delta_tensor.square().mean().sqrt()
                # Clip large updates, but NEVER amplify small ones.
                ceiling = (0.01 / delta_rms.clamp_min(1e-12)).clamp(max=1.0)
                updated = (leaf - delta_tensor * ceiling).detach()
                delta = (updated - latents.float()).square().mean().sqrt().item()
                latent_rms = latents.float().square().mean().sqrt().item()
                store.guidance_diagnostics.append(
                    {'step': step_index, 'loss': loss.item(), 'gradient_rms': rms.item(),
                     'update_rms': delta, 'learning_rate': rate,
                     'clipped': bool(ceiling.item() < 1.0), 'latent_rms': latent_rms,
                     'update_ratio': delta / max(latent_rms, 1e-12)})
                return updated
            finally:
                store.end_forward(accumulate=False)

    @torch.no_grad()
    def sample(self, cfg, store, relations=(), negative_prompt=''):
        device = self._execution_device
        do_cfg = cfg.guidance_scale > 1.0
        store.reset()
        positive, negative = self.encode_prompt(cfg.prompt, device, 1, do_cfg,
                                                negative_prompt=negative_prompt or '')
        embeddings = torch.cat([negative, positive]) if do_cfg else positive
        self.scheduler.set_timesteps(cfg.n_inference_steps, device=device)
        generator = torch.Generator(device='cpu').manual_seed(cfg.seeds[0])
        latents = self.prepare_latents(1, self.unet.config.in_channels, cfg.height, cfg.width,
                                       positive.dtype, device, generator, None).float()
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
        try:
            for i, timestep in enumerate(self.scheduler.timesteps):
                if cfg.token_groups and i < cfg.max_iter_to_alter and cfg.attend_excite_lr > 0:
                    latents = self._update_latents_with_fuzzy_loss(
                        latents, store, cfg, timestep, positive, relations, i)
                # Predict noise from UPDATED latents at the SAME timestep.
                model_input = torch.cat([latents, latents]) if do_cfg else latents
                model_input = self.scheduler.scale_model_input(model_input, timestep)
                store.begin_forward(cfg=do_cfg)
                noise = self.unet(model_input.to(embeddings.dtype), timestep,
                                  encoder_hidden_states=embeddings, return_dict=False)[0]
                store.end_forward()
                if do_cfg:
                    unconditional, conditional = noise.chunk(2)
                    noise = unconditional + cfg.guidance_scale * (conditional - unconditional)
                latents = self.scheduler.step(noise.float(), timestep, latents, **extra_step_kwargs).prev_sample
            decode_dtype = next(self.vae.post_quant_conv.parameters()).dtype
            decoded = self.vae.decode(latents.to(decode_dtype) / self.vae.config.scaling_factor,
                                      return_dict=False)[0]
            return self.image_processor.postprocess(decoded, output_type='pil')[0]
        finally:
            store.end_forward(accumulate=False)
            self.maybe_free_model_hooks()


def configure(pipeline, device, fuzzy=None):
    pipeline.fuzzy = fuzzy
    for component in (pipeline.unet, pipeline.vae, pipeline.text_encoder):
        component.eval().requires_grad_(False)
    if fuzzy is not None:
        pipeline.unet.enable_gradient_checkpointing()
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config, algorithm_type='dpmsolver++', use_karras_sigmas=True)
    if device.type == 'cuda':
        pipeline.enable_model_cpu_offload(gpu_id=0)
    else:
        pipeline.to(device)
    return pipeline


def build_pipeline(model_id, device, fuzzy=None):
    """fuzzy=None loads the plain backbone; a definitions namespace enables guidance."""
    # fp16 has no CPU kernels, so a CPU run must stay in fp32.
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    pipeline_class = StableDiffusionPipeline if fuzzy is None else FuzzyStableDiffusionPipeline
    pipeline = pipeline_class.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    return configure(pipeline, device, fuzzy)


def build_config(fuzzy, tokenizer, prompt, words, seed, output_path, steps=50, guidance=7.5,
                 size=512, learning_rate=0.2, updates=25, sharpness=100.0, t_norm='min',
                 membership_mode='share', hedge_guidance=False):
    cfg = fuzzy['RunConfig'](prompt=prompt, seeds=[seed], n_inference_steps=steps,
                             guidance_scale=guidance, height=size, width=size,
                             max_iter_to_alter=updates, attend_excite_lr=learning_rate,
                             membership_sharpness=sharpness, t_norm=t_norm, membership_mode=membership_mode,
                             output_path=Path(output_path), hedge_guidance=hedge_guidance)
    groups = fuzzy['get_token_groups'](tokenizer, prompt, words)
    # Membership is a phrase's share of the prompt's own tokens, so the start/end
    # markers and the padding tail are excluded from the distribution.
    cfg.text_span = (1, len(tokenizer.encode(prompt)) - 1)
    cfg.token_groups = [groups[word] for word in words]
    cfg.token_indices = sorted({index for group in groups.values() for index in group})
    return cfg, groups


# Mirror of the notebook's RELATION_VERBS, needed before the notebook is loaded.
# tests/test_sd_fuzzy.py asserts this stays in step with the canonical list.
HEDGES = ('slightly', 'somewhat', 'relatively', 'moderately', 'considerably', 'much')
RELATION_VERBS = (('left_of', 'right_of', 'above', 'below', 'bound_to',
                   'larger_than', 'smaller_than')
                  + tuple(f'{hedge}_{direction}_than'
                          for hedge in HEDGES for direction in ('larger', 'smaller')))


def parse_relations(values):
    """Parse --relate "phrase:verb:phrase" arguments."""
    relations = []
    for value in values or ():
        parts = [part.strip() for part in value.split(':')]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f'--relate expects "phrase:verb:phrase", got {value!r}.')
        subject, verb, obj = parts
        if verb not in RELATION_VERBS:
            raise ValueError(f'Unsupported relation {verb!r}; use one of {", ".join(RELATION_VERBS)}.')
        relations.append((subject, verb, obj))
    return relations


def resolve_relations(groups, relations):
    """Turn phrase-name relations into token-group relations."""
    resolved = []
    for subject, verb, obj in relations:
        for phrase in (subject, obj):
            if phrase not in groups:
                raise ValueError(f'Relation operand {phrase!r} is not a tracked phrase; add it to --words.')
        resolved.append((groups[subject], verb, groups[obj]))
    return resolved


def resolve_bindings(groups, bindings):
    """Turn ('attribute', 'object') phrase pairs into token-group relations."""
    return resolve_relations(groups, [(attribute, 'bound_to', obj) for attribute, obj in bindings])


def parse_bindings(values):
    """Parse --bind "attribute>object" arguments."""
    bindings = []
    for value in values or ():
        attribute, separator, obj = value.partition('>')
        if not separator or not attribute.strip() or not obj.strip():
            raise ValueError(f'--bind expects "attribute>object", got {value!r}.')
        bindings.append((attribute.strip(), obj.strip()))
    return bindings


def generate_fuzzy(pipeline, fuzzy, cfg, groups, negative_prompt='', relations=()):
    if cfg.hedge_guidance:
        # Relations arrive resolved; match token groups back to their phrase names.
        names = {tuple(tokens): phrase for phrase, tokens in groups.items()}
        named_relations = [(names[tuple(a)], verb, names[tuple(b)]) for a, verb, b in relations]
        cfg.hedge_targets = fuzzy['resolve_hedge_targets'](groups, named_relations)
        if not cfg.hedge_targets:
            raise ValueError('No supported hedge prefix found in tracked attribute phrases.')
    else:
        cfg.hedge_targets = {}
    store = fuzzy['AttentionStore'](attn_res=cfg.attn_res)
    store.hedge_targets = cfg.hedge_targets
    store.token_groups = groups
    store.text_span = cfg.text_span
    store.membership_mode = cfg.membership_mode
    pipeline.register_attention_control(store)
    return pipeline.sample(cfg, store, relations=relations, negative_prompt=negative_prompt), store


def generate_plain(pipeline, prompt, seed, steps=50, guidance=7.5, size=512, negative_prompt=''):
    return pipeline(prompt=prompt, negative_prompt=negative_prompt, height=size, width=size,
                    num_inference_steps=steps, guidance_scale=guidance,
                    generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]


def missing_words(tokenizer, prompt, words):
    """A phrase the tokenizer cannot find is almost always a typo, not a model failure."""
    ids = tokenizer.encode(prompt)
    missing = []
    for word in words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        spans = (ids[start:start + len(word_ids)] for start in range(len(ids) - len(word_ids) + 1))
        if not word_ids or word_ids not in spans:
            missing.append(word)
    return missing


def main(argv=None):
    from fuzzydiff.cli.sd15 import main as entry
    return entry(argv)


from .session import ModelSession


class PlainSD15Session(ModelSession):
    def __init__(self, method, options):
        self.method, self.options = method, options
        self.torch = torch
        self.pipelines, self.ns, self.last_details = [], {}, {}
        self.original_processors = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.base = build_pipeline(DEFAULT_MODEL, self.device)
        self.pipelines = [self.base]
        self.original_processors = [dict(self.base.unet.attn_processors)]

    def generate(self, seed, ledger=None):
        o = self.options
        return generate_plain(self.base, o['prompt'], seed, o['steps'], o['guidance'], o['size'], o['negative'])

    def metadata(self):
        return {'checkpoints': [DEFAULT_MODEL], 'dtype': str(self.base.unet.dtype),
                'effective_sampling_settings': {k: self.options[k] for k in ('steps', 'guidance', 'size')},
                'schedulers': [type(self.base.scheduler).__name__], 'limitations': []}
