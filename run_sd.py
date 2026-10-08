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
NOTEBOOK = Path(__file__).resolve().parent / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb'
# Cell prefixes holding backbone-independent fuzzy logic: config, membership and truth,
# the loss, attention collection and tracked-phrase tokenizer alignment.
FUZZY_DEFINITIONS = ('@dataclass', 'def soft_truth', 'def compute_fuzzy_loss',
                     'class AttentionStore', 'class AttendExciteAttnProcessor',
                     'def get_token_groups')


def load_fuzzy_definitions():
    namespace = {'torch': torch, 'F': F, 'math': math, 'Path': Path,
                 'dataclass': dataclass, 'field': field, 'asdict': asdict,
                 'List': List, 'Dict': Dict, 'Optional': Optional, 'Union': Union}
    for cell in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']:
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        if cell['cell_type'] == 'code' and source.startswith(FUZZY_DEFINITIONS):
            exec(compile(source, str(NOTEBOOK), 'exec'), namespace)
    return namespace


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
                                                        getattr(cfg, 'membership_mode', 'share'))
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
                 membership_mode='share'):
    cfg = fuzzy['RunConfig'](prompt=prompt, seeds=[seed], n_inference_steps=steps,
                             guidance_scale=guidance, height=size, width=size,
                             max_iter_to_alter=updates, attend_excite_lr=learning_rate,
                             membership_sharpness=sharpness, t_norm=t_norm, membership_mode=membership_mode,
                             output_path=Path(output_path))
    groups = fuzzy['get_token_groups'](tokenizer, prompt, words)
    # Membership is a phrase's share of the prompt's own tokens, so the start/end
    # markers and the padding tail are excluded from the distribution.
    cfg.text_span = (1, len(tokenizer.encode(prompt)) - 1)
    cfg.token_groups = [groups[word] for word in words]
    cfg.token_indices = sorted({index for group in groups.values() for index in group})
    return cfg, groups


def resolve_bindings(groups, bindings):
    """Turn ('attribute', 'object') phrase pairs into token-group relations."""
    relations = []
    for attribute, obj in bindings:
        for phrase in (attribute, obj):
            if phrase not in groups:
                raise ValueError(f'Binding operand {phrase!r} is not a tracked phrase; add it to --words.')
        relations.append((groups[attribute], 'bound_to', groups[obj]))
    return relations


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
    store = fuzzy['AttentionStore'](attn_res=cfg.attn_res)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prompt', required=True, help='Text prompt.')
    parser.add_argument('--words', default='',
                        help='Comma-separated object words/phrases, each appearing verbatim in the prompt.')
    parser.add_argument('--fuzzy', action='store_true', help='Apply FuzzyDiff guidance; requires --words.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='SD 1.x checkpoint (local path or Hub id).')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.5, help='Classifier-free guidance scale.')
    parser.add_argument('--size', type=int, default=512, help='Square image size; SD 1.x is trained at 512.')
    parser.add_argument('--lr', type=float, default=0.2, help='Fuzzy guidance step size.')
    parser.add_argument('--updates', type=int, default=25, help='Number of guided denoising steps.')
    parser.add_argument('--sharpness', type=float, default=100.0,
                        help='Membership sharpness; lower keeps truths graded instead of near-binary.')
    parser.add_argument('--membership-mode', choices=('share', 'relative'), default='share',
                        help='Experimental relative mode normalizes each phrase by its own spatial peak.')
    parser.add_argument('--tnorm', choices=('min', 'product'), default='min',
                        help="Fuzzy conjunction: 'min' gradients only the weakest phrase, 'product' all of them.")
    parser.add_argument('--bind', action='append', metavar='ATTR>OBJECT',
                        help='Require an attribute to hold where an object is, e.g. '
                             '--bind "yellow>clock". Both sides must be tracked phrases.')
    parser.add_argument('--binding-weight', type=float, default=1.0,
                        help='Weight of each binding conjunct in the loss.')
    parser.add_argument('--negative', default='', help='Negative prompt.')
    parser.add_argument('--output', default=None, help='Output directory.')
    options = parser.parse_args()

    words = [word.strip() for word in options.words.split(',') if word.strip()]
    if options.fuzzy and not words:
        parser.error('--fuzzy needs --words: there is nothing to ground without tracked phrases.')
    output = Path(options.output) if options.output else (
        (Path('/kaggle/working/outputs') if Path('/kaggle/working').exists()
         else Path('./outputs')) / ('sd_fuzzy' if options.fuzzy else 'sd_baseline'))
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('No GPU found; running on CPU in float32. This is slow.', flush=True)

    fuzzy = load_fuzzy_definitions() if options.fuzzy else None
    print(f'Loading {options.model} ...', flush=True)
    pipeline = build_pipeline(options.model, device, fuzzy)
    absent = missing_words(pipeline.tokenizer, options.prompt, words)
    if absent:
        print('Warning: these object words are not in the prompt:', absent, flush=True)

    truths, diagnostics, settings = None, [], {}
    if options.fuzzy:
        cfg, groups = build_config(fuzzy, pipeline.tokenizer, options.prompt, words, options.seed,
                                   output, options.steps, options.guidance, options.size,
                                   options.lr, options.updates, options.sharpness, options.tnorm,
                                   options.membership_mode)
        cfg.binding_loss_weight = options.binding_weight
        relations = resolve_bindings(groups, parse_bindings(options.bind))
        print('Tracked phrase tokens:', groups, flush=True)
        if relations:
            print('Bindings:', parse_bindings(options.bind), flush=True)
        image, store = generate_fuzzy(pipeline, fuzzy, cfg, groups, options.negative, relations)
        truths = fuzzy['phrase_truth_scores'](store, cfg.alpha, cfg.membership_sharpness)
        diagnostics = store.guidance_diagnostics
        settings = {**asdict(cfg), 'output_path': str(cfg.output_path)}
    else:
        image = generate_plain(pipeline, options.prompt, options.seed, options.steps,
                               options.guidance, options.size, options.negative)
        settings = {'steps': options.steps, 'guidance_scale': options.guidance,
                    'height': options.size, 'width': options.size}

    stem = f'seed_{options.seed}'
    image_path = output / f'{stem}.png'
    image.save(image_path)
    (output / f'{stem}_metadata.json').write_text(json.dumps({
        'backbone': options.model,
        'method': 'FuzzyDiff guidance' if options.fuzzy else 'plain Stable Diffusion, no fuzzy guidance',
        'prompt': options.prompt, 'object_words': words, 'words_not_in_prompt': absent,
        'seed': options.seed, 'negative_prompt': options.negative,
        'scheduler': 'DPMSolverMultistep dpmsolver++ karras', 'settings': settings,
        'phrase_truth': truths, 'guidance_diagnostics': diagnostics, 'image': str(image_path),
        'truth_note': 'Attention-derived objective, not visual accuracy or attribute intensity.',
        'versions': {'torch': torch.__version__, 'diffusers': __import__('diffusers').__version__},
    }, indent=2), encoding='utf-8')
    if truths:
        print('Phrase presence truth:', truths, flush=True)
    if diagnostics:
        print('Guidance:', fuzzy['summarize_guidance'](diagnostics), flush=True)
    print('Saved:', image_path.resolve(), flush=True)


if __name__ == '__main__':
    main()
