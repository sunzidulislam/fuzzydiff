"""Model access for FuzzyDiff: conditioning, guided velocity, clean prediction, decode, encode.

Guidance and SAR depend only on `Backend`, so the CPU tests run them on `tiny_backend()` (random
weights, no downloads). The CFG batch is [cond, uncond], so the conditional branch is batch 0.

diffusers 0.31.0 checkpoints transformer blocks only when the module's own `training` flag is set.
`_enable_checkpointing` sets that flag on the top-level module only; every submodule stays in eval
mode, so dropout and normalization behave as in inference.
"""
from __future__ import annotations

import gc
import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzydiff_sd3_attention import FuzzyJointAttnProcessor  # noqa: E402
from grounding import CLIP_TOKENS, map_tokens  # noqa: E402

T5_TOKENS = 256


@dataclass
class TextCond:
    embeds: torch.Tensor
    pooled: torch.Tensor
    uncond_embeds: torch.Tensor
    uncond_pooled: torch.Tensor
    valid_text: List[int]
    phrase_tokens: Dict[str, List[int]]
    t5_offsets: List[Tuple[int, int]]
    unmapped: List[str] = field(default_factory=list)


@dataclass
class VelocityOut:
    v: torch.Tensor
    v_cond: torch.Tensor
    v_uncond: torch.Tensor
    maps: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    shares: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    text_mass: Dict[int, torch.Tensor] = field(default_factory=dict)


def checkpoint_vae_resnets(vae):
    """Recompute one decoder resnet at a time in the backward pass. Per-block checkpointing rebuilds
    the last up block whole, about 8 GiB at 1024^2 in 16 bits and twice that in float32 (measured
    on a T4), which does not fit on the GPU."""
    for module in vae.decoder.modules():
        if isinstance(module, ResnetBlock2D):
            module.forward = _checkpointed(module.forward)


def _checkpointed(forward):
    def run(hidden_states, temb=None):
        if not torch.is_grad_enabled():
            return forward(hidden_states, temb)
        return torch.utils.checkpoint.checkpoint(forward, hidden_states, temb, use_reentrant=False)
    return run


def _enable_checkpointing(transformer, vae):
    transformer.enable_gradient_checkpointing()
    transformer.training = True
    checkpoint_vae_resnets(vae)


class Backend:
    def __init__(self, transformer, vae, cfg: float, shift: float, read_blocks: Sequence[int],
                 sar_blocks: Sequence[int], tf_device, vae_device, dtype, latent_size: int,
                 encoder=None, defer_transformer=False):
        self.transformer, self.vae, self.cfg, self.shift = transformer.eval(), vae.eval(), cfg, shift
        self.read_blocks, self.sar_blocks = list(read_blocks), list(sar_blocks)
        self.tf_device, self.vae_device, self.dtype = torch.device(tf_device), torch.device(vae_device), dtype
        # 16-bit transformer weights; latents, estimators, losses, updates, and the VAE stay wider.
        self.latent_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self.vae_dtype = next(vae.parameters()).dtype
        self.latent_size = latent_size
        self._encoder = encoder
        self._deferred = defer_transformer  # T5 and the transformer do not fit on one T4 together
        self._text_cache: Dict[Tuple, TextCond] = {}
        for module in (transformer, vae):
            module.requires_grad_(False)
        _enable_checkpointing(transformer, vae)
        names = list(transformer.attn_processors.keys())
        self.procs = {int(n.split(".")[1]): FuzzyJointAttnProcessor() for n in names}
        transformer.set_attn_processor({n: self.procs[int(n.split(".")[1])] for n in names})
        cc = transformer.config
        self.n_img = (latent_size // cc.patch_size) ** 2
        self.in_channels = cc.in_channels
        self.n_text: Optional[int] = None

    # ------------------------------------------------------------------------------ schedule
    def sigmas(self, steps: int) -> torch.Tensor:
        s = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=self.shift)
        s.set_timesteps(steps)
        return s.sigmas.detach().cpu().to(torch.float64)

    def noise_like_latent(self, seed: int) -> torch.Tensor:
        g = torch.Generator("cpu").manual_seed(int(seed))
        z = torch.randn(1, self.in_channels, self.latent_size, self.latent_size, generator=g, dtype=self.latent_dtype)
        return z.to(self.tf_device)

    # ------------------------------------------------------------------------------ text
    def encode_prompt(self, prompt: str, phrases: Optional[Dict[str, Tuple[int, int]]] = None) -> TextCond:
        phrases = phrases or {}
        key = (prompt, tuple(sorted(phrases.items())))
        if key not in self._text_cache:
            if self._encoder is None:
                raise RuntimeError(f"prompt not cached and text encoders are unloaded: {prompt!r}")
            cond = self._encoder(prompt)
            cond.phrase_tokens, cond.unmapped = map_tokens(cond.t5_offsets, phrases, T5_TOKENS)
            self._text_cache[key] = cond
            self.n_text = cond.embeds.shape[1]
        return self._text_cache[key]

    def unload_text_encoders(self) -> None:
        self._encoder = None
        if hasattr(self, "_pipe"):
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                setattr(self._pipe, name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._deferred:
            self.transformer.to(self.tf_device)
            self._deferred = False

    # ------------------------------------------------------------------------------ velocity
    def _timestep(self, sigma, n):
        """Computed wide: a 16-bit sigma would move the timestep by up to about 2 of 1000."""
        return (sigma.to(self.tf_device, self.latent_dtype) * 1000).reshape(1).repeat(n)

    def _forward(self, z, sigma, embeds, pooled):
        if self._deferred:
            raise RuntimeError("encode all prompts, then call unload_text_encoders, before running the transformer")
        t = self._timestep(sigma, z.shape[0])
        return self.transformer(hidden_states=z, encoder_hidden_states=embeds, pooled_projections=pooled,
                                timestep=t, return_dict=False)[0]

    def _configure(self, readout=None, valid_text=None, sar_log_r=None):
        for i, p in self.procs.items():
            p.readout = readout if (readout and i in self.read_blocks) else None
            p.valid_text = list(valid_text) if (p.readout and valid_text is not None) else None
            p.readout_batch = 0
            p.sar_log_r = sar_log_r if (sar_log_r is not None and i in self.sar_blocks) else None
            p.sar_batch = [0]
            p.maps, p.shares, p.text_mass = {}, {}, None

    def velocity(self, z, sigma, cond: TextCond, readout: bool = False, cfg: Optional[float] = None) -> VelocityOut:
        """Guided velocity v_g at z. Gradients follow the caller's grad mode.

        Checkpointed blocks rerun their processors during backward, so the processor state set here
        stays installed until the next call: run backward passes before the next forward pass."""
        w = self.cfg if cfg is None else cfg
        self._configure(readout=cond.phrase_tokens if readout else None, valid_text=cond.valid_text)
        v = self._forward(z.to(self.tf_device, self.dtype).repeat(2, 1, 1, 1), sigma,
                          torch.cat([cond.embeds, cond.uncond_embeds]),
                          torch.cat([cond.pooled, cond.uncond_pooled]))
        v = v.to(z.device, z.dtype)
        v_cond, v_uncond = v[:1], v[1:]
        out = VelocityOut(v=v_uncond + w * (v_cond - v_uncond), v_cond=v_cond, v_uncond=v_uncond)
        if readout and cond.phrase_tokens:
            for i in self.read_blocks:
                p = self.procs[i]
                for q in cond.phrase_tokens:
                    out.maps.setdefault(q, {})[i] = p.maps[q]
                    out.shares.setdefault(q, {})[i] = p.shares[q]
                out.text_mass[i] = p.text_mass
        return out

    def velocity_cond(self, z, sigma, cond: TextCond, sar_log_r: torch.Tensor) -> torch.Tensor:
        """Conditional branch only, with the SAR logit bias (paper App. B, pass 2 of a Stage-2 step)."""
        self._configure(sar_log_r=sar_log_r.to(self.tf_device, self.dtype))
        return self._forward(z.to(self.tf_device, self.dtype), sigma, cond.embeds, cond.pooled).to(z.device, z.dtype)

    @staticmethod
    def clean(z, sigma, v):
        return z - sigma.to(z.device, z.dtype) * v

    # ------------------------------------------------------------------------------ VAE
    def decode(self, z0) -> torch.Tensor:
        """[1, 3, H, W] in [0, 1]; the clamp passes gradients straight through."""
        c = self.vae.config
        x = self.vae.decode(z0.to(self.vae_device, self.vae_dtype) / c.scaling_factor + c.shift_factor).sample
        x = ((x + 1) / 2).to(self.latent_dtype)
        return x + (x.clamp(0, 1) - x).detach()

    def encode(self, x) -> torch.Tensor:
        """Deterministic (posterior mode), so every Stage-2 arm starts from the same latent."""
        c = self.vae.config
        h = self.vae.encode(2 * x.to(self.vae_device, self.vae_dtype) - 1).latent_dist.mode()
        return ((h - c.shift_factor) * c.scaling_factor).to(self.tf_device, self.latent_dtype)


# ---------------------------------------------------------------------------------- real checkpoint
def _pipeline_encoder(pipe, device):
    def encode(prompt: str) -> TextCond:
        with torch.no_grad():
            embeds, uncond, pooled, uncond_pooled = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt="", device=torch.device(device),
                do_classifier_free_guidance=True, max_sequence_length=T5_TOKENS)
        t5 = pipe.tokenizer_3(prompt, padding="max_length", max_length=T5_TOKENS, truncation=True,
                              return_offsets_mapping=True)
        clip = pipe.tokenizer(prompt, padding="max_length", max_length=CLIP_TOKENS, truncation=True)
        valid = [j for j, m in enumerate(clip["attention_mask"]) if m]
        valid += [CLIP_TOKENS + j for j, m in enumerate(t5["attention_mask"]) if m]
        return TextCond(embeds, pooled, uncond, uncond_pooled, valid, {}, [tuple(o) for o in t5["offset_mapping"]])
    return encode


def load_backend(model_id, read_blocks, sar_blocks=None, tf_device="cuda:0", vae_device=None, dtype=torch.float16,
                 cfg=7.0, resolution=1024) -> Backend:
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    vae_device = vae_device or tf_device
    pipe.vae.to(vae_device, torch.float32)
    for enc in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        enc.to(tf_device)
    backend = Backend(pipe.transformer, pipe.vae, cfg, pipe.scheduler.config.shift, read_blocks,
                      sar_blocks if sar_blocks is not None else read_blocks, tf_device, vae_device, dtype,
                      resolution // pipe.vae_scale_factor, encoder=_pipeline_encoder(pipe, tf_device),
                      defer_transformer=True)
    backend._pipe = pipe
    return backend


# ---------------------------------------------------------------------------------- tiny test double
def _tiny_encoder(dt, n_t5=8):
    def rand(text, shape):
        seed = int(hashlib.sha256(f"{text}|{shape}".encode()).hexdigest()[:8], 16)
        return torch.randn(*shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=dt)

    def encode(prompt: str) -> TextCond:
        offsets, pos = [], 0
        for word in prompt.split():
            a = prompt.index(word, pos)
            pos = a + len(word)
            offsets.append((a, pos))
        offsets = offsets[:n_t5] + [(0, 0)] * max(0, n_t5 - len(offsets))
        n = CLIP_TOKENS + n_t5
        valid = list(range(4)) + [CLIP_TOKENS + j for j, (a, b) in enumerate(offsets) if b > a]
        return TextCond(rand(prompt, (1, n, 32)), rand(prompt, (1, 16)), rand("", (1, n, 32)), rand("", (1, 16)),
                        valid, {}, offsets)
    return encode


def tiny_backend(seed: int = 0, read_blocks=(0, 1), sar_blocks=None, defer_transformer=False, vae_dtype=None) -> Backend:
    """Random 3-block transformer and 4-level VAE on CPU in float64; 16 latent -> 128 pixels."""
    torch.manual_seed(seed)
    dt = torch.float64
    tr = SD3Transformer2DModel(sample_size=16, patch_size=2, in_channels=4, num_layers=3, attention_head_dim=8,
                               num_attention_heads=4, joint_attention_dim=32, caption_projection_dim=32,
                               pooled_projection_dim=16, out_channels=4, pos_embed_max_size=32).to(dt)
    with torch.no_grad():
        for p in tr.parameters():
            if p.dim() > 1:
                p.mul_(3.0)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(4,) * 4, layers_per_block=1,
                        latent_channels=4, norm_num_groups=2, scaling_factor=1.5305, shift_factor=0.0609,
                        use_quant_conv=False, use_post_quant_conv=False).to(vae_dtype or dt)
    b = Backend(tr, vae, 7.0, 3.0, read_blocks, sar_blocks if sar_blocks is not None else read_blocks, "cpu", "cpu",
                dt, 16, encoder=_tiny_encoder(dt), defer_transformer=defer_transformer)
    b.n_text = CLIP_TOKENS + 8
    return b
