"""Tests of sd3_backend.py on the tiny random model (python -I test_backend.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sd3_backend import tiny_backend  # noqa: E402


def close(a, b, tol=1e-9):
    return (a - b).abs().max().item() < tol


def test_cfg_combination_and_clean():
    b = tiny_backend()
    c = b.encode_prompt("x")
    z = b.noise_like_latent(0)
    o = b.velocity(z, torch.tensor(0.8), c, cfg=7.0)
    assert close(o.v, o.v_uncond + 7.0 * (o.v_cond - o.v_uncond))
    assert close(b.clean(z, torch.tensor(0.8, dtype=torch.float64), o.v), z - 0.8 * o.v)


def test_gradient_reaches_latent():
    b = tiny_backend()
    c = b.encode_prompt("x")
    z = b.noise_like_latent(0).requires_grad_()
    b.decode(b.clean(z, torch.tensor(0.8), b.velocity(z, torch.tensor(0.8), c, cfg=7.0).v)).mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_decode_range_and_roundtrip_shape():
    b = tiny_backend()
    x = b.decode(b.noise_like_latent(0))
    assert x.min() >= 0 and x.max() <= 1 and b.encode(x).shape == b.noise_like_latent(0).shape


def test_decode_uses_scale_and_shift():
    b = tiny_backend()
    z = b.noise_like_latent(1)
    assert b.vae.config.scaling_factor == 1.5305 and b.vae.config.shift_factor == 0.0609
    with torch.no_grad():
        raw = b.vae.decode(z / 1.5305 + 0.0609).sample
    assert close(b.decode(z), ((raw + 1) / 2).clamp(0, 1))


def test_sar_only_changes_conditional():
    b = tiny_backend()
    c = b.encode_prompt("x")
    z = b.noise_like_latent(0)
    s = torch.tensor(0.5)
    plain = b.velocity(z, s, c, cfg=7.0).v_cond
    zero = b.velocity_cond(z, s, c, torch.zeros(b.n_img, b.n_text))
    assert close(plain, zero)
    bias = torch.zeros(b.n_img, b.n_text)
    bias[:, 77] = 3.0
    assert not close(plain, b.velocity_cond(z, s, c, bias))
    after = b.velocity(z, s, c, cfg=7.0).v_cond                 # bias does not stay installed
    assert close(plain, after)


def test_readout_per_phrase_and_block():
    b = tiny_backend(read_blocks=(0, 1))
    c = b.encode_prompt("a red mug", {"mug": (6, 9)})
    assert c.phrase_tokens["mug"] == [79]
    z = b.noise_like_latent(0)
    o = b.velocity(z, torch.tensor(0.5), c, readout=True)
    assert set(o.maps["mug"]) == {0, 1} and o.maps["mug"][0].shape == (b.n_img,)
    assert set(o.text_mass) == {0, 1} and o.shares["mug"][1].shape == (b.n_img,)
    assert not b.velocity(z, torch.tensor(0.5), c).maps          # readout off leaves no maps


def test_gradient_checkpointing_is_active():
    b = tiny_backend()
    calls = []
    original = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(1)
        return original(*a, **k)

    torch.utils.checkpoint.checkpoint = spy
    try:
        z = b.noise_like_latent(0).requires_grad_()
        b.velocity(z, torch.tensor(0.5), b.encode_prompt("x")).v.sum().backward()
        tf_calls = len(calls)
        calls.clear()
        b.decode(z).sum().backward()
        vae_calls = len(calls)
    finally:
        torch.utils.checkpoint.checkpoint = original
    assert tf_calls >= 3, "transformer checkpointing silently off"
    assert vae_calls >= 1, "VAE checkpointing silently off"
    assert not b.transformer.transformer_blocks[0].training      # no dropout or norm change below the flag


def test_checkpointed_readout_gradient_matches_plain():
    b = tiny_backend(read_blocks=(0, 1))
    c = b.encode_prompt("a red mug", {"mug": (6, 9)})

    def grad():
        z = b.noise_like_latent(0).requires_grad_()
        o = b.velocity(z, torch.tensor(0.5), c, readout=True)
        (o.v.sum() + o.maps["mug"][0].sum() + o.maps["mug"][1].sum()).backward()
        return z.grad

    with_ckpt = grad()
    b.transformer.training = False
    assert close(with_ckpt, grad(), 1e-7)


def test_sigmas_match_flow_schedule():
    b = tiny_backend()
    s = b.sigmas(28)
    assert len(s) == 29 and abs(s[0] - 1.0) < 1e-6 and s[-1] == 0 and (s[1:] < s[:-1]).all()


def main():
    test_cfg_combination_and_clean()
    test_gradient_reaches_latent()
    test_decode_range_and_roundtrip_shape()
    test_decode_uses_scale_and_shift()
    test_sar_only_changes_conditional()
    test_readout_per_phrase_and_block()
    test_gradient_checkpointing_is_active()
    test_checkpointed_readout_gradient_matches_plain()
    test_sigmas_match_flow_schedule()
    print("backend: pass")


if __name__ == "__main__":
    main()
