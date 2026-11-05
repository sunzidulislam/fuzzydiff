"""Tests of grounding.py (python -I test_grounding.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding import aggregate_maps, binding, binding_keys, map_tokens, phrase_spans  # noqa: E402
from predicates import translate  # noqa: E402
from sd3_backend import TextCond, VelocityOut, tiny_backend  # noqa: E402


def fake_offsets(prompt, limit=256):
    out, pos = [], 0
    for w in prompt.split():
        a = prompt.index(w, pos)
        pos = a + len(w)
        out.append((a, pos))
    return out[:limit]


def rand_map(n=64, seed=None):
    g = torch.Generator().manual_seed(seed if seed is not None else n)
    return torch.rand(n, generator=g, dtype=torch.float64)


def overlap(a, q, eps=1e-6):
    return (a * q).sum() / (q.sum() + eps)


def test_t5_shift_and_repeats():
    p = "a red mug and a red teapot"
    s = phrase_spans(p, translate(p))
    toks, bad = map_tokens(fake_offsets(p), s)
    assert toks["red#0"] != toks["red#1"] and min(toks["red#0"]) >= 77 and not bad
    assert toks["mug"] == [79] and toks["teapot"] == [83]


def test_unique_words_keep_plain_keys_and_binding_keys_resolve():
    p = "a greenish-gray mug and a turquoise-green teapot"
    prog = translate(p)
    s = phrase_spans(p, prog)
    assert p[slice(*s["greenish-gray"])] == "greenish-gray" and p[slice(*s["teapot"])] == "teapot"
    assert binding_keys(p, prog)[("turquoise-green", "teapot")] == ("turquoise-green", "teapot")


def test_binding_keys_pick_the_occurrence_before_the_object():
    p = "a red mug and a red teapot"
    k = binding_keys(p, translate(p))
    assert k[("red", "mug")] == ("red#0", "mug") and k[("red", "teapot")] == ("red#1", "teapot")


def test_truncated_phrase_is_reported():
    toks, bad = map_tokens(fake_offsets("x " * 400 + "mug"), {"mug": (800, 803)})
    assert bad == ["mug"] and "mug" not in toks


def test_binding_range_and_single_competitor():
    a, q, r = rand_map(seed=1), rand_map(seed=2), rand_map(seed=3)
    b = binding(a, q, [r], 0.02, 1e-6)
    assert -1 <= b <= 1 and abs(b - (overlap(a, q) - overlap(a, r))) < 1e-6
    wide = binding(a, q, [r, rand_map(seed=4), rand_map(seed=5)], 0.02, 1e-6)
    assert -1 <= wide <= 1


def _one(b_map, share, mass, tokens=1, valid=10, thresholds=(2.0, 0.5, 0.0)):
    n = len(b_map)
    cond = TextCond(None, None, None, None, list(range(valid)), {"mug": list(range(tokens))}, [])
    out = VelocityOut(None, None, None, maps={"mug": {0: b_map}}, shares={"mug": {0: torch.full((n,), share)}},
                      text_mass={0: torch.full((n,), mass)})
    return aggregate_maps(out, cond, [0], thresholds)["mug"]


def test_confidence_uses_raw_map():
    flat = torch.full((4096,), 1e-3)
    flat[0] += 5e-4
    g = _one(flat, share=0.1, mass=0.5)
    assert g.peak < 2.0 and g.informative is False
    assert g.P.max() > 0.99                                      # min-max alone would have looked sharp


def test_informative_map_needs_all_three_statistics():
    peaked = torch.full((64,), 1e-3)
    peaked[10] = 0.5
    assert _one(peaked, 0.1, 0.5).informative is True
    assert _one(peaked, 0.01, 0.5).informative is False          # low share
    assert _one(peaked, 0.1, 0.5, thresholds=(2.0, 0.5, 0.9)).informative is False  # low absolute mass


def test_blocks_are_weighted_uniformly_and_map_stays_differentiable():
    a = torch.zeros(64, dtype=torch.float64, requires_grad=True)
    a0, a1 = a + rand_map(seed=1), a + rand_map(seed=2)
    cond = TextCond(None, None, None, None, list(range(10)), {"mug": [0]}, [])
    out = VelocityOut(None, None, None, maps={"mug": {0: a0, 1: a1}},
                      shares={"mug": {0: torch.ones(64) * .1, 1: torch.ones(64) * .1}},
                      text_mass={0: torch.ones(64), 1: torch.ones(64)})
    g = aggregate_maps(out, cond, [0, 1], (2.0, 0.5, 0.0))["mug"]
    assert (g.B - (a0 + a1) / 2).abs().max() < 1e-9
    g.P.sum().backward()
    assert a.grad.abs().sum() > 0


def test_backend_drops_unmappable_phrases_and_reports_them():
    b = tiny_backend()
    c = b.encode_prompt("a b c d e f g h i j mug", {"mug": (20, 23), "a": (0, 1)})
    assert "mug" not in c.phrase_tokens and c.unmapped == ["mug"] and c.phrase_tokens["a"] == [77]


def test_attention_stats_on_tiny_model():
    b = tiny_backend(read_blocks=(0, 1))
    c = b.encode_prompt("a red mug", {"red": (2, 5), "mug": (6, 9)})
    out = b.velocity(b.noise_like_latent(0), torch.tensor(0.5), c, readout=True)
    stats = aggregate_maps(out, c, [0, 1], (1.0, 0.0, 0.0))
    assert stats["mug"].P.shape == (8, 8) and stats["red"].P.max() <= 1 + 1e-9
    assert stats["mug"].informative and 0 < stats["mug"].mass <= 1


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("grounding: pass")


if __name__ == "__main__":
    main()
