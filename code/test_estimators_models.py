"""Tests of the model-based estimators on tiny random models (python -I test_estimators_models.py)."""
import os
import sys

import torch
from transformers import CLIPSegConfig, CLIPSegForImageSegmentation, SiglipConfig, SiglipModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import ClipSegSupport, FakeSigLIP, FakeSupport, SigLIPEvidence, area, centroid, extent, region_crop, text_templates  # noqa: E402

DT = torch.float64


def blob_mask(n=64, a=20, b=40):
    s = torch.zeros(n, n, dtype=DT)
    s[a:b, a:b] = 1
    return s


def tiny_clipseg():
    torch.manual_seed(0)
    cfg = CLIPSegConfig(text_config={"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 1,
                                     "num_attention_heads": 2, "max_position_embeddings": 8, "vocab_size": 32,
                                     "projection_dim": 8},
                        vision_config={"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 4,
                                       "num_attention_heads": 2, "image_size": 32, "patch_size": 16,
                                       "projection_dim": 8},
                        projection_dim=8, reduce_dim=8, decoder_num_attention_heads=2, extract_layers=[1, 2, 3])
    model = CLIPSegForImageSegmentation(cfg)

    def encode_text(phrase):
        return torch.ones(1, 8) * (len(phrase) % 3 + 1) / 3

    return ClipSegSupport(model, encode_text, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), size=32, grid=64)


def tiny_siglip():
    torch.manual_seed(0)
    cfg = SiglipConfig(text_config={"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 1,
                                    "num_attention_heads": 2, "max_position_embeddings": 8, "vocab_size": 32},
                       vision_config={"hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 1,
                                      "num_attention_heads": 2, "image_size": 32, "patch_size": 16})
    model = SiglipModel(cfg)

    def encode_text(text):
        return torch.ones(1, 16) * (len(text) % 5 + 1)

    return SigLIPEvidence(model, encode_text, size=32)


def test_region_crop_frame_is_detached_and_pixels_differentiable():
    x = torch.rand(3, 128, 128, dtype=DT, requires_grad=True)
    s = blob_mask()
    r = region_crop(x, s, 2.5)
    assert r.shape[-2:] == (384, 384)
    r.mean().backward()
    assert x.grad.abs().sum() > 0


def test_region_crop_follows_the_object_through_soft_mask_only():
    x = torch.rand(3, 64, 64, dtype=DT)
    s = blob_mask().requires_grad_()
    r = region_crop(x, s, 2.5, size=32)
    r.mean().backward()
    assert s.grad.abs().sum() > 0                                # through the soft mask
    moved = blob_mask(a=40, b=60)
    assert not torch.allclose(region_crop(x, moved, 2.5, size=32), r.detach())


def test_region_crop_fills_outside_with_gray():
    x = torch.zeros(3, 64, 64, dtype=DT)
    s = blob_mask(a=30, b=34)
    corners = region_crop(x, s, 8.0, size=64)[0, :, 0, 0]
    assert (corners - 0.5).abs().max() < 0.2


def test_empty_support_is_finite():
    s = torch.zeros(64, 64, dtype=DT, requires_grad=True)
    v = area(s) + centroid(s, 1e-6).sum()
    v.backward()
    assert torch.isfinite(s.grad).all()
    assert torch.isfinite(region_crop(torch.rand(3, 64, 64, dtype=DT), s, 2.5, size=32)).all()
    assert torch.isfinite(extent(torch.zeros(64, 64, dtype=DT), "y"))


def test_extent_and_area_for_box():
    s = torch.zeros(64, 64, dtype=DT)
    s[16:48, :] = 1
    assert abs(extent(s, "y") - 0.5) < 0.02 and abs(extent(s, "x") - 1.0) < 0.02
    assert abs(area(s) - 0.5) < 1e-9
    c = centroid(s, 1e-6)
    assert abs(c[0] - 0.5) < 0.02 and abs(c[1] - 0.5) < 0.02


def test_clipseg_gradient_reaches_pixels_with_tiny_model():
    m = tiny_clipseg()
    x = torch.rand(3, 128, 128, requires_grad=True)
    out = m(x, ["mug"])["mug"]
    assert out.shape == (64, 64) and 0 <= out.min() and out.max() <= 1
    out.mean().backward()
    assert x.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in m.model.parameters())


def test_clipseg_caches_text_embeddings():
    m = tiny_clipseg()
    calls = []
    original = m.encode_text
    m.encode_text = lambda p: (calls.append(p), original(p))[1]
    x = torch.rand(3, 32, 32)
    m(x, ["mug", "teapot"])
    m(x, ["mug"])
    assert calls == ["mug", "teapot"]


def test_text_templates():
    assert text_templates("towel") == ["a photo of a towel", "a close-up photo of a towel", "an image of a towel"]
    assert text_templates("apple")[0] == "a photo of an apple"
    assert text_templates("uniform")[0] == "a photo of a uniform"


def test_siglip_logits_are_differentiable_scalars():
    m = tiny_siglip()
    x = torch.rand(3, 64, 64, requires_grad=True)
    region = torch.rand(1, 3, 32, 32)
    g, r = m.logits(x, region, "mug")
    assert g.dim() == 0 and r.dim() == 0
    (g + r).backward()
    assert x.grad.abs().sum() > 0


def test_fakes_match_the_real_interfaces():
    masks = {"mug": blob_mask()}
    assert FakeSupport(masks)(torch.rand(3, 8, 8), ["mug"])["mug"].shape == (64, 64)
    g, r = FakeSigLIP(lambda phrase, image: 2.0).logits(torch.rand(3, 8, 8), torch.rand(1, 3, 8, 8), "mug")
    assert float(g) == 2.0 and float(r) == 2.0


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("estimator models: pass")


if __name__ == "__main__":
    main()
