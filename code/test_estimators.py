"""Tests of estimators.py (python -I test_estimators.py). No weights are downloaded."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimators import erode_core, object_photometric, scene_lightness, srgb_to_cielab  # noqa: E402

DT = torch.float64


def test_lab_known_values():
    white = torch.ones(3, 8, 8, dtype=DT)
    black = torch.zeros_like(white)
    assert abs(scene_lightness(white) - 100) < 1e-3 and abs(scene_lightness(black)) < 1e-3
    gray = torch.full((3, 8, 8), 0.5, dtype=DT)
    assert abs(scene_lightness(gray) - 53.39) < 0.1          # sRGB 0.5
    assert srgb_to_cielab(gray)[1:].abs().max() < 1e-2       # achromatic
    red = torch.zeros(3, 2, 2, dtype=DT)
    red[0] = 1
    lab = srgb_to_cielab(red)[:, 0, 0]
    assert abs(lab[0] - 53.24) < 0.1 and abs(lab[1] - 80.09) < 0.2 and abs(lab[2] - 67.2) < 0.2


def test_lightness_gradient_is_finite_at_black():
    x = torch.zeros(3, 8, 8, dtype=DT, requires_grad=True)
    scene_lightness(x).backward()
    assert torch.isfinite(x.grad).all()
    x = torch.zeros(3, 8, 8, dtype=DT, requires_grad=True)
    object_photometric(x, torch.ones(64, 64, dtype=DT), "chroma").backward()
    assert torch.isfinite(x.grad).all()


def test_lightness_accepts_batched_image_and_grad_flows():
    x = torch.full((1, 3, 16, 16), 0.5, dtype=DT, requires_grad=True)
    scene_lightness(x).backward()
    assert (x.grad > 0).all()


def test_support_weights_get_no_gradient():
    x = torch.rand(3, 64, 64, dtype=DT)
    s = torch.rand(64, 64, dtype=DT).requires_grad_()
    v = object_photometric(x, s, "lightness")
    assert not v.requires_grad                               # stop-gradient on the support


def test_object_photometric_measures_inside_support():
    x = torch.zeros(3, 64, 64, dtype=DT)
    x[:, 20:40, 20:40] = 1.0
    s = torch.zeros(64, 64, dtype=DT)
    s[22:38, 22:38] = 1
    assert abs(object_photometric(x, s, "lightness") - 100) < 1e-2
    assert object_photometric(x, torch.zeros(64, 64, dtype=DT), "lightness").isfinite()


def test_erode_shrinks_mask():
    s = torch.zeros(64, 64, dtype=DT)
    s[20:40, 20:40] = 1
    assert erode_core(s).sum() < s.sum() and erode_core(s)[30, 30] == 1 and erode_core(s, 3)[20, 20] == 0


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("estimators: pass")


if __name__ == "__main__":
    main()
