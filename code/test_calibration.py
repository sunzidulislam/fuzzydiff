"""Tests of calibration.py (python -I test_calibration.py)."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration, domain_weight, ramp_weight  # noqa: E402
from settings import load_settings  # noqa: E402


def test_provisional_lightness_and_tag():
    c = ProvisionalCalibration(load_settings())
    assert c.tag == "provisional" and c.valid("F2a", 1)
    assert abs(c.phi("F2a", "scene_lightness", 4, torch.tensor(50.0)) - 0.5) < 1e-6
    assert c.domain("F2a", "scene_lightness", 4) == (0.0, 100.0)


def test_hedge_interval_from_table():
    l, u, k = ProvisionalCalibration(load_settings()).interval("F2a", "lightness", "slightly", 1)
    assert abs(l - 0.05) < 1e-9 and abs(u - 0.25) < 1e-9 and k == 0.1
    l, u, _ = ProvisionalCalibration(load_settings()).interval("F2a", "lightness", "extremely", 1)
    assert abs(l - 0.85) < 1e-9 and u == 1.0                 # clipped to the scale


def test_size_interval_and_domain_weight():
    l, u, _ = ProvisionalCalibration(load_settings()).interval("F8", "area", "moderately", 1)
    assert (l, u) == (0.5, 1.0)
    assert domain_weight(0.0, 0, 100, 0.1) == 0 and domain_weight(-1, 0, 100, 0.1) == 0
    assert domain_weight(50, 0, 100, 0.1) == 1 and 0 < domain_weight(5, 0, 100, 0.1) < 1
    assert abs(domain_weight(5, 0, 100, 0.1) - 0.5) < 1e-9
    assert ramp_weight(0, 2) == 0 and ramp_weight(1, 2) == 0.5 and ramp_weight(2, 2) == 1


def test_native_families_have_one_sided_intervals():
    c = ProvisionalCalibration(load_settings())
    l, u, k = c.interval("presence", None, None, 1)
    assert abs(l - math.log(0.9 / 0.1)) < 1e-6 and math.isinf(u) and k == 1.0
    l, u, _ = c.interval("binding", None, None, 1)
    assert l == 0.05 and math.isinf(u)
    l, u, _ = c.interval("leakage", None, None, -1)
    assert math.isinf(l) and u == 0.25


def test_remaining_interface():
    c = ProvisionalCalibration(load_settings())
    assert c.size_quantiles({}) == (0.1, 0.2) and c.psi("mug", {}) == 1.0
    assert c.drift_thresholds("small") == (0.5, 0.5) and c.attention_ok("objects")
    assert c.presence_logit(4, torch.tensor(2.0), torch.tensor(4.0)) == 3.0
    white = c.prototype("white")
    assert white.shape == (3,) and white[0] > 0.9 and white[1:].abs().max() < 0.01
    try:
        c.prototype("no-such-colour")
        assert False
    except KeyError:
        pass


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("calibration: pass")


if __name__ == "__main__":
    main()
