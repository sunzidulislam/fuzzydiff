"""Tests of constraints.py (python -I test_constraints.py)."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from constraints import (Evidence, charbonnier, freeze_step, in_domain, membership, new_state,  # noqa: E402
                         score, tier_losses, violation)
from predicates import Constraint, Program  # noqa: E402
from settings import load_settings  # noqa: E402

PROV = ProvisionalCalibration(load_settings())


def lightness_program(hedge=None, o=1):
    return Program(objects=[], constraints=[Constraint("F2a", ("room",), hedge, "scene_lightness", o)])


def evidence_with_L(value):
    return Evidence(x0=None, scene={"scene_lightness": torch.tensor(float(value), dtype=torch.float64)})


def test_membership_and_charbonnier():
    assert membership(torch.tensor(.5), .4, .6, .1) == 1
    assert abs(membership(torch.tensor(.8), .4, .6, .1) - math.exp(-2)) < 1e-6
    assert charbonnier(torch.tensor(0.), .1) == 0
    d = torch.tensor(5., requires_grad=True)
    charbonnier(d, .1).backward()
    assert 0 < d.grad < 10                                        # bounded, non-vanishing


def test_violation_with_half_lines():
    assert abs(violation(torch.tensor(.2), .5, math.inf) - .3) < 1e-6
    assert violation(torch.tensor(9.), .5, math.inf) == 0
    assert violation(torch.tensor(-9.), -math.inf, 0.25) == 0


def test_out_of_domain_gets_zero_weight_and_gate_false():
    f = freeze_step(lightness_program(), evidence_with_L(-5.0), PROV, 4, new_state())[0]
    assert f.weight == 0 and f.gates["domain"] is False


def test_frozen_set_is_not_recomputed_at_candidate():
    ev = evidence_with_L(50.0)
    f = freeze_step(lightness_program(), ev, PROV, 4, new_state())
    moved = evidence_with_L(99.9)                                 # would have a lower weight if re-frozen
    assert tier_losses(f, moved, PROV)[3] > 0 and f[0].weight == 1


def test_orientation_flips_the_score():
    ev = evidence_with_L(20.0)
    dark = freeze_step(lightness_program(o=-1), ev, PROV, 4, new_state())[0]
    bright = freeze_step(lightness_program(o=1), ev, PROV, 4, new_state())[0]
    assert abs(score(dark, ev, PROV) - 0.8) < 1e-9 and abs(score(bright, ev, PROV) - 0.2) < 1e-9
    assert tier_losses([dark], ev, PROV)[3] < tier_losses([bright], ev, PROV)[3]


def test_commanded_target_overrides_interval():
    ev = evidence_with_L(50.0)
    f = freeze_step(lightness_program(), ev, PROV, 4, new_state(), targets={0: (0.2, 0.22)})[0]
    assert (f.low, f.high) == (0.2, 0.22)


def test_in_domain_checks_raw_input_at_candidate():
    f = freeze_step(lightness_program(), evidence_with_L(50.0), PROV, 4, new_state())[0]
    assert in_domain(f, evidence_with_L(60.0), PROV) and not in_domain(f, evidence_with_L(101.0), PROV)


def test_invalid_family_is_inactive_then_ramps_in():
    class Late(ProvisionalCalibration):
        def valid(self, family, b):
            return b >= 2

    late, st = Late(load_settings()), new_state(ramp=2)
    ev = evidence_with_L(50.0)
    st.step = 0
    f0 = freeze_step(lightness_program(), ev, late, 1, st)[0]
    assert f0.weight == 0 and f0.gates["validity"] is False
    weights = []
    for step in (1, 2, 3):
        st.step = step
        weights.append(freeze_step(lightness_program(), ev, late, 2, st)[0].weight)
    assert weights == [0.0, 0.5, 1.0]


def test_family_without_a_scorer_is_recorded_not_guided():
    prog = Program(objects=[], constraints=[Constraint("F9", ("mug", "teapot"), None, "left", 1)])
    st = new_state()
    assert freeze_step(prog, evidence_with_L(50.0), PROV, 4, st) == []
    assert "F9" in st.unguided


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("constraints: pass")


if __name__ == "__main__":
    main()
