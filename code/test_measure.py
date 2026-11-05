"""Tests of measure.py and the tier 1-3 scorers (python -I test_measure.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from color_stats import oklab_to_srgb, srgb_to_oklab  # noqa: E402
from constraints import Evidence, freeze_step, in_domain, membership, new_state, score, tier_losses  # noqa: E402
from estimators import FakeSigLIP, FakeSupport  # noqa: E402
from grounding import phrase_spans  # noqa: E402
from guidance import GuideCfg, run_stage1  # noqa: E402
from measure import EstimatorDeps, Measure  # noqa: E402
from predicates import translate  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

DT = torch.float64
S = load_settings()


class Prov(ProvisionalCalibration):
    def attention_thresholds(self, kind):
        return (1.0, 0.0, 0.0)


PROV = Prov(S)
MIX = "a greenish-gray mug and a turquoise-green teapot"


class MemoryLog:
    def __init__(self):
        self.records = []

    def write(self, **rec):
        self.records.append(rec)


def half_masks():
    left = torch.zeros(64, 64, dtype=DT)
    left[8:56, 4:30] = 1
    right = torch.zeros(64, 64, dtype=DT)
    right[8:56, 34:60] = 1
    return {"mug": left, "teapot": right}


def solid(rgb_left, rgb_right):
    x = torch.zeros(3, 128, 128, dtype=DT)
    for i in range(3):
        x[i, :, :64] = rgb_left[i]
        x[i, :, 64:] = rgb_right[i]
    return x


def proto_rgb(term):
    return S.get("prototypes", dict)[term]


def deps(presence=1.0, masks=None, calib=PROV):
    seg = FakeSupport(masks if masks is not None else half_masks())

    def logit(phrase, image):
        value = presence[phrase] if isinstance(presence, dict) else presence
        return image.mean() * 0 + value

    return EstimatorDeps(seg, FakeSigLIP(logit), calib, S)


def measure_x(x, prompt=MIX, presence=1.0, masks=None, calib=PROV, bin=4):
    program = translate(prompt)
    m = Measure(deps(presence, masks, calib), prompt, None)
    return program, m(None, x, None, program, bin)


def test_absent_object_only_gets_presence():
    prog, ev = measure_x(solid((.5, .5, .5), (.5, .5, .5)), presence={"mug": -3.0, "teapot": 3.0})
    active = {(f.constraint.family, f.constraint.args[0]) for f in freeze_step(prog, ev, PROV, 4, new_state()) if f.weight > 0}
    assert ("presence", "mug") in active and ("F6", "mug") not in active and ("F6", "teapot") in active


def _leak_loss(left, right, prompt="a red mug and a blue teapot"):
    prog, ev = measure_x(solid(left, right), prompt)
    frozen = [f for f in freeze_step(prog, ev, PROV, 4, new_state()) if f.constraint.family == "leakage"]
    return tier_losses(frozen, ev, PROV)[2], frozen, ev


def test_leakage_penalizes_colour_on_other_object():
    clean, *_ = _leak_loss(proto_rgb("red"), proto_rgb("blue"))
    both, *_ = _leak_loss(proto_rgb("red"), proto_rgb("red"))
    assert both > clean + 1e-3


def test_native_scores_are_not_tapered_at_their_bounds():
    _, frozen, ev = _leak_loss(proto_rgb("red"), proto_rgb("red"))
    red_on_teapot = next(f for f in frozen if f.constraint.args[0] == "red")
    assert float(ev.pairs[("leakage", "red", "teapot")]) > 0.99 and red_on_teapot.weight == 1.0


def _f6_mu(x):
    prog, ev = measure_x(x, "a greenish-gray mug and a blue teapot")
    frozen = [f for f in freeze_step(prog, ev, PROV, 4, new_state()) if f.constraint.family == "F6"]
    assert {f.component for f in frozen} == {"main", "e", "varpi", "zeta"}
    product = 1.0
    for f in frozen:
        product *= float(membership(score(f, ev, PROV), f.low, f.high, f.kappa))
    return product


def _blend_rgb(fraction):
    gray, green = PROV.prototype("gray"), PROV.prototype("green")
    return oklab_to_srgb((gray + fraction * (green - gray)).view(3, 1, 1)).view(3).tolist()


def test_blend_rejects_two_tone():
    uniform = solid(_blend_rgb(0.15), (0, 0, 1))
    two_tone = uniform.clone()
    a, b = proto_rgb("gray"), proto_rgb("green")
    for r in range(0, 128, 32):
        for i in range(3):
            two_tone[i, r:r + 16, :64] = a[i]
            two_tone[i, r + 16:r + 32, :64] = b[i]
    assert _f6_mu(uniform) > _f6_mu(two_tone) + 0.1


def test_mixture_support_weight_has_no_gradient():
    mask = half_masks()["mug"].clone().requires_grad_()
    masks = {"mug": mask, "teapot": half_masks()["teapot"]}
    prog, ev = measure_x(solid((.4, .5, .4), (0, 0, 1)), "a greenish-gray mug and a blue teapot", masks=masks)
    f = next(f for f in freeze_step(prog, ev, PROV, 4, new_state()) if f.constraint.family == "F6")
    assert not score(f, ev, PROV).requires_grad


def test_mixture_gradient_reaches_pixels():
    x = solid((.4, .5, .4), (0, 0, 1)).requires_grad_()
    prog, ev = measure_x(x, "a greenish-gray mug and a blue teapot")
    f = next(f for f in freeze_step(prog, ev, PROV, 4, new_state()) if f.constraint.family == "F6")
    score(f, ev, PROV).backward()
    assert x.grad.abs().sum() > 0


def test_object_photometric_uses_its_own_support():
    prompt = "a slightly dark mug on a wooden table"
    masks = {"mug": half_masks()["mug"]}
    prog, ev = measure_x(solid((.2, .2, .2), (.9, .9, .9)), prompt, masks=masks)
    f = next(f for f in freeze_step(prog, ev, PROV, 4, new_state()) if f.constraint.family == "F2a")
    assert f.weight > 0 and f.constraint.quantity == "lightness"
    assert abs(score(f, ev, PROV) - (1 - float(ev.objects["mug"]["lightness"]) / 100)) < 1e-9
    assert float(ev.objects["mug"]["lightness"]) < 30


def test_empty_support_turns_constraints_off_without_nans():
    masks = {"mug": torch.zeros(64, 64, dtype=DT), "teapot": half_masks()["teapot"]}
    prog, ev = measure_x(solid((.5, .5, .5), (.5, .5, .5)), masks=masks)
    frozen = freeze_step(prog, ev, PROV, 4, new_state())
    f6 = next(f for f in frozen if f.constraint.family == "F6" and f.constraint.args[0] == "mug")
    assert f6.weight == 0 and f6.gates["support"] is False
    assert all(torch.isfinite(l) for l in tier_losses(frozen, ev, PROV).values())


def test_frozen_constraint_stays_computable_when_support_vanishes():
    prog, ev = measure_x(solid((.5, .5, .5), (.5, .5, .5)))
    frozen = freeze_step(prog, ev, PROV, 4, new_state())
    gone = {"mug": torch.zeros(64, 64, dtype=DT), "teapot": half_masks()["teapot"]}
    _, ev2 = measure_x(solid((.5, .5, .5), (.5, .5, .5)), masks=gone)
    losses = tier_losses(frozen, ev2, PROV)
    assert all(torch.isfinite(l) for l in losses.values())


def _tiny_run(prompt=MIX, calib=PROV, phrases=None, steps=2):
    b = tiny_backend(read_blocks=(0, 1))
    program = translate(prompt)
    cond = b.encode_prompt(prompt, phrases if phrases is not None else phrase_spans(prompt, program))

    def mask(phrase, x):
        return torch.nn.functional.adaptive_avg_pool2d(x[None], 64)[0].mean(0)

    m = Measure(EstimatorDeps(FakeSupport(mask), FakeSigLIP(lambda p, img: img.mean() * 2 + 0.5), calib, S), prompt, cond)
    cfg = GuideCfg(steps=steps, cfg=7.0, sigma_lo=0.0, eta0=0.1, delta=0.1, dmax=0.2, eps={1: 0.01, 2: 0.01}, backtrack=1, ramp=2)
    log = MemoryLog()
    return run_stage1(b, program, cond, 0, calib, m, cfg, log, prompt=prompt), log


def test_three_tiers_project_on_tiny_model():
    out, log = _tiny_run()
    s = next(r for r in log.records if r["type"] == "step")
    assert set(s["tiers"]) == {1, 2, 3} and set(s["removed"]) <= {"1", "2", "3"} and s["removed"]["1"] == 0.0
    assert set(s["cos"]) <= {"1-2", "1-3", "2-3"}
    assert out.traces and out.supports["mug"].shape == (64, 64) and "mug" in out.presence


def test_attention_off_disables_binding_only():
    class Fails(Prov):
        def attention_ok(self, kind):
            return False

    fails = Fails(S)
    _, log = _tiny_run(calib=fails)
    records = [r for r in log.records if r["type"] == "constraint"]
    assert not any(r["active"] for r in records if r["family"] == "binding")
    assert any(r["active"] for r in records if r["family"] == "leakage")


def test_unmapped_phrase_degrades_binding_but_keeps_presence():
    prompt = "a red mug and a blue teapot"
    out, log = _tiny_run(prompt, phrases={"mug": (6, 9)})        # red, blue, teapot have no tokens
    records = [r for r in log.records if r["type"] == "constraint"]
    assert not any(r["active"] for r in records if r["family"] == "binding")
    assert any(r["active"] for r in records if r["family"] == "presence")


def test_final_records_omit_binding_because_no_attention_is_measured():
    out, _ = _tiny_run()
    assert not any(f["family"] == "binding" for f in out.finals)
    assert any(f["family"] == "presence" for f in out.finals)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("measure: pass")


if __name__ == "__main__":
    main()
