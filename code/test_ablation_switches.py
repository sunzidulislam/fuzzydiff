"""Each Tab. full-ablations switch changes what it says, at the place it acts (python -I test_ablation_switches.py).

The tiny random model rarely accepts a guided step and SAR falls back under the provisional calibration, so
whole-image runs cannot show a switch working; these tests check the quantity each switch changes."""
import dataclasses
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from constraints import Evidence, freeze_step, new_state  # noqa: E402
from fuzzydiff_sd3_attention import _post_softmax_output  # noqa: E402
from guidance import _reject_reason  # noqa: E402
from predicates import translate  # noqa: E402
from run_experiment import ABLATION_ARMS, SAR_ARMS, sar_cfg  # noqa: E402
from sar import build_priors  # noqa: E402
from settings import load_settings  # noqa: E402

S = load_settings()
CAL = ProvisionalCalibration(S)


def _evidence(prompt, lightness=50.0, p=0.9, area_=0.2):
    """Hand-made evidence: every object present with a square support."""
    program = translate(prompt)
    ev = Evidence(x0=None)
    sup = torch.zeros(16, 16)
    sup[4:12, 4:12] = 1.0
    for q in dict.fromkeys(program.objects):
        ev.objects[q] = {"S": sup, "core": sup, "A": torch.tensor(area_), "c": torch.tensor([0.5, 0.5]),
                         "h": torch.tensor(math.log(p / (1 - p)), requires_grad=True), "g": torch.tensor(0.0),
                         "r": torch.tensor(0.0), "p": torch.tensor(p), "psi": 1.0,
                         "lightness": torch.tensor(lightness), "chroma": torch.tensor(lightness)}
    return program, ev


def _frozen(prompt, state_kw=None, **ev_kw):
    program, ev = _evidence(prompt, **ev_kw)
    state = new_state(**{"ramp": 0, **(state_kw or {})})
    return freeze_step(program, ev, CAL, 4, state), program, ev


def _get(frozen, family, component="main"):
    return next(f for f in frozen if f.constraint.family == family and f.component == component)


def test_hard_interval_scales_kappa():
    base = _get(_frozen("a slightly dark mug")[0], "F2a")
    hard = _get(_frozen("a slightly dark mug", ABLATION_ARMS["abl_hard_interval"]["state"])[0], "F2a")
    assert math.isclose(hard.kappa, base.kappa * 1e-3)


def test_no_presence_zeroes_presence_weights_only():
    frozen = _frozen("a slightly dark mug", ABLATION_ARMS["abl_no_presence"]["state"])[0]
    assert _get(frozen, "presence").weight == 0.0 and _get(frozen, "F2a").weight > 0


def test_linear_domain_opens_the_domain_gate_outside_E():
    outside = dict(lightness=150.0)                     # L* = 150 is outside the provisional domain [0, 100]
    assert not _get(_frozen("a slightly dark mug", **outside)[0], "F2a").gates["domain"]
    f = _get(_frozen("a slightly dark mug", ABLATION_ARMS["abl_linear_domain"]["state"], **outside)[0], "F2a")
    assert f.gates["domain"] and f.weight > 0


def test_no_core_opens_the_support_gate():
    small = dict(area_=1e-5)                            # below a_min: the support gate closes
    assert not _get(_frozen("a slightly dark mug", **small)[0], "F2a").gates["support"]
    state = ABLATION_ARMS["abl_no_core"]["state"]
    assert _get(_frozen("a slightly dark mug", state, **small)[0], "F2a").gates["support"]
    assert ABLATION_ARMS["abl_no_core"]["measure"] == "no_core"


def test_size_without_margin_targets_m_above_zero():
    def low(state_kw):
        program, ev = _evidence("a mug slightly larger than a bottle")
        ev.pairs[("F8", "mug", "bottle", "area")] = (torch.tensor(0.2), torch.tensor(0.1))
        state = new_state(**{"ramp": 0, **state_kw})
        f = _get(freeze_step(program, ev, CAL, 4, state), "F8")
        return f.low, state.size_regime
    with_margin, regime = low({})
    without, _ = low(ABLATION_ARMS["abl_size_no_margin"]["state"])
    if "direction" in regime.values():                  # the margin applies to the direction regime only
        assert with_margin == S.get("size_c95") and without == 0.0


def test_acceptance_without_tier2_ignores_a_tier2_rise():
    from guidance import GuideCfg
    frozen, program, ev = _frozen("a slightly dark mug")
    cfg = GuideCfg(steps=1, cfg=7.0, sigma_lo=0, eta0=0.1, delta=0.1, dmax=1, eps={1: 0.0, 2: 0.0}, backtrack=0, ramp=0)
    before = {"2": -1.0}                                # tier 2 "rose" from -1 to its current value
    assert _reject_reason(frozen, ev, CAL, before, cfg) in ("tier2", "total")
    assert _reject_reason(frozen, ev, CAL, before, dataclasses.replace(cfg, accept_tier2=False)) in (None, "total")


def test_sar_prior_variants():
    trace = torch.zeros(8, 8)
    trace[0:4, 0:4] = 1.0
    support = torch.zeros(8, 8)
    support[2:6, 2:6] = 1.0
    base = sar_cfg(S)
    product = build_priors({"mug": trace}, {"mug": support}, base)["mug"]
    attention = build_priors({"mug": trace}, {"mug": support}, dataclasses.replace(base, **SAR_ARMS["stage1_sar_attn_prior"]))["mug"]
    assert float(product[0, 0]) == 0.0 and float(attention[0, 0]) == 1.0     # attention alone places a prior
    far = torch.zeros(8, 8)
    far[4:8, 4:8] = 1.0
    traces, supports = {"mug": support, "vase": far}, {"mug": support, "vase": far}
    rel = {("mug", "vase"): "overlap"}                  # not exclusive: no competition unless for all pairs
    comp = dataclasses.replace(base, competition=True)
    some = build_priors(traces, supports, comp, relations=rel)
    every = build_priors(traces, supports, dataclasses.replace(comp, **SAR_ARMS["stage1_sar_comp_all"]), relations=rel)
    assert torch.equal(some["mug"], build_priors(traces, supports, base)["mug"]) and not torch.equal(every["mug"], some["mug"])


def test_post_softmax_differs_from_logit_bias():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 10, 4), torch.randn(2, 10, 4), torch.randn(2, 10, 4)
    log_r = torch.full((6, 4), math.log(2.0))
    post = _post_softmax_output(q, k, v, 6, 0.5, log_r)
    mask = torch.zeros(10, 10)
    mask[:6, 6:] = log_r
    pre = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask, scale=0.5)[0]
    assert (post - pre).abs().max() > 1e-3


def test_time_matched_plain_keeps_the_noise_range():
    """Same start and end noise as the plain pass, twice the steps (read back from the run's step log)."""
    import json
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from grounding import phrase_spans
    from guidance import Stage1Out
    from reporting import StepLog
    from sar import run_stage2
    from sd3_backend import tiny_backend
    b = tiny_backend()
    prompt = "a slightly dark mug"
    program = translate(prompt)
    cond = b.encode_prompt(prompt, phrase_spans(prompt, program))
    stub = Stage1Out(latent=None, image=torch.rand(1, 3, 128, 128, dtype=b.latent_dtype), traces={}, supports={}, finals=[])
    seen = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, extra in (("plain", {}), ("tm", SAR_ARMS["stage1_plain_tm"])):
            cfg = dataclasses.replace(sar_cfg(S), k=3, **extra)
            log = StepLog(str(Path(tmp) / f"{name}.jsonl"), "dev", CAL.tag)
            run_stage2(b, stub, cond, program, prompt, CAL, None, cfg, 0, log, mode="plain", steps=8)
            log.close()
            seen[name] = [json.loads(l)["sigma"] for l in open(Path(tmp) / f"{name}.jsonl") if '"sar_step"' in l]
    assert len(seen["tm"]) == 2 * len(seen["plain"]) and math.isclose(seen["tm"][0], seen["plain"][0])


def test_cond_branch_gradient_flows_only_through_v_cond():
    from guidance import GuideCfg, Stage1Ctx, _measure
    from sd3_backend import tiny_backend
    from grounding import phrase_spans
    b = tiny_backend()
    prompt = "a slightly dark mug"
    program = translate(prompt)
    cond = b.encode_prompt(prompt, phrase_spans(prompt, program))
    cfg = GuideCfg(steps=4, cfg=7.0, sigma_lo=0, eta0=0.1, delta=0.1, dmax=1, eps={1: 0, 2: 0}, backtrack=0, ramp=0)

    def grad(c):
        ctx = Stage1Ctx(b, program, cond, CAL, lambda *a: Evidence(x0=a[1]), c, new_state(), None, None, prompt, 0)
        z = b.noise_like_latent(0).requires_grad_(True)
        _, ev = _measure(ctx, z, b.sigmas(4)[1], 2, False)
        return torch.autograd.grad(ev.x0.sum(), z)[0], ev.x0.detach()
    g_cfg, x_cfg = grad(cfg)
    g_cond, x_cond = grad(dataclasses.replace(cfg, grad_branch="cond"))
    assert torch.allclose(x_cfg, x_cond) and not torch.allclose(g_cfg, g_cond)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ablation switches: pass")


if __name__ == "__main__":
    main()
