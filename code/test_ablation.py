"""Each ablation arm of Tab. ablation changes exactly its component."""
import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guidance  # noqa: E402
from constraints import Evidence, charbonnier, floored_log_loss, freeze_step, new_state  # noqa: E402
from estimators import FakeSigLIP  # noqa: E402
from grounding import binding, overlap  # noqa: E402
from guidance import GuideCfg  # noqa: E402
from measure import EstimatorDeps, Measure  # noqa: E402
from predicates import Constraint, Program, translate  # noqa: E402
from run_experiment import ABLATION_ARMS, ABLATION_TABLE_ARMS, STAGE2_MODES  # noqa: E402
from sar import SarCfg  # noqa: E402
from test_guidance import CFG, PROV, Worse, _ctx, _step, close  # noqa: E402
from test_measure import PROV as MPROV, S, half_masks, solid  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
import test_sar as sar_tests  # noqa: E402

LIGHT = Program(objects=[], constraints=[Constraint("F2a", ("room",), "slightly", "scene_lightness", 1)])
PRESENT = Program(objects=["mug"], constraints=[Constraint("presence", ("mug",), None, None, 1)])


def _frozen(program, **state):
    ev = Evidence(scene={"scene_lightness": torch.tensor(50.0, dtype=torch.float64)},
                  objects={"mug": {"h": torch.tensor(0.0, dtype=torch.float64)}})
    return freeze_step(program, ev, PROV, 4, new_state(**state))


def test_point_and_one_sided_change_only_degree_targets():
    low, high = _frozen(LIGHT)[0].low, _frozen(LIGHT)[0].high
    assert low < high < 1
    point = _frozen(LIGHT, target="point")[0]
    assert point.low == point.high == (low + high) / 2
    one = _frozen(LIGHT, target="one_sided")[0]
    assert one.low == low and math.isinf(one.high)
    for target in ("point", "one_sided"):
        f, = _frozen(PRESENT, target=target)
        g, = _frozen(PRESENT)
        assert (f.low, f.high) == (g.low, g.high)


def test_floored_log_loss_saturates_where_charbonnier_does_not():
    kappa = 0.1
    d = torch.tensor(6 * kappa, requires_grad=True)
    floored_log_loss(d, kappa).backward()
    floored = float(d.grad)
    d.grad = None
    charbonnier(d, kappa).backward()
    assert floored < 0.01 * float(d.grad)
    f, = [f for f in _frozen(LIGHT, loss="floored")]
    assert f.loss == "floored" and _frozen(LIGHT)[0].loss == "charbonnier"


def test_weighted_sum_skips_the_priority_projection():
    called = []
    original = guidance.project_tier
    guidance.project_tier = lambda g, higher: called.append(1) or original(g, higher)
    try:
        _step(_ctx(tiny_backend(), targets={0: (0.9, 1.0)}))
        assert called
        called.clear()
        _step(_ctx(tiny_backend(), targets={0: (0.9, 1.0)}, cfg=GuideCfg(**{**CFG.__dict__, "priority": False})))
        assert not called
    finally:
        guidance.project_tier = original


def test_without_acceptance_the_first_candidate_is_taken():
    worse = Worse(S)
    out = _step(_ctx(tiny_backend(), targets={0: (0.9, 1.0)}, calib=worse))
    assert out.accepted is False
    out = _step(_ctx(tiny_backend(), targets={0: (0.9, 1.0)}, calib=worse,
                     cfg=GuideCfg(**{**CFG.__dict__, "acceptance": False})))
    assert out.accepted is True and out.n_tried == 1 and not close(out.z_used, out.latent)


def test_conditional_x0_measures_the_conditional_clean_prediction():
    seen = []

    def record(backend, x0, out, program, bin):
        seen.append((x0.detach(), out))
        return Evidence(x0=x0, scene={"scene_lightness": x0.mean() * 100})

    for cond_x0 in (False, True):
        seen.clear()
        b = tiny_backend()
        out = _step(_ctx(b, record, targets={0: (0.9, 1.0)}, cfg=GuideCfg(**{**CFG.__dict__, "cond_x0": cond_x0})))
        x0, v = seen[0]
        with torch.no_grad():
            want = b.decode(b.clean(out.latent, out.sigma, v.v_cond if cond_x0 else v.v))
        assert close(x0, want, 1e-6)
    assert not close(v.v, v.v_cond, 1e-6)


def _grounded(maps):
    return {q: SimpleNamespace(P=p, informative=True, peak=2.0, share=1.0, mass=1.0) for q, p in maps.items()}


class NoSeg:
    def __call__(self, x, phrases):
        raise AssertionError("attention-only evidence must not call the segmenter")


def test_attention_only_uses_phrase_maps_as_supports():
    prompt = "a red mug and a blue teapot"
    masks = half_masks()
    est = EstimatorDeps(NoSeg(), FakeSigLIP(lambda phrase, image: image.mean() * 0 + 3.0), MPROV, S)
    m = Measure(est, prompt, None, "attention_only")
    coarse = {q: torch.nn.functional.avg_pool2d(p[None, None], 2)[0, 0] for q, p in masks.items()}
    m._grounded = lambda out: _grounded(coarse)
    ev = m(None, solid((.8, .1, .1), (.1, .2, .8)), SimpleNamespace(maps={"mug": {0: None}}), translate(prompt), 4)
    s = ev.objects["mug"]["S"]
    assert s.shape == (64, 64) and float((s * masks["mug"]).sum()) > 0.9 * float(s.sum())
    est = EstimatorDeps(NoSeg(), est.siglip, MPROV, S)
    m = Measure(est, prompt, None, "attention_only")
    m._grounded = lambda out: {}
    ev = m(None, solid((.8, .1, .1), (.1, .2, .8)), None, translate(prompt), 4)
    assert float(ev.objects["mug"]["S"].sum()) == 0


def test_overlap_ablation_scores_binding_without_the_competitor():
    prompt = "a red mug and a blue teapot"
    est = EstimatorDeps(sar_tests.ScriptedSeg(), FakeSigLIP(lambda p, i: i.mean() * 0 + 3.0), MPROV, S)
    maps = {k: torch.rand(32, 32, generator=torch.Generator().manual_seed(i), dtype=torch.float64)
            for i, k in enumerate(("red", "mug", "blue", "teapot"))}
    out = SimpleNamespace(maps={"mug": {0: None}})
    scores = {}
    for ablation in ("", "overlap"):
        m = Measure(est, prompt, None, ablation)
        m._grounded = lambda o: _grounded(maps)
        m._objects = lambda *a, **k: None
        ev = Evidence()
        m._attention(out, translate(prompt), ev)
        scores[ablation] = float(ev.pairs[("binding", "red", "mug")])
    others = [maps["teapot"]]
    assert abs(scores["overlap"] - float(overlap(maps["red"], maps["mug"]))) < 1e-9
    assert abs(scores[""] - float(binding(maps["red"], maps["mug"], others, m.tau))) < 1e-9
    assert abs(scores[""] - scores["overlap"]) > 1e-6


def test_sar_without_consistency_check_keeps_every_prior_but_keeps_the_final_check():
    o, _, stage1, log = sar_tests.run(seg=sar_tests.ScriptedSeg(moved_calls={0}))
    assert "mug" in o.disabled                                      # with the check: disabled at step 0
    cfg = SarCfg(**{**sar_tests.CFG.__dict__, "consistency": False})
    o, _, stage1, log = sar_tests.run(seg=sar_tests.ScriptedSeg(moved_calls={0}), cfg=cfg)
    steps = [r for r in log.records if r["type"] == "sar_step"]
    assert not o.disabled and o.rollbacks == 0 and all("mug" in s["enabled"] for s in steps)
    assert o.used_fallback and o.fallback_reason == "spatial"       # call 0 is now the final check


def test_ablation_table_arms_run_stage1_before_stage2():
    arms = list(ABLATION_TABLE_ARMS)
    assert arms.index("stage1") < min(arms.index(a) for a in arms if a in STAGE2_MODES)
    assert set(ABLATION_ARMS) <= set(arms) and "stage1_sar_nocheck" in STAGE2_MODES
    for spec in ABLATION_ARMS.values():
        assert set(spec) <= {"cfg", "state", "measure"} and len(spec) == 1
    from annotate import ABLATION_ARM_NAMES
    assert set(ABLATION_ARM_NAMES) == set(ABLATION_TABLE_ARMS)


def test_exp3_rows_fill_our_three_methods_and_keep_the_rest_tbd():
    from annotate import EXP3_ARMS, exp3_table_rows
    assert EXP3_ARMS == ("unguided", "stage1", "stage1_sar")
    human = {"missing": {"lenient": {"stage1": 10.0}, "strict": {"stage1": 20.0}},
             "size_violation": {"stage1": 35.04}, "fidelity": {"stage1": 41.256}}
    auto = {"text_image": {"stage1": 0.268}, "text_text": {"stage1": 0.731}, "clip_iqa": {"stage1": 0.8123}}
    lines = exp3_table_rows(human, auto).splitlines()
    assert lines[1] == "S1 & 10.0 / 20.0 & 35.0 & 41.26 & 0.27 / 0.73 & 0.812\\\\"
    assert lines[0] == "SD3 & \\tbd & \\tbd & \\tbd & \\tbd & \\tbd\\\\"
    assert lines[2].startswith("\\textbf{FuzzyDiff} & \\tbd") and all(l.count("&") == 5 for l in lines)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ablation: pass")


if __name__ == "__main__":
    main()
