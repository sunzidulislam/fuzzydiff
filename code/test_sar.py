"""Tests of sar.py on the tiny model with scripted segmentation (python -I test_sar.py)."""
import dataclasses
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from constraints import Evidence  # noqa: E402
from grounding import phrase_spans  # noqa: E402
from guidance import Stage1Out  # noqa: E402
from predicates import translate  # noqa: E402
from sar import SarCfg, attribute_priors, build_log_r, build_priors, drift_ok, relation_types, run_stage2, stratum  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

DT = torch.float64
S = load_settings()
PROV = ProvisionalCalibration(S)
CFG = SarCfg(strength=0.75, k=3, alpha=1.0, beta=0.5, temp=0.1, gamma_min=0.3, overlap=0.3, nest=0.8)
PROMPT = "a red mug and a blue teapot"


def close(a, b, tol=1e-9):
    return (a - b).abs().max().item() < tol


def blob(a, b, n=64):
    s = torch.zeros(n, n, dtype=DT)
    s[a:a + 16, b:b + 16] = 1
    return s


class MemoryLog:
    def __init__(self):
        self.records = []

    def write(self, **rec):
        self.records.append(rec)


REF = {"mug": blob(10, 10), "teapot": blob(40, 40)}
MOVED = blob(40, 10)


class ScriptedSeg:
    """Returns the reference supports, except the mug has moved on the listed call numbers."""

    def __init__(self, moved_calls=(), trace=None):
        self.moved, self.n, self.trace = set(moved_calls), 0, trace

    def __call__(self, x, phrases):
        moved = self.n in self.moved
        self.n += 1
        if self.trace is not None:
            self.trace.append("seg")
        return {p: (MOVED if moved and p == "mug" else REF[p]) for p in phrases}


class Spy:
    """Wraps a backend and records calls, without changing behaviour."""

    def __init__(self, backend):
        self.b, self.calls, self.cond_v, self.cfg_v = backend, [], [], []

    def __getattr__(self, name):
        return getattr(self.b, name)

    def velocity(self, *a, **k):
        self.calls.append("cfg")
        out = self.b.velocity(*a, **k)
        self.cfg_v.append(out)
        return out

    def velocity_cond(self, *a, **k):
        self.calls.append("cond_only")
        v = self.b.velocity_cond(*a, **k)
        self.cond_v.append(v)
        return v

    def decode(self, *a, **k):
        self.calls.append("decode")
        return self.b.decode(*a, **k)


def _fixture():
    b = tiny_backend(read_blocks=(0, 1))
    program = translate(PROMPT)
    cond = b.encode_prompt(PROMPT, phrase_spans(PROMPT, program))
    with torch.no_grad():
        image = b.decode(b.noise_like_latent(3))
    traces = {"mug": REF["mug"], "teapot": REF["teapot"], "red": REF["mug"], "blue": REF["teapot"]}
    stage1 = Stage1Out(b.noise_like_latent(3), image, traces, dict(REF), [{"cid": 0, "component": "main", "mu": 0.9}],
                       {"mug": 0.9, "teapot": 0.9})
    return b, program, cond, stage1


def run(mode="sar", seg=None, calib=PROV, cfg=CFG, measure=None, backend_wrap=None, priors=None):
    b, program, cond, stage1 = _fixture()
    spy = (backend_wrap or Spy)(b)
    log = MemoryLog()
    m = measure or (lambda backend, x, out, prog, bin: Evidence(x0=x))
    out = run_stage2(spy, stage1, cond, program, PROMPT, calib, seg if seg is not None else ScriptedSeg(), cfg, 0, log,
                     mode=mode, measure=m, steps=4, priors=priors)
    return out, spy, stage1, log


# ---------------------------------------------------------------------------- priors (Task 14)
def test_geometric_mean_blocks_attention_alone():
    assert build_priors({"mug": blob(10, 10)}, {"mug": blob(40, 40)}, CFG)["mug"].max() < 0.1
    agree = build_priors({"mug": blob(10, 10)}, {"mug": blob(10, 10)}, CFG)["mug"]
    assert agree.max() > 0.99


def test_attribute_traces_do_not_become_object_priors():
    assert set(build_priors({"mug": blob(10, 10), "red": blob(10, 10)}, {"mug": blob(10, 10)}, CFG)) == {"mug"}


def test_prior_resizes_coarse_traces():
    coarse = torch.zeros(8, 8, dtype=DT)
    coarse[2:4, 2:4] = 1
    assert build_priors({"mug": coarse}, {"mug": blob(16, 16)}, CFG)["mug"].shape == (64, 64)


def test_log_r_values_and_off_keys():
    r = build_log_r({"mug": torch.ones(4096)}, {"mug": [80]}, 333, 1.0, 0.5, {"mug"})
    assert close(r[:, 80], torch.full((4096,), math.log(2.0))) and r[:, 81].abs().max() == 0
    z = build_log_r({"mug": torch.zeros(4096)}, {"mug": [80]}, 333, 1.0, 0.5, {"mug"})
    assert close(z[:, 80], torch.full((4096,), math.log(0.5)))


def test_log_r_pools_grid_priors_to_image_tokens():
    r = build_log_r({"mug": torch.ones(64, 64)}, {"mug": [80]}, 85, 1.0, 0.5, {"mug"}, n_img=64)
    assert r.shape == (64, 85)


def test_disabled_phrase_is_identity():
    r = build_log_r({"mug": torch.ones(4096)}, {"mug": [80]}, 333, 1.0, 0.5, set())
    assert r.abs().max() == 0


def test_attributes_use_their_object_prior():
    pri = {"mug": blob(10, 10), "teapot": blob(40, 40)}
    a = attribute_priors(PROMPT, translate(PROMPT), pri)
    assert a["red"] is pri["mug"] and a["blue"] is pri["teapot"]


def test_competition_only_when_flagged_and_exclusive():
    traces = {"a": blob(10, 10) * 0.98 + 0.02, "b": blob(40, 40) * 0.98 + 0.02}
    supports = {"a": blob(10, 10) * 0.98 + 0.02, "b": blob(40, 40) * 0.98 + 0.02}
    plain = build_priors(traces, supports, CFG)["a"]
    comp = dataclasses.replace(CFG, competition=True)
    assert close(build_priors(traces, supports, comp)["a"], plain)
    assert not close(build_priors(traces, supports, comp, relations={("a", "b"): "exclusive"})["a"], plain)
    assert close(build_priors(traces, supports, comp, relations={("a", "b"): "overlap"})["a"], plain)
    nested = {"a": blob(10, 10), "b": blob(10, 10)}
    assert close(build_priors(nested, nested, comp, relations={("a", "b"): "exclusive"})["a"],
                 build_priors(nested, nested, CFG)["a"])


def test_relation_types():
    assert relation_types(PROMPT, translate(PROMPT))[("mug", "teapot")] == "exclusive"
    p = "a mug on a teapot"
    assert relation_types(p, translate("a mug and a teapot"))[("mug", "teapot")] == "overlap"


# ---------------------------------------------------------------------------- drift (Task 15)
def test_drift_and_strata():
    t = (0.5, 0.5)
    assert drift_ok(REF["mug"], REF["mug"], t) and not drift_ok(MOVED, REF["mug"], t)
    assert stratum(0.01) == "small" and stratum(0.05) == "medium" and stratum(0.2) == "large"


def test_order_and_uncond_reuse():
    o, spy, _, _ = run()
    assert spy.calls[:3] == ["cfg", "decode", "cond_only"]       # segmentation runs between decode and pass 2
    assert spy.calls.count("cfg") == CFG.k and spy.calls.count("cond_only") == CFG.k
    assert o.passes == 2 * CFG.k and o.rollbacks == 0 and not o.used_fallback


def test_conditional_pass_reuses_pass_one_unconditional():
    o, spy, _, _ = run()
    first = spy.cfg_v[0]
    v = first.v_uncond + 7.0 * (spy.cond_v[0] - first.v_uncond)
    s = spy.b.sigmas(4)
    z1 = o.renoised + (s[2] - s[1]) * v
    second_input = o.latents[1]
    assert close(z1, second_input, 1e-6)


def test_drift_on_first_step_disables_without_rollback():
    o, _, _, log = run(seg=ScriptedSeg(moved_calls={0}))
    steps = [r for r in log.records if r["type"] == "sar_step"]
    assert "mug" in o.disabled and o.rollbacks == 0 and "mug" not in steps[0]["enabled"]
    assert all("mug" not in s["enabled"] for s in steps)


def test_post_step_failure_rolls_back_and_recomputes():
    o, spy, _, _ = run(seg=ScriptedSeg(moved_calls={1}))
    assert o.rollbacks == 1 and "mug" in o.disabled
    assert close(o.latents[1], o.sar_free[0], 1e-6)              # restored to the SAR-free update
    assert spy.calls.count("cfg") == CFG.k + 1                   # one extra SAR-free pass


def test_final_failure_returns_stage1_image():
    o, _, stage1, _ = run(seg=ScriptedSeg(moved_calls={3}))
    assert o.used_fallback and close(o.image, stage1.image)
    assert o.fallback_reason == "spatial"


def test_attribute_regression_returns_stage1_image():
    calib = PROV

    def measure(backend, x, out, program, bin):
        return Evidence(x0=x, objects={q: {"h": torch.tensor(-9.0), "A": torch.tensor(0.1), "psi": 1.0,
                                           "p": torch.tensor(0.9)} for q in program.objects})

    b, program, cond, stage1 = _fixture()
    pres = next(i for i, c in enumerate(program.constraints) if c.family == "presence")
    stage1.finals = [{"cid": pres, "component": "main", "mu": 0.9}]
    o = run_stage2(Spy(b), stage1, cond, program, PROMPT, calib, ScriptedSeg(), CFG, 0, MemoryLog(), mode="sar",
                   measure=measure, steps=4)
    assert o.used_fallback and o.fallback_reason == "attribute" and close(o.image, stage1.image)


def test_plain_matches_identity_sar():
    a, _, _, _ = run(mode="plain")
    b_, _, _, _ = run(mode="sar", cfg=dataclasses.replace(CFG, alpha=0.0, beta=0.0))
    assert close(a.latent, b_.latent, 1e-6) and a.passes == CFG.k


def test_plain_has_no_drift_logic():
    o, spy, _, _ = run(mode="plain", seg=ScriptedSeg(moved_calls={0, 1, 2, 3}))
    assert spy.calls.count("cond_only") == 0 and not o.disabled and not o.used_fallback


def test_no_thresholds_means_no_sar():
    class NoThresholds(ProvisionalCalibration):
        def drift_thresholds(self, stratum_name):
            return None

    o, spy, _, _ = run(calib=NoThresholds(S))
    assert spy.calls.count("cond_only") == 0 and o.fallback_objects == {"mug", "teapot"}


def test_invalid_bin_suspends_sar_without_disabling():
    class Suspended(ProvisionalCalibration):
        def stage2_valid(self, bin):
            return False

    seg = ScriptedSeg(moved_calls={0, 1, 2})
    o, spy, _, _ = run(calib=Suspended(S), seg=seg)
    assert spy.calls.count("cond_only") == 0 and o.suspended_steps == CFG.k and not o.disabled and seg.n == 1


def test_attention_failure_gives_plain_pass_and_attribute_failure_keeps_objects():
    class None_(ProvisionalCalibration):
        def attention_ok(self, kind):
            return False

    o, spy, _, _ = run(calib=None_(S))
    assert spy.calls.count("cond_only") == 0

    class NoAttr(ProvisionalCalibration):
        def attention_ok(self, kind):
            return kind == "objects"

    seen = []

    class Capture(Spy):
        def velocity_cond(self, z, sigma, cond, log_r):
            seen.append(log_r.clone())
            return super().velocity_cond(z, sigma, cond, log_r)

    run(calib=NoAttr(S), backend_wrap=Capture)
    red = 77 + PROMPT.split().index("red")
    mug = 77 + PROMPT.split().index("mug")
    assert seen[0][:, red].abs().max() == 0 and seen[0][:, mug].abs().max() > 0


def test_confidence_scaling_is_off_by_default_and_disables_weak_priors():
    weak = dataclasses.replace(CFG, confidence=True, gamma_min=2.0)     # no prior can reach gamma 2
    o, spy, _, _ = run(cfg=weak)
    assert spy.calls.count("cond_only") == 0 and o.passes == CFG.k


def test_seed_shared_across_modes():
    assert close(run(mode="plain")[0].renoised, run(mode="sar")[0].renoised)


def test_placebo_priors_change_the_bias_only():
    other = {"mug": blob(40, 40), "teapot": blob(10, 10)}
    a = run(mode="sar")[0]
    p = run(mode="sar", priors=other)[0]
    assert close(a.renoised, p.renoised) and not close(a.latent, p.latent, 1e-9)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("sar: pass")


if __name__ == "__main__":
    main()
