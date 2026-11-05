"""Tests of guidance.py on the tiny model with a lightness measurement (python -I test_guidance.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from constraints import Evidence, new_state  # noqa: E402
from estimators import scene_lightness  # noqa: E402
from guidance import GuideCfg, Stage1Ctx, bounded_step, guided_step, project_tier, run_stage1  # noqa: E402
from predicates import Constraint, Program  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

PROV = ProvisionalCalibration(load_settings())
CFG = GuideCfg(steps=4, cfg=7.0, sigma_lo=0.0, eta0=0.5, delta=0.1, dmax=0.2, eps={1: 0.01, 2: 0.01}, backtrack=2,
               ramp=2)
PROGRAM = Program(objects=[], constraints=[Constraint("F2a", ("room",), None, "scene_lightness", 1)])


def close(a, b, tol=1e-9):
    return (a - b).abs().max().item() < tol


class MemoryLog:
    def __init__(self):
        self.records = []

    def write(self, **rec):
        self.records.append(rec)


def measure(backend, x0, out, program, bin):
    return Evidence(x0=x0, scene={"scene_lightness": scene_lightness(x0)})


def nan_measure(backend, x0, out, program, bin):
    return Evidence(x0=x0, scene={"scene_lightness": scene_lightness(x0) * torch.tensor(float("nan"))})


def test_projection_properties():
    a = torch.tensor([1., 0.])
    g = torch.tensor([-1., 1.])
    p = project_tier(g, [a])
    assert torch.dot(p, a) >= -1e-9 and close(p, torch.tensor([0., 1.]))
    assert close(project_tier(torch.tensor([1., 1.]), [a]), torch.tensor([1., 1.]))
    b = torch.tensor([0., 1.])
    p2 = project_tier(torch.tensor([-1., -1.]), [a, b])
    assert torch.dot(p2, a) >= -1e-9 and torch.dot(p2, b) >= -1e-9
    assert close(project_tier(torch.tensor([-1., 0.]), [a, a]), torch.zeros(2))   # collinear
    assert close(project_tier(g, []), g)
    assert close(project_tier(g, [torch.zeros(2)]), g)                            # empty higher tier


def test_projection_preserves_tensor_shape():
    a = torch.ones(1, 2, 3, 3)
    assert project_tier(-torch.ones(1, 2, 3, 3), [a]).shape == a.shape


def test_step_clip_bounds_every_element():
    g = torch.zeros(4)
    g[0] = 100.
    z2, _ = bounded_step(torch.zeros(4), g, torch.tensor(1.), CFG)
    assert z2.abs().max() <= CFG.dmax + 1e-6 and z2[0] < 0
    z3, eta = bounded_step(torch.zeros(4), torch.ones(4), torch.tensor(1.), GuideCfg(**{**CFG.__dict__, "eta0": 0.01}))
    assert abs(eta - 0.01 * 1 / 1.1) < 1e-6 and abs(z3[0] + eta) < 1e-6           # RMS-normalized, no clip


def _ctx(backend, measure_fn=measure, log=None, targets=None, cfg=CFG, calib=PROV):
    return Stage1Ctx(backend=backend, program=PROGRAM, cond=backend.encode_prompt("a room"), calib=calib,
                     measure=measure_fn, cfg=cfg, state=new_state(), log=log or MemoryLog(), targets=targets,
                     prompt="a room", seed=0)


def _step(ctx):
    b = ctx.backend
    return guided_step(b.noise_like_latent(0), torch.tensor(0.8, dtype=torch.float64),
                       torch.tensor(0.6, dtype=torch.float64), 0, ctx)


def test_nan_gradient_falls_back_to_euler():
    out = _step(_ctx(tiny_backend(), nan_measure, targets={0: (0.9, 1.0)}))
    assert close(out.latent_next, out.latent + (out.sigma_next - out.sigma) * out.v_plain)
    assert out.accepted is False and ("nonfinite" in out.rejections or "no_active" in out.rejections)


def test_nonfinite_gradient_only_is_caught():
    calls = []

    def half_nan(backend, x0, out, program, bin):
        s = scene_lightness(x0)
        s = s + (s * 0).detach()
        calls.append(1)
        hooked = s.clone()
        hooked.register_hook(lambda g: g * float("nan"))
        return Evidence(x0=x0, scene={"scene_lightness": hooked})

    out = _step(_ctx(tiny_backend(), half_nan, targets={0: (0.9, 1.0)}))
    assert "nonfinite" in out.rejections and out.accepted is False and torch.isfinite(out.latent_next).all()


def test_loss_with_no_path_to_the_latent_falls_back_to_euler():
    def constant(backend, x0, out, program, bin):
        return Evidence(x0=x0, scene={"scene_lightness": torch.tensor(50.0, dtype=torch.float64)})

    out = _step(_ctx(tiny_backend(), constant, targets={0: (0.9, 1.0)}))
    assert out.accepted is False and "small_gradient" in out.rejections
    assert close(out.latent_next, out.latent + (out.sigma_next - out.sigma) * out.v_plain)


class Worse(ProvisionalCalibration):
    """Reports every candidate as worse: evidence measured without gradients scores below the scale."""

    def phi(self, family, quantity, bin, e):
        return e / 100 if e.requires_grad else torch.full_like(e, -1.0)


def test_rejection_backtracks_then_keeps_latent():
    out = _step(_ctx(tiny_backend(), targets={0: (0.9, 1.0)}, calib=Worse(load_settings())))
    assert out.n_tried == CFG.backtrack + 1 and out.accepted is False and close(out.z_used, out.latent)
    assert close(out.latent_next, out.latent + (out.sigma_next - out.sigma) * out.v_plain)
    assert "total" in out.rejections


def test_accepted_step_logs_reporting_schema():
    log = MemoryLog()
    _step(_ctx(tiny_backend(), log=log, targets={0: (0.9, 1.0)}))
    con = next(r for r in log.records if r["type"] == "constraint")
    assert set(con["gates"]) == {"validity", "presence", "support", "domain", "confidence"}
    assert {"prompt", "seed", "step", "sigma", "bin", "family", "cid", "tier", "active", "accepted", "dL"} <= set(con)
    st = next(r for r in log.records if r["type"] == "step")
    assert {"tiers", "loss_before", "loss_after", "cos", "removed", "rejections", "accepted", "eps"} <= set(st)


def _run(guided, target=None, cfg=None):
    b = tiny_backend()
    c = cfg or GuideCfg(**{**CFG.__dict__, "guided": guided})
    return b, run_stage1(b, PROGRAM, b.encode_prompt("a room"), 0, PROV, measure, c, MemoryLog(),
                         targets={0: target} if target else None)


def test_unguided_matches_manual_euler():
    b, out = _run(False)
    z = b.noise_like_latent(0)
    s = b.sigmas(CFG.steps)
    c = b.encode_prompt("a room")
    with torch.no_grad():
        for i in range(CFG.steps):
            z = z + (s[i + 1] - s[i]) * b.velocity(z, s[i], c).v
    assert close(out.latent, z)


def test_guidance_moves_lightness_on_tiny_model():
    _, a = _run(False)
    _, b = _run(True, target=(0.9, 1.0))
    assert b.finals[0]["s"] > a.finals[0]["s"]


def test_window_skips_low_noise_steps():
    log = MemoryLog()
    b = tiny_backend()
    run_stage1(b, PROGRAM, b.encode_prompt("a room"), 0, PROV, measure,
               GuideCfg(**{**CFG.__dict__, "sigma_lo": 0.9}), log, targets={0: (0.9, 1.0)})
    guided_sigmas = [r["sigma"] for r in log.records if r["type"] == "step"]
    assert guided_sigmas and all(s >= 0.9 for s in guided_sigmas)


def test_guidance_can_start_after_the_noisiest_steps():
    log = MemoryLog()
    b = tiny_backend()
    run_stage1(b, PROGRAM, b.encode_prompt("a room"), 0, PROV, measure,
               GuideCfg(**{**CFG.__dict__, "sigma_hi": 0.75}), log, targets={0: (0.9, 1.0)})
    guided_sigmas = [r["sigma"] for r in log.records if r["type"] == "step"]
    assert guided_sigmas and all(s <= 0.75 for s in guided_sigmas)
    assert len(guided_sigmas) < CFG.steps


def test_a_met_target_is_not_reported_as_a_small_gradient():
    b = tiny_backend()
    z = b.noise_like_latent(0)
    s0 = torch.tensor(0.8, dtype=torch.float64)
    with torch.no_grad():
        x0 = b.decode(b.clean(z, s0, b.velocity(z, s0, b.encode_prompt("a room")).v))
    now = float(scene_lightness(x0)) / 100
    out = _step(_ctx(b, targets={0: (now - 0.2, now + 0.2)}))
    assert out.rejections == ["target_met"] and out.accepted is False


def test_final_record_has_score_and_mu():
    _, out = _run(False)
    f = out.finals[0]
    assert {"family", "cid", "tier", "s", "mu", "raw"} <= set(f) and 0 <= f["mu"] <= 1


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("guidance: pass")


if __name__ == "__main__":
    main()
