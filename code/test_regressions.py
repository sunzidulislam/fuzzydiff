"""One test per review finding on the GPU and fitting paths (python -I test_regressions.py).

No GPU is available in CI, so device tests use the `meta` device: a cross-device operation raises a
device error there, while correctly placed tensors only fail later, when a value is read.
"""
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import color_stats  # noqa: E402
from annotate import build_rating_page  # noqa: E402,F401
from calibration import ProvisionalCalibration  # noqa: E402
from calibration_fit import FittedCalibration  # noqa: E402
from color_stats import mixture_stats, srgb_to_oklab, trimmed_mean  # noqa: E402
from constraints import Evidence, Frozen, new_state, tier_losses  # noqa: E402
from estimators import processor_size, region_crop  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from fit_calibration import fit_main, rows_from_run  # noqa: E402
from grounding import aggregate_maps, missing_phrases, phrase_spans  # noqa: E402
from guidance import GuideCfg, Stage1Ctx, guided_step, run_stage1  # noqa: E402
from measure import measure_scene  # noqa: E402
from predicates import Constraint, Program, translate  # noqa: E402
from reporting import LeakageError, ProvisionalError, StepLog, read_jsonl, tier_rows  # noqa: E402
from run_experiment import Deps, Job, _is_finished, check_test_run, image_id, merge_logs, peak_memory, run_jobs, summarize_arms  # noqa: E402
from sar import SarCfg, drift_ok, run_stage2  # noqa: E402
from sd3_backend import TextCond, VelocityOut, tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402
from test_sar import MemoryLog as SarLog, REF, ScriptedSeg, _fixture  # noqa: E402

S = load_settings()
PROV = ProvisionalCalibration(S)
DT = torch.float64
CFG = GuideCfg(steps=4, cfg=7.0, sigma_lo=0.0, eta0=0.2, delta=0.1, dmax=0.2, eps={1: 0.01, 2: 0.01}, backtrack=1, ramp=2)
SAR = SarCfg(strength=0.75, k=3, alpha=1.0, beta=0.5, temp=0.1, gamma_min=0.3, overlap=0.3, nest=0.8)
ROOM = Program(objects=[], constraints=[Constraint("F2a", ("room",), None, "scene_lightness", 1)])


def with_tmp(fn):
    d = tempfile.mkdtemp()
    try:
        fn(Path(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def assert_no_device_mismatch(fn):
    try:
        fn()
    except Exception as e:
        assert "expected device" not in str(e) and "same device" not in str(e), e


class Log:
    def __init__(self):
        self.records = []

    def write(self, **r):
        self.records.append(r)


# ---------------------------------------------------------------------------- Critical
def test_c1_drift_check_moves_the_reference_to_the_current_device():
    assert_no_device_mismatch(lambda: drift_ok(REF["mug"].to("meta"), REF["mug"], (0.5, 0.5)))


def test_c2_colour_statistics_create_tensors_on_the_image_device():
    seen = []
    arange, tensor = torch.arange, torch.tensor

    def spy_arange(*a, **k):
        seen.append(("arange", k.get("device")))
        return arange(*a, **k)

    def spy_tensor(*a, **k):
        seen.append(("tensor", k.get("device")))
        return tensor(*a, **k)

    lab = srgb_to_oklab(torch.rand(3, 16, 16, dtype=DT))
    w = torch.ones(16, 16, dtype=DT)
    torch.arange, torch.tensor = spy_arange, spy_tensor
    try:
        color_stats._blur(lab, 2.0)
        trimmed_mean(lab, w)
    finally:
        torch.arange, torch.tensor = arange, tensor
    assert {name for name, _ in seen} == {"arange", "tensor"} and all(device is not None for _, device in seen), seen


def _frozen(family, args, tier):
    return Frozen(Constraint(family, args, None, None, 1, tier=tier), 0, tier, 4, 1.0, 0.5, math.inf, 0.1, {})


def test_c3_losses_from_attention_and_estimators_can_be_summed():
    binding = _frozen("binding", ("red", "mug"), 2)
    presence = _frozen("presence", ("mug",), 1)
    ev = Evidence(objects={"mug": {"h": torch.tensor(0.0)}}, pairs={("binding", "red", "mug"): torch.tensor(0.0).to("meta")})
    losses = tier_losses([binding, presence], ev, PROV)
    assert len({v.device for v in losses.values()}) == 1
    sum(losses.values())


def test_c3_attention_statistics_are_float32():
    maps = torch.full((64,), 1e-3, dtype=torch.bfloat16)
    cond = TextCond(None, None, None, None, list(range(10)), {"mug": [0]}, [])
    out = VelocityOut(None, None, None, maps={"mug": {0: maps}}, shares={"mug": {0: maps}}, text_mass={0: maps})
    assert aggregate_maps(out, cond, [0], (1.0, 0.0, 0.0))["mug"].P.dtype == torch.float32


def test_c4_transformer_waits_until_the_text_encoders_are_gone():
    b = tiny_backend(defer_transformer=True)
    calls = []
    original = b.transformer.to
    b.transformer.to = lambda *a, **k: (calls.append(a), original(*a, **k))[1]
    b.encode_prompt("x")
    assert not calls
    try:
        b.velocity(b.noise_like_latent(0), torch.tensor(0.5), b.encode_prompt("x"))
        assert False, "the transformer must not run before text encoding has finished"
    except RuntimeError as e:
        assert "unload_text_encoders" in str(e)
    b.unload_text_encoders()
    assert len(calls) == 1
    b.velocity(b.noise_like_latent(0), torch.tensor(0.5), b.encode_prompt("x"))


def test_c4_verify_checkpoint_does_not_keep_encoders_with_the_transformer():
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "verify_checkpoint.py")).read()
    assert "torch_dtype=dt).to(dev)" not in src and "del_text_encoders" in src


def test_c4_nothing_keeps_the_text_encoders_alive_after_deletion():
    import gc
    import inspect
    import weakref
    from types import SimpleNamespace

    import verify_checkpoint as vc
    tree = torch.nn.Linear
    pipe = SimpleNamespace(text_encoder=tree(1, 1), text_encoder_2=tree(1, 1), text_encoder_3=tree(1, 1))
    t5 = weakref.ref(pipe.text_encoder_3)
    vc.encoders_to(pipe, "cpu")
    vc.del_text_encoders(pipe)
    gc.collect()
    assert t5() is None
    # main() holds a reference for its whole life if it loops over the encoders itself
    assert "for encoder in" not in inspect.getsource(vc.main)


def test_t4_vae_can_run_at_higher_precision_than_the_transformer():
    b = tiny_backend(vae_dtype=torch.float32)
    z = b.noise_like_latent(0)
    assert b.decode(z).shape[1] == 3 and b.encode(b.decode(z)).shape == z.shape


def test_t4_timestep_is_not_rounded_to_transformer_precision():
    b = tiny_backend()
    b.dtype, b.latent_dtype = torch.float16, torch.float32
    t = b._timestep(torch.tensor(0.8137, dtype=DT), 2)
    assert t.dtype == torch.float32 and t.shape == (2,) and abs(t[0].item() - 813.7) < 0.01


def test_t4_attention_readout_keeps_small_probabilities_in_float16():
    from fuzzydiff_sd3_attention import chunked_text_attention
    torch.manual_seed(0)
    q, k = torch.randn(2, 256, 16, dtype=DT) * 3, torch.randn(2, 300, 16, dtype=DT) * 3
    idx = torch.arange(20)
    exact = chunked_text_attention(q, k, 256, idx, 0.25, None, 64)
    half = chunked_text_attention(q.to(torch.float16), k.to(torch.float16), 256, idx, 0.25, None, 64)
    assert half.dtype == torch.float32
    assert (half.double() - exact).abs().max() < 1e-2 and (half > 0).all()


def test_t4_defaults_use_a_precision_this_gpu_can_run():
    import inspect

    import run_experiment
    import verify_checkpoint
    assert 'default="float16"' in inspect.getsource(run_experiment.main)
    assert 'default="float16"' in inspect.getsource(verify_checkpoint.main)
    assert "bf16" not in S.get("cfgPrecision", str)


def _count_resnets(vae):
    from diffusers.models.resnet import ResnetBlock2D
    return sum(isinstance(m, ResnetBlock2D) for m in vae.decoder.modules())


def test_t4_vae_backward_rebuilds_one_resnet_at_a_time():
    b = tiny_backend()
    calls = []
    original = torch.utils.checkpoint.checkpoint

    def spy(*a, **k):
        calls.append(1)
        return original(*a, **k)

    z = b.noise_like_latent(0).requires_grad_()
    torch.utils.checkpoint.checkpoint = spy
    try:
        with_ckpt = b.decode(z).sum()
    finally:
        torch.utils.checkpoint.checkpoint = original
    assert len(calls) == _count_resnets(b.vae)                   # per resnet, and not also per block
    with_ckpt.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_t4_vae_gradient_is_unchanged_by_checkpointing():
    from sd3_backend import checkpoint_vae_resnets
    plain = tiny_backend()
    for m in plain.vae.decoder.modules():
        m.__dict__.pop("forward", None)                          # undo the per-resnet wrapper
    z = plain.noise_like_latent(0).requires_grad_()
    plain.decode(z).sum().backward()
    wrapped = tiny_backend()
    y = wrapped.noise_like_latent(0).requires_grad_()
    wrapped.decode(y).sum().backward()
    assert (z.grad - y.grad).abs().max() < 1e-9
    assert checkpoint_vae_resnets is not None


def test_t4_vae_without_gradients_uses_no_checkpoint():
    b = tiny_backend()
    calls = []
    original = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        with torch.no_grad():
            b.decode(b.noise_like_latent(0))
    finally:
        torch.utils.checkpoint.checkpoint = original
    assert not calls


def test_verify_checkpoint_can_write_its_tables_to_another_folder():
    import verify_checkpoint as vc
    from table_rows import write_rows

    def body(tmp):
        vc.copy_table_inputs(tmp)
        write_rows(tmp)
        assert (tmp / "processor_check_rows.tex").exists()
    with_tmp(body)


def test_kaggle_readout_gradient_survives_zero_text_mass():
    b = tiny_backend(read_blocks=(0, 1))
    with torch.no_grad():                                        # image keys dominate, as they can on real weights
        for block in b.transformer.transformer_blocks:
            block.attn.to_q.weight.mul_(300)
            block.attn.to_k.weight.mul_(300)
    c = b.encode_prompt("a red mug", {"mug": (6, 9)})
    z = b.noise_like_latent(0).requires_grad_()
    out = b.velocity(z, torch.tensor(0.5, dtype=DT), c, readout=True)
    assert out.text_mass[0].min() == 0, "this test needs an image query with no text mass"
    assert torch.isfinite(out.shares["mug"][0]).all()
    out.maps["mug"][0].sum().backward()
    assert torch.isfinite(z.grad).all()


def test_kaggle_peak_memory_helpers_use_the_requested_gpu():
    import verify_checkpoint as vc
    seen = []
    original = torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats
    torch.cuda.max_memory_allocated = lambda device=None: (seen.append(device), 0)[1]
    torch.cuda.reset_peak_memory_stats = lambda device=None: seen.append(device)
    try:
        dev = torch.device("cuda:1")
        vc.peak_gb(dev)
        vc.reset_peak(dev)
    finally:
        torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats = original
    assert seen == [dev, dev]


def test_kaggle_oom_record_shows_every_gpu():
    def body(tmp):
        def stage1(*a, **k):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        d = Deps(tiny_backend(), PROV, measure_scene, CFG, S, stage1=stage1)
        run_jobs([Job("unguided", "a room", 0, "dev", "m1_eval", ROOM)], d, tmp)
        meta = json.load(open(tmp / "unguided" / image_id("a room", 0) / "meta.json"))
        assert meta["failed"] == "oom" and "peak_by_gpu" in meta and "CUDA out of memory" in meta["error"]
    with_tmp(body)


def test_kaggle_vae_defaults_to_16_bits():
    import inspect

    import run_experiment
    import sd3_backend
    import verify_checkpoint
    assert inspect.signature(sd3_backend.load_backend).parameters["vae_dtype"].default == torch.float16
    assert '"--vae-dtype", default="float16"' in inspect.getsource(run_experiment.main)
    assert '"--vae-dtype", default="float16"' in inspect.getsource(verify_checkpoint.main)
    assert "fp16 transformer and VAE" in S.get("cfgPrecision", str)


# ---------------------------------------------------------------------------- Important
def test_i1_mixture_uses_the_object_not_the_background():
    x = torch.full((3, 64, 64), 0.05, dtype=DT)
    x[:, 26:38, 26:38] = 1.0                                     # a small bright object on a dark scene
    lab = srgb_to_oklab(x)
    core = torch.zeros(64, 64, dtype=DT)
    core[28:36, 28:36] = 1
    c1, c2 = lab[:, 30, 30], lab[:, 30, 30] + torch.tensor([0.0, 0.1, 0.1], dtype=DT)
    clean = mixture_stats(lab, core, c1, c2, sigma_p=0.5)
    with_eps = mixture_stats(lab, core + 1e-8, c1, c2, sigma_p=0.5)
    assert abs(clean["rho"]) < 0.05 and abs(with_eps["rho"]) < 0.05
    assert torch.abs(with_eps["cbar"] - lab[:, 30, 30]).max() < 0.02


def test_i1_empty_support_stays_finite():
    lab = srgb_to_oklab(torch.rand(3, 16, 16, dtype=DT))
    empty = torch.zeros(16, 16, dtype=DT)
    c1, c2 = torch.tensor([0.5, 0.0, 0.0], dtype=DT), torch.tensor([0.5, 0.1, 0.1], dtype=DT)
    assert all(torch.isfinite(v).all() for v in mixture_stats(lab, empty, c1, c2, sigma_p=1.0).values())
    assert torch.isfinite(trimmed_mean(lab, empty)).all()


def test_i2_clipseg_uses_the_processor_size():
    class Processor:
        size = {"height": 352, "width": 352}

    assert processor_size(Processor(), 224) == 352
    Processor.size = 352
    assert processor_size(Processor(), 224) == 352
    Processor.size = None
    assert processor_size(Processor(), 224) == 224


def test_i3_nan_in_one_tier_reaches_the_euler_fallback():
    prog = Program(objects=["room"], constraints=[Constraint("presence", ("room",), None, None, 1, tier=1),
                                                  Constraint("F2a", ("room",), None, "scene_lightness", 1)])

    def measure(backend, x0, out, program, bin):
        light = x0.mean() * 100
        hooked = light.clone()
        hooked.register_hook(lambda g: g * float("nan"))
        return Evidence(x0=x0, scene={"scene_lightness": hooked},
                        objects={"room": {"h": x0.mean(), "p": torch.tensor(0.9), "A": torch.tensor(0.5), "psi": 1.0}})

    b = tiny_backend()
    ctx = Stage1Ctx(b, prog, b.encode_prompt("a room"), PROV, measure, CFG, new_state(), Log(), None, "a room", 0)
    s0, s1 = torch.tensor(0.8, dtype=DT), torch.tensor(0.6, dtype=DT)
    out = guided_step(b.noise_like_latent(0), s0, s1, 0, ctx)
    assert "nonfinite" in out.rejections and out.accepted is False and torch.isfinite(out.latent_next).all()


def test_i4_every_step_and_the_final_image_can_feed_calibration():
    def body(tmp):
        job = Job("unguided", "a room", 0, "cal", "m1_eval", ROOM, overrides={"log_all": True})
        d = Deps(tiny_backend(), PROV, measure_scene, CFG, S)
        assert run_jobs([job], d, tmp).done == 1
        item = f"unguided/{image_id('a room', 0)}"
        rows = rows_from_run(tmp, ratings={item: 0.5})["ratings"]
        assert {r["bin"] for r in rows} >= {1, 4} and all(r["item"] == item for r in rows)
        assert [r for r in rows if r["bin"] == 4]
    with_tmp(body)


def test_i4_plain_runs_do_not_spend_estimator_passes():
    calls = []

    def measure(*a):
        calls.append(1)
        return measure_scene(*a)

    b = tiny_backend()
    run_stage1(b, ROOM, b.encode_prompt("a room"), 0, PROV, measure, GuideCfg(**{**CFG.__dict__, "guided": False}), Log())
    assert len(calls) == 1                                       # the final image only


def _run_with_log(tmp, records, meta=None):
    d = tmp / "stage1" / "img"
    d.mkdir(parents=True)
    json.dump({"prompt": "a room", **(meta or {})}, open(d / "meta.json", "w"))
    with open(d / "log.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _constraint(**k):
    return {"type": "constraint", "split": "cal", "family": "F6", "quantity": "gray:green", "bin": 2, "raw": 0.3,
            "component": "main", **k}


def test_i5_only_the_mixture_fraction_feeds_the_perceptual_map():
    def body(tmp):
        _run_with_log(tmp, [_constraint(), _constraint(component="e", raw=9.0), _constraint(component="zeta", raw=9.0)])
        rows = rows_from_run(tmp, ratings={"stage1/img": 0.5})["ratings"]
        assert [r["e"] for r in rows] == [0.3]
    with_tmp(body)


def test_i6_test_records_cannot_enter_fitting():
    def body(tmp):
        _run_with_log(tmp, [_constraint(split="test")])
        try:
            rows_from_run(tmp, ratings={"stage1/img": 0.5})
            assert False
        except LeakageError:
            pass
    with_tmp(body)


def test_i6_test_images_need_a_fitted_calibration_and_a_manifest():
    def body(tmp):
        manifest = tmp / "support_manifest.json"
        manifest.write_text("{}")
        check_test_run("dev", "provisional", None)
        for tag, path in (("provisional", manifest), ("fitted", None), ("fitted", tmp / "absent.json")):
            try:
                check_test_run("test", tag, path)
                assert False, (tag, path)
            except LeakageError:
                pass
        check_test_run("test", "fitted", manifest)
    with_tmp(body)


def test_i7_untagged_and_partly_provisional_records_are_refused():
    base = {"type": "step", "tiers": [3], "loss_before": {"3": 1.0}, "loss_after": {"3": 0.5}, "accepted": True}
    for bad in (base, {**base, "calibration": "provisional"},
                {**base, "calibration": "fitted", "provisional": ["cfgEtaZero"]}):
        try:
            tier_rows([bad])
            assert False, bad
        except ProvisionalError:
            pass
    tier_rows([{**base, "calibration": "fitted", "provisional": []}])


def test_i7_steplog_writes_what_was_provisional_at_write_time():
    def body(tmp):
        used = set()
        log = StepLog(str(tmp / "a.jsonl"), "dev", "fitted", lambda: used)
        log.write(type="step")
        used.add("F2a|chroma|very|1")
        log.write(type="step")
        log.close()
        a, b = read_jsonl(tmp / "a.jsonl")
        assert a["provisional"] == [] and b["provisional"] == ["F2a|chroma|very|1"]
    with_tmp(body)


def test_i7_a_calibration_built_on_provisional_parts_is_not_fitted():
    def body(tmp):
        (tmp / "input.json").write_text("{}")
        c = fit_main(tmp / "input.json", tmp / "c.json", S)
        assert c.tag == "provisional" and "prototypes" in c.parts["provisional_parts"]
        assert FittedCalibration.load(tmp / "c.json").tag == "provisional"
    with_tmp(body)


def test_stage2_validity_ends_at_the_final_bin_without_gaps():
    def body(tmp):
        cells = {"1": {"n": 500, "rho_lb": 0.9}, "2": {"n": 500, "rho_lb": 0.0},
                 "3": {"n": 500, "rho_lb": 0.9}, "4": {"n": 500, "rho_lb": 0.9}}
        (tmp / "input.json").write_text(json.dumps({"stage2_cells": cells}))
        c = fit_main(tmp / "input.json", tmp / "c.json", S, min_images=100)
        assert [c.stage2_valid(b) for b in (1, 2, 3, 4)] == [False, False, True, True]
    with_tmp(body)


def test_i8_missing_presence_fit_leaves_objects_inactive_not_crashed():
    def body(tmp):
        (tmp / "input.json").write_text("{}")
        c = fit_main(tmp / "input.json", tmp / "c.json", S)
        h = c.presence_logit(4, torch.tensor(1.0), torch.tensor(1.0))
        assert torch.isnan(h) and not c.valid("presence", 4)
        assert c.parts["not_valid"] and "leakage" in c.parts["not_valid"]
    with_tmp(body)


def test_i9_validity_counts_images_not_steps():
    rows = []
    for image in range(40):                                      # 40 images, 5 rows each
        for step in range(5):
            e = (image * 2.5) + step * 0.01
            rows.append({"family": "F2a", "quantity": "scene_lightness", "bin": 1, "e": e, "rating": e / 100,
                         "base": f"b{image}", "item": f"img{image}"})

    def body(tmp):
        json.dump({"ratings": rows}, open(tmp / "input.json", "w"))
        calib = fit_main(tmp / "input.json", tmp / "c.json", S, min_images=100)
        assert calib.parts["cells"]["F2a"]["scene_lightness"]["1"]["n"] == 40
        assert calib.valid("F2a", 1) is False
    with_tmp(body)


def _sar_run(calib=PROV, cfg=SAR, seg=None, traces=None, priors=None, tokens=True):
    b, program, cond, stage1 = _fixture()
    if traces is not None:
        stage1.traces = traces
    if not tokens:
        cond = TextCond(cond.embeds, cond.pooled, cond.uncond_embeds, cond.uncond_pooled, cond.valid_text, {}, cond.t5_offsets)
    return run_stage2(b, stage1, cond, program, "a red mug and a blue teapot", calib, seg or ScriptedSeg(), cfg, 0, SarLog(),
                      mode="sar", measure=lambda *a: Evidence(), steps=4, priors=priors)


def test_i10_a_sar_run_with_a_bias_reports_how_many_steps_used_it():
    o = _sar_run()
    assert o.bias_steps == SAR.k and o.no_op_reason is None


def test_i10_no_informative_traces_is_reported_as_a_no_op():
    o = _sar_run(traces={})
    assert o.bias_steps == 0 and o.no_op_reason == "no_priors"


def test_i10_all_objects_disabled_first_is_reported():
    o = _sar_run(seg=ScriptedSeg(moved_calls={0}), cfg=SarCfg(**{**SAR.__dict__}))
    assert o.bias_steps >= 1                                     # the teapot keep its bias
    both_moved = lambda x, phrases: {p: torch.zeros(64, 64, dtype=DT) for p in phrases}  # noqa: E731
    o = _sar_run(seg=both_moved)
    assert o.bias_steps == 0 and o.no_op_reason == "all_disabled"


def test_i10_phrases_without_tokens_are_not_logged_as_biased():
    o = _sar_run(tokens=False)
    assert o.bias_steps == 0 and o.no_op_reason == "no_tokens"


def test_i10_attention_failure_is_reported():
    class Off(ProvisionalCalibration):
        def attention_ok(self, kind):
            return False

    assert _sar_run(calib=Off(S)).no_op_reason == "attention_failed"


def test_i10_summary_reports_share_without_a_bias():
    def body(tmp):
        for i, steps in enumerate((0, 3)):
            d = tmp / "stage1_sar" / f"i{i}"
            d.mkdir(parents=True)
            json.dump({"arm": "stage1_sar", "bias_steps": steps}, open(d / "meta.json", "w"))
        assert summarize_arms(tmp)["stage1_sar"]["no_bias_share"] == 0.5
    with_tmp(body)


def test_i11_peak_memory_is_reported_for_every_gpu():
    original = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.max_memory_allocated)
    torch.cuda.is_available, torch.cuda.device_count = (lambda: True), (lambda: 2)
    torch.cuda.max_memory_allocated = lambda device=None: {0: 2 ** 30, 1: 3 * 2 ** 30}[device]
    try:
        peaks = peak_memory()
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.max_memory_allocated = original
    assert peaks == {"cuda:0": 1.0, "cuda:1": 3.0}


def test_i11_a_gpu_that_was_never_used_does_not_fail_the_image():
    from run_experiment import _reset_peaks
    original = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.reset_peak_memory_stats)
    reset = []

    def fake_reset(i):
        if i == 1:
            raise RuntimeError("Invalid device argument 1: did you call init?")
        reset.append(i)

    torch.cuda.is_available, torch.cuda.device_count, torch.cuda.reset_peak_memory_stats = (lambda: True), (lambda: 2), fake_reset
    try:
        _reset_peaks()
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.reset_peak_memory_stats = original
    assert reset == [0]


def test_i12_a_finished_step_keeps_no_gradient_graph():
    b = tiny_backend()
    ctx = Stage1Ctx(b, ROOM, b.encode_prompt("a room"), PROV, measure_scene, CFG, new_state(), Log(), {0: (0.9, 1.0)}, "a room", 0)
    out = guided_step(b.noise_like_latent(0), torch.tensor(0.8, dtype=DT), torch.tensor(0.6, dtype=DT), 0, ctx)
    assert out.evidence.x0 is None and not out.evidence.scene and not out.latent_next.requires_grad


def test_i12_estimator_vision_encoders_are_checkpointed():
    from test_estimators_models import tiny_clipseg, tiny_siglip

    def counted(layer):
        calls = []
        original = layer._gradient_checkpointing_func
        layer._gradient_checkpointing_func = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
        return calls

    seg = tiny_clipseg()
    seg_calls = counted(seg.model.clip.vision_model.encoder.layers[0])
    seg(torch.rand(3, 32, 32, requires_grad=True), ["mug"])["mug"].sum().backward()
    sig = tiny_siglip()
    sig_calls = counted(sig.model.vision_model.encoder.layers[0])
    sig.logits(torch.rand(3, 32, 32, requires_grad=True), torch.rand(1, 3, 32, 32), "mug")[0].backward()
    assert seg_calls and sig_calls


def test_i13_a_partial_copy_does_not_count_as_finished():
    def body(tmp):
        d = tmp / "x"
        d.mkdir()
        (d / "meta.json").write_text('{"arm": "a"}')
        assert not _is_finished(d)                               # no image
        (d / "meta.json").write_text('{"arm')
        assert not _is_finished(d)                               # truncated metadata
        (d / "meta.json").write_text('{"arm": "a", "skipped": "headroom"}')
        assert _is_finished(d)
    with_tmp(body)


def test_i13_a_copy_that_dies_leaves_no_partial_image_directory():
    def body(tmp):
        d = Deps(tiny_backend(), PROV, measure_scene, GuideCfg(**{**CFG.__dict__, "steps": 1, "guided": False}), S)
        job = Job("unguided", "a room", 0, "dev", "m1_eval", ROOM)
        run_jobs([job], d, tmp / "old")
        original = shutil.copytree

        def dies(src, dest):
            Path(dest).mkdir(parents=True)
            shutil.copy(Path(src) / "meta.json", Path(dest) / "meta.json")
            raise OSError("session died mid-copy")

        shutil.copytree = dies
        try:
            run_jobs([job], d, tmp / "new", resume_from=tmp / "old")
            assert False
        except OSError:
            pass
        finally:
            shutil.copytree = original
        assert not (tmp / "new" / "unguided" / image_id("a room", 0)).exists()
        assert run_jobs([job], d, tmp / "new", resume_from=tmp / "old").skipped == 1
    with_tmp(body)


def test_i14_punctuation_and_missing_phrases():
    p = "a red mug and a blue teapot."
    assert p[slice(*phrase_spans(p, translate("a red mug and a blue teapot"))["teapot"])] == "teapot"
    assert missing_phrases(p, translate("a red mug and a blue teapot")) == []
    gone = "a mug and a blue teapot"
    assert missing_phrases(gone, translate("a red mug and a blue teapot")) == ["red"]


def test_i15_failed_images_stay_out_of_merged_logs_and_fitting():
    def body(tmp):
        _run_with_log(tmp, [_constraint()], meta={"failed": "oom"})
        assert merge_logs(tmp, tmp / "all.jsonl") == 0
        assert rows_from_run(tmp, ratings={"stage1/img": 0.5})["ratings"] == []
    with_tmp(body)


def test_stratum_without_thresholds_does_not_fail_the_final_check():
    class SmallOff(ProvisionalCalibration):
        def drift_thresholds(self, name):
            return None if name == "medium" else (0.5, 0.5)

    o = _sar_run(calib=SmallOff(S))
    assert o.fallback_objects == {"mug", "teapot"} and not o.used_fallback


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("regressions: pass")


if __name__ == "__main__":
    main()
