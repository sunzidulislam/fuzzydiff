"""Tests of run_experiment.py and annotate.py on the tiny model (python -I test_runner.py)."""
import json
import os
import random as random_module
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import (build_comparison_page, build_rating_page, export_mask_tasks, export_rating_tasks,  # noqa: E402
                      import_masks, import_ratings, score_m1)
from calibration_fit import FittedCalibration  # noqa: E402
from fit_calibration import fit_main, rows_from_run  # noqa: E402
from reporting import read_jsonl, tier_rows  # noqa: E402
from calibration import ProvisionalCalibration  # noqa: E402
from guidance import GuideCfg, run_stage1  # noqa: E402
from estimators import FakeSigLIP, FakeSupport  # noqa: E402
from measure import EstimatorDeps, Measure, measure_scene  # noqa: E402
from predicates import Constraint, Program, translate  # noqa: E402
from run_experiment import (Deps, Job, arm_jobs, commanded_targets, describe_program, image_id, m1_jobs, merge_logs,  # noqa: E402
                            only_prompts, placebo_source, prompt_rows, run_jobs, summarize_arms)
from sar import SarCfg  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

SETTINGS = load_settings()
CFG = GuideCfg(steps=2, cfg=7.0, sigma_lo=0.0, eta0=0.2, delta=0.1, dmax=0.2, eps={1: 0.01, 2: 0.01}, backtrack=1, ramp=2)


def deps(stage1=run_stage1):
    return Deps(backend=tiny_backend(), calib=ProvisionalCalibration(SETTINGS), measure=measure_scene, cfg=CFG,
                settings=SETTINGS, stage1=stage1)


def program(noun="room"):
    return Program(objects=[], constraints=[Constraint("F2a", (noun,), None, "scene_lightness", 1)])


def jobs():
    return [Job("unguided", "a room", 0, "dev", "m1_eval", program()),
            Job("unguided", "a hall", 0, "dev", "m1_eval", program("hall"))]


def count_records(out):
    n = 0
    for arm in Path(out).iterdir():
        if arm.is_dir() and not arm.name.startswith("."):
            for img in arm.iterdir():
                log = img / "log.jsonl"
                if log.exists():
                    n += sum(1 for _ in open(log))
    return n


def with_tmp(fn):
    d = tempfile.mkdtemp()
    try:
        fn(Path(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resume_skips_done_and_redoes_partial():
    def body(tmp):
        run_jobs(jobs()[:1], deps(), tmp)
        (tmp / ".tmp-x").mkdir()
        (tmp / ".tmp-x" / "p.jsonl").write_text('{"trunc')     # dead session
        s = run_jobs(jobs(), deps(), tmp)
        assert s.skipped == 1 and s.done == 1 and not (tmp / ".tmp-x").exists()
        assert merge_logs(tmp, tmp / "all.jsonl") == count_records(tmp) > 0
        for line in open(tmp / "all.jsonl"):
            json.loads(line)
    with_tmp(body)


def test_resume_from_earlier_session():
    def body(tmp):
        run_jobs(jobs()[:1], deps(), tmp / "old")
        assert run_jobs(jobs(), deps(), tmp / "new", resume_from=tmp / "old").skipped == 1
        assert (tmp / "new" / "unguided" / image_id("a room", 0) / "image.png").exists()
    with_tmp(body)


def test_oom_records_failure_and_continues():
    calls = []

    def stage1(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return run_stage1(*a, **k)

    def body(tmp):
        s = run_jobs(jobs(), deps(stage1), tmp)
        assert s.failed == 1 and s.done == 1
        meta = json.load(open(tmp / "unguided" / image_id("a room", 0) / "meta.json"))
        assert meta["failed"] == "oom" and "peak_gb" in meta
        retry = run_jobs(jobs(), deps(), tmp)                  # a failed image is retried, a done one is not
        assert retry.done == 1 and retry.skipped == 1
    with_tmp(body)


def test_commanded_targets():
    lo, hi = commanded_targets(0.5, -0.3, 0.05, 0.10)
    assert abs((lo + hi) / 2 - 0.2) < 1e-9 and hi - lo < 0.05
    assert commanded_targets(0.12, -0.3, 0.05, 0.10) is None   # below headroom
    lo, hi = commanded_targets(0.9, 0.3, 0.05, 0.10) or (None, None)
    assert lo is None                                          # 0.95 is only 0.05 away


def test_commanded_arm_reads_unguided_and_skips_without_headroom():
    def body(tmp):
        run_jobs(jobs()[:1], deps(), tmp)
        bright = Job("brighter", "a room", 0, "dev", "m1_eval", program())
        run_jobs([bright], deps(), tmp)
        meta = json.load(open(tmp / "brighter" / image_id("a room", 0) / "meta.json"))
        base = json.load(open(tmp / "unguided" / image_id("a room", 0) / "meta.json"))["finals"][0]["s"]
        assert "skipped" in meta or abs(meta["ds"] - 0.3) < 1e-6
        assert "skipped" in meta or abs(meta["dy"] - (meta["finals"][0]["s"] - base)) < 1e-9

        missing = run_jobs([Job("darker", "a hall", 0, "dev", "m1_eval", program("hall"))], deps(), tmp)
        assert missing.failed == 1
    with_tmp(body)


def test_records_carry_tag_and_page_hides_condition():
    def body(tmp):
        run_jobs(jobs(), deps(), tmp)
        merge_logs(tmp, tmp / "a.jsonl")
        assert all(r["calibration"] == "provisional" for r in map(json.loads, open(tmp / "a.jsonl")))
        images = sorted(tmp.glob("*/*/image.png"))
        build_rating_page(images, tmp / "rate.html", 0)
        html = (tmp / "rate.html").read_text()
        assert "unguided" not in html and "darker" not in html and "a room" not in html
        key = json.load(open(tmp / "rate.key.json"))
        assert len(key) == 2 and all("unguided" in v for v in key.values())
    with_tmp(body)


def test_m1_job_lists():
    ev = m1_jobs(False)
    pairs = {(j.prompt, j.seed) for j in ev if j.arm == "darker"}
    assert len(pairs) == 40 and sum(j.arm == "unguided" for j in ev) == 40
    order = [j.arm for j in ev]
    assert order.index("unguided") < order.index("darker")
    assert {"ladder_unguided", "ladder_guided"} <= set(order)
    ladder = next(j for j in ev if j.arm == "ladder_guided")
    assert {c.family for c in ladder.program.constraints} == {"F2a"}
    tune = m1_jobs(True)
    assert {j.overrides.get("eta0") for j in tune if j.overrides} == {0.02, 0.05, 0.1}
    assert {j.overrides.get("sigma_hi") for j in tune if j.overrides} == {0.75, 0.5}
    assert tune[5].overrides["eta0"] == 0.05 and tune[5].overrides["sigma_hi"] == 0.75   # the likeliest setting runs first
    assert {j.prompt for j in tune}.isdisjoint({j.prompt for j in ev})


def test_only_prompts_keeps_arm_order_for_the_chosen_prompts():
    jobs = only_prompts(m1_jobs(True), ["a bakery", "a cellar"])
    assert {j.prompt for j in jobs} == {"a bakery", "a cellar"}
    assert [j.arm for j in jobs[:6]] == ["unguided"] * 2 + ["darker@eta0.05@hi0.75"] * 2 + ["brighter@eta0.05@hi0.75"] * 2


def test_only_prompts_refuses_a_prompt_the_suite_does_not_have():
    try:
        only_prompts(m1_jobs(True), ["a bakery", "a bakry"])
    except ValueError as e:
        assert "a bakry" in str(e)
    else:
        raise AssertionError("a misspelled prompt must not silently run nothing")


def test_score_m1_against_known_outcome():
    def body(tmp):
        for i in range(10):
            for arm, s, ds, dy in (("unguided", .5, None, None), ("darker", .3, -.3, -.2), ("brighter", .7, .3, .2)):
                d = tmp / arm / f"img{i}"
                d.mkdir(parents=True)
                Image.new("RGB", (8, 8)).save(d / "image.png")
                meta = {"prompt": f"p{i}", "seed": 0, "arm": arm, "finals": [{"s": s}]}
                if ds is not None:
                    meta.update(ds=ds, dy=dy)
                json.dump(meta, open(d / "meta.json", "w"))
        build_rating_page(sorted(tmp.glob("*/*/image.png")), tmp / "rate.html", 0)
        key = json.load(open(tmp / "rate.key.json"))
        ratings = []
        for anon, path in key.items():
            arm = Path(path).parts[-3]
            ratings.append({"id": anon, "rating": {"unguided": 50, "darker": 30, "brighter": 70}[arm], "broken": False})
        json.dump(ratings, open(tmp / "ratings.json", "w"))
        r = score_m1(tmp / "ratings.json", tmp / "rate.key.json", tmp, SETTINGS)
        assert r["dir_accuracy"] == 1.0 and abs(r["transfer_slope"] - 2 / 3) < 1e-9
        assert r["human_direction_agreement"] == 1.0 and r["broken_share"] == 0.0 and r["passed"] is True
    with_tmp(body)


def _blob(a, b):
    m = torch.zeros(64, 64, dtype=torch.float64)
    m[a:a + 16, b:b + 16] = 1
    return m


MASKS = {"mug": _blob(10, 10), "teapot": _blob(40, 40), "vase": _blob(10, 40), "lamp": _blob(40, 10), "chair": _blob(30, 30)}
SAR_CFG = SarCfg(strength=0.5, k=1, alpha=1.0, beta=0.5, temp=0.1, gamma_min=0.3, overlap=0.3, nest=0.8)


def four_arm_deps(stage1=run_stage1):
    calib = ProvisionalCalibration(SETTINGS)
    seg = FakeSupport(MASKS)
    est = EstimatorDeps(seg, FakeSigLIP(lambda phrase, image: 1.0), calib, SETTINGS)
    return Deps(backend=tiny_backend(read_blocks=(0, 1)), calib=calib, measure=None, cfg=CFG, settings=SETTINGS, stage1=stage1,
                measure_factory=lambda prompt, cond: Measure(est, prompt, cond), seg=seg, sar_cfg=SAR_CFG)


def rows():
    from prompts import PromptRow
    return [PromptRow(p, p, "dev", "plain_color") for p in ("a red mug and a blue teapot", "a green vase and a pink lamp")]


def test_arms_share_stage1_and_renoising():
    runs = []

    def counted(*a, **k):
        runs.append(1)
        return run_stage1(*a, **k)

    def body(tmp):
        jobs_ = arm_jobs(rows()[:2], (0,), ["unguided", "stage1", "stage1_plain", "stage1_sar", "stage1_placebo"])
        s = run_jobs(jobs_, four_arm_deps(counted), tmp)
        assert s.failed == 0, s.failures
        assert len(runs) == 4                                    # unguided and stage1 per prompt, none for Stage 2
        i = image_id(rows()[0].prompt, 0)
        metas = {a: json.load(open(tmp / a / i / "meta.json")) for a in ("stage1_plain", "stage1_sar", "stage1_placebo")}
        assert len({m["renoised_sha"] for m in metas.values()}) == 1
        assert metas["stage1_placebo"]["placebo_from"] == rows()[1].prompt
        assert metas["stage1_sar"]["passes"] > metas["stage1_plain"]["passes"]
        assert (tmp / "stage1" / i / "stage1.pt").exists()
    with_tmp(body)


def _job(prompt, arm="stage1_placebo"):
    return Job(arm, prompt, 0, "dev", "plain_color", translate(prompt))


def test_placebo_uses_another_image_same_object_count():
    a, same, other = _job("a red mug and a blue teapot"), _job("a green vase and a pink lamp"), _job("a red chair")
    assert placebo_source(a, [a, same, other]) is same
    assert placebo_source(other, [a, same, other]) is None


def test_placebo_without_match_records_failure():
    def body(tmp):
        lone = [PromptRow_("a red mug and a blue teapot")]
        s = run_jobs(arm_jobs(lone, (0,), ["stage1", "stage1_placebo"]), four_arm_deps(), tmp)
        assert s.failed == 1
        meta = json.load(open(tmp / "stage1_placebo" / image_id(lone[0].prompt, 0) / "meta.json"))
        assert meta["failed"] == "no_placebo_source"
    with_tmp(body)


def PromptRow_(p):
    from prompts import PromptRow
    return PromptRow(p, p, "dev", "plain_color")


def test_stage2_without_stage1_fails_clearly():
    def body(tmp):
        s = run_jobs(arm_jobs(rows()[:1], (0,), ["stage1_sar"]), four_arm_deps(), tmp)
        assert s.failed == 1 and "stage1" in s.failures[0].lower()
    with_tmp(body)


def test_resume_after_stage1_only_redoes_stage2():
    runs = []

    def counted(*a, **k):
        runs.append(1)
        return run_stage1(*a, **k)

    def body(tmp):
        run_jobs(arm_jobs(rows()[:1], (0,), ["stage1"]), four_arm_deps(counted), tmp)
        s = run_jobs(arm_jobs(rows()[:1], (0,), ["stage1", "stage1_plain"]), four_arm_deps(counted), tmp)
        assert s.skipped == 1 and s.done == 1 and len(runs) == 1
    with_tmp(body)


def test_summary_and_blind_page():
    def body(tmp):
        run_jobs(arm_jobs(rows(), (0,), ["stage1", "stage1_plain", "stage1_sar", "stage1_placebo"]), four_arm_deps(), tmp)
        s = summarize_arms(tmp)
        assert {"fallback_rate", "rollbacks", "suspended_steps", "n", "time_s"} <= set(s["stage1_sar"])
        assert s["stage1_sar"]["n"] == 2
        build_comparison_page(tmp, tmp / "c.html", 0)
        html = (tmp / "c.html").read_text()
        assert "stage1_sar" not in html and "placebo" not in html and "a red mug" in html
        key = json.load(open(tmp / "c.key.json"))
        assert all(set(v.values()) == {"stage1", "stage1_plain", "stage1_sar", "stage1_placebo"} for v in key.values())
    with_tmp(body)


def test_own_prompt_runs_through_every_stage():
    def body(tmp):
        prompt = "a red mug and a blue teapot"
        rows = prompt_rows([prompt])
        assert rows[0].prompt == prompt and rows[0].group == "dev" and rows[0].suite == "custom"
        s = run_jobs(arm_jobs(rows, (0,), ["unguided", "stage1", "stage1_sar"]), four_arm_deps(), tmp)
        assert s.done == 3 and s.failed == 0, s.failures
        for arm in ("unguided", "stage1", "stage1_sar"):
            assert (tmp / arm / image_id(prompt, 0) / "image.png").exists()
    with_tmp(body)


def test_describe_program_says_what_will_and_will_not_be_guided():
    text = describe_program("a slightly red mug and a few fast cars")
    assert "not guided" in text.lower()
    guided = describe_program("a slightly dark mug on a wooden table")
    assert "F2a" in guided and "presence" in guided and "slightly" in guided
    assert "F9" in describe_program("a mug left of a teapot") and "not guided" in describe_program("a mug left of a teapot")


class FullySet:
    """Settings as they will be once the YAML holds every run value, with nothing provisional behind them."""

    provisional_used = frozenset()

    def __init__(self, settings):
        self.settings = settings

    def get(self, *a, **k):
        return self.settings.get(*a, **k)


def _ratings_csv(path, n_per_item=3, bad_checks=0):
    lines = ["worker,item,rating,check"]
    for item in ("a", "b"):
        for w in range(n_per_item):
            lines.append(f"w{w},{item},{50 + w},")
    for w in range(n_per_item):
        lines.append(f"w{w},check1,90,pass")
    for b in range(bad_checks):
        lines.append(f"bad{b},a,5,")
        lines.append(f"bad{b},check1,1,fail")
    path.write_text("\n".join(lines))
    return path


def test_import_filters_and_requires_three():
    def body(tmp):
        rows = import_ratings(_ratings_csv(tmp / "r.csv", bad_checks=1))
        assert all(r["n"] >= 3 for r in rows) and not any(r["worker"].startswith("bad") for r in rows)
        assert not any(r["item"] == "check1" for r in rows)
        try:
            import_ratings(_ratings_csv(tmp / "r2.csv", n_per_item=2))
            assert False, "fewer than 3 ratings per item must fail"
        except ValueError:
            pass
    with_tmp(body)


def test_exports_are_blind_and_include_attention_checks():
    def body(tmp):
        run_jobs(jobs(), deps(), tmp / "run")
        export_rating_tasks(tmp / "run", "brightness", tmp / "tasks.csv", ratings_per_item=3, attention_checks=0.5, seed=0)
        text = (tmp / "tasks.csv").read_text()
        assert "unguided" not in text and "check" in text and "a room" not in text
        export_mask_tasks(tmp / "run", tmp / "masks.json", seed=0)
        tasks = json.load(open(tmp / "masks.json"))
        assert len(tasks) == 2 and "image" in tasks[0]["data"] and "unguided" not in json.dumps(tasks[0]["data"])
    with_tmp(body)


def test_import_masks_rasterizes_polygons():
    def body(tmp):
        shape = {"type": "polygonlabels", "value": {"points": [[0, 0], [50, 0], [50, 100], [0, 100]], "polygonlabels": ["mug"]}}
        json.dump([{"data": {"item": "x"}, "annotations": [{"result": [shape]}]}], open(tmp / "m.json", "w"))
        m = import_masks(tmp / "m.json")["x"]["mug"]
        assert m.shape == (64, 64) and abs(m.mean() - 0.5) < 0.05 and m[30, 5] == 1 and m[30, 60] == 0
    with_tmp(body)


def test_steps_log_raw_values_for_fitting():
    def body(tmp):
        run_jobs(jobs()[:1], deps(), tmp)   # unguided: no guided steps, so also run a guided job
        guided = Job("stage1", "a room", 0, "dev", "m1_eval", program())
        run_jobs([guided], deps(), tmp)
        merge_logs(tmp, tmp / "a.jsonl")
        rec = next(r for r in read_jsonl(tmp / "a.jsonl") if r["type"] == "constraint")
        assert 0 <= rec["raw"] <= 100
    with_tmp(body)


def _synthetic_inputs(tmp):
    """Ratings that follow L*, with every other section supplied directly."""
    guided = [Job("stage1", f"a room{i}", 0, "dev", "m1_eval", program(f"room{i}")) for i in range(8)]
    run_jobs(guided, deps(), tmp / "cal")
    rng = random_module.Random(0)
    ratings = {}
    for meta_path in (tmp / "cal").glob("*/*/meta.json"):
        m = json.load(open(meta_path))
        item = f"{meta_path.parts[-3]}/{meta_path.parts[-2]}"
        ratings[item] = m["finals"][0]["raw"] / 100
    rows = rows_from_run(tmp / "cal", ratings=ratings)
    extra = []
    for r in list(rows["ratings"]):
        for i in range(60):
            e = rng.uniform(0, 100)
            extra.append({**r, "e": e, "rating": e / 100, "base": f"synthetic{i}", "item": f"s{len(extra)}"})
    rows["ratings"] += extra
    rows["acceptance"] = [{"family": "F2a", "quantity": "scene_lightness", "hedge": "None", "orientation": 1,
                           "s": s, "accept": int(0.6 <= s <= 0.8), "base": f"b{i % 40}"}
                          for i, s in enumerate(rng.uniform(0, 1) for _ in range(800))]
    rows.update(prototypes={"red": [0.6, 0.2, 0.1]}, bounds={"e": [0.4, 0.1]}, leakage_scale=0.08, psi=1.0,
                size=[{"A_q": 0.1 * (i % 5 + 1), "A_r": 0.05, "abs_error": 0.1, "bin": 4, "estimate": i, "human": i, "base": f"b{i}"}
                      for i in range(60)])
    return rows


def test_synthetic_annotations_to_fitted_run():
    def body(tmp):
        data = _synthetic_inputs(tmp)
        json.dump(data, open(tmp / "input.json", "w"))
        fit_main(tmp / "input.json", tmp / "calibration.json", SETTINGS, min_images=50)
        calib = FittedCalibration.load(tmp / "calibration.json")
        assert calib.valid("F2a", 4)
        assert calib.tag == "fitted"
        d = deps()
        d.calib = calib
        d.settings = FullySet(SETTINGS)
        s = run_jobs(jobs(), d, tmp / "out")
        assert s.done == 2 and s.failed == 0, s.failures
        guided = Job("stage1", "a room", 0, "dev", "m1_eval", program())
        assert run_jobs([guided], d, tmp / "out").done == 1
        merge_logs(tmp / "out", tmp / "a.jsonl")
        records = read_jsonl(tmp / "a.jsonl")
        assert all(r["calibration"] == "fitted" for r in records)
        tier_rows(records)                                       # not refused
    with_tmp(body)


def test_loop_files_do_not_branch_on_calibration_type():
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("guidance.py", "constraints.py", "sar.py"):
        src = open(os.path.join(here, f)).read()
        assert "ProvisionalCalibration" not in src and "FittedCalibration" not in src


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("runner: pass")


if __name__ == "__main__":
    main()
