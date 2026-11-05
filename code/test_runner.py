"""Tests of run_experiment.py and annotate.py on the tiny model (python -I test_runner.py)."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate import build_rating_page, score_m1  # noqa: E402
from calibration import ProvisionalCalibration  # noqa: E402
from guidance import GuideCfg, run_stage1  # noqa: E402
from measure import measure_scene  # noqa: E402
from predicates import Constraint, Program  # noqa: E402
from run_experiment import (Deps, Job, commanded_targets, image_id, m1_jobs, merge_logs, run_jobs)  # noqa: E402
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
    assert {j.overrides.get("sigma_lo") for j in tune if j.overrides} == {0.5, 0.25}
    assert {j.prompt for j in tune}.isdisjoint({j.prompt for j in ev})


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


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("runner: pass")


if __name__ == "__main__":
    main()
