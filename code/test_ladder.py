"""Tests of the hedge-ladder table: prompts, prompt weighting, measurement, rating page and scoring."""
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import (LADDER_ARM_NAMES, build_ladder_page, ladder_images, ladder_question, ladder_table_rows,  # noqa: E402
                    measure, score_ladder, summarize_auto)
from predicates import translate  # noqa: E402
from prompts import LADDER_HEDGES, degree_word, select_ladders  # noqa: E402
from run_experiment import HEDGE_WEIGHTS, LADDER_ARMS, Job, image_id, run_jobs  # noqa: E402
from sd3_backend import CLIP_TOKENS, tiny_backend, weight_span  # noqa: E402
import test_runner as runner_tests  # noqa: E402


def with_tmp(fn):
    d = tempfile.mkdtemp()
    try:
        fn(Path(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_each_base_prompt_comes_at_every_hedge_in_order_and_bases_spread_over_words():
    for suite, n_words in (("ladder_attribute", 7), ("ladder_mixture", 12), ("ladder_size", 2)):
        rows = select_ladders(suite, "dev", 6)
        assert len(rows) == 30 and all(r.group == "dev" for r in rows)
        for i in range(0, 30, 5):
            ladder = rows[i:i + 5]
            assert len({r.base for r in ladder}) == 1
            assert [next(w for w in r.prompt.split() if w in LADDER_HEDGES) for r in ladder] == list(LADDER_HEDGES)
        assert len({degree_word(r.prompt) for r in rows[::5]}) == min(6, n_words)
        assert [r.prompt for r in rows] == [r.prompt for r in select_ladders(suite, "dev", 6)]


def test_weighting_scales_only_the_degree_words_tokens():
    b = tiny_backend()
    prompt = "a very dark mug"
    cond = b.encode_prompt(prompt)
    start = prompt.index("dark")
    w = weight_span(cond, (start, start + 4), 1.5)
    dark = [3, CLIP_TOKENS + 2]                 # CLIP: BOS, a, very, dark; T5: a, very, dark
    assert torch.allclose(w.embeds[:, dark], cond.embeds[:, dark] * 1.5)
    rest = [j for j in range(cond.embeds.shape[1]) if j not in dark]
    assert torch.equal(w.embeds[:, rest], cond.embeds[:, rest])
    assert torch.equal(w.uncond_embeds, cond.uncond_embeds) and torch.equal(w.pooled, cond.pooled)
    assert torch.equal(b.encode_prompt(prompt).embeds, cond.embeds)       # the cached encoding is untouched


def test_prompt_weight_arm_is_sd3_with_the_hedge_as_weight():
    def body(tmp):
        prompts = ("a very dark mug", "a moderately dark mug")
        jobs = [Job(arm, p, 0, "dev", "ladder_attribute", translate(p)) for arm in ("unguided", "prompt_weight")
                for p in prompts]
        s = run_jobs(jobs, runner_tests.four_arm_deps(), tmp)
        assert s.failed == 0, s.failures

        def image(arm, p):
            return torch.from_numpy(np.asarray(Image.open(tmp / arm / image_id(p, 0) / "image.png"), dtype=np.float32))
        meta = json.load(open(tmp / "prompt_weight" / image_id(prompts[0], 0) / "meta.json"))
        assert meta["prompt_weight"] == {"word": "dark", "weight": HEDGE_WEIGHTS["very"]}
        assert not torch.equal(image("unguided", prompts[0]), image("prompt_weight", prompts[0]))
        assert torch.equal(image("unguided", prompts[1]), image("prompt_weight", prompts[1]))   # weight 1.0
    with_tmp(body)


def _flat(rgb, size=64):
    return torch.tensor(rgb, dtype=torch.float64).view(3, 1, 1).expand(3, size, size).clone()


FULL = (0, 0, 64, 64)


def _prototype(term):
    from calibration import ProvisionalCalibration
    from settings import load_settings
    return ProvisionalCalibration(load_settings()).prototype(term)


def test_measurements_follow_the_prompts_direction():
    dark = translate("a very dark mug").constraints[0]
    assert measure(_flat((.2, .2, .2)), dark, {"mug": FULL}, _prototype) > measure(_flat((.7, .7, .7)), dark, {"mug": FULL}, _prototype)
    assert measure(_flat((.2, .2, .2)), dark, {"mug": None}, _prototype) is None
    vivid = translate("a very vivid mug").constraints[0]
    assert measure(_flat((.9, .1, .1)), vivid, {"mug": FULL}, _prototype) > measure(_flat((.6, .45, .45)), vivid, {"mug": FULL}, _prototype)
    mix = translate("a very greenish-gray mug").constraints[0]
    greener = measure(_flat((.35, .6, .35)), mix, {"mug": FULL}, _prototype)
    assert greener > measure(_flat((.5, .5, .5)), mix, {"mug": FULL}, _prototype)
    small = translate("a mug very smaller than a bottle").constraints[0]
    boxes = {"mug": (0, 0, 10, 10), "bottle": (0, 0, 20, 20)}
    assert abs(measure(_flat((.5, .5, .5)), small, boxes, _prototype) - math.log(4)) < 1e-9
    assert measure(_flat((.5, .5, .5)), small, {"mug": FULL, "bottle": None}, _prototype) is None


BASES = ("a dark mug", "a dark vase")


def _ladder_runs(tmp, arms=LADDER_ARM_NAMES, s_of=lambda arm, rank: rank / 4):
    """A dark-mug ladder per arm: stage1 and stage1_sar get darker with the hedge; the others do not."""
    for arm in arms:
        for base in BASES:
            obj = base.split()[-1]
            for rank, hedge in enumerate(LADDER_HEDGES):
                prompt = f"a {hedge} dark {obj}"
                d = tmp / "ladder_attribute" / arm / image_id(prompt, 0)
                d.mkdir(parents=True)
                grey = int(200 - 40 * rank) if arm.startswith("stage1") else 150
                Image.new("RGB", (32, 32), (grey, grey, grey)).save(d / "image.png")
                finals = [{"family": "F2a", "component": "main", "s": s_of(arm, rank)}]
                (d / "meta.json").write_text(json.dumps({"prompt": prompt, "arm": arm, "suite": "ladder_attribute",
                                                         "finals": finals}))
    return [tmp / "ladder_attribute"]


class FullBox:
    def boxes(self, img, names):
        return {n: (0, 0) + img.size for n in names}


def test_automatic_rho_is_one_for_an_ordered_ladder_and_zero_for_a_flat_one():
    from ladder import measure_run

    def body(tmp):
        rows = measure_run(ladder_images(_ladder_runs(tmp)), FullBox(), _prototype)
        s = summarize_auto(rows)["rho"]["attribute"]
        assert s["stage1"]["value"] == 1.0 and s["stage1"]["n"] == 2
        assert s["unguided"]["value"] == 0.0
    with_tmp(body)


def test_a_ladder_with_an_undetected_image_is_left_out():
    from ladder import measure_run

    class MissesOne(FullBox):
        def boxes(self, img, names):
            found = super().boxes(img, names)
            return {n: None for n in names} if img.getpixel((0, 0))[0] == 120 else found

    def body(tmp):
        out = summarize_auto(measure_run(ladder_images(_ladder_runs(tmp)), MissesOne(), _prototype))
        assert out["rho"]["attribute"]["stage1"]["n"] == 0 and out["excluded_ladders"]["attribute"]["stage1"] == 2
        assert out["rho"]["attribute"]["unguided"]["n"] == 2
    with_tmp(body)


def test_ladder_page_hides_the_hedge_and_the_method():
    def body(tmp):
        build_ladder_page(_ladder_runs(tmp), tmp / "ladder.html", seed=0)
        page = (tmp / "ladder.html").read_text(encoding="utf-8")
        key = json.loads((tmp / "ladder.key.json").read_text())
        assert len(key) == len(BASES) * len(LADDER_ARM_NAMES) and page.count("<figure") == 5 * len(key)
        visible = re.sub(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", "", page)
        for word in (*LADDER_HEDGES, "stage1", "prompt_weight", "unguided"):
            assert not re.search(rf"\b{word}\b", visible), word
        assert "How dark is the mug?" in page and "<h2>a dark mug</h2>" in page
    with_tmp(body)


def test_questions_say_what_0_and_100_mean():
    assert ladder_question("a slightly greenish-gray mug").startswith("How far is the mug's colour from gray toward green?")
    assert ladder_question("a mug very smaller than a bottle").startswith("How much smaller is the mug than the bottle?")


def _rate(key, degree_of):
    return [{"group": g, "panel": letter, "degree": degree_of(e["arm"], p["rank"])}
            for g, e in key.items() for letter, p in e["panels"].items()]


def test_pair_accuracy_counts_hedge_pairs_in_order_and_ties_as_half():
    def body(tmp):
        build_ladder_page(_ladder_runs(tmp), tmp / "ladder.html", seed=0)
        key = json.loads((tmp / "ladder.key.json").read_text())

        def degree(arm, rank):
            return {"stage1": rank * 20, "stage1_sar": 100 - rank * 20, "unguided": 50}.get(arm)
        (tmp / "r.json").write_text(json.dumps(_rate(key, degree)))
        s = score_ladder(tmp / "r.json", tmp / "ladder.key.json")
        pair = s["pair"]["attribute"]
        assert pair["stage1"]["value"] == 100.0 and pair["stage1_sar"]["value"] == 0.0
        assert pair["unguided"]["value"] == 50.0 and pair["prompt_weight"]["n"] == 0
        assert s["unrated"] == 5 * len(BASES)
    with_tmp(body)


def test_evidence_row_correlates_fuzzydiffs_score_with_the_human_degree():
    def body(tmp):
        build_ladder_page(_ladder_runs(tmp), tmp / "ladder.html", seed=0)
        key = json.loads((tmp / "ladder.key.json").read_text())
        (tmp / "r.json").write_text(json.dumps(_rate(key, lambda arm, rank: rank * 20 + (arm == "unguided"))))
        e = score_ladder(tmp / "r.json", tmp / "ladder.key.json")["evidence_vs_human"]["attribute"]
        assert e["value"] > 0.9 and e["n"] == 5 * len(BASES) * len(LADDER_ARM_NAMES)
    with_tmp(body)


def test_evidence_is_not_flipped_twice_for_words_that_point_down():
    """The scorers already orient s (1 - s for dark, o * log ratio for smaller); _evidence must keep it."""
    from constraints import Photometric
    from ladder import _evidence
    dark = translate("a very dark mug").constraints[0]
    assert dark.orientation == -1

    class Calib:
        def phi(self, family, quantity, bin, raw):
            return raw / 100                     # lightness L* in [0, 100] -> [0, 1]
    scorer = Photometric()
    scorer.raw = lambda c, evidence, component: evidence     # evidence here is the L* value itself
    s_dark, s_light = scorer.score(dark, 20.0, Calib(), 4, "main"), scorer.score(dark, 80.0, Calib(), 4, "main")
    assert s_dark > s_light
    for s in (s_dark, s_light):
        assert _evidence({"finals": [{"family": "F2a", "component": "main", "s": s}]}, dark) == s
    small = translate("a mug very smaller than a bottle").constraints[0]
    assert _evidence({"finals": [{"family": "F8", "component": "main", "s": 0.7}]}, small) == 0.7


def test_table_rows_keep_sliders_and_spo_tbd_and_match_the_runner():
    assert LADDER_ARM_NAMES == LADDER_ARMS
    human = {"pair": {"attribute": {"stage1": {"value": 87.5}}}, "evidence_vs_human": {"size": {"value": 0.41}}}
    auto = {"rho": {"attribute": {"stage1": {"value": 0.8}}}}
    lines = ladder_table_rows(human, auto).splitlines()
    assert all(line.count("&") == 6 for line in lines if "&" in line)
    assert lines[4] == "FuzzyDiff Stage 1 & 87.5 & 0.80 & \\tbd & \\tbd & \\tbd & \\tbd\\\\"
    assert lines[2].startswith("Concept Sliders & \\tbd") and lines[3].startswith("SPO & \\tbd")
    assert lines[-1] == "Evidence $s_k$ vs.\\ human degree & -- & \\tbd & -- & \\tbd & -- & 0.41\\\\"


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ladder: pass")


if __name__ == "__main__":
    main()
