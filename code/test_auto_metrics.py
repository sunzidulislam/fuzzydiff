"""Tests of auto_metrics.py (python -I test_auto_metrics.py)."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subprocess  # noqa: E402

from auto_metrics import cosine, score_run, summarize  # noqa: E402

ARMS = ("unguided", "stage1", "stage1_plain", "stage1_sar")
WORDS = {"red": torch.tensor([1.0, 0.0, 0.0]), "green": torch.tensor([0.0, 1.0, 0.0]), "blue": torch.tensor([0.0, 0.0, 1.0])}


class FakeClip:
    """Images embed as their mean colour; text as the sum of the colour words it names."""

    def image(self, img):
        return torch.tensor(img.convert("RGB").resize((1, 1)).getpixel((0, 0)), dtype=torch.float32)

    def text(self, s):
        return sum((v for w, v in WORDS.items() if w in s), torch.full((3,), 1e-3))


class FakeCaptioner:
    def caption(self, img):
        r, g, b = img.convert("RGB").resize((1, 1)).getpixel((0, 0))
        return {r: "a red thing", g: "a green thing", b: "a blue thing"}[max(r, g, b)]


class FakeIQA:
    """Brighter images count as better quality."""

    def score(self, img):
        return sum(img.convert("L").resize((1, 1)).getpixel((0, 0)) for _ in (0,)) / 255


def _run(tmp, colours):
    for (arm, prompt), rgb in colours.items():
        d = tmp / arm / f"{abs(hash(prompt)) % 10 ** 8}"
        d.mkdir(parents=True)
        Image.new("RGB", (16, 16), rgb).save(d / "image.png")
        (d / "meta.json").write_text(json.dumps({"prompt": prompt, "arm": arm}))


def with_tmp(fn):
    d = tempfile.mkdtemp()
    try:
        fn(Path(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cosine():
    assert abs(cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) - 1.0) < 1e-6
    assert abs(cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 3.0]))) < 1e-6


def test_each_image_gets_both_similarities_and_its_caption():
    def body(tmp):
        _run(tmp, {("stage1", "a red mug"): (255, 0, 0), ("unguided", "a red mug"): (0, 0, 255)})
        rows = {r["arm"]: r for r in score_run(tmp, ("unguided", "stage1"), FakeClip(), FakeCaptioner())}
        assert rows["stage1"]["text_image"] > 0.99 and rows["unguided"]["text_image"] < 0.01
        assert rows["stage1"]["caption"] == "a red thing" and rows["stage1"]["text_text"] > 0.99
        assert rows["unguided"]["text_text"] < 0.01 and rows["stage1"]["prompt"] == "a red mug"
    with_tmp(body)


def test_summary_reports_each_arm_and_compares_them_on_the_same_prompts():
    def body(tmp):
        colours = {}
        for prompt, good in (("a red mug", (255, 0, 0)), ("a green vase", (0, 255, 0))):
            colours[("unguided", prompt)] = (0, 0, 255)
            for arm in ARMS[1:]:
                colours[(arm, prompt)] = good
        _run(tmp, colours)
        s = summarize(score_run(tmp, ARMS, FakeClip(), FakeCaptioner()))
        assert s["n_prompts"] == 2 and s["text_image"]["unguided"] < 0.01 and s["text_image"]["stage1"] > 0.99
        first = s["comparisons"]["text_image"]["stage1 - unguided"]
        assert first["n"] == 2 and first["better"] == 2 and first["difference"] > 0.99
        assert abs(s["comparisons"]["text_text"]["stage1_sar - stage1_plain"]["difference"]) < 1e-6
        assert "CLIP ViT-B/16" in s["models"]
    with_tmp(body)


def test_clip_iqa_is_scored_per_image_and_summarized_like_the_other_metrics():
    def body(tmp):
        colours = {}
        for prompt in ("a red mug", "a green vase"):
            colours[("unguided", prompt)] = (40, 40, 40)
            for arm in ARMS[1:]:
                colours[(arm, prompt)] = (200, 200, 200)
        _run(tmp, colours)
        rows = score_run(tmp, ARMS, FakeClip(), FakeCaptioner(), FakeIQA())
        assert all(0 <= r["clip_iqa"] <= 1 for r in rows)
        s = summarize(rows)
        assert s["clip_iqa"]["stage1"] > s["clip_iqa"]["unguided"]
        assert s["comparisons"]["clip_iqa"]["stage1 - unguided"]["better"] == 2
        assert "CLIP-IQA" in s["models"]
        assert "clip_iqa" not in summarize(score_run(tmp, ARMS, FakeClip(), FakeCaptioner()))
    with_tmp(body)


def test_clip_iqa_loads_without_torchvision():
    """Kaggle removes torchvision, and piq's package init imports it."""
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] == 'torchvision': raise ImportError('torchvision blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "from auto_metrics import piq_clip_iqa\n"
        "assert piq_clip_iqa().__name__ == 'CLIPIQA'\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr[-600:]


def test_an_arm_with_no_images_is_left_out_without_failing():
    def body(tmp):
        _run(tmp, {("stage1", "a red mug"): (255, 0, 0)})
        s = summarize(score_run(tmp, ARMS, FakeClip(), FakeCaptioner()))
        assert set(s["text_image"]) == {"stage1"} and s["comparisons"]["text_image"]["stage1 - unguided"]["n"] == 0
    with_tmp(body)


def test_a_run_restored_from_its_rating_page_scores_like_the_run():
    from annotate import build_fmas_page
    from restore_run import restore

    def body(tmp):
        colours = {}
        for prompt, good in (("a red mug", (255, 0, 0)), ("a green vase", (0, 255, 0))):
            colours[("unguided", prompt)] = (0, 0, 255)
            for arm in ARMS[1:]:
                colours[(arm, prompt)] = good
        _run(tmp / "run", colours)
        build_fmas_page(tmp / "run", tmp / "fmas.html", seed=0)
        assert restore(tmp / "fmas.html", tmp / "fmas.key.json", tmp / "restored") == 8
        meta = json.loads(next((tmp / "restored" / "stage1").glob("*/meta.json")).read_text())
        assert meta["restored_from"] == "fmas.html"
        before = summarize(score_run(tmp / "run", ARMS, FakeClip(), FakeCaptioner()))
        after = summarize(score_run(tmp / "restored", ARMS, FakeClip(), FakeCaptioner()))
        for arm in ARMS:
            assert abs(before["text_image"][arm] - after["text_image"][arm]) < 0.02
        assert after["n_prompts"] == 2
    with_tmp(body)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("auto_metrics: pass")


if __name__ == "__main__":
    main()
