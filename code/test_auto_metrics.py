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


def test_an_arm_with_no_images_is_left_out_without_failing():
    def body(tmp):
        _run(tmp, {("stage1", "a red mug"): (255, 0, 0)})
        s = summarize(score_run(tmp, ARMS, FakeClip(), FakeCaptioner()))
        assert set(s["text_image"]) == {"stage1"} and s["comparisons"]["text_image"]["stage1 - unguided"]["n"] == 0
    with_tmp(body)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("auto_metrics: pass")


if __name__ == "__main__":
    main()
