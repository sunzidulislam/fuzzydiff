"""Tests of layout_check.py on synthetic logs and images with known answers.
Usage: python -I test_layout_check.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_check import check, pixel_change, saturated_share, step_rows, summarize, verdict  # noqa: E402


def step(i, accepted, eta, rejections=(), before=2.0, after=1.0):
    return {"type": "step", "step": i, "sigma": 1 - i / 10, "accepted": accepted, "eta": eta, "rejections": list(rejections),
            "loss_before": {"1": 0.0, "3": before}, "loss_after": {"1": 0.0, "3": after if accepted else before}}


def size(i, raw, active=True):
    return {"type": "constraint", "step": i, "family": "F8", "component": "main", "active": active, "raw": raw,
            "areas": [0.2, 0.1]}


def final(s, mu):
    return {"family": "F8", "component": "main", "s": s, "mu": mu, "low": 0.5, "high": 1.0}


def test_step_rows_halve_the_step_per_rejection():
    records = [step(1, True, 0.04, ["tier1", "tier1"]), size(1, 0.3), step(0, True, 0.02), size(0, 0.1),
               step(2, False, 0.04, ["tier1"] * 3), size(2, 0.3, active=False), step(3, False, None, ["target_met"])]
    rows = step_rows(records)
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert [r["moved"] for r in rows] == [0.02, 0.01, 0.0, 0.0]
    assert [r["size_active"] for r in rows] == [True, True, False, False]
    assert rows[3]["size_score"] is None and rows[1]["loss_after"] == 1.0
    s = summarize(rows, final(0.6, 1.0), final(0.1, 0.0))
    assert (s["guided_steps"], s["accepted"], s["size_steps"]) == (4, 2, 2)
    assert abs(s["moved_total"] - 0.03) < 1e-12 and s["score_sd3"] == 0.1 and s["score"] == 0.6


def test_verdicts():
    moved = {"score": 0.6, "score_sd3": 0.1}
    same = {"score": 0.15, "score_sd3": 0.1}
    assert verdict(3.0, 0.02, same).startswith("same picture as SD3")
    assert verdict(3.0, 0.02, moved).startswith("same picture, but the score moved")
    assert verdict(40.0, 0.02, moved).startswith("picture changed")
    assert verdict(90.0, 0.6, moved).startswith("broken")
    assert verdict(3.0, 0.02, {"score": None, "score_sd3": 0.1}).startswith("same picture as SD3")


def test_image_measures():
    grey, lighter = Image.new("RGB", (8, 8), (100, 100, 100)), Image.new("RGB", (8, 8), (110, 100, 90))
    assert pixel_change(grey, grey) == 0 and abs(pixel_change(lighter, grey) - 20 / 3) < 1e-9
    assert saturated_share(grey) == 0
    half = Image.new("RGB", (8, 8), (100, 100, 100))
    half.paste((255, 60, 0), (0, 0, 4, 8))
    assert saturated_share(half) == 0.5
    assert saturated_share(Image.new("RGB", (8, 8), (255, 255, 0))) == 1.0


def test_check_reads_run_folders():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def run(setting, arm, image, s, records):
            d = root / setting / arm / "job_s0"
            d.mkdir(parents=True)
            image.save(d / "image.png")
            (d / "meta.json").write_text(json.dumps({"prompt": "a bowl moderately larger than a mug", "finals": [final(s, 0.5)]}))
            (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

        photo = Image.new("RGB", (16, 16), (120, 110, 100))
        flat = Image.new("RGB", (16, 16), (255, 60, 0))
        run("a", "unguided", photo, 0.0, [])
        run("a", "stage1", photo, 0.05, [step(0, True, 0.01), size(0, 0.0)])
        run("b", "stage1", flat, 0.9, [step(0, True, 0.05), size(0, 0.0)])
        results = check(root)
        assert list(results) == ["a", "b"]
        assert results["a"]["verdict"].startswith("same picture as SD3")
        assert results["b"]["verdict"].startswith("broken")
        assert (root / "sheet.jpg").exists() and Image.open(root / "sheet.jpg").size == (3 * 524, 560)
        lines = (root / "steps.csv").read_text().splitlines()
        assert len(lines) == 3 and lines[1].startswith("a,0,1.000,1,0.0100,")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
