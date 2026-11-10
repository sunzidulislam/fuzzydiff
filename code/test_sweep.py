"""Tests of prompt screening (screen.py) and the guidance-strength sweep (sweep.py)."""
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import measure_run  # noqa: E402
from prompts import LADDER_HEDGES  # noqa: E402
from run_experiment import image_id  # noqa: E402
from screen import keep, screen  # noqa: E402
from sweep import SETTINGS, summarize_sweep, sweep_flags, sweep_images, sweep_table  # noqa: E402


def with_tmp(fn):
    d = tempfile.mkdtemp()
    try:
        fn(Path(d))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _meta(d, prompt, arm, finals, **extra):
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"prompt": prompt, "arm": arm, "finals": finals, **extra}))


def test_screen_keeps_prompts_sd3_fails_worst_first():
    def body(tmp):
        mus = {"a red mug and a blue vase": (1.0, 1.0), "a pink cup and a green hat": (0.9, 0.2),
               "a grey lamp and a teal bowl": (0.0, 1.0)}
        for prompt, (a, b) in mus.items():
            _meta(tmp / "unguided" / image_id(prompt, 0), prompt, "unguided",
                  [{"family": "F6", "component": "main", "mu": a}, {"family": "presence", "component": "main", "mu": b}])
        _meta(tmp / "unguided" / "x", "a broken one", "unguided", [], failed="oom")
        rows = screen(tmp)
        assert [r["prompt"] for r in rows] == ["a grey lamp and a teal bowl", "a pink cup and a green hat",
                                               "a red mug and a blue vase"]
        assert rows[0]["failing"] == ["F6/main mu=0.00"] and rows[2]["failing"] == []
        # the cup prompt fails only because the hat is missing, so a colour ablation has nothing to fix there
        assert [r["objects_present"] for r in rows] == [True, False, True]
        assert keep(rows, 5) == ["a grey lamp and a teal bowl", "a pink cup and a green hat"]
        assert keep(rows, 5, objects_present=True) == ["a grey lamp and a teal bowl"]
        assert keep(rows, 1) == ["a grey lamp and a teal bowl"]
    with_tmp(body)


class AreaBox:
    """The first object's box side is the image's red value; the second object's is fixed at 100."""

    def boxes(self, img, names):
        side = img.getpixel((0, 0))[0]
        return {names[0]: (0, 0, side, side), names[1]: (0, 0, 100, 100)}


def _ladder(root, folder, arm, side_of, evidence_of):
    for rank, hedge in enumerate(LADDER_HEDGES):
        prompt = f"a mug {hedge} larger than a bottle"
        d = root / folder / arm / image_id(prompt, 0)
        _meta(d, prompt, arm, [{"family": "F8", "component": "main", "s": evidence_of(rank)}], time_s=10.0)
        Image.new("RGB", (8, 8), (side_of(rank), 0, 0)).save(d / "image.png")


def test_sweep_scores_each_setting_by_measured_size_against_sd3():
    def body(tmp):
        _ladder(tmp, "unguided", "unguided", lambda r: 100, lambda r: 0.5)
        _ladder(tmp, "stronger", "stage1", lambda r: 100 + 20 * r, lambda r: r / 4)      # size follows the hedge
        _ladder(tmp, "earlier", "stage1", lambda r: 100, lambda r: r / 4)               # only its own scorer moves
        rows = measure_run(sweep_images(tmp), AreaBox(), prototype=None)
        s = summarize_sweep(rows, tmp)
        assert set(s) == {"unguided", "stronger", "earlier"}
        assert s["stronger"]["rho"]["value"] == 1.0 and s["earlier"]["rho"]["value"] == 0.0
        assert s["earlier"]["evidence_rho"]["value"] == 1.0 and s["earlier"]["moved"] == 0.0
        expected = sum(abs(2 * math.log((100 + 20 * r) / 100)) for r in range(5)) / 5
        assert abs(s["stronger"]["moved"] - expected) < 1e-9
        assert s["stronger"]["time_s"] == 10.0 and s["stronger"]["flags"] == "--dmax 1.0 --eta0 0.2"
        table = sweep_table(s)
        assert table.splitlines()[0].startswith("setting") and "stronger" in table and "both" not in table
    with_tmp(body)


def test_settings_form_a_two_by_two_design_on_the_two_suspected_causes():
    assert SETTINGS["current"] == {} and sweep_flags("current") == ""
    assert SETTINGS["both"] == {**SETTINGS["stronger"], **SETTINGS["earlier"]}
    assert sweep_flags("earlier") == "--sigma-hi 1.0"


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("sweep: pass")


if __name__ == "__main__":
    main()
