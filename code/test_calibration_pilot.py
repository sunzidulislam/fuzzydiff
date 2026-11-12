"""Tests of calibration_pilot.py (python -I test_calibration_pilot.py)."""
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_pilot import (GRID, _items, box_mask, build_page, degree_question, fit,  # noqa: E402
                               labels_to_annotations, table_rows)
from predicates import translate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_degree_questions_name_the_word_of_each_family():
    c, q = degree_question("a slightly dark mug")
    assert c.family == "F2a" and q.startswith("How dark is the mug?")
    c, q = degree_question("a kitten moderately larger than a pillow")
    assert c.family == "F8" and "larger is the kitten than the pillow" in q
    assert degree_question("a red mug and a blue teapot")[0] is None or degree_question("a red mug and a blue teapot")[0].family in ("F2a", "F6")


def test_box_mask_area_and_annotations():
    m = box_mask([0.0, 0.0, 0.5, 0.25])
    assert m.shape == (GRID, GRID) and abs(m.mean() - 0.125) < 1e-9
    labels = {"unguided/a": {"rating": 40, "present": {"mug": True, "teapot": False},
                             "boxes": {"mug": [0.1, 0.1, 0.3, 0.3], "teapot": [0, 0, 1, 1]}}}
    ratings, presence, masks = labels_to_annotations(labels)
    assert ratings == {"unguided/a": 0.4} and presence == {"unguided/a": {"mug": 1, "teapot": 0}}
    assert list(masks["unguided/a"]) == ["mug"]                      # no mask for an object said to be absent


def _labels(run):
    out = {}
    rng = random.Random(0)
    for item, prompt, _ in _items(run):
        objects = list(dict.fromkeys(translate(prompt).objects))
        out[item] = {"rating": rng.randint(0, 100), "present": {o: True for o in objects},
                     "boxes": {o: [0.1 * i, 0.1, 0.1 * i + 0.4, 0.6] for i, o in enumerate(objects)}}
    return out


def test_page_fit_and_table_on_a_tiny_run():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp, "calib")
        for suite in ("soft", "size"):
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", suite,
                            "--group", "cal", "--n", "3", "--arms", "unguided,stage1", "--log-all-steps", "--out", str(run)],
                           check=True, capture_output=True)
        assert build_page(run, run / "labels.html") == 6
        assert (run / "labels.html").read_text(encoding="utf-8").count("<section") == 6
        (run / "labels.json").write_text(json.dumps(_labels(run)))
        counts, _ = fit(run, run / "labels.json", run)
        assert counts["ratings"] > 0 and counts["presence"] > 0
        rows = table_rows(run)
        assert rows.count("\\\\") == 8 and "ins." in rows
        assert json.loads((run / "out_of_domain.json").read_text())      # stage1 logs give the last column


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("calibration_pilot: pass")


if __name__ == "__main__":
    main()
