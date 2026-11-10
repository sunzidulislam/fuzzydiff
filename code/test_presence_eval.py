"""Tests of presence_eval.py (python -I test_presence_eval.py)."""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from measure import EstimatorDeps  # noqa: E402
from presence_eval import (PROMPTS, SUBSETS, build_page, generate, queries, score, subset_rows,  # noqa: E402
                           table_rows)
from prompts import vocab  # noqa: E402
from run_experiment import _tiny_estimators  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402


def test_ten_prompts_per_subset_and_crowded_name_at_least_four():
    for subset in ("ordinary", "small", "crowded", "context"):
        assert sum(p["subset"] == subset for p in PROMPTS) == 10
    assert all(len(p["present"]) >= 4 for p in PROMPTS if p["subset"] == "crowded")
    assert all(p["context"] for p in PROMPTS if p["subset"] == "context")


def test_queries_add_one_absent_object_not_in_the_prompt():
    entry = PROMPTS[0]
    q = queries(entry)
    assert q[:2] == entry["present"] and len(q) == 3
    assert q[2] in vocab()["objects"] and q[2] not in entry["prompt"]
    assert queries(entry) == q                                    # reproducible


def _rows(n=40, separated=True, seed=0):
    rng = random.Random(seed)
    rows, labels = [], []
    for i in range(n):
        subset = SUBSETS[1:][i % 3] if i % 4 else "ordinary"
        for obj, present in (("a", True), ("b", False)):
            centre = (3.0 if present else -3.0) if separated else 0.0
            r = {"image": f"img{i}", "subset": subset, "prompt": f"prompt {i}", "object": obj, "named": True,
                 "context_absent": obj == "b" and subset == "context", "n_named": 5 if subset == "crowded" else 2,
                 "bin": 4, "global": centre + rng.gauss(0, 1), "region": centre + rng.gauss(0, 1),
                 "h": centre + rng.gauss(0, 1), "area": 0.01 if subset == "small" else 0.2}
            rows.append(r)
            labels.append({"image": r["image"], "object": obj, "present": present})
    return rows, labels


def _write(tmp, rows, labels):
    out = Path(tmp)
    (out / "presence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "labels.json").write_text(json.dumps(labels))
    return out


def test_separable_evidence_scores_auroc_one_on_every_subset():
    with tempfile.TemporaryDirectory() as tmp:
        out = _write(tmp, *_rows())
        result = score(out, out / "labels.json")
        for key, cells in result["table"].items():
            for s in SUBSETS:
                assert cells[s]["auroc"] == 1.0, (key, s)
                assert 0 <= cells[s]["brier"] < 0.1 and 0 <= cells[s]["ece"] <= 1
        assert table_rows(result).count("\\\\") == 3


def test_labels_come_from_the_rater_not_the_prompt():
    rows, labels = _rows()
    for x in labels:
        x["present"] = not x["present"]                           # the rater says the opposite of the prompt
    with tempfile.TemporaryDirectory() as tmp:
        out = _write(tmp, rows, labels)
        result = score(out, out / "labels.json")
        assert result["table"]["combined"]["all"]["auroc"] == 0.0
        assert result["named_but_absent"] == 40


def test_unlabelled_pairs_are_left_out():
    rows, labels = _rows()
    labels[0]["present"] = None
    with tempfile.TemporaryDirectory() as tmp:
        out = _write(tmp, rows, labels)
        assert score(out, out / "labels.json")["n_pairs"] == len(rows) - 1


def test_subset_rules():
    rows = [{"label": True, "area": 0.01, "subset": "ordinary", "n_named": 2, "named": True, "context_absent": False},
            {"label": True, "area": 0.30, "subset": "small", "n_named": 1, "named": True, "context_absent": False},
            {"label": False, "area": 0.00, "subset": "small", "n_named": 1, "named": False, "context_absent": False},
            {"label": False, "area": 0.00, "subset": "context", "n_named": 1, "named": False, "context_absent": False},
            {"label": False, "area": 0.00, "subset": "context", "n_named": 1, "named": False, "context_absent": True}]
    assert subset_rows(rows, "small") == [rows[0], rows[2]]       # small by measured area, absent from small images
    assert subset_rows(rows, "context") == [rows[4]]              # only the scene-suggested absent object
    assert subset_rows(rows, "crowded") == []


def test_generate_and_page_on_the_tiny_model():
    s = load_settings()
    seg, siglip = _tiny_estimators()
    backend = tiny_backend()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        prompts = [PROMPTS[0], PROMPTS[30]]
        generate(backend, EstimatorDeps(seg, siglip, ProvisionalCalibration(s), s), out, prompts, 0, 4, 7.0)
        rows = [json.loads(line) for line in open(out / "presence.jsonl")]
        assert len(rows) == len(queries(prompts[0])) + len(queries(prompts[1]))
        assert {"global", "region", "h", "area"} <= set(rows[0])
        generate(backend, EstimatorDeps(seg, siglip, ProvisionalCalibration(s), s), out, prompts, 0, 4, 7.0)
        assert len(open(out / "presence.jsonl").read().splitlines()) == len(rows)    # a rerun resumes
        build_page(out, out / "labels.html")
        page = (out / "labels.html").read_text(encoding="utf-8")
        assert page.count("data-object=") == len(rows) and "a mug and a teapot" not in page   # caption hidden


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("presence_eval: pass")


if __name__ == "__main__":
    main()
