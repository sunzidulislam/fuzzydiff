"""Tests of context_invariance.py (python -I test_context_invariance.py)."""
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from context_invariance import (FAMILIES, build_page, categories, fit_all, generate, holm, jobs,  # noqa: E402
                                table_rows)
from prompts import assign_group, base_prompt  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402


def test_four_categories_per_family_all_in_g_cal():
    for family in FAMILIES:
        assert len(categories(family)) == 4
    assert all(assign_group(base_prompt(j["prompt"])) == "cal" for j in jobs(1))
    assert len(jobs()) == 3 * 4 * 2 * 15


def test_holm_is_monotone_and_capped():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.9, 0.8]) == [1.0, 1.0]


def _labels(out, shift):
    """Synthetic judgments: accept when s is near a hedge centre, the centre moved by `shift` per category."""
    rng, labels = random.Random(0), {}
    centre = {"slightly": 0.3, "very": 0.8, "moderately": 0.55, "considerably": 0.7}
    for meta_path in out.glob("*/meta.json"):
        m = json.loads(meta_path.read_text())
        cats = sorted({j["category"] for j in jobs(1) if j["family"] == m["family"]})
        c = centre[m["hedge"]] + shift * cats.index(m["category"])
        s = rng.random()
        if m["family"] == "F8":
            m["m"] = s
            meta_path.write_text(json.dumps(m))
        labels[meta_path.parent.name] = {"accept": abs(s - c) < 0.15, "degree": round(100 * s)}
    return labels


def test_generate_page_and_fit_on_the_tiny_model():
    backend = tiny_backend()
    todo = jobs(15)
    for p in dict.fromkeys(j["prompt"] for j in todo):
        backend.encode_prompt(p)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        generate(backend, out, todo[:30] + todo[120:150] + todo[240:270], 2, 7.0)
        assert build_page(out, out / "context.html") == 90
        rows = fit_all(out, _labels(out, 0.0))
        assert [r["family"] for r in rows] == ["F2a", "F6", "F8"]
        assert table_rows(rows).count("\\\\") == 3


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("context_invariance: pass")


if __name__ == "__main__":
    main()
