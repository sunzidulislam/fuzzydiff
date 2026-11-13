"""Tests of shift_sys.py (python -I test_shift_sys.py)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import load_settings  # noqa: E402
from shift_sys import build_page, build_records, key, shift_stats, source_of, table_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_sources():
    assert source_of("unguided") == "unguided" and source_of("stage1_sar") == "guided"
    assert source_of("abl_point") == "variants"


def test_records_page_and_score_on_tiny_runs():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for suite in ("soft", "size"):
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", suite,
                            "--group", "sys", "--n", "2", "--arms", "unguided,stage1,stage1_sar,abl_point",
                            "--out", str(out)], check=True, capture_output=True)
        records = build_records(out, load_settings())
        assert records and {r["source"] for r in records} == {"unguided", "guided", "variants"}
        assert {r["family"] for r in records} <= {"F2a", "F6", "F8"}
        n = build_page(out, out / "shift.html")
        assert n == len({(r["arm"], r["image"]) for r in records})
        for i, r in enumerate(records):
            r.update(rating=10 * (i % 7), accept=i % 2 == 0)
        stats = shift_stats(records)
        assert sum(v["n"] for v in stats.values()) == len(records)
        assert table_rows(stats).count("\\\\") == 3
        assert key(records[0]).count("|") == 1


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("shift_sys: pass")


if __name__ == "__main__":
    main()
