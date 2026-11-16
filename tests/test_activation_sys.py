"""Tests of activation_sys.py (python tests/test_activation_sys.py)."""
import subprocess
import sys
import tempfile
from pathlib import Path

from fuzzydiff.evaluation.reporting import activation_stats
from fuzzydiff.experiments.activation_sys import build_records, extra_prompts, table_rows
from fuzzydiff.language.prompts import assign_group, base_prompt


def test_extra_prompts_are_in_g_sys():
    ps = extra_prompts()
    assert len(ps) == 8 and all(assign_group(base_prompt(p)) == "sys" for p in ps)


def test_table_on_a_tiny_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, "-m", "fuzzydiff.experiments.run_experiment", "--tiny", "--suite", "soft",
                        "--group", "sys", "--n", "2", "--steps", "12", "--arms", "unguided,stage1,stage1_sar", "--out", str(out)],
                       check=True, capture_output=True)
        records = build_records(out)
        assert any(r["type"] == "final" for r in records) and any(r["type"] == "constraint" for r in records)
        stats = activation_stats(records, 3)
        rows = table_rows(stats)
        assert rows.count("\\\\") == 6 and "F2a" in stats


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("activation_sys: pass")


if __name__ == "__main__":
    main()
