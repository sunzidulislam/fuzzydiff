"""Tests of activation_sys.py (python -I test_activation_sys.py)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activation_sys import build_records, extra_prompts, table_rows  # noqa: E402
from prompts import assign_group, base_prompt  # noqa: E402
from reporting import activation_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_extra_prompts_are_in_g_sys():
    ps = extra_prompts()
    assert len(ps) == 8 and all(assign_group(base_prompt(p)) == "sys" for p in ps)


def test_table_on_a_tiny_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", "soft",
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
