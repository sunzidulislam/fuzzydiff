"""Tests of paired_table.py (python -I test_paired_table.py)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_table import ARMS, load_metrics, strongest, table_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_strongest_respects_direction():
    table = {"p1": {"spo": 10.0, "ae": 30.0}, "p2": {"spo": 20.0, "ae": 40.0}}
    assert strongest(table, higher=True) == "ae" and strongest(table, higher=False) == "spo"


def _ratings(page, key):
    """One complete synthetic answer per panel: arm-dependent so differences are non-zero."""
    good = {"stage1_sar": True, "stage1": True, "unguided": False, "spo": False, "ae": True}
    out = []
    for group, e in key.items():
        for letter, arm in e["panels"].items():
            if page == "fmas":
                out.append({"group": group, "panel": letter, "rating": 5 if good[arm] else 2})
            elif page == "success":
                out.append({"group": group, "panel": letter, "present": True, "acceptable": good[arm]})
            else:
                for q in e["objects"]:
                    out.append({"group": group, "panel": letter, "object": q, "present": True,
                                "colour": good[arm] if q in e.get("coloured", []) else None})
                field = "leakage" if page == "mixture" else "size"
                value = (not good[arm]) if field == "leakage" else good[arm]
                out.append({"group": group, "panel": letter, field: value, "fidelity": 80 if good[arm] else 40})
    return out


def test_pages_and_table_on_tiny_runs():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for suite in ("soft", "mixture", "size"):
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", suite,
                            "--group", "dev", "--n", "2", "--arms", "unguided,stage1,stage1_sar",
                            "--out", str(out / suite)], check=True, capture_output=True)
            for base in ("spo", "ae"):                            # stand-ins for the baseline images
                shutil.copytree(out / suite / "unguided", out / suite / base)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "paired_table.py"), "pages", "--out", str(out)],
                       check=True, capture_output=True)
        for page in ("fmas", "success", "mixture", "size"):
            key = json.load(open(out / f"{page}.key.json"))
            assert key and all(len(e["panels"]) == len(ARMS) for e in key.values())
            (out / f"{page}.ratings.json").write_text(json.dumps(_ratings(page, key)))
        rows = table_rows(load_metrics(out))
        assert rows.count("\\\\") == 10 and "A\\&E" in rows and "SPO" in rows
        assert "--" not in rows.split("\\midrule")[1]


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("paired_table: pass")


if __name__ == "__main__":
    main()
