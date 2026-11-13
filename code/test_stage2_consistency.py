"""Tests of stage2_consistency.py (python -I test_stage2_consistency.py)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration_fit import DriftChoice  # noqa: E402
from run_experiment import _tiny_estimators  # noqa: E402
from stage2_consistency import _decide, _rates, build_page, drift_records, score, table_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_decisions_follow_the_stratum_thresholds():
    choice = {"large": DriftChoice(0.5, 0.5, 0.0, 0.0), "small": None}
    assert _decide({"stratum": "large", "iou": 0.4, "shift_rel": 0.1}, choice) == "disabled"
    assert _decide({"stratum": "large", "iou": 0.9, "shift_rel": 0.1}, choice) == "kept"
    assert _decide({"stratum": "large", "iou": 0.9, "shift_rel": 0.9}, choice) == "disabled"
    assert _decide({"stratum": "small", "iou": 0.9, "shift_rel": 0.1}, choice) == "fallback"
    assert _decide({"stratum": "large", "iou": 0.55, "shift_rel": 0.1}, choice, theta_shift=0.1) == "disabled"


def test_rates_count_false_disables_and_misses():
    rows = [{"moved": False, "decision": "disabled"}, {"moved": False, "decision": "kept"},
            {"moved": True, "decision": "kept"}, {"moved": True, "decision": "fallback"}]
    r = _rates(rows)
    assert r["disabled"] == 0.25 and r["false_dis"] == 0.5 and r["miss"] == 1.0 and r["fallback"] == 0.25


def test_drift_page_and_score_on_a_tiny_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", "mixture",
                        "--group", "cons", "--n", "3", "--arms", "stage1,stage1_sar_nocheck", "--out", str(out)],
                       check=True, capture_output=True)
        seg, _ = _tiny_estimators()
        rows = drift_records(seg, out)
        assert rows and {"iou", "shift_rel", "stratum", "variant"} <= set(rows[0])
        with open(out / "drift.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        assert build_page(out, out / "page.html") == 3
        labels = {}
        for r in rows:
            labels.setdefault(r["image"], {})[r["object"]] = r["iou"] < 0.99
        result = score(out, labels, 0.10, ci=False)
        assert set(result) >= {"default", "temp 2", "delta x2"}
        assert table_rows(result).count("\\\\") == len(result) + 1


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("stage2_consistency: pass")


if __name__ == "__main__":
    main()
