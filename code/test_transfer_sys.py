"""Tests of transfer_sys.py (python -I test_transfer_sys.py)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import assign_group, base_prompt  # noqa: E402
from transfer_sys import _direction, build_page, f2b_prompts, table_rows, transfer_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_f2b_prompts_are_in_g_sys():
    ps = f2b_prompts()
    assert len(ps) == 6 and all(assign_group(base_prompt(p)) == "sys" for p in ps)


def test_direction_points_into_the_interval():
    assert _direction(0.2, 0.4, 0.6) == 1 and _direction(0.8, 0.4, 0.6) == -1 and _direction(0.5, 0.4, 0.6) == 0
    assert _direction(-0.1, 0.0, float("inf")) == 1


def _pairs():
    out = []
    for i in range(12):
        fam = ("F2a", "F6", "F8", "binding")[i % 4]
        ds = 0.1 + 0.02 * i
        out.append({"image": f"img{i}", "prompt": "a slightly dark mug", "base": f"b{i % 6}", "cid": 0, "family": fam,
                    "args": ["mug"], "ds": ds, "dy": 0.5 * ds, "direction": 1, "n_active": 1 + i % 3, "mu_est": 0.8,
                    "human_accept": i % 2 == 0})
    return out


def test_transfer_statistics_follow_reporting():
    stats = transfer_stats(_pairs(), n_boot=50)
    f2a = stats["F2a"]
    assert f2a["dir_acc"] == 1.0 and abs(f2a["slope"] - 0.5) < 1e-9 and f2a["exploitation"] == 0.0
    assert stats["F2b"]["n"] == 0
    assert table_rows(stats).count("\\\\") == 5


def test_page_on_a_tiny_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", "soft",
                        "--group", "sys", "--n", "2", "--arms", "unguided,stage1,stage1_sar", "--log-all-steps",
                        "--out", str(out)], check=True, capture_output=True)
        images = [p.parent.name for p in (out / "stage1_sar").glob("*/image.png")]
        assert images
        with open(out / "pairs.jsonl", "w") as f:
            for iid in images:
                f.write(json.dumps({"image": iid, "prompt": json.loads((out / "stage1_sar" / iid / "meta.json").read_text())["prompt"],
                                    "cid": 0, "family": "F2a", "args": ["x"], "ds": 0.1, "dy": 0.1}) + "\n")
        assert build_page(out, out / "t.html") == len(images)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("transfer_sys: pass")


if __name__ == "__main__":
    main()
