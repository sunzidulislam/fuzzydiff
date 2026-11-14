"""Tests of refine_arms.py and refined_table.py (python -I test_refined_table.py)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refined_table import ROWS, collect, differences, table_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_chain_on_tiny_runs():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for setting in ("mixture", "size"):
            subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_experiment.py"), "--tiny", "--suite", setting,
                            "--group", "dev", "--n", "2", "--arms", "unguided,stage1,stage1_plain,stage1_sar",
                            "--out", str(out / setting)], check=True, capture_output=True)
            for base in ("sd1x", "ae", "spo", "spo_refiner"):        # stand-ins for the baseline images
                shutil.copytree(out / setting / "unguided", out / setting / base)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "refine_arms.py"), "--tiny", "--out", str(out),
                        "--arms", "unguided,sd1x,ae,spo"], check=True, capture_output=True)
        assert len(list((out / "mixture" / "ae_r").glob("*/image.png"))) == 2
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "refined_table.py"), "pages", "--out", str(out)],
                       check=True, capture_output=True)
        good = {"stage1_sar": True, "stage1_plain": True, "stage1": True}
        for name, page in (("refined_mixture", "mixture"), ("refined_size", "size")):
            key = json.load(open(out / f"{name}.key.json"))
            assert all("pd_r" not in e["panels"].values() for e in key.values())   # no PD images: not on the page
            arms = {a for e in key.values() for a in e["panels"].values()}
            good_by_arm = {a: good.get(a, False) for a in arms}
            ratings = [r for r in _ratings_for(page, key, good_by_arm)]
            (out / f"{name}.ratings.json").write_text(json.dumps(ratings))
        with open(out / "refined_auto.jsonl", "w") as f:
            for setting, metric in (("mixture", "vqa"), ("size", "owl")):
                for meta in (out / setting / "unguided").glob("*/meta.json"):
                    prompt = json.loads(meta.read_text())["prompt"]
                    for arm, _ in (r for r in ROWS if r[0] != "pd_r"):     # no PD images, so no PD scores
                        f.write(json.dumps({"setting": setting, "arm": arm, "prompt": prompt, metric: 0.5}) + "\n")
        metrics = collect(out)
        rows = table_rows(metrics)
        assert rows.count("\\\\") == len(ROWS)
        pd_row = next(line for line in rows.splitlines() if line.startswith("PD + R"))
        assert set(pd_row.replace("PD + R &", "").replace("\\\\", "").replace("/", "").replace("&", "").split()) == {"--"}
        assert differences(metrics)


def _ratings_for(page, key, good):
    out = []
    for group, e in key.items():
        for letter, arm in e["panels"].items():
            for q in e["objects"]:
                out.append({"group": group, "panel": letter, "object": q, "present": True,
                            "colour": good[arm] if q in e.get("coloured", []) else None})
            field = "leakage" if page == "mixture" else "size"
            out.append({"group": group, "panel": letter, field: (not good[arm]) if field == "leakage" else good[arm],
                        "fidelity": 80 if good[arm] else 40})
    return out


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("refined_table: pass")


if __name__ == "__main__":
    main()
