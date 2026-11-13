"""Tests of gradient_validity.py (python -I test_gradient_validity.py)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_validity import build_page, commanded_band, rows, score, table_rows  # noqa: E402
from prompts import assign_group, base_prompt  # noqa: E402
from settings import load_settings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def test_every_row_has_two_g_int_prompts():
    for key, _, _, _, _, prompts in rows():
        assert len(prompts) == 2, key
        assert all(assign_group(base_prompt(p)) == "int" for p in prompts), key


def test_commanded_bands_respect_each_scale():
    s = load_settings()
    up = commanded_band("F2a", 0.4, +1, s)
    assert up and abs((up[0] + up[1]) / 2 - 0.7) < 1e-9
    assert commanded_band("F2a", 0.97, +1, s) is None                 # no headroom at the top of [0, 1]
    assert commanded_band("binding", 0.9, +1, s) is None and commanded_band("binding", 0.0, -1, s)
    assert commanded_band("presence", 3.0, -1, s)[0] < 3.0           # logit scale, unbounded


def test_run_page_and_score_on_the_tiny_model():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "gradient_validity.py"), "run", "--tiny", "--out", str(out),
                        "--only", "F2a_l,F9,binding,presence"], check=True, capture_output=True)
        recs = list(out.glob("*/*/record.json"))
        assert len(recs) == 8
        labels = {}
        for p in recs:                                            # a measurement and a judgment for every run
            rec = json.loads(p.read_text())
            for name, run in rec["runs"].items():
                run["y"] = 0.0 if name == "base" else (0.2 if name == "up" else -0.2)
                if name != "base":
                    key = f"{p.parent.parent.name}/{p.parent.name}/{name}"
                    labels[key] = ({"left": "neither", "right": rec["constraint"][1][1]} if rec["family"] == "binding"
                                   else {"change": "more" if name == "up" else "less"})
            p.write_text(json.dumps(rec))
        n = build_page(out, out / "gv.html")
        result = score(out, labels, load_settings())
        assert table_rows(result).count("\\\\") == len(rows())
        done = [r for r in result.values() if r.get("n")]
        assert n == sum(r["n"] for r in done)
        f2a = result["F2a_l"]
        assert f2a["n"] == 0 or (f2a["dir_h"] == 1.0 and f2a["dir_a"] == 1.0)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("gradient_validity: pass")


if __name__ == "__main__":
    main()
