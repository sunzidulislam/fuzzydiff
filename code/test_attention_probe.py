"""Tests of attention_probe.py (python -I test_attention_probe.py)."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attention_probe  # noqa: E402
from attention_probe import build_page, groups, probe_prompts, run, score, table_rows  # noqa: E402
from sd3_backend import tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402


def test_prompts_are_the_calibration_ones_plus_binding():
    p = probe_prompts(2, 1, 1)
    assert [k for _, k in p] == ["calib", "calib", "calib", "binding"]


def test_block_groups_cover_sd3_medium():
    g = groups(load_settings())
    assert g["blocks 0-7"] == list(range(8)) and g["blocks 16-23"][-1] == 23 and g["selected L"]


def test_run_page_and_score_on_the_tiny_model():
    s = load_settings()
    attention_probe.WINDOW = (0.0, 1.0)                  # the tiny schedule has 4 steps
    backend = tiny_backend(read_blocks=(0, 1, 2))
    from grounding import phrase_spans
    from predicates import translate
    prompts = probe_prompts(1, 1, 1)
    for prompt, _ in prompts:
        backend.encode_prompt(prompt, phrase_spans(prompt, translate(prompt)))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        run(backend, s, out, prompts, 4)
        assert len(list(out.glob("*/records.jsonl"))) == 3
        assert build_page(out, out / "binding.html") == 1
        boxes, bind = {}, {}
        for meta in out.glob("*/meta.json"):
            m = json.loads(meta.read_text())
            objs = translate(m["prompt"]).objects
            boxes[f"unguided/{meta.parent.name}"] = {"present": {o: True for o in objs},
                                                    "boxes": {o: [0.0, 0.0, 0.6, 0.6] for o in objs}}
            attrs = [c.args[0] for c in translate(m["prompt"]).constraints if c.family == "binding"]
            if attrs:
                bind[meta.parent.name] = {a: objs[i % len(objs)] for i, a in enumerate(attrs)}
        (out / "boxes.json").write_text(json.dumps(boxes))
        (out / "bind.json").write_text(json.dumps(bind))
        table = score(out, out / "boxes.json", out / "bind.json")
        assert table["blocks 0-7"]["n_point"] > 0 and table["blocks 0-7"]["n_maps"] > 0   # the tiny model has 3 blocks, no L
        assert table["blocks 0-7"]["n_pairs"] > 0
        assert table_rows(table).count("\\\\") == 5


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("attention_probe: pass")


if __name__ == "__main__":
    main()
