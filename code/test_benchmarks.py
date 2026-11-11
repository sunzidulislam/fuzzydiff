"""Tests of benchmarks.py on a fake run folder. No torch needed.
Usage: python -I test_benchmarks.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import export_geneval, export_t2i, read_t2i  # noqa: E402


def fake_id(prompt, seed):
    return f"{abs(hash(prompt)) % 10 ** 8}_s{seed}"


def make_run(root, arm, prompts, n, skip=()):
    for prompt in prompts:
        for seed in range(n):
            if (prompt, seed) in skip:
                continue
            d = root / arm / fake_id(prompt, seed)
            d.mkdir(parents=True)
            Image.new("RGB", (4, 4), (seed, 0, 0)).save(d / "image.png")


def test_geneval_layout_keeps_metadata_verbatim_and_drops_incomplete_prompts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lines = ['{"tag": "single_object", "include": [{"class": "cow", "count": 1}], "prompt": "a photo of a cow"}',
                 '{"tag": "colors", "include": [{"class": "car", "count": 1, "color": "red"}], "prompt": "a photo of a red car"}',
                 '{"tag": "two_object", "include": [], "prompt": "a photo of a cat and a dog"}']
        (root / "meta.jsonl").write_text("\n".join(lines) + "\n")
        make_run(root / "run", "stage1", ["a photo of a cow", "a photo of a red car", "a photo of a cat and a dog"], 2,
                 skip={("a photo of a red car", 1)})
        done, missing = export_geneval(root / "run", "stage1", root / "meta.jsonl", root / "out", n=2, image_id=fake_id)
        assert (done, missing) == (2, 1)
        assert sorted(p.name for p in (root / "out").iterdir()) == ["00000", "00002"]   # index = line in the file
        assert (root / "out" / "00002" / "metadata.jsonl").read_text() == lines[2] + "\n"
        assert sorted(p.name for p in (root / "out" / "00000" / "samples").iterdir()) == ["0000.png", "0001.png"]
        assert Image.open(root / "out" / "00000" / "samples" / "0001.png").getpixel((0, 0))[0] == 1   # image k = seed k


def test_t2i_names_files_prompt_and_running_number():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "color_val.txt").write_text("a green bench and a blue bowl\na red cake and a blue bench\n\n")
        make_run(root / "run", "unguided", ["a green bench and a blue bowl", "a red cake and a blue bench"], 3)
        assert export_t2i(root / "run", "unguided", root / "color_val.txt", root / "out", n=3, image_id=fake_id) == (2, 0)
        names = sorted(p.name for p in (root / "out" / "samples").iterdir())
        assert names[0] == "a green bench and a blue bowl_000000.png" and len(names) == 6
        assert "a red cake and a blue bench_000005.png" in names
        # the official scorer reads the prompt before the first '_' and sorts on the number after the last
        assert all(n.split("_")[0] in ("a green bench and a blue bowl", "a red cake and a blue bench") for n in names)
        (root / "bad.txt").write_text("a snake_case prompt\n")
        try:
            read_t2i(root / "bad.txt")
            raise AssertionError("a prompt with '_' must be refused")
        except ValueError:
            pass


def test_prompt_lists_are_what_run_experiment_reads():
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "m.jsonl").write_text('{"tag": "a", "prompt": "a photo of a cow"}\n{"tag": "b", "prompt": "a photo of a dog"}\n')
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([sys.executable, "-I", os.path.join(here, "benchmarks.py"), "prompts", "--geneval",
                        str(root / "m.jsonl"), "--out", str(root / "p.json")], check=True, capture_output=True)
        assert json.loads((root / "p.json").read_text()) == ["a photo of a cow", "a photo of a dog"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
