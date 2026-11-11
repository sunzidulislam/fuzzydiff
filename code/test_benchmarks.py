"""Tests of benchmarks.py on fake prompt files and run folders. No GPU or torch needed.
Usage: python -I test_benchmarks.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmarks import BENCHMARKS, export, subset  # noqa: E402

GENEVAL_LINES = ['{"tag": "single_object", "include": [{"class": "cow", "count": 1}], "prompt": "a photo of a cow"}',
                 '{"tag": "colors", "include": [{"class": "car", "count": 1, "color": "red"}], "prompt": "a photo of a red car"}',
                 '{"tag": "two_object", "include": [], "prompt": "a photo of a cat and a dog"}']
COLOR_PROMPTS = ["a green bench and a blue bowl", "a red cake and a blue bench", "a red book and a brown sheep"]


def sources(root):
    """Prompt files where the notebook clones them, relative to a sources folder."""
    (root / "geneval/prompts").mkdir(parents=True)
    (root / "geneval/prompts/evaluation_metadata.jsonl").write_text("\n".join(GENEVAL_LINES) + "\n")
    (root / "T2I-CompBench/examples/dataset").mkdir(parents=True)
    (root / "T2I-CompBench/examples/dataset/color_val.txt").write_text("\n".join(COLOR_PROMPTS) + "\n\n")
    return root


def run(root, arm, prompts, seeds, skip=(), failed=()):
    """A run folder as run_experiment.py leaves it: one folder per image with meta.json and image.png."""
    for i, prompt in enumerate(prompts):
        for seed in seeds:
            if (prompt, seed) in skip:
                continue
            d = root / arm / f"job{i}_s{seed}"
            d.mkdir(parents=True)
            meta = {"prompt": prompt, "seed": seed, **({"failed": "oom"} if (prompt, seed) in failed else {})}
            (d / "meta.json").write_text(json.dumps(meta))
            Image.new("RGB", (4, 4), (seed, 0, 0)).save(d / "image.png")
    return root


def test_geneval_export_keeps_each_prompt_line_and_its_position():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = sources(tmp / "src")
        prompts = [json.loads(line)["prompt"] for line in GENEVAL_LINES]
        r = run(tmp / "run", "stage1", prompts, [0, 1], skip={(prompts[1], 1)})
        counts = export(BENCHMARKS["geneval"], r, "stage1", tmp / "out", src)
        assert counts == {"exported": 2, "incomplete": 1, "not_run": 0, "seeds": 2}
        assert sorted(p.name for p in (tmp / "out").iterdir()) == ["00000", "00002"]
        assert (tmp / "out/00002/metadata.jsonl").read_text() == GENEVAL_LINES[2] + "\n"
        assert sorted(p.name for p in (tmp / "out/00000/samples").iterdir()) == ["0000.png", "0001.png"]
        assert Image.open(tmp / "out/00000/samples/0001.png").getpixel((0, 0))[0] == 1      # image k is seed k


def test_t2i_export_names_files_like_the_official_scorer_reads_them():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = sources(tmp / "src")
        r = run(tmp / "run", "unguided", COLOR_PROMPTS[:2], range(3), failed={(COLOR_PROMPTS[1], 2)})
        counts = export(BENCHMARKS["color"], r, "unguided", tmp / "out", src)
        assert counts == {"exported": 1, "incomplete": 1, "not_run": 1, "seeds": 3}       # a failed image is unfinished
        names = sorted(p.name for p in (tmp / "out/samples").iterdir())
        assert names == [f"{COLOR_PROMPTS[0]}_{k:06d}.png" for k in range(3)]
        assert all(n.split("_")[0] == COLOR_PROMPTS[0] for n in names)                   # prompt before the first '_'


def test_pilot_exports_with_the_same_command_and_keeps_full_run_numbering():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = sources(tmp / "src")
        r = run(tmp / "run", "stage1", [COLOR_PROMPTS[2]], [0])
        assert export(BENCHMARKS["color"], r, "stage1", tmp / "out", src)["exported"] == 1
        # third prompt of the file, 10 images per prompt in the full protocol: numbers start at 20
        assert [p.name for p in (tmp / "out/samples").iterdir()] == [f"{COLOR_PROMPTS[2]}_000020.png"]


def test_subset_is_fixed_and_in_file_order():
    items = [f"prompt {i}" for i in range(50)]
    assert subset(items, 3) == subset(items, 3) and subset(items, 3) == [p for p in items if p in subset(items, 3)]
    assert len({subset(items, 1, seed=s)[0] for s in range(10)}) > 1
    assert subset(items) == items and subset(items, 99) == items


def test_command_line():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = sources(tmp / "src")

        def call(*args):
            return subprocess.run([sys.executable, "-I", os.path.join(HERE, "benchmarks.py"), *args, "--sources", str(src)],
                                  check=True, capture_output=True, text=True).stdout
        assert call("seeds", "geneval").strip() == "0,1,2,3" and call("seeds", "spatial").strip() == ",".join(map(str, range(10)))
        call("prompts", "geneval", "--out", str(tmp / "p.json"))
        assert json.loads((tmp / "p.json").read_text()) == ["a photo of a cow", "a photo of a red car", "a photo of a cat and a dog"]
        call("prompts", "color", "--n", "1", "--out", str(tmp / "one.json"))
        assert len(json.loads((tmp / "one.json").read_text())) == 1
        bad = tmp / "bad"
        (bad / "T2I-CompBench/examples/dataset").mkdir(parents=True)
        (bad / "T2I-CompBench/examples/dataset/shape_val.txt").write_text("a snake_case prompt\n")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "benchmarks.py"), "prompts", "shape", "--out",
                            str(tmp / "x.json"), "--sources", str(bad)], capture_output=True, text=True)
        assert r.returncode != 0 and "not allowed" in r.stderr


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
