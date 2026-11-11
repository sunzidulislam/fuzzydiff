"""Standard benchmarks (Tab. standard benchmarks): GenEval and T2I-CompBench++, scored with their official code.

Generation is ordinary run_experiment.py on the benchmark's own prompts; this file only converts in both
directions:

  prompts  the official prompt file -> the JSON list run_experiment.py --prompts-file reads
  export   finished images of one arm -> the folder layout the official scorer reads

Official layouts (read from each repository's README and evaluation scripts):
  GenEval          <out>/<00000>/metadata.jsonl (the prompt's line of evaluation_metadata.jsonl, verbatim)
                   <out>/<00000>/samples/<0000>.png ...                       4 images per prompt
  T2I-CompBench++  <out>/samples/<prompt>_<000000>.png, one running number over all images   10 per prompt

Image k of a prompt is the run's seed k, the same seeds for every method. A prompt with an image missing is
left out and counted, so a partial run never mixes complete and incomplete prompts.

  python -I code/benchmarks.py prompts --geneval geneval/prompts/evaluation_metadata.jsonl --out outputs/geneval.json
  python -I code/benchmarks.py prompts --t2i T2I-CompBench/examples/dataset/color_val.txt --out outputs/t2i_color.json
  python -I code/benchmarks.py export --run outputs/bench_geneval --arm stage1_sar --images 4 \
      --geneval geneval/prompts/evaluation_metadata.jsonl --out outputs/geneval_stage1_sar
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

GENEVAL_IMAGES = 4
T2I_IMAGES = 10


def read_geneval(path):
    """(prompt, metadata line) per prompt, in file order."""
    lines = [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(json.loads(line)["prompt"], line) for line in lines]


def read_t2i(path):
    """Prompts of a T2I-CompBench++ *_val.txt file, in file order. The scorer splits file names at '_'."""
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [p for p in prompts if "_" in p or "/" in p]
    if bad:
        raise ValueError(f"T2I-CompBench++ names files <prompt>_<n>.png, so a prompt cannot contain '_' or '/': {bad[:3]}")
    return prompts


def _images(run, arm, prompt, n, image_id):
    paths = [Path(run) / arm / image_id(prompt, seed) / "image.png" for seed in range(n)]
    return paths if all(p.exists() for p in paths) else None


def export_geneval(run, arm, metadata, out, n=GENEVAL_IMAGES, image_id=None):
    """Copies the arm's images into GenEval's layout. Returns (exported, missing) prompt counts."""
    image_id = image_id or _image_id()
    out = Path(out)
    done = missing = 0
    for i, (prompt, line) in enumerate(read_geneval(metadata)):
        paths = _images(run, arm, prompt, n, image_id)
        if paths is None:
            missing += 1
            continue
        folder = out / f"{i:05d}"
        (folder / "samples").mkdir(parents=True, exist_ok=True)
        (folder / "metadata.jsonl").write_text(line + "\n", encoding="utf-8")
        for k, p in enumerate(paths):
            shutil.copyfile(p, folder / "samples" / f"{k:04d}.png")
        done += 1
    return done, missing


def export_t2i(run, arm, prompt_file, out, n=T2I_IMAGES, image_id=None):
    """Copies the arm's images into <out>/samples, T2I-CompBench++'s layout. Returns (exported, missing)."""
    image_id = image_id or _image_id()
    samples = Path(out) / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    done = missing = 0
    for i, prompt in enumerate(read_t2i(prompt_file)):
        paths = _images(run, arm, prompt, n, image_id)
        if paths is None:
            missing += 1
            continue
        for k, p in enumerate(paths):
            shutil.copyfile(p, samples / f"{prompt}_{i * n + k:06d}.png")
        done += 1
    return done, missing


def _image_id():
    from run_experiment import image_id    # needs torch; tests pass their own
    return image_id


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--geneval")
    p.add_argument("--t2i")
    p.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--arm", required=True)
    e.add_argument("--geneval")
    e.add_argument("--t2i")
    e.add_argument("--images", type=int, default=None, help=f"per prompt; default {GENEVAL_IMAGES} (GenEval), {T2I_IMAGES} (T2I)")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if bool(args.geneval) == bool(args.t2i):
        ap.error("give exactly one of --geneval and --t2i")
    if args.cmd == "prompts":
        prompts = [p for p, _ in read_geneval(args.geneval)] if args.geneval else read_t2i(args.t2i)
        Path(args.out).write_text(json.dumps(prompts, indent=1), encoding="utf-8")
        print(f"{len(prompts)} prompts -> {args.out}")
        return
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if args.geneval:
        done, missing = export_geneval(args.run, args.arm, args.geneval, args.out, args.images or GENEVAL_IMAGES)
    else:
        done, missing = export_t2i(args.run, args.arm, args.t2i, args.out, args.images or T2I_IMAGES)
    print(f"{args.arm}: exported {done} prompts to {args.out}; {missing} left out (an image not finished)")
    if missing:
        print("the benchmark score is over the exported prompts only: finish the run before reporting it")


if __name__ == "__main__":
    main()
