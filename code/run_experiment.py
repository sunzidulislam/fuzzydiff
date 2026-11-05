"""Run experiment arms with per-image saving, so a Kaggle timeout costs at most the current image.

Each image is written into a temporary directory and published by one atomic rename to
<out>/<arm>/<image id>/ (image.png, meta.json, log.jsonl). A finished directory is never redone; a
failed one is retried on the next run; a temporary directory left by a dead session is discarded.

Usage:
  python -I code/run_experiment.py --suite m1_tune --out outputs/m1_tune
  python -I code/run_experiment.py --suite m1_eval --out outputs/m1 [--resume-from DIR] [--vae-device cuda:1]
  python -I code/run_experiment.py --suite m1_eval --tiny --out outputs/smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibration import ProvisionalCalibration  # noqa: E402
from grounding import phrase_spans  # noqa: E402
from guidance import GuideCfg, run_stage1  # noqa: E402
from measure import measure_scene  # noqa: E402
from predicates import Constraint, Program, translate  # noqa: E402
from prompts import enumerate_suite  # noqa: E402
from reporting import StepLog  # noqa: E402
from sd3_backend import load_backend, tiny_backend  # noqa: E402
from settings import load_settings  # noqa: E402

COMMANDED = {"darker": -1, "brighter": +1}
TARGET_BAND = 0.02
UNGUIDED_ARMS = {"unguided", "ladder_unguided"}


@dataclass
class Job:
    arm: str
    prompt: str
    seed: int
    split: str
    suite: str
    program: Program
    overrides: Dict[str, float] = field(default_factory=dict)
    base_arm: str = "unguided"


@dataclass
class Deps:
    backend: object
    calib: object
    measure: Callable
    cfg: GuideCfg
    settings: object
    stage1: Callable = run_stage1
    unload_text: bool = False
    measure_factory: Optional[Callable] = None     # (prompt, cond) -> measure; needs phrase readout


@dataclass
class Summary:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)


def image_id(prompt: str, seed: int) -> str:
    return f"{hashlib.sha256(prompt.encode()).hexdigest()[:12]}_s{seed}"


def commanded_targets(unguided_s, shift, margin, headroom):
    """Target interval for a commanded shift, or None when the direction has too little headroom
    (App. gradient-validity test: the target stays `margin` inside the scale)."""
    target = min(max(unguided_s + shift, margin), 1 - margin)
    if abs(target - unguided_s) < headroom:
        return None
    return (target - TARGET_BAND, target + TARGET_BAND)


def _is_finished(path: Path) -> bool:
    meta = path / "meta.json"
    if not meta.exists():
        return False
    with open(meta) as f:
        return "failed" not in json.load(f)


def _save_image(image, path):
    x = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((x * 255).round().astype(np.uint8)).save(path)


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float("nan")


def _with_overrides(cfg: GuideCfg, overrides) -> GuideCfg:
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def _read_meta(out: Path, arm: str, job: Job) -> Optional[dict]:
    path = out / arm / image_id(job.prompt, job.seed) / "meta.json"
    if not path.exists():
        return None
    with open(path) as f:
        meta = json.load(f)
    return None if "failed" in meta else meta


def _final_records(job, finals, base_finals, log):
    by_key = {(f["cid"], f["component"]): f["mu"] for f in (base_finals or [])}
    for f in finals:
        guided = f["mu"]
        unguided = guided if base_finals is None else by_key.get((f["cid"], f["component"]), float("nan"))
        log.write(type="final", prompt=job.prompt, seed=job.seed, family=f["family"], cid=f["cid"],
                  component=f["component"], tier=f["tier"], mu_unguided=unguided, mu_guided=guided, s=f["s"])


def _cond(job: Job, deps: Deps):
    spans = phrase_spans(job.prompt, job.program) if deps.measure_factory else None
    return deps.backend.encode_prompt(job.prompt, spans)


def _execute(job: Job, deps: Deps, out: Path, tmp: Path) -> dict:
    """Run one job into tmp; returns its metadata."""
    meta = {"arm": job.arm, "prompt": job.prompt, "seed": job.seed, "split": job.split, "suite": job.suite,
            "calibration": deps.calib.tag, "overrides": job.overrides,
            "provisional_used": sorted(deps.settings.provisional_used)}
    targets, base_finals = None, None
    cfg = _with_overrides(deps.cfg, job.overrides)
    if job.arm.split("@")[0] in COMMANDED:
        base = _read_meta(out, job.base_arm, job)
        if base is None:
            raise LookupError(f"missing {job.base_arm} image for {job.prompt!r} seed {job.seed}")
        base_finals = base["finals"]
        s = base_finals[0]["s"]
        shift = COMMANDED[job.arm.split("@")[0]] * deps.settings.get("cfgGVShift")
        band = commanded_targets(s, shift, deps.settings.get("cfgGVMargin"), deps.settings.get("cfgGVHeadroom"))
        if band is None:
            meta["skipped"] = "headroom"
            meta["finals"] = []
            return meta
        targets = {0: band}
        meta["ds"] = (band[0] + band[1]) / 2 - s
    if job.arm in UNGUIDED_ARMS:
        cfg = dataclasses.replace(cfg, guided=False)
    elif job.arm == "ladder_guided":
        base = _read_meta(out, "ladder_unguided", job)
        base_finals = base["finals"] if base else None

    cond = _cond(job, deps)
    meta["unmapped"] = cond.unmapped
    measure = deps.measure_factory(job.prompt, cond) if deps.measure_factory else deps.measure
    log = StepLog(str(tmp / "log.jsonl"), job.split, deps.calib.tag)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        result = deps.stage1(deps.backend, job.program, cond, job.seed, deps.calib, measure, cfg, log,
                             targets=targets, prompt=job.prompt)
        _final_records(job, result.finals, base_finals, log)
    finally:
        log.close()
    _save_image(result.image, tmp / "image.png")
    meta.update(finals=result.finals, time_s=time.perf_counter() - start, peak_gb=_peak_gb())
    if "ds" in meta:
        meta["dy"] = result.finals[0]["s"] - base_finals[0]["s"]
    return meta


def run_image(job: Job, deps: Deps, out: Path) -> dict:
    """Publish one image atomically. Out-of-memory and other errors become a failure record."""
    tmp = out / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    try:
        meta = _execute(job, deps, out, tmp)
    except torch.cuda.OutOfMemoryError:
        meta = {"arm": job.arm, "prompt": job.prompt, "seed": job.seed, "failed": "oom", "peak_gb": _peak_gb()}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        meta = {"arm": job.arm, "prompt": job.prompt, "seed": job.seed, "failed": type(e).__name__,
                "error": str(e), "traceback": traceback.format_exc()}
    with open(tmp / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    dest = out / job.arm / image_id(job.prompt, job.seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    return meta


def run_jobs(jobs: List[Job], deps: Deps, out, resume_from=None) -> Summary:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(".tmp-*"):
        shutil.rmtree(stale, ignore_errors=True)
    summary = Summary()
    pending = []
    for job in jobs:
        rel = Path(job.arm) / image_id(job.prompt, job.seed)
        if resume_from is not None and not (out / rel).exists() and _is_finished(Path(resume_from) / rel):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(resume_from) / rel, out / rel)
        if _is_finished(out / rel):
            summary.skipped += 1
        else:
            pending.append(job)
    for job in pending:
        _cond(job, deps)
    if deps.unload_text:
        deps.backend.unload_text_encoders()
    for job in pending:
        meta = run_image(job, deps, out)
        if "failed" in meta:
            summary.failed += 1
            summary.failures.append(f"{job.arm} {job.prompt!r} seed {job.seed}: {meta['failed']}")
        else:
            summary.done += 1
    return summary


def merge_logs(out_dir, into) -> int:
    """Concatenate per-image logs of every published image; returns the number of records."""
    n = 0
    with open(into, "w") as dest:
        for log in sorted(Path(out_dir).glob("*/*/log.jsonl")):
            if log.parts[-3].startswith("."):
                continue
            with open(log) as f:
                for line in f:
                    dest.write(line)
                    n += 1
    return n


def _scene_program(noun):
    return Program(objects=[], constraints=[Constraint("F2a", (noun,), None, "scene_lightness", 1)])


def _degree_only(prompt):
    p = translate(prompt)
    return Program(objects=p.objects, constraints=[c for c in p.constraints if c.family == "F2a"])


def _neutral_jobs(suite, seeds, arms):
    jobs = []
    for arm, overrides in arms:
        for r in enumerate_suite(suite):
            noun = r.prompt.split()[-1]
            for seed in seeds:
                jobs.append(Job(arm, r.prompt, seed, r.group, suite, _scene_program(noun), overrides))
    return jobs


def m1_jobs(tune: bool) -> List[Job]:
    if tune:
        arms = [("unguided", {})]
        for eta0 in (0.02, 0.05, 0.1):
            for lo in (0.5, 0.25):
                for arm in COMMANDED:
                    arms.append((f"{arm}@eta{eta0}@lo{lo}", {"eta0": eta0, "sigma_lo": lo}))
        return _neutral_jobs("m1_tune", (0,), arms)
    jobs = _neutral_jobs("m1_eval", (0, 1), [("unguided", {}), ("darker", {}), ("brighter", {})])
    for arm in ("ladder_unguided", "ladder_guided"):
        for r in enumerate_suite("m1_ladder"):
            for seed in (0, 1):
                jobs.append(Job(arm, r.prompt, seed, r.group, "m1_ladder", _degree_only(r.prompt)))
    return jobs


def guide_cfg(s) -> GuideCfg:
    return GuideCfg(steps=s.get("cfgSteps", int), cfg=s.get("cfgCFG"), sigma_lo=s.get("cfgSigmaLo"),
                    eta0=s.get("cfgEtaZero"), delta=s.get("cfgDelta"), dmax=s.get("cfgMaxPert"),
                    eps={1: s.get("cfgEpsOne"), 2: s.get("cfgEpsTwo")}, backtrack=s.get("cfgBacktrack", int),
                    ramp=s.get("cfgRampSteps", int), g_min=s.get("g_min"), eps_g=s.get("eps_g"),
                    c_sigma=s.get("cfgCSigma"),
                    state_kw={"p_gate": s.get("cfgPMin"), "a_min": s.get("cfgAMin"), "psi_min": s.get("cfgSupportConfMin"),
                              "band": s.get("cfgDomBand"), "lam": s.get("cfgLambda")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=["m1_tune", "m1_eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N jobs")
    args = ap.parse_args()

    s = load_settings()
    cfg = guide_cfg(s)
    if args.tiny:
        backend = tiny_backend()
        cfg = dataclasses.replace(cfg, steps=2)
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, getattr(torch, args.dtype), s.get("cfgCFG"), args.resolution)
    jobs = m1_jobs(args.suite == "m1_tune")[:args.limit]
    deps = Deps(backend, ProvisionalCalibration(s), measure_scene, cfg, s, unload_text=not args.tiny)
    result = run_jobs(jobs, deps, args.out, resume_from=args.resume_from)
    n = merge_logs(args.out, Path(args.out) / "all.jsonl")
    print(json.dumps({"done": result.done, "skipped": result.skipped, "failed": result.failed,
                      "failures": result.failures[:20], "log_records": n}, indent=1))
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()
