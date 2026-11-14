"""Run experiment arms with per-image saving, so a Kaggle timeout costs at most the current image.

Each image is written into a temporary directory and published by one atomic rename to
<out>/<arm>/<image id>/ (image.png, meta.json, log.jsonl). A finished directory is never redone; a
failed one is retried on the next run; a temporary directory left by a dead session is discarded.

Stage-2 arms reuse the latent and attention traces the `stage1` arm saved for the same prompt and
seed, so every arm refines the same Stage-1 image from the same re-noised latent.

Usage:
  python -I code/run_experiment.py --suite m1_tune --out outputs/m1_tune
  python -I code/run_experiment.py --suite m1_eval --out outputs/m1 [--resume-from DIR] [--vae-device cuda:1]
  python -I code/run_experiment.py --suite mixture --group dev --n 20 --out outputs/sar
  python -I code/run_experiment.py --suite mixture --group dev --n 2 --tiny --out outputs/smoke
  python -I code/run_experiment.py --prompt "a slightly red mug and a blue teapot" --out outputs/mine
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
from calibration_fit import FittedCalibration  # noqa: E402
from constraints import DEGREE_FAMILIES, FAMILY_SCORERS, new_state  # noqa: E402
from estimators import ClipSegSupport, FakeSigLIP, FakeSupport, SigLIPEvidence  # noqa: E402
from grounding import missing_phrases, phrase_spans  # noqa: E402
from guidance import BINS, GuideCfg, Stage1Out, run_stage1, score_image  # noqa: E402
from measure import EstimatorDeps, Measure, measure_scene  # noqa: E402
from negative import negative_prompt  # noqa: E402
from predicates import Constraint, Program, translate  # noqa: E402
from prompts import PromptRow, base_prompt, degree_word, enumerate_suite, hedge_of, select, select_ladders  # noqa: E402
from reporting import LeakageError, StepLog  # noqa: E402
from sar import SarCfg, build_priors, run_stage2  # noqa: E402
from sd3_backend import load_backend, tiny_backend, weight_span  # noqa: E402
from settings import load_settings  # noqa: E402

COMMANDED = {"darker": -1, "brighter": +1}
TARGET_BAND = 0.02
UNGUIDED_ARMS = {"unguided", "ladder_unguided", "prompt_weight", "unguided_neg"}
STAGE2_MODES = {"stage1_plain": "plain", "stage1_sar": "sar", "stage1_placebo": "sar", "stage1_sar_nocheck": "sar",
                "stage1_sar_neg": "sar", "stage1_sar_conf": "sar", "stage1_sar_comp": "sar"}
# Stage 2 ablations: the SarCfg fields each arm changes (stage1_sar is the minimal form, both off)
SAR_ARMS = {"stage1_sar_nocheck": {"consistency": False}, "stage1_sar_conf": {"confidence": True},
            "stage1_sar_comp": {"competition": True}}
# Arms with the negative prompt derived from the program (negative.py). unguided_neg is the control:
# SD3 with the same negative prompt, so a gain of the method is not the negative prompt alone.
NEGATIVE_ARMS = {"unguided_neg", "stage1_neg", "stage1_sar_neg"}
STAGE1_OF = {"stage1_sar_neg": "stage1_neg"}        # Stage 2 arms start from this Stage 1 arm (default stage1)
STAGE1_ARMS = {"stage1", "stage1_neg"}
COMPARISON_ARMS = ("stage1", "stage1_plain", "stage1_sar", "stage1_placebo")
# Tab. ablation: each Stage-1 arm changes one component. Values are GuideCfg fields, StepState fields
# (state_kw), or the Measure ablation. "w/o gradient-validity gate" and "w/o noise-bin validity gates" have
# no arm: provisional calibration has neither gate (every family is valid in every bin), so both equal Stage 1.
ABLATION_ARMS = {
    "abl_point": {"state": {"target": "point"}},
    "abl_one_sided": {"state": {"target": "one_sided"}},
    "abl_attention_only": {"measure": "attention_only"},
    "abl_overlap": {"measure": "overlap"},
    "abl_cond_x0": {"cfg": {"cond_x0": True}},
    "abl_floored": {"state": {"loss": "floored"}},
    "abl_weighted_sum": {"cfg": {"priority": False}},
    "abl_no_acceptance": {"cfg": {"acceptance": False}},
    "abl_no_backtrack": {"cfg": {"backtrack": 0}},
    # hard domain gate: no cosine taper inside the domain (band 0) and no ramp after a gate opens (ramp 0)
    "abl_hard_domain": {"state": {"band": 0.0, "ramp": 0}},
    # w/o satisfaction gating: delta -> 0 makes eta = eta0 c(sigma) L / (L + delta) constant while L > 0
    "abl_no_satisfaction": {"cfg": {"delta": 1e-12}},
}
ABLATION_TABLE_ARMS = ("unguided", "stage1", *ABLATION_ARMS, "stage1_plain", "stage1_sar", "stage1_placebo",
                       "stage1_sar_conf", "stage1_sar_comp", "stage1_sar_nocheck")
# Tab. hedge ladder. prompt_weight is SD3 with the word the hedge modifies weighted by the hedge
# (A1111-style weights, linear in hedge rank); the hedge stays in the prompt.
LADDER_ARMS = ("unguided", "prompt_weight", "stage1", "stage1_sar")
LADDER_SUITES = ("ladder_attribute", "ladder_mixture", "ladder_size")
HEDGE_WEIGHTS = {"barely": 0.5, "slightly": 0.75, "moderately": 1.0, "very": 1.25, "extremely": 1.5}


class JobFailure(RuntimeError):
    """An expected reason a job cannot run; its message becomes the failure code."""


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
    measure: Optional[Callable]
    cfg: GuideCfg
    settings: object
    stage1: Callable = run_stage1
    unload_text: bool = False
    measure_factory: Optional[Callable] = None     # (prompt, cond) -> measure; needs phrase readout
    seg: object = None
    sar_cfg: Optional[SarCfg] = None
    jobs: List[Job] = field(default_factory=list)


@dataclass
class Summary:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    failures: List[str] = field(default_factory=list)


def manifest_no_degree(path) -> frozenset:
    """Degree families the frozen support manifest does not mark "degree-guided" (absent counts as not)."""
    if path is None:
        return frozenset()
    with open(path) as f:
        status = json.load(f)["status"]
    return frozenset(fam for fam in DEGREE_FAMILIES if status.get(fam, {}).get("status") != "degree-guided")


def check_test_run(group, calibration_tag, manifest) -> None:
    """Test images exist only after every selection decision is frozen (paper: App. no_leakage)."""
    if group != "test":
        return
    if calibration_tag != "fitted":
        raise LeakageError("test images need fully fitted calibration")
    if manifest is None or not Path(manifest).exists():
        raise LeakageError("test images need a frozen support manifest: run freeze_manifest.py first")


def peak_memory() -> Dict[str, float]:
    """Peak allocated GiB per GPU: the transformer and the estimators can sit on different ones."""
    if not torch.cuda.is_available():
        return {}
    return {f"cuda:{i}": torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(torch.cuda.device_count())}


def _reset_peaks():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(i)
            except RuntimeError:
                pass                        # a GPU this process has not used yet has no statistics


def image_id(prompt: str, seed: int) -> str:
    return f"{hashlib.sha256(prompt.encode()).hexdigest()[:12]}_s{seed}"


def commanded_targets(unguided_s, shift, margin, headroom):
    """Target interval for a commanded shift, or None when the direction has too little headroom
    (App. gradient-validity test: the target stays `margin` inside the scale)."""
    target = min(max(unguided_s + shift, margin), 1 - margin)
    if abs(target - unguided_s) < headroom:
        return None
    return (target - TARGET_BAND, target + TARGET_BAND)


def placebo_source(job: Job, jobs: List[Job]) -> Optional[Job]:
    """Next job of the same suite, group, seed, and object count with a different prompt."""
    same = [j for j in jobs if j.arm == job.arm and j.suite == job.suite and j.split == job.split
            and j.seed == job.seed and len(j.program.objects) == len(job.program.objects)]
    if job not in same:
        return None
    i = same.index(job)
    return next((j for j in same[i + 1:] + same[:i] if j.prompt != job.prompt), None)


def _is_finished(path: Path) -> bool:
    """Finished means complete metadata and, unless the job was skipped, its image."""
    try:
        with open(path / "meta.json") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    return "failed" not in meta and ("skipped" in meta or (path / "image.png").exists())


def _save_image(image, path):
    x = image[0].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((x * 255).round().astype(np.uint8)).save(path)


def _peak_gb():
    return max(peak_memory().values(), default=float("nan"))


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
        log.write(type="final", prompt=job.prompt, seed=job.seed, family=f["family"], cid=f["cid"], bin=BINS,
                  component=f["component"], tier=f["tier"], mu_unguided=unguided, mu_guided=guided, s=f["s"],
                  raw=f.get("raw"), quantity=f.get("quantity"))


def _negative(job: Job, deps: Deps) -> str:
    if job.arm not in NEGATIVE_ARMS:
        return ""
    return negative_prompt(job.program, deps.settings.get("hedge_centers", dict), deps.settings.get("default_center"))


def _cond(job: Job, deps: Deps):
    spans = phrase_spans(job.prompt, job.program) if deps.measure_factory else None
    return deps.backend.encode_prompt(job.prompt, spans, _negative(job, deps))


def _prompt_weighted(cond, prompt):
    """The degree word weighted by its hedge, e.g. "dark" in "a very dark mug" times 1.25."""
    hedge, word = hedge_of(prompt), degree_word(prompt)
    words = prompt.split(" ")
    start = len(" ".join(words[:words.index(hedge) + 1])) + 1
    weight = HEDGE_WEIGHTS[hedge]
    return weight_span(cond, (start, start + len(word)), weight), {"word": word, "weight": weight}


def _save_stage1(result, tmp):
    torch.save({"latent": result.latent.detach().cpu(), "traces": {k: v.cpu() for k, v in result.traces.items()},
                "supports": {k: v.cpu() for k, v in result.supports.items()}, "finals": result.finals,
                "presence": result.presence}, tmp / "stage1.pt")


def _load_stage1(out: Path, prompt: str, seed: int, backend, arm: str = "stage1") -> Stage1Out:
    path = out / arm / image_id(prompt, seed) / "stage1.pt"
    if not path.exists():
        raise JobFailure("missing_stage1")
    data = torch.load(path, weights_only=True)
    latent = data["latent"].to(backend.tf_device, backend.latent_dtype)
    with torch.no_grad():
        image = backend.decode(latent)
    return Stage1Out(latent, image, data["traces"], data["supports"], data["finals"], data["presence"])


def _placebo_priors(job, deps, out, cond):
    source = placebo_source(job, deps.jobs)
    if source is None:
        raise JobFailure("no_placebo_source")
    other = _load_stage1(out, source.prompt, source.seed, deps.backend)
    built = build_priors(other.traces, other.supports, deps.sar_cfg)
    priors = {}
    mine_keys = _object_keys(job.program.objects, cond.phrase_tokens)
    their_keys = _object_keys(source.program.objects, built)
    for mine, theirs in zip(mine_keys, their_keys):
        if theirs in built:
            priors[mine] = built[theirs]
    return priors, source.prompt


def _object_keys(objects, known):
    """Phrase key of each object in order: its name, or name#i when the object is repeated."""
    seen: Dict[str, int] = {}
    keys = []
    for head in objects:
        i = seen.get(head, 0)
        seen[head] = i + 1
        keys.append(head if head in known else f"{head}#{i}")
    return keys


def _run_stage2_arm(job, deps, out, tmp, meta, cond, measure, log):
    mode = STAGE2_MODES[job.arm]
    stage1 = _load_stage1(out, job.prompt, job.seed, deps.backend, STAGE1_OF.get(job.arm, "stage1"))
    priors = None
    sar = dataclasses.replace(deps.sar_cfg, **SAR_ARMS.get(job.arm, {}))
    if job.arm == "stage1_placebo":
        priors, meta["placebo_from"] = _placebo_priors(job, deps, out, cond)
    result = run_stage2(deps.backend, stage1, cond, job.program, job.prompt, deps.calib, deps.seg, sar, job.seed, log,
                        mode=mode, measure=measure, steps=deps.cfg.steps, priors=priors)
    with torch.no_grad():
        finals, _ = score_image(job.program, measure, deps.backend, deps.calib, result.image,
                                new_state(**deps.cfg.state_kw))
    meta.update(finals=finals, used_fallback=result.used_fallback, fallback_reason=result.fallback_reason,
                rollbacks=result.rollbacks, disabled=sorted(result.disabled),
                fallback_objects=sorted(result.fallback_objects), suspended_steps=result.suspended_steps,
                passes=result.passes, bias_steps=result.bias_steps, no_op_reason=result.no_op_reason,
                renoised_sha=hashlib.sha256(result.renoised.detach().cpu().numpy().tobytes()).hexdigest()[:16])
    return result.image, finals


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
            raise JobFailure(f"missing_{job.base_arm}")
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
    elif job.arm in UNGUIDED_ARMS:
        cfg = dataclasses.replace(cfg, guided=False)
    else:
        base = _read_meta(out, "ladder_unguided" if job.arm == "ladder_guided" else job.base_arm, job)
        base_finals = base["finals"] if base else None

    ablation = ABLATION_ARMS.get(job.arm, {})
    if ablation:
        cfg = dataclasses.replace(cfg, **ablation.get("cfg", {}),
                                  state_kw={**cfg.state_kw, **ablation.get("state", {})})
        meta["ablation"] = ablation
    cond = _cond(job, deps)
    if job.arm in NEGATIVE_ARMS:
        meta["negative_prompt"] = _negative(job, deps)
    if job.arm == "prompt_weight":
        cond, meta["prompt_weight"] = _prompt_weighted(cond, job.prompt)
    meta["unmapped"] = cond.unmapped + (missing_phrases(job.prompt, job.program) if deps.measure_factory else [])
    if deps.measure_factory and "measure" in ablation:
        measure = deps.measure_factory(job.prompt, cond, ablation["measure"])
    else:
        measure = deps.measure_factory(job.prompt, cond) if deps.measure_factory else deps.measure
    log = StepLog(str(tmp / "log.jsonl"), job.split, deps.calib.tag,
                  lambda: set(deps.settings.provisional_used) | set(getattr(deps.calib, "heuristic_used", ())))
    start = time.perf_counter()
    _reset_peaks()
    try:
        if job.arm in STAGE2_MODES:
            image, finals = _run_stage2_arm(job, deps, out, tmp, meta, cond, measure, log)
        else:
            result = deps.stage1(deps.backend, job.program, cond, job.seed, deps.calib, measure, cfg, log,
                                 targets=targets, prompt=job.prompt)
            image, finals = result.image, result.finals
            if job.arm in STAGE1_ARMS:
                _save_stage1(result, tmp)
        _final_records(job, finals, base_finals, log)
    finally:
        log.close()
    _save_image(image, tmp / "image.png")
    meta.update(finals=finals, time_s=time.perf_counter() - start, peak_gb=_peak_gb(), peak_by_gpu=peak_memory())
    if "ds" in meta:
        meta["dy"] = finals[0]["s"] - base_finals[0]["s"]
    return meta


def run_image(job: Job, deps: Deps, out: Path) -> dict:
    """Publish one image atomically. Out-of-memory and other errors become a failure record."""
    tmp = out / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    head = {"arm": job.arm, "prompt": job.prompt, "seed": job.seed}
    try:
        meta = _execute(job, deps, out, tmp)
    except torch.cuda.OutOfMemoryError as e:
        meta = {**head, "failed": "oom", "peak_gb": _peak_gb(), "peak_by_gpu": peak_memory(), "error": str(e)[:400],
                "traceback": traceback.format_exc()[-1500:]}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except JobFailure as e:
        meta = {**head, "failed": str(e)}
    except Exception as e:
        meta = {**head, "failed": type(e).__name__, "error": str(e), "traceback": traceback.format_exc()}
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
    deps.jobs = jobs
    summary = Summary()
    pending = []
    for job in jobs:
        rel = Path(job.arm) / image_id(job.prompt, job.seed)
        if resume_from is not None and not (out / rel).exists() and _is_finished(Path(resume_from) / rel):
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            tmp = out / f".tmp-{uuid.uuid4().hex}"
            shutil.copytree(Path(resume_from) / rel, tmp)
            os.replace(tmp, out / rel)
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
    """Concatenate per-image logs of every finished image; returns the number of records."""
    n = 0
    with open(into, "w") as dest:
        for log in sorted(Path(out_dir).glob("*/*/log.jsonl")):
            if log.parts[-3].startswith(".") or _failed(log.parent):
                continue
            with open(log) as f:
                for line in f:
                    dest.write(line)
                    n += 1
    return n


def _failed(path: Path) -> bool:
    try:
        with open(path / "meta.json") as f:
            return "failed" in json.load(f)
    except (OSError, ValueError):
        return True


def summarize_arms(out_dir) -> Dict[str, dict]:
    """Per arm: what happened, including how often SAR rolled back or fell back to Stage 1."""
    metas: Dict[str, List[dict]] = {}
    for path in Path(out_dir).glob("*/*/meta.json"):
        with open(path) as f:
            meta = json.load(f)
        metas.setdefault(meta.get("arm", path.parts[-3]), []).append(meta)

    def mean(rows, key):
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else 0.0

    out = {}
    for arm, rows in sorted(metas.items()):
        done = [r for r in rows if "failed" not in r]
        out[arm] = {"n": len(done), "failed": len(rows) - len(done),
                    "fallback_rate": mean([{"x": float(bool(r.get("used_fallback")))} for r in done], "x"),
                    "rollbacks": mean(done, "rollbacks"), "suspended_steps": mean(done, "suspended_steps"),
                    "disabled": mean([{"x": len(r.get("disabled", []))} for r in done], "x"),
                    "no_bias_share": mean([{"x": float(r["bias_steps"] == 0)} for r in done if "bias_steps" in r], "x"),
                    "passes": mean(done, "passes"), "time_s": mean(done, "time_s"), "peak_gb": mean(done, "peak_gb")}
    return out


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
        # At the noisiest steps a push decides what the image shows, not only how bright it is, so
        # guidance starts later (Kaggle: starting at sigma 1 turned a bakery into a logo).
        for eta0 in (0.05, 0.02, 0.1):
            for hi in (0.75, 0.5):
                for arm in COMMANDED:
                    arms.append((f"{arm}@eta{eta0}@hi{hi}", {"eta0": eta0, "sigma_hi": hi}))
        for arm in COMMANDED:
            arms.append((f"{arm}@eta0.1@hi0.75@max0.4", {"eta0": 0.1, "sigma_hi": 0.75, "dmax": 0.4}))
        return _neutral_jobs("m1_tune", (0,), arms)
    jobs = _neutral_jobs("m1_eval", (0, 1), [("unguided", {}), ("darker", {}), ("brighter", {})])
    for arm in ("ladder_unguided", "ladder_guided"):
        for r in enumerate_suite("m1_ladder"):
            for seed in (0, 1):
                jobs.append(Job(arm, r.prompt, seed, r.group, "m1_ladder", _degree_only(r.prompt)))
    return jobs


def only_prompts(jobs: List[Job], prompts) -> List[Job]:
    """The jobs for exactly these prompts, in their original order."""
    missing = set(prompts) - {j.prompt for j in jobs}
    if missing:
        raise ValueError(f"not in this suite: {sorted(missing)}")
    return [j for j in jobs if j.prompt in prompts]


def only_settings(jobs: List[Job], settings) -> List[Job]:
    """The unguided jobs and those tuned with exactly these settings, such as "eta0.1@hi0.75"."""
    def setting(job):
        return job.arm.split("@", 1)[1] if "@" in job.arm else None
    missing = set(settings) - {setting(j) for j in jobs}
    if missing:
        raise ValueError(f"not in this suite: {sorted(missing)}")
    return [j for j in jobs if j.arm == "unguided" or setting(j) in settings]


def with_overrides(jobs: List[Job], **overrides) -> List[Job]:
    """The same jobs with these guidance settings; None leaves a setting alone."""
    chosen = {k: v for k, v in overrides.items() if v is not None}
    return [dataclasses.replace(j, overrides={**j.overrides, **chosen}) for j in jobs] if chosen else jobs


def arm_jobs(rows, seeds, arms) -> List[Job]:
    """Arms in the given order, so Stage 1 of every prompt finishes before any Stage-2 arm starts."""
    jobs = []
    for arm in arms:
        for r in rows:
            for seed in seeds:
                base = "unguided"
                jobs.append(Job(arm, r.prompt, seed, r.group, r.suite, r.program, base_arm=base))
    return jobs


def prompt_rows(prompts) -> List[PromptRow]:
    """Your own prompts: development data, never part of any calibration or test group."""
    return [PromptRow(p, base_prompt(p), "dev", "custom") for p in prompts]


def describe_program(prompt: str) -> str:
    """What the pipeline understood from a prompt, and what it will not guide."""
    program = translate(prompt)
    lines = [f"Prompt: {prompt}", f"Objects: {', '.join(program.objects) or 'none found'}"]
    if program.degraded:
        lines.append("Outside the grammar: the pipeline only checks that objects appear.")
    not_guided = list(program.unsupported)
    for c in program.constraints:
        text = f"  {c.family} {c.args}" + (f" hedge={c.hedge}" if c.hedge else "") + (f" {c.quantity}" if c.quantity else "")
        if c.family in FAMILY_SCORERS:
            lines.append(text)
        else:
            not_guided.append(text.strip())
    if not_guided:
        lines.append("Recognized but not guided: " + "; ".join(map(str, not_guided)))
    return "\n".join(lines)


def guide_cfg(s) -> GuideCfg:
    return GuideCfg(steps=s.get("cfgSteps", int), cfg=s.get("cfgCFG"), sigma_lo=s.get("cfgSigmaLo"), sigma_hi=s.get("cfgSigmaHi"),
                    eta0=s.get("cfgEtaZero"), delta=s.get("cfgDelta"), dmax=s.get("cfgMaxPert"),
                    eps={1: s.get("cfgEpsOne"), 2: s.get("cfgEpsTwo")}, backtrack=s.get("cfgBacktrack", int),
                    ramp=s.get("cfgRampSteps", int), g_min=s.get("g_min"), eps_g=s.get("eps_g"),
                    c_sigma=s.get("cfgCSigma"),
                    state_kw={"p_gate": s.get("cfgPMin"), "a_min": s.get("cfgAMin"), "psi_min": s.get("cfgSupportConfMin"),
                              "band": s.get("cfgDomBand"), "lam": s.get("cfgLambda")})


def sar_cfg(s) -> SarCfg:
    return SarCfg(strength=s.get("cfgStrength"), k=s.get("cfgKSteps", int), alpha=s.get("cfgAlpha"), beta=s.get("cfgBeta"),
                  temp=s.get("cfgTemp"), gamma_min=s.get("cfgGammaMin"), overlap=s.get("cfgOverlap"), nest=s.get("cfgNest"))


def _tiny_estimators():
    def mask(phrase, x):
        return torch.nn.functional.adaptive_avg_pool2d(x[None], 64)[0].mean(0)

    return FakeSupport(mask), FakeSigLIP(lambda phrase, image: image.mean() * 2 + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["m1_tune", "m1_eval", "plain_color", "mixture", "size", "soft", *LADDER_SUITES])
    ap.add_argument("--prompt", action="append", help="your own prompt instead of a suite; can be repeated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="dev")
    ap.add_argument("--n", type=int, default=20, help="prompts per suite (suites other than M1); base prompts for ladders")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default="unguided," + ",".join(COMPARISON_ARMS),
                    help='comma-separated arms, "ablation" for every arm of Tab. ablation, or "ladder" for Tab. hedge ladder')
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3-medium-diffusers")
    ap.add_argument("--tf-device", default="cuda:0")
    ap.add_argument("--vae-device", default=None)
    ap.add_argument("--resolution", type=int, default=1024)
    # bfloat16 needs the full attention matrix on a T4: one attention step exceeds 14.5 GiB (measured).
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    # With CLIPSeg and SigLIP on its GPU, a float32 VAE ran out of memory on a T4 (measured).
    ap.add_argument("--vae-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--estimator-dtype", default="float32", choices=["float16", "float32"],
                    help="SigLIP and CLIPSeg weights; the completed runs used float32")
    ap.add_argument("--tiny", action="store_true", help="smoke test on the tiny random model (CPU)")
    ap.add_argument("--only-prompt", action="append", help="run only this prompt from the suite; can be repeated")
    ap.add_argument("--only-setting", action="append", help="run only this tuning setting and unguided; can be repeated")
    ap.add_argument("--prompts-file", default=None, help="JSON list of prompts (as screen.py writes) instead of a suite")
    ap.add_argument("--dmax", type=float, default=None, help="largest change to one latent value per step, for trying other limits")
    ap.add_argument("--eta0", type=float, default=None, help="base guidance step size, for the sweep")
    ap.add_argument("--sigma-hi", type=float, default=None, help="noise level where guidance starts (1.0 = from the first step), for the sweep")
    ap.add_argument("--clip", default=None, choices=["peak", "rms"], help="dmax limits the largest latent change or its RMS")
    ap.add_argument("--steps", type=int, default=None, help="sampling steps of Stage 1 and unguided images, for comparing step counts")
    ap.add_argument("--limit", type=int, default=None, help="only the first N jobs")
    ap.add_argument("--calibration", default=None, help="fitted calibration file; provisional values if omitted")
    ap.add_argument("--manifest", default=None, help="frozen support manifest; required for --group test")
    ap.add_argument("--log-all-steps", action="store_true", help="log raw values at every step (calibration runs)")
    args = ap.parse_args()
    if args.prompts_file:
        with open(args.prompts_file) as f:
            args.prompt = (args.prompt or []) + json.load(f)
    if bool(args.suite) == bool(args.prompt):
        ap.error("give either --suite or --prompt / --prompts-file")
    if args.prompt and args.arms == ap.get_default("arms"):
        args.arms = "unguided,stage1,stage1_sar"    # a placebo needs another image, so it is left out here
    for prompt in args.prompt or []:
        print(describe_program(prompt))

    s = load_settings()
    cfg, sar = guide_cfg(s), sar_cfg(s)
    calib = FittedCalibration.load(args.calibration) if args.calibration else ProvisionalCalibration(s)
    check_test_run(args.group, calib.tag, args.manifest)
    cfg = dataclasses.replace(cfg, log_all=args.log_all_steps,
                              state_kw={**cfg.state_kw, "no_degree": manifest_no_degree(args.manifest)})
    if args.tiny:
        backend = tiny_backend()
        cfg = dataclasses.replace(cfg, steps=2)
        sar = dataclasses.replace(sar, k=1)
    else:
        backend = load_backend(args.model, s.get("cfgReadBlocks", list), s.get("cfgSARBlocks", list), args.tf_device,
                               args.vae_device, getattr(torch, args.dtype), s.get("cfgCFG"), args.resolution, getattr(torch, args.vae_dtype))
    if args.suite and args.suite.startswith("m1"):
        jobs = m1_jobs(args.suite == "m1_tune")
        deps = Deps(backend, calib, measure_scene, cfg, s, unload_text=not args.tiny)
    else:
        if args.prompt:
            rows = prompt_rows(args.prompt)
        elif args.suite in LADDER_SUITES:
            rows = select_ladders(args.suite, args.group, args.n)
        else:
            rows = select(args.suite, args.group, args.n)
        arms = {"ablation": list(ABLATION_TABLE_ARMS), "ladder": list(LADDER_ARMS)}.get(args.arms, args.arms.split(","))
        jobs = arm_jobs(rows, [int(x) for x in args.seeds.split(",")], arms)
        if args.tiny:
            seg, siglip = _tiny_estimators()
        else:
            device = args.vae_device or args.tf_device
            est_dtype = getattr(torch, args.estimator_dtype)
            seg = ClipSegSupport.from_pretrained(s.get("cfgCLIPSeg", str), device, est_dtype)
            siglip = SigLIPEvidence.from_pretrained(s.get("cfgSigLIP", str), device, est_dtype)
        estimators = EstimatorDeps(seg, siglip, calib, s)
        deps = Deps(backend, calib, None, cfg, s, unload_text=not args.tiny, seg=seg, sar_cfg=sar,
                    measure_factory=lambda prompt, cond, ablation="": Measure(estimators, prompt, cond, ablation))
    if args.only_prompt:
        jobs = only_prompts(jobs, args.only_prompt)
    if args.only_setting:
        jobs = only_settings(jobs, args.only_setting)
    jobs = with_overrides(jobs, dmax=args.dmax, steps=args.steps, eta0=args.eta0, sigma_hi=args.sigma_hi,
                          clip=args.clip)
    result = run_jobs(jobs[:args.limit], deps, args.out, resume_from=args.resume_from)
    n = merge_logs(args.out, Path(args.out) / "all.jsonl")
    print(json.dumps({"done": result.done, "skipped": result.skipped, "failed": result.failed,
                      "failures": result.failures[:20], "log_records": n, "arms": summarize_arms(args.out)}, indent=1))
    sys.exit(1 if result.failed else 0)


if __name__ == "__main__":
    main()
