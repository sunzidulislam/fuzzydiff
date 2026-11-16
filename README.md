# FuzzyDiff

Code for the CVPR submission *FuzzyDiff: hedged fuzzy attributes in text-to-image diffusion*.

Prompts such as "a slightly faded sofa", "a bluish-purple plate" or "a kettle somewhat bigger than a shoe" ask for a
*degree* of an attribute, which current text-to-image models mostly ignore. FuzzyDiff turns such a prompt into
a small program of fuzzy constraints and steers Stable Diffusion 3 (Medium) toward it at sampling time, with no
fine-tuning:

1. **Prompt → predicates.** A rule-based parser maps each hedged phrase to a constraint family (visibility,
   photometric attribute, open-vocabulary attribute, whiteness, colour mixture, relative size, position) with a
   target interval for the hedge.
2. **Stage 1: latent guidance.** At each guided step the clean prediction is decoded, measured with frozen
   estimators (CLIPSeg supports, SigLIP evidence, analytic CIELAB/OKLab statistics) and pulled toward the
   constraint memberships through a Charbonnier loss. Constraints are gated by presence, support and estimator
   validity, and guidance only runs in the noise window where the estimators are reliable.
3. **Stage 2: Spatial Attention Refiner (SAR).** A short re-noise and resample of the Stage-1 image with an
   additive bias on the image→text joint attention logits, which keeps each attribute on its own object.

Everything the paper reports is produced by the scripts in `fuzzydiff/experiments` and `fuzzydiff/evaluation`;
each one names the table it fills in its docstring.

## Installation

Python 3.10 or newer. A GPU is needed for SD3-Medium (we used 2× T4 16 GB); the unit tests and the `--tiny`
smoke runs work on a CPU.

```bash
git clone <this repository> fuzzydiff && cd fuzzydiff
pip install -r requirements.txt
pip install -e . --no-deps
```

SD3-Medium is a gated model: accept its licence on Hugging Face and log in (`huggingface-cli login`, or set
`HF_TOKEN`) before the first real run. CLIPSeg (`CIDAS/clipseg-rd64-refined`) and SigLIP
(`google/siglip-so400m-patch14-384`) are downloaded on first use.

## Quick start

A smoke test on a tiny random model (CPU, under a minute, no downloads):

```bash
python -m fuzzydiff.experiments.run_experiment --suite mixture --group dev --n 2 --tiny --out outputs/smoke
```

Your own prompt with the full model: SD3 alone, Stage 1, and Stage 1 + SAR:

```bash
python -m fuzzydiff.experiments.run_experiment \
    --prompt "a bluish-purple plate and a coral-orange book" \
    --arms unguided,stage1,stage1_sar --seeds 0 --out outputs/mine \
    --tf-device cuda:0 --vae-device cuda:1
```

Each image is written to `outputs/mine/<arm>/<image id>/` together with `meta.json` (the parsed program, final
memberships) and `log.jsonl` (per-step measurements and gates). For relative size we used a wider guidance window,
`--sigma-hi 0.9 --clip rms --dmax 0.02 --eta0 0.04`.

To see how a prompt is parsed:

```python
from fuzzydiff.language.predicates import translate
print(translate("a kettle slightly bigger than a shoe"))
```

## Repository layout

```
fuzzydiff/
  language/      prompt -> constraint program (predicates.py), prompt suites and data splits (prompts.py)
  models/        SD3 backend, custom joint-attention processor, frozen estimators, colour statistics
  guidance/      grounding in joint attention, constraints and memberships, Stage 1 (guidance.py), Stage 2 (sar.py)
  calibration/   provisional and fitted calibration, fitting from annotations
  evaluation/    statistics and table rows, blind rating pages, automatic metrics, standard benchmarks
  experiments/   one script per experiment or analysis table; run_experiment.py generates all images
  checks/        processor checks against a materialized-softmax reference, on random and real weights
  config/        settings loader; configs/fuzzydiff_sd3.yaml is the single source of every reported value
baselines/       Attend-and-Excite, SPO, Concept Sliders and plain SD 1.4, generated on the same prompts and seeds
configs/         settings, provisional thresholds, vocabulary
notebooks/       the Kaggle notebook we ran the experiments with (2x T4)
results/         small result files the paper reads (processor checks, schedule, translation challenge, pilots)
tests/           unit and regression tests
```

## Reproducing the paper

All settings come from `configs/fuzzydiff_sd3.yaml`. `python -m fuzzydiff.config.make_config --status` lists the
values that are still marked *proposed* rather than measured.

| What | Command |
| --- | --- |
| Images for every arm and suite | `python -m fuzzydiff.experiments.run_experiment --suite {soft,mixture,size,...}` |
| Processor checks (random weights / real checkpoint) | `python tests/test_processor.py`, `python -m fuzzydiff.checks.arch_check`, `python -m fuzzydiff.checks.verify_checkpoint` |
| Noise schedule and guided window | `python -m fuzzydiff.guidance.activation_schedule` |
| Prompt translation challenge set | `python -m fuzzydiff.experiments.translation_challenge` |
| Attention validation (pointing, binding AUROC) | `python -m fuzzydiff.experiments.attention_probe` |
| Estimator validity per noise bin | `python -m fuzzydiff.experiments.calibration_pilot` |
| Calibration fit from annotations | `python -m fuzzydiff.calibration.fit_calibration` |
| Hedge intervals and their context invariance | `python -m fuzzydiff.experiments.acceptance`, `... context_invariance` |
| Presence evidence | `python -m fuzzydiff.experiments.presence_eval` |
| Gradient validity (single family) | `python -m fuzzydiff.experiments.gradient_validity` |
| Activation and transfer in full sampling | `python -m fuzzydiff.experiments.activation_sys`, `... transfer_sys`, `... shift_sys` |
| Support manifest (frozen before any test image) | `python -m fuzzydiff.evaluation.freeze_manifest` |
| Hedge ladders | `python -m fuzzydiff.evaluation.ladder` |
| Paired comparison with baselines | `python baselines/generate_paired.py`, then `python -m fuzzydiff.evaluation.paired_table` |
| Refinement-matched comparison (+R) | `python -m fuzzydiff.experiments.refine_arms`, `python -m fuzzydiff.evaluation.refined_table` |
| Stage-2 consistency | `python -m fuzzydiff.experiments.stage2_consistency` |
| Memory and runtime | `python -m fuzzydiff.experiments.memory_probe` |
| Automatic metrics, GenEval, T2I-CompBench++ | `python -m fuzzydiff.evaluation.auto_metrics`, `python -m fuzzydiff.evaluation.benchmarks` |

Every script has `--help`. The notebook in `notebooks/` runs them in order on Kaggle; each section says which
table it fills.

### Data splits and the test set

Prompts are assigned to calibration, intervention, system, consistency, development and test groups by a
salted hash of the hedge-free prompt (`fuzzydiff/language/prompts.py`), so every hedge of a base prompt lands in
the same group. `run_experiment.py` refuses to generate test-group images until a fitted calibration and a frozen
support manifest are passed, and every selection function in `fuzzydiff/evaluation/reporting.py` raises on test
records.

### Human ratings

`fuzzydiff/evaluation/annotate.py` builds self-contained blind rating pages (method order shuffled per item, key
stored separately). Ratings come back as JSON and are read by the table scripts. Files under `results/ai_pilot/`
are pilot labels used to test the pipeline end to end. They are not human annotations and are not reported as such.

## Tests

```bash
pip install pytest
pytest
```

The suite uses tiny random models and synthetic images, so it runs on a CPU without any downloads.

## Licence

Code: MIT (see `LICENSE`). Model weights are under their own licences (SD3-Medium: Stability AI Community Licence; CLIPSeg,
SigLIP, OWLv2 and the baselines as published by their authors).
