# FuzzyDiff SD3 attention processor

* `fuzzydiff_sd3_attention.py` – drop-in replacement for diffusers' `JointAttnProcessor2_0`
  with (a) chunked log-sum-exp readout of image→text joint attention (Stage 1) and
  (b) the SAR additive logit bias (Stage 2).
* `test_processor.py` – unit tests on a reduced random SD3 transformer (float64, CPU):
  stock equivalence, readout, phrase text-share, and absolute text mass vs. materialized softmax (values, gradients),
  SAR equivalence, finite-difference gradcheck. Writes `processor_check.json`.
* `verify_checkpoint.py` – the same checks on the real checkpoint (GPU), plus T5 token indexing,
  pipeline image identity, and peak memory / runtime (the SAR step is timed as sequenced in
  App. B: SAR-free CFG pass + VAE decode + conditional-only pass with the bias; CLIPSeg excluded). Writes `checkpoint_check.json` and
  `memory_check.tex`. `--tiny` runs all code paths on a random model (no weights needed).
* `arch_check.py` – the same processor checks with random weights at the exact SD3-Medium layer
  dimensions (24 heads x 64, joint dim. 4096, 77+256 text tokens, 2 blocks, last block
  `context_pre_only`), 256^2, float64, CPU (~30 s, <6 GB RAM). Writes `arch_check.json`.
* `table_rows.py` – builds `processor_check_rows.tex` (read by the paper) from the JSON files.
* `make_config.py` – generates `../config.tex` from `../configs/fuzzydiff_sd3.yaml`, the single
  source of all settings (the experiment code should load the same YAML via `make_config.load()`);
  `--status` lists settings that are still proposed defaults.
* `activation_schedule.py` – sigma schedule of the sampler, noise bins, and guided window
  (paper: activation table); runs offline from the diffusers scheduler.
* `reporting.py` – log schema and statistics for the result tables; selection functions raise
  `LeakageError` on test records. `test_reporting.py` checks it on synthetic logs.
* `freeze_manifest.py` – freezes which families are degree-guided (non-test data only) into
  `support_manifest.json` with SHA-256; writes `support_rows.tex` (status column).
* `make_tables.py` – run logs -> `results_rows.tex` (activation, transfer, tiers, shift, intervals).
* `color_stats.py`, `test_color_unimodality.py` – OKLab mixture statistics and pattern index;
  synthetic control (blends / two-tone / mottled) -> `color_synthetic_rows.tex`.
* `translation_challenge.py` – author-written challenge set with gold programs -> `translation_rows.tex`
  (`translation_challenge_firstrun.json`: results before the bugs it exposed were fixed).
* `predicates.py`, `test_predicates.py` – reference implementation of the prompt-to-predicate
  translation (constraint program, hedge lexicon, template grammar) and its tests.

```
pip install torch diffusers==0.31.0
python -I test_processor.py
```

Verified environment: diffusers 0.31.0, torch 2.15.0 (CPU, float64).

Facts checked in the diffusers 0.31.0 source and used in the paper:
* joint attention concatenates [image, text] queries/keys/values and splits the output in that order;
* the SD3 pipeline concatenates CLIP-L‖CLIP-G (77 tokens, zero-padded to the T5 width) before T5
  along the token axis (`max_sequence_length` default 256), so T5 token j is joint key N_I + 77 + j;
* pipeline defaults: 28 steps, guidance 7.0; decoding uses `latents / scaling_factor + shift_factor`;
* FlowMatchEulerDiscreteScheduler step: `prev = sample + (sigma_next - sigma) * model_output`.

To fill the "Checkpoint" column of the processor table, the measured memory rows, and the
checkpoint-level settings (`backbone_config.tex`, read by `config.tex`):

```
python -I verify_checkpoint.py --model stabilityai/stable-diffusion-3-medium-diffusers
# then recompile the paper
```
Needs a GPU with enough memory for one guided step at 1024^2 (gradient checkpointing is enabled)
and access to the gated SD3 weights.

## Pipeline

* `settings.py` - typed settings: `configs/fuzzydiff_sd3.yaml`, with placeholders filled from `configs/provisional.yaml`.
* `prompts.py` - suites from `configs/vocab.yaml`; a hash of the hedge-free base prompt decides the group.
* `sd3_backend.py` - model access: text, guided velocity, clean prediction, VAE. `tiny_backend()` is the CPU test double.
* `grounding.py` - phrase tokens, localization fields, confidence statistics, contrastive binding.
* `estimators.py`, `measure.py` - frozen estimators (analytic colour, CLIPSeg, SigLIP) and the evidence for one latent.
* `calibration.py` - the interface the loop uses, with provisional values.
* `calibration_fit.py`, `fit_calibration.py` - fit maps, presence, validity, hedges, size, and thresholds; `FittedCalibration`.
* `constraints.py` - memberships, Charbonnier losses, gates, the constraint set frozen per step.
* `guidance.py` - Stage 1: tier projection, bounded step, acceptance. `sar.py` - Stage 2: priors, drift checks, rollback, fallback.
* `run_experiment.py` - arms, per-image saving, resume. `annotate.py` - blind pages, annotation export and import.
* `../notebooks/fuzzydiff_kaggle.ipynb` - thin Kaggle driver: verify, M1, then the SAR comparison.

Tests run on CPU with no downloads: `for f in code/test_*.py; do python -I "$f"; done`.
Records from provisional calibration show that the pipeline runs; `reporting.py` refuses them for paper tables.

## Backbone scope

The method in the paper is backbone-agnostic (any flow-matching MM-DiT with joint attention).
The code here is the **SD3 adapter** used in the experiments: `FuzzyJointAttnProcessor` replaces
diffusers' `JointAttnProcessor2_0` and assumes SD3's layout (image tokens before text tokens,
CLIP-L||CLIP-G (77) before T5, classifier-free guidance with a conditional and an unconditional branch).
Porting to another MM-DiT needs a processor for each of its joint-attention block types, its own
segment order and tokenizer offsets, and, for guidance-distilled models, a single conditional
evaluation in place of the CFG pair (see App. C, "Other MM-DiT backbones").

See ../RUNNING.md for the order in which the GPU runs fill every table.
