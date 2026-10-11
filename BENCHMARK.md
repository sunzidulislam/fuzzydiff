# Prompt-based whole-model benchmark

`run_benchmark.py` runs the five archived analysis implementations and full
FuzzyDiff in isolated, sequential GPU processes. It reads notebook definitions;
it does not edit notebooks or execute their saved batch cells. The main table has
one row per method. Base generation and refinement are combined.

This is an exploratory benchmark, not a validation of the archived baselines.
In particular, `attend-f3.ipynb` detaches the attention used for guidance. The
runner reports that limitation rather than repairing or relabeling the result.
The predicate implementation also needs validation before use as a published
baseline. Timing a broken or inactive method is not a fair method comparison.

## Kaggle command

Copy the three new Python files (`run_benchmark.py`, `benchmark_models.py`,
`benchmark_timing.py`) into the repository on Kaggle. Keep `run_sd.py`, the
canonical notebook, and the five analysis notebooks there, including
`pipeline_fuzzy/analysis/enhancing-f3 (1).ipynb`.
Use the repository's existing requirements and a GPU with Internet enabled for
the initial cache preparation. Run from the repository directory:

```python
import subprocess
import sys

prompt = "A red ball beside a blue ball."
words = "red ball,blue ball"

subprocess.run([
    sys.executable, "run_benchmark.py",
    "--prompt", prompt, "--words", words,
    "--seeds", "1024,999,42,142",
    "--steps", "50", "--size", "512", "--guidance", "7.5",
    "--warmup", "1", "--repeats", "3",
    "--sharpness", "20", "--tnorm", "product", "--lr", "5",
    "--predicate-strength", "0.5",
    "--prepare-cache",
    "--output", "/kaggle/working/outputs/balls_whole_model_benchmark",
], check=True)
```

For a quick pilot use `--seeds 42 --repeats 1`. Append `--dry-run` to validate the
plan without loading models, downloading files, or writing results. Use a new
output directory for every experiment; existing nonempty directories are refused.

Preparation loads checkpoints in separate processes with downloads enabled, then
exits. Measurements start in fresh processes with offline loading enforced.
Preparation duration is recorded separately and is not included in model-load
time. If everything is already cached, omit `--prepare-cache`. A missing cache
then produces an explicit load failure rather than silently downloading weights.

Six methods, four seeds, three repetitions and one warmup means 72 saved images
and 150 complete inference invocations: 72 ordinary passes, 72 profiling passes,
and six warmups. Repetitions with the same seed are timing replicates, not
additional independent quality samples. Two-stage invocations include both stages.

The `enhancing` method is the archived SDXL-Turbo notebook, with no refiner or
fuzzy objective. It always uses its native 4 steps and CFG=0; these effective
settings are recorded in metadata. Its timing is not a comparison at matched
step counts. Optional xFormers is omitted; standard attention and VAE slicing/
tiling are used. Run its pilot separately with:

```python
subprocess.run([
    sys.executable, "run_benchmark.py",
    "--prompt", "A red ball beside a blue ball.",
    "--words", "red ball,blue ball", "--methods", "enhancing",
    "--seeds", "142", "--repeats", "1",
    "--steps", "4", "--size", "512", "--guidance", "0",
    "--prepare-cache",
    "--output", "/kaggle/working/outputs/enhancing_pilot",
], check=True)
```

## Outputs

- `comparison.md`: whole-model table; no separate base/refiner rows and no CLIP scores.
- `timings.csv`: per-seed/per-repetition timings and failure status.
- `results.json`: full records, summaries, environment and implementation limitations.
- `config.json`, `source_hashes.json`: requested configuration and code fingerprints.
- `<method>/model.json`: combined cached loading time, component loading details,
  effective precision/schedulers, checkpoint IDs and available component revisions.
- `<method>/seed_<seed>_repeat_<n>.png`: ordinary-pass generated images.
- `<method>/benchmark.log`, `runs.json`: execution details and incremental results.
- `<method>/prepare.log`: separate cache-preparation log when requested.

Workers continue after another method fails. The final process exits nonzero if
preparation, inference, or profiling fails; inspect the saved reports and logs.
Missing measurements are N/A, never zero-valued successful results.

## What the numbers mean

`load_s` is one cached loading observation for all components plus device/offload
configuration. It excludes dependency imports, notebook definition loading,
downloads, warmup and generation. Filesystem caches can be warm. Repeat the
entire experiment in fresh processes if reporting loading variability.

`inference_s` is synchronized wall-clock time for an ordinary call from prompt
input to the complete final image in memory. For FuzzyDiff this includes guided
base generation **and** predicate refinement. It includes tokenization, text
encoding, VAE operations, transfers and inherent algorithm overhead. Saving,
image display and CLIP evaluation are excluded. CUDA synchronization brackets
the call. Model weights remain loaded across seeds. Attention processors are
reset and models offloaded before each sample, outside the timer. Offload hook
resets can also empty the device cache before each sample; this benchmark does
not represent uninterrupted serving with a persistent GPU allocator cache.

The separate profiling call wraps tokenizer methods, text-encoder forwards, VAE
encode/decode, and application preprocessing. Exclusive nested accounting avoids
double-counting tokenizer calls inside phrase alignment or preprocessing. Here:

- preprocessing means configuration, phrase alignment, controller setup and
  predicate-mask preparation, excluding measured tokenization;
- tokenization includes positive/negative text and tracked phrase encoding;
- text encoding includes the text encoders and their offload transfers;
- generation is the remaining inference work **including VAE work and both
  generation stages**; it is not a pure UNet-kernel measurement.

These four profiled components sum to profiled time, not ordinary `inference_s`.
Synchronizing individual components adds overhead and can perturb performance,
especially millisecond-scale measurements. Use ordinary total inference for the
main efficiency claim. The report records profiled UNet forward calls and whether
the profiled and ordinary passes produced identical pixels; a difference needs
investigation before trusting the stage breakdown. Gradients stay enabled where
the original method needs them.

GPU memory is peak allocated/reserved memory from the ordinary inference pass,
including any already-resident tensors. It is not peak loading memory or host RAM.
There is also explicit cache eviction after warmup. Summary memory is the maximum
observed, while latency is mean, sample SD,
and median across successful prompt/seed/repetition runs. Profile failures are
reported separately and do not erase a successful ordinary image.

## Experimental interpretation

The runner applies common size, step-count and CFG arguments, and model CPU
offload, to every adapter. This is a documented benchmark configuration rather
than a byte-for-byte replay of the archived batch scripts:

- `stable`: SD 1.4, FP16, checkpoint scheduler and attention/VAE slicing.
- `attend`: SD 1.5, FP16, DDIM; archived guidance remains disconnected. The
  maintained SD 1.5 checkpoint repository replaces the historical repository ID.
- `predicate`: SD 1.4, FP32. Targets use the final token of each tracked phrase;
  `--bind "attribute>object"` adds correspondence pairs. No leakage or possession
  constraints are automatically inferred. Check the recorded token mapping.
- `aesthetic`: SDXL base + refiner, FP16, a 0.4 denoising split. The refiner uses
  its own copy of the same fixed VAE checkpoint to avoid shared offload hooks.
- `fuzzydiff`: canonical SDXL guidance + predicate refinement, FP16, default
  30 base guidance updates, 30 requested refinement steps with strength 0.4.

The default negative prompt is empty because the archived implementations do
not consistently apply negative prompts. A nonempty `--negative` preserves and
reports those archived limitations; it does not make them equivalent.

512-square output equalizes pixel count but is not SDXL's native resolution.
The archived attend/predicate collectors require actual 16-square attention maps;
when those methods are selected, the runner accepts only 128, 256, 512 or 1024
square images. It rejects sizes such as 768 before loading models.
Different architectures, precision, scheduler calls, extra forward/backward
passes and refinement schedules still imply different computational workloads.
This table compares configured complete systems. It cannot isolate fuzzy
guidance overhead; use a separate matched-backbone ablation for that claim.

CPU tests validate accounting, definition loading, failure reporting and a real
random-weight diffusion profiling path. They do not establish GPU performance,
visual quality or correctness of the archived algorithms. No quality ranking is
manufactured by this benchmark.
