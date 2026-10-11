# FuzzyDiff repository restructuring design

Status: architecture and written specification approved in conversation on 2026-10-12.
Baseline: `6f23ae4`, inspected on 2026-10-12.

## Purpose and success criteria

The user wants a maintainable research repository instead of a collection of
root-level scripts. The approved structure separates model adapters, ordinary
image generation, experimental orchestration, reports, and command entry points.
It must support the existing Kaggle workflows and the recent single-image usage.

Success means that implementation ownership is clear, ordinary generation does
not require a benchmark run, the six benchmark methods remain available, and
existing commands, imports, output naming, and experimental safeguards continue
to work. A directory rearrangement alone is insufficient: the existing model
loading and generation branches must become reusable adapters.

This refactor does not change the FuzzyDiff method or improve generated images.

## Constraints

1. `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb` remains the only implementation
   of fuzzy membership, loss, attention control, and refinement mathematics.
   Its cells and their first definitions are not moved or renamed.
2. Archived notebooks in `pipeline_fuzzy/analysis/` remain byte-identical.
3. No new runtime dependencies, pretrained downloads, or local GPU experiments.
4. Existing root commands remain usable from a repository checkout on Kaggle.
5. Existing research limitations remain visible. SDXL base/refiner is not SPO;
   SDXL-Turbo is not C3; archived Attend is not a validated reproduction.
6. CPU verification uses random-weight models. Its results cannot establish
   pretrained image quality or GPU performance.
7. `REVIEW.md` is append-only. Generated outputs and user assets are not moved,
   deleted, rewritten, or included in the migration commit.

## Selected architecture

Use a checkout-local `fuzzydiff/` Python package. It supports package imports and
`python -m` execution without adding a mandatory editable-install step to Kaggle.
Keep package initializers lightweight: importing configuration, reporting,
planning, or CLI help must not load Torch, CUDA, notebooks, or checkpoints.

```text
fuzzydiff/                         # repository root
  fuzzydiff/                      # importable Python package
    __init__.py
    paths.py
    config.py
    notebook_runtime.py
    models/
      __init__.py
      registry.py
      sd15.py
      sdxl.py
      sdxl_turbo.py
      archived.py
    generation/
      __init__.py
      requests.py
      runner.py
    experiments/
      __init__.py
      benchmark.py
      comparison.py
      snow/
        __init__.py
        study.py
        gpu.py
        artifacts.py
        control.py
        control_source.py
        control_figure.py
        ratings.py
    reporting/
      __init__.py
      timing.py
      tables.py
    cli/
      __init__.py
      generate.py
      full_pipeline.py
      sd15.py
      benchmark.py
      snow.py
      snow_control.py
      snow_ratings.py
  pipeline_fuzzy/                  # canonical and archived notebooks, unchanged
  configs/                        # runnable JSON examples, no YAML dependency
  tests/                          # preserve unittest discovery and existing imports
  docs/
  outputs/                        # generated results, still ignored
  run_fullpipeline.py              # compatibility entry points
  run_sd.py
  run_benchmark.py
  run_snow_experiment.py
  run_snow_control.py
  analyze_snow_ratings.py
  requirements.txt
  README.md
  AGENTS.md
```

Other legacy import filenames remain as compatibility modules where needed.
The diagram lists the main interfaces rather than every compatibility filename.
There is no database repository abstraction or MVC layer: those patterns do not
serve this image-generation research workflow.

## Responsibilities and dependency direction

- `paths.py` resolves the repository, notebook, and compatibility-entry locations
  independently of the current working directory. Output paths remain explicit.
- `config.py` holds shared data-only command settings and validation. It does not
  duplicate notebook `RunConfig` or import model runtimes.
- `notebook_runtime.py` reads and compiles notebooks and supports the existing
  distinct loading policies. Full execution, prefix-selected fuzzy definitions,
  archived AST definitions, and snow's guarded cell selection are separate named
  operations sharing source reading and compilation. The loader does not evaluate
  a notebook batch unless the full execution caller explicitly requests it.
- `models/` owns loading, device placement, scheduler setup, generation, and
  resource cleanup. Adapters delegate the scientific algorithms to notebooks.
- `generation/` validates a request, invokes the selected adapter, and returns
  an image with diagnostics and effective settings. It does not benchmark or
  display images as an implicit side effect.
- `experiments/` owns repeated runs, warmups, worker isolation, resumable manifests,
  comparisons, and experiment artifact orchestration.
- `reporting/` formats already measured data and instruments exclusive timings.
  It does not invent results or own checkpoint loading.
- `cli/` parses arguments and explicitly saves or displays outputs. IPython
  display is optional and imported only when requested.

Dependencies flow from CLI to generation/experiments to model adapters and the
notebook runtime. Reporting and shared data definitions are reusable downstream
modules. The package's implementation never imports a legacy root compatibility
module to reach another package module.

## Model interfaces and truthful identities

Define a small adapter interface with generation, reset, close, metadata, and
access to owned pipelines for the existing profiler. Context-manager support
ensures cleanup even when generation fails. Internal adapters may retain distinct
loading strategies, but callers do not need to understand notebook globals.

The registry uses descriptive identities and retains existing names as aliases:

| Existing method | Descriptive identity | Actual behavior |
|---|---|---|
| stable | sd14 | Archived plain SD 1.4, PNDM |
| attend | attend_archived | Archived SD 1.5 Attend implementation |
| predicate | predicate_archived | Archived SD 1.4 Predicate implementation, FP32 |
| aesthetic | sdxl_base_refiner | SDXL base plus refiner; no SPO objective |
| enhancing | sdxl_turbo | Stock SDXL-Turbo; no C3 feature amplification |
| fuzzydiff | fuzzydiff_full | Canonical FuzzyDiff guidance plus refinement |

Plain SD 1.5 remains available through the existing SD CLI and the new generation
interface, without changing the `stable` benchmark alias to SD 1.5. Existing
comparison modes preserve their separate plain, fuzzy-base, and full conditions.
The migration adds no published baseline implementations or checkpoints.

Keep `benchmark_models.NotebookModel`, `NOTEBOOKS`, `LABELS`, `LIMITATIONS`, and
existing helpers import-compatible. The old `NotebookModel` facade delegates to
the registry; it is not a second independent generation implementation.

## Ordinary image generation

`GenerationRequest` is a data-only request containing model identity, prompt,
seed, dimensions, steps, CFG, negative conditioning, tracked phrases, and an
explicit mapping of method-specific options. Defaults are resolved by the chosen
adapter, then reported as effective settings. Unsupported options must produce a
clear error or an explicitly recorded existing limitation, not a silent change.

`GenerationResult` contains the image, effective model/settings metadata, and
diagnostics. Saving and displaying are explicit caller operations. A helper can
save an image plus JSON with the same provenance, without entering the benchmark.

The new `python -m fuzzydiff.cli.generate` command accepts `--model`, `--prompt`,
`--seed`, `--words`, `--negative`, dimensions/steps/CFG, method options, `--output`,
and optional `--display`. JSON examples use these same settings. Configuration
files provide defaults; explicitly supplied flags override them. This interface
must be documented with one-image examples for the six existing methods and plain
SD 1.5. It is separate from the existing full notebook command.

Maintain the existing effective behaviors, including Turbo's four steps/CFG=0,
Predicate's FP32, archived negative-prompt limitations, fuzzy guidance settings,
and separate base/refiner outputs where the existing command emits them.

## Concrete migration map

| Current implementation | Package owner |
|---|---|
| run_fullpipeline.py parsing/dispatch | cli/full_pipeline.py |
| run_sd.py sampling and pipeline setup | models/sd15.py |
| run_sd.py command parsing | cli/sd15.py |
| Existing notebook readers | notebook_runtime.py |
| benchmark_models.py loader/generation branches | models/registry.py and model adapters |
| benchmark_models.py preprocessing profiler hooks | reporting/timing.py |
| benchmark_timing.py | reporting/timing.py |
| run_benchmark.py worker orchestration | experiments/benchmark.py |
| run_benchmark.py parsing/entry dispatch | cli/benchmark.py |
| run_benchmark.py summaries/report writing | reporting/tables.py |
| compare_fullpipeline.py and four_way.py | experiments/comparison.py |
| snow_experiment.py | experiments/snow/study.py |
| snow_gpu.py | experiments/snow/gpu.py |
| snow_artifacts.py | experiments/snow/artifacts.py |
| snow_control*.py | corresponding experiments/snow modules |
| analyze_snow_ratings.py analysis | experiments/snow/ratings.py |
| Snow command parsers | corresponding cli modules |

Keep unrelated `pipeline_fuzzy/` visualization scripts, evaluation notebooks,
architecture images, and attention assets in their current locations. They are
not needed to establish the approved generation/experiment architecture.

## Compatibility details

- Root commands call package entry points. Root import facades expose the existing
  names and preserve patchable shared module state. A naive `from module import *`
  wrapper is insufficient where tests or callers replace module globals.
- `importlib.reload` in long-lived Kaggle kernels must reload the actual SD and
  comparison implementation, not only a facade. Add a documented reload helper
  and preserve existing reload points.
- Benchmark workers launch with a supported `python -m` entry point or the root
  compatibility command, not the moved implementation file as a bare script.
  Workers receive a resolved repository working directory and absolute config
  and output paths so external caller working directories work.
- Benchmark source hashes include actual package implementation files and source
  notebooks, using stable repository-relative keys. Hashing only compatibility
  wrappers is inadequate provenance.
- Snow fingerprints similarly include the actual package implementations.
  Existing studies produced by old code hashes continue to be rejected after the
  refactor. Start new output directories; do not rewrite hashes or relax resume
  checks to make old manifests appear compatible.
- Preserve output filenames and existing report fields. New provenance fields
  can be additive. Existing timing components remain exclusive and separate from
  ordinary whole-model inference; refiners stay combined in main benchmark rows.
- Preserve the existing generation defaults and exact negative-prompt strings.
  Loading policies must include constants referenced by notebook definitions,
  including `FUZZY_ATTRIBUTE_TARGETS` when those definitions can use it.

## Documentation and configuration

Update README with installation, package layout, ordinary generation, full
FuzzyDiff, benchmarking, snow studies, and known implementation limitations.
Update AGENTS.md's path ownership and loader descriptions while preserving its
scientific restrictions. Update benchmark/hedge/snow command documentation only
as needed for the new interfaces. Do not rewrite past REVIEW.md findings.

Add JSON examples for one-image generation and benchmark pilots. No checkpoint
paths, credentials, generated results, or unsupported metric claims belong in
example configuration. Narrow any ignore-rule change to documentation needed for
the refactor; do not accidentally track existing generated attention/results data.

## Verification and completion criteria

1. Record baseline notebook hashes and run the current CPU suite before moving
   implementation. Record unrelated failures separately if present.
2. Keep all current tests passing through compatibility imports. Add targeted
   tests for root/module entry points, external working directories, worker
   launching, reload semantics, and package implementation fingerprints.
3. Test lightweight imports/help, notebook compile checks, benchmark dry runs,
   and snow planning without GPU access or checkpoint downloads.
4. Use existing random-weight diffusion fixtures to compare deterministic
   generation before/after adapter extraction with identical settings. Exercise
   both SD 1.x and SDXL paths and preserve profiling instrumentation behavior.
5. Verify plain generation saves/displays the requested final image without
   warmups, repetitions, or profiling. Benchmark behavior remains separate.
6. Verify invalid bindings/configurations and unsupported options fail clearly;
   generation cleanup executes on exceptions and restores attention processors.
7. Verify archived and canonical notebook hashes remain unchanged. Preserve
   resume refusal for changed code fingerprints.
8. Finish with the complete CPU suite, Python compilation, notebook `--check`,
   dry runs, and an append-only REVIEW.md entry. GPU behavior remains unverified
   locally and must not be reported as tested image quality or measured speed.

## Execution stages after specification and plan approval

1. Introduce package paths and notebook runtime with compatibility coverage.
2. Extract model adapters and add ordinary generation requests/results.
3. Move benchmark, comparison, reporting, and snow implementations to their owners.
4. Replace legacy files with compatible entry/import facades, fix workers/reloads
   and provenance, and update JSON examples/documentation.
5. Verify the complete migration and provide the actual resulting structure and
   runnable Kaggle commands.

Implementation should happen in an isolated Git worktree. The written
implementation plan will specify tests and changes for each stage. No migration
implementation is authorized by approval of the directory sketch alone under the
brainstorming workflow; the next gate is review of this specification.
