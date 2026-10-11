# Repository Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Refactor the research scripts into the approved Python package while preserving existing commands and adding ordinary single-image generation independent of benchmarking.

**Architecture:** A checkout-local `fuzzydiff/` package owns notebook loading, model adapters, generation, experiments, reporting, and CLI parsing. Root files become compatible entry/import facades. Scientific logic remains in the canonical notebook; archived notebook algorithms remain unchanged.

**Tech Stack:** Python, standard-library dataclasses/argparse/JSON/unittest, existing pinned Torch/Diffusers/Transformers/Pillow dependencies, Git worktrees; no new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-10-12-repository-structure-design.md` (approved by the user on 2026-10-12).

## Global Constraints

- `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb` remains the only implementation of fuzzy membership, loss, attention control, and refinement mathematics.
- Archived notebooks in `pipeline_fuzzy/analysis/` remain byte-identical.
- No new runtime dependencies, pretrained downloads, or local GPU experiments.
- Existing root commands remain usable from a repository checkout on Kaggle.
- `REVIEW.md` is append-only. Generated outputs and user assets are not moved, deleted, rewritten, or included in the migration commit.
- Existing studies produced by old code hashes continue to be rejected after the refactor. Start new output directories.
- Plain SD 1.5 does not replace the SD 1.4 `stable` benchmark alias.
- Turbo retains 4 steps/CFG=0; Predicate retains FP32; published-method limitations remain visible.
- Retain original output filenames, negative strings, scheduler settings, timing definitions, and full/base ablation distinctions.

## Review Focus

1. Calling a CLI from an external working directory must still resolve notebooks and launch workers (Tasks 1, 5).
2. A fresh subprocess must import the package without editable installation or inherited notebook kernel state (Task 5).
3. Importing planning/reporting or requesting help must not import CUDA/model runtimes or download checkpoints (Tasks 1, 4, 7).
4. Legacy module monkeypatches and Kaggle reloads must affect the actual implementation rather than a stale facade (Tasks 2, 5, 7).
5. Notebook constants and signed hedge refinement must survive every definition-loading policy that uses them (Tasks 1, 3).

## File Ownership

The approved specification supplies the final tree and migration map. Tasks below
lock the interfaces at each seam. Preserve the existing flat `tests/` discovery;
new tests live beside the existing regression tests. Package `__init__.py` files
contain documentation only and never import heavy adapters eagerly.

Root compatibility modules retain public helpers in addition to `main()` where
existing tests or user cells import them. Core package modules use package imports,
never root compatibility imports. The unrelated visualization scripts and image
assets remain outside this migration.

## Task 1: Paths and notebook runtime

**Files:** Create `fuzzydiff/__init__.py`, `fuzzydiff/paths.py`, `fuzzydiff/notebook_runtime.py`; modify the notebook reader portions of `run_sd.py`, `run_fullpipeline.py`, `benchmark_models.py`, `snow_gpu.py`; create `tests/test_notebook_runtime.py`.

**Interfaces produced:**
- `repository_root() -> Path`, `canonical_notebook() -> Path` in `paths.py`.
- `notebook_cells(path: Path) -> list[tuple[int, str]]`: original cell indices and normalized code strings, excluding installation lines at the callers' existing policy.
- `compile_notebook(path: Path) -> list[tuple[int, CodeType]]`: preserves `:cell-N` filenames used by the full/snow callers and tests.
- `load_prefix_definitions(path: Path, prefixes: tuple[str, ...], namespace: dict) -> dict`: executes only matching cells, retains supplied namespace globals.
- `definition_code(path: Path) -> CodeType`, `load_definitions(path: Path) -> dict`: archived AST loading with the existing import filters and module registration for dataclasses.
- `source_hashes(paths: Iterable[Path]) -> dict[str, str]`: repository-relative keys, actual source bytes, no silent omission of required files.

- [x] Write tests asserting root resolution after changing cwd, string/list notebook source support, cell-index preservation, unchanged notebook bytes, prefix loading, and skipped batch/download calls. Include `FUZZY_ATTRIBUTE_TARGETS` in an AST fixture and exercise a function referencing it.
- [x] Run `.venv-cpu/Scripts/python.exe -m unittest discover -s tests -p test_notebook_runtime.py`; verify missing-package/function failures before implementation.
- [x] Implement paths/runtime helpers using the existing loader bodies and explicit policies. Add `FUZZY_ATTRIBUTE_TARGETS` to the allowed assignment names alongside `HEDGE_TARGETS` and `RELATION_VERBS`; do not broaden arbitrary notebook execution.
- [x] Delegate existing readers to these helpers without changing caller namespaces, prefixes, skipped cell numbers, or model initialization.
- [x] Run the new runtime tests and current fullpipeline/SD/benchmark/snow-GPU tests. `python run_fullpipeline.py --check` must compile the same code-cell count without model imports.
- [x] Commit only Task 1 source/tests: `refactor: centralize notebook runtime and repository paths`.

## Task 2: SD 1.x adapter and compatible import seam

**Files:** Create `fuzzydiff/models/__init__.py`, `fuzzydiff/models/sd15.py`, `fuzzydiff/cli/__init__.py`, `fuzzydiff/cli/sd15.py`; modify `run_sd.py`; create `tests/test_package_sd15.py`; retain `tests/test_sd_fuzzy.py`.

**Interfaces consumed:** Task 1 path/runtime helpers.

**Interfaces produced:** Preserve signatures of `FuzzyStableDiffusionPipeline`, `load_fuzzy_definitions`, `configure`, `build_pipeline`, `build_config`, relation/binding helpers, `generate_fuzzy`, and `generate_plain` in `models/sd15.py`. `cli/sd15.py:main(argv: list[str] | None = None) -> int` owns parsing and file output.

- [x] Write a compatibility test asserting legacy `run_sd` imports resolve the same callable objects and shared globals as `fuzzydiff.models.sd15`. Verify patching a legacy helper affects the sampling caller and reloading preserves availability.
- [x] Run the new test and observe import/identity failures.
- [x] Move the existing sampling/setup/helpers into `models/sd15.py`, delegating notebook loading to Task 1. Move CLI code into `cli/sd15.py`; expose legacy `main` through an explicit facade callable with a lazy CLI import.
- [x] Implement a root `run_sd.py` facade that executes CLI main when run as a script and preserves shared import state when imported. Avoid copied function implementations and avoid star-import snapshots of mutable globals.
- [x] Run package compatibility tests and all `test_sd_fuzzy.py` random-weight tests. Preserve the maintained SD 1.5 checkpoint, DPMSolver++ Karras scheduler, and seeded pixels.
- [x] Commit: `refactor: extract SD generation adapter and CLI`.

## Task 3: Registry and reusable model sessions

**Files:** Create `fuzzydiff/models/registry.py`, `fuzzydiff/models/sdxl.py`, `fuzzydiff/models/sdxl_turbo.py`, `fuzzydiff/models/archived.py`; modify `benchmark_models.py`; create `tests/test_model_registry.py`.

**Interfaces consumed:** Tasks 1 and 2.

**Interfaces produced:**
- `ModelSpec`: data-only record with `id`, `aliases`, `label`, `notebook`, `limitations`.
- `resolve_model(name: str) -> ModelSpec`; canonical descriptive IDs and old aliases follow the specification. Also register `sd15` for ordinary generation.
- `create_session(model: str, options: dict) -> ModelSession`: lazy adapter import.
- Session interface: `generate(seed: int, ledger=None) -> Image.Image`, `reset() -> None`, `close() -> None`, `metadata() -> dict`, `profile_preprocessing(ledger)` context manager, `pipelines`, `last_details`; context-manager entry returns the session, exit closes it.
- Legacy `NotebookModel(method, options)` delegates to `create_session`, preserving its existing methods/properties without a second branch-based implementation.

- [x] Test every old alias against its descriptive identity; `stable` must resolve to SD 1.4, `enhancing` to Turbo, `aesthetic` to ordinary SDXL base/refiner. Test unknown IDs and data-only registry import with Torch unavailable.
- [x] Test factory selection, ownership/cleanup on exceptions, Turbo's 4 steps/CFG=0, Predicate's FP32, and effective settings/provenance using injected or patched loaders that never download weights.
- [x] Run registry tests and verify they fail on absent interfaces.
- [x] Extract the existing `NotebookModel` branches into concrete adapters, preserving generation bodies and existing limitations. Each adapter delegates notebook-defined algorithms; share lifecycle/loading utilities only where behavior is actually common.
- [x] Preserve base/refiner VAE separation, offload-hook ordering, attention restoration, negative-prompt limitations, and profile access. Fuzzy session loading must include the constants and helper definitions used by hedge refinement.
- [x] Run existing benchmark definition/metadata/instrumentation tests plus new registry tests. Compare SD 1.x and SDXL random-weight seeded outputs across the extracted adapter seam.
- [x] Commit: `refactor: introduce model registry and generation sessions`.

## Task 4: Ordinary generation request, runner, and command

**Files:** Create `fuzzydiff/config.py`, `fuzzydiff/generation/__init__.py`, `fuzzydiff/generation/requests.py`, `fuzzydiff/generation/runner.py`, `fuzzydiff/cli/generate.py`, `configs/generate_fuzzydiff.json`, `configs/generate_sdxl_turbo.json`; create `tests/test_generation_runner.py`.

**Interfaces consumed:** Task 3 factory/spec/session interface.

**Interfaces produced:**
- Frozen `GenerationRequest(model: str, prompt: str, seed: int = 142, size: int = 512, steps: int | None = None, guidance: float | None = None, negative: str = '', words: tuple[str, ...] = (), method_options: dict = field(default_factory=dict))`.
- `GenerationResult(image, metadata: dict, diagnostics: dict)`; images are caller-owned after session cleanup.
- `generate_image(request: GenerationRequest) -> GenerationResult`: one generation, no warmup/profiling/benchmark repetitions.
- `save_result(result: GenerationResult, directory: Path, seed: int) -> dict[str, Path]`: final `seed_{seed}.png` and `seed_{seed}_metadata.json`, preserving scientific measurement notes.
- `load_request_config(path: Path) -> dict`, `merge_request_config(config: dict, explicit_flags: dict) -> GenerationRequest` in `config.py`: flags override JSON; reject unknown config fields.
- `cli/generate.py:main(argv=None) -> int`: model/prompt/seed/words/negative/size/steps/guidance, existing fuzzy method options, `--config`, `--output`, `--display`.

- [x] Test exactly one session generation, no timing/profile calls, cleanup on failure, default/effective settings, unknown options, invalid seeds, malformed JSON, and explicit-flag precedence. Mock display and assert only the final requested image is displayed once.
- [x] Run generation tests and verify missing interface failures.
- [x] Implement data-only request validation and registry-driven defaults. Convert request fields into existing adapter options; do not change mathematical defaults or repurpose attention truth as visual accuracy.
- [x] Implement the runner and explicit image/JSON saving. Heavy imports occur only inside generation; `--help` and config validation remain CPU/lightweight. Import IPython only for `--display`.
- [x] Add JSON examples and test them against the same parser/schema. Use dummy model sessions and existing random-weight fixtures; never download pretrained checkpoints.
- [x] Run runner/config tests and single-image dry validation. Confirm benchmark and snow code are not imported by ordinary generation.
- [x] Commit: `feat: add direct model generation interface`.

## Task 5: Benchmark, comparisons, reporting, and full CLI

**Files:** Create `fuzzydiff/reporting/__init__.py`, `fuzzydiff/reporting/timing.py`, `fuzzydiff/reporting/tables.py`, `fuzzydiff/experiments/__init__.py`, `fuzzydiff/experiments/benchmark.py`, `fuzzydiff/experiments/comparison.py`, `fuzzydiff/cli/benchmark.py`, `fuzzydiff/cli/full_pipeline.py`; modify `benchmark_timing.py`, `run_benchmark.py`, `run_fullpipeline.py`, `compare_fullpipeline.py`, `four_way.py`, `benchmark_models.py`; create `tests/test_package_commands.py`.

**Interfaces consumed:** Tasks 1–3. Ordinary generation and benchmarking share adapter sessions, not timing loops.

**Interfaces produced:** Preserve existing `TimeLedger`, instrumentation functions, summary/report helpers, `run_worker`, `launch_worker`, `run_comparison`, and `run_four_way` signatures. `cli/benchmark.py:main(argv=None) -> int` supports the existing internal `--worker` dispatch. `cli/full_pipeline.py:main(argv=None)` preserves existing flags/dispatch and notebook execution.

- [x] Test old and `python -m` commands with the same arguments; capture dispatch/settings without models. Test `--worker` in a fresh CPU-only subprocess, external cwd, root import patching, and package reload state.
- [x] Test report schemas/whole-model stage grouping and exclusive ledger accounting through the new ownership. Assert source hashes cover the registry, model adapters, runtime, orchestration, reports, and notebooks rather than only facades.
- [x] Run command tests and verify missing package command failures.
- [x] Move timing to reporting and split report formatting from worker orchestration. Move comparisons into their package owner and retain the plain -> fuzzy-base -> full distinction and current grid labels/CLIP behavior.
- [x] Launch workers as `python -m fuzzydiff.cli.benchmark --worker ...` with `cwd=repository_root()` and absolute config/output paths. Retain offline loading for measured workers and separate cache-preparation workers.
- [x] Add reload handling for actual adapter/comparison modules, preserving existing `%run` defenses. Preserve the canonical notebook full-run entry cell, its filename convention, and RUN_OPTIONS forwarding.
- [x] Replace root modules with compatible script/import facades. Update provenance paths using Task 1 helpers and maintain existing JSON field names, failures, and nonempty-directory refusal.
- [x] Run current benchmark, runner-option, SD/grid tests and new command tests. Run root/package `--check`, all-model benchmark dry run, and Enhancing-only dry run at 4 steps/CFG=0.
- [x] Commit: `refactor: package benchmark comparison and reporting workflows`.

## Task 6: Snow study and control workflows

**Files:** Create `fuzzydiff/experiments/snow/` modules from the migration map and `fuzzydiff/cli/snow.py`, `snow_control.py`, `snow_ratings.py`; modify all root `snow_*.py`, `run_snow_*.py`, `analyze_snow_ratings.py`; create `tests/test_package_snow.py`.

**Interfaces consumed:** Task 1 paths/runtime/provenance; existing canonical notebook interfaces. No change to snow sampling/rendering objectives.

**Interfaces produced:** Preserve `make_design`, `jobs_for`, `run_study`, `atomic_json`, `SnowBackend`, attention serialization, control-design/render/export helpers, and rating analysis through compatibility imports. Package CLI `main(argv=None)` functions preserve existing argument sets.

- [x] Test planning/artifact generation with no Torch/CUDA imports where the existing path permits it. Test actual package-file fingerprints, a changed implementation hash causing resume rejection, and unchanged artifact schema/path names.
- [x] Run package snow tests and verify missing interfaces/fingerprint failures.
- [x] Move existing implementation bodies and update imports/paths. Preserve guarded notebook cell indices and attention checkpoint payloads. Hash real source modules, never only root facades.
- [x] Preserve atomic writes, original failed-run upgrades, missing/bad-asset detection, procedural-renderer labeling, and resume checks. Do not convert old study manifests to new hashes.
- [x] Run all existing snow-study/control/artifact/GPU-adapter CPU tests plus package snow tests. Run original/package snow `--plan` paths; do not invoke a GPU generation phase.
- [x] Commit: `refactor: organize snow experiments and controls`.

## Task 7: Final compatibility, configuration, and documentation

**Files:** Modify `README.md`, `AGENTS.md`, `BENCHMARK.md`, `HEDGE_GUIDANCE.md`, `SNOW_EXPERIMENT.md`, `SNOW_CONTROL.md`, `.gitignore` only if necessary; create `configs/benchmark_pilot.json`; extend `tests/test_package_commands.py`; append `REVIEW.md`.

**Interfaces consumed:** All previous tasks; existing root commands remain supported.

- [x] Test clean-interpreter imports of package data/reporting/help with blocked Torch and checkpoint loaders. Assert all supported root helpers still import and legacy monkeypatches/reloads reach their package owners.
- [x] Verify old and new CLI flags, image names, model aliases, JSON fields, and single-display behavior against the specification. Assert canonical/archived notebook bytes equal the captured baseline hashes.
- [x] Run final compatibility tests and observe any remaining broken assumptions before fixing facades/docs. Fix actual implementation gaps rather than weakening tests or changing notebook algorithms.
- [x] Update README with real package layout, direct generation examples for six methods plus SD 1.5, benchmark/snow commands, and known baseline limitations. Update AGENTS path/loader ownership without removing scientific restrictions. Keep example configs CPU-valid and free of secrets/generated results.
- [x] If docs need tracking, narrow the existing broad ignore rule only for authored documentation; leave old attention/generated directories ignored. Do not stage user artifacts.
- [x] Run complete `unittest discover -s tests`, compile package and root Python files, root/package notebook checks, and dry-run commands. Inspect Git diff and notebook hashes. Scope results to CPU verification; no GPU/image-quality claims.
- [x] Append REVIEW with structural changes, verification results, and the requirement for fresh snow-study output directories. Commit: `docs: document structured repository and compatibility`.

## Execution Preparation and Handoff

- [x] Obtain user review of this plan and their chosen execution approach.
- [x] Use `superpowers:using-git-worktrees` to create an isolated worktree for this refactor. Inspect cleanliness before branching; preserve user changes. Confirm the approved spec and plan are committed or otherwise available inside that worktree.
- [x] Record baseline notebook SHA256 values and run the baseline full CPU suite using the existing `.venv-cpu` interpreter. Use the same interpreter by absolute path from the isolated worktree rather than installing dependencies. Windows temporary-directory access may require the already established elevated-test approval.
- [x] Implement tasks in order; update checkboxes only after each verification passes. Keep code moves separate from necessary compatibility fixes in reviewable commits.
- [x] Complete the required whole-branch review, address findings, and report the actual file layout, new commands, legacy compatibility, checks performed, and GPU limitations.
- [ ] Integrate the verified branch only through the user's chosen finishing workflow. Do not delete existing worktrees or discard user changes.

## Plan Self-Review

The tasks cover paths/runtime, adapters, ordinary generation, benchmarks/reports,
comparisons, snow, configuration/documentation, and compatibility. All five Review
Focus conditions have explicit tests assigned. Signatures referenced by later tasks
are defined in earlier Interfaces blocks; current helper signatures are preserved
where callers already depend on them. Notebook mathematics, archived baseline
repairs, and image-quality experiments are outside this plan.
