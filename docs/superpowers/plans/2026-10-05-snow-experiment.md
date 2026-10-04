# Snow experiment implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Ship the confirmed 15-image pilot and resumable 225-image study.

**Architecture:** A pure Python study ledger owns settings, jobs and persistence.
A GPU adapter executes the existing notebook, while CPU artifact and rating
modules render outputs and support independent human review.

**Tech Stack:** Python, Pillow, matplotlib, existing pinned SDXL/Diffusers stack.

**Spec:** [SNOW_EXPERIMENT.md](../../../SNOW_EXPERIMENT.md)

## Global constraints

- Preserve the notebook's guidance algorithm and all confirmed study settings.
- Pilot: alpine scene, seed 142, five levels, three methods; full: 225 images.
- Persist each image and failures; resume without replacing valid completed outputs.
- Match prompts/seeds; disclose precision and native-reference confounds.
- Separate private mappings from the shared review pack; require two reviewers.

## Review focus

- Interrupted writes or missing attention files must trigger a safe retry.
- Changed code/settings must not silently mix experiments.
- Refiner errors must preserve successful stage-one output.
- Anonymous exports must not expose method or target-level filenames/metadata.
- Invalid, duplicate, or incomplete human ratings must not produce a complete-study claim.

### Task 1: Study ledger and execution seam

Files: `snow_experiment.py`, `tests/test_snow_experiment.py`.
Interface: `make_design()`, `jobs_for(design, full)`, `run_study(root, design, backend, full=False)`.

- [x] Write tests for 15/225 matched jobs, resume, missing dependencies, errors and design mismatch.
- [x] Run tests and confirm they fail before implementation.
- [x] Implement atomic manifest persistence and an injectable generation backend.
- [x] Run tests to green.

### Task 2: GPU adapter and CLI

Files: `snow_gpu.py`, `run_snow_experiment.py`.
Interface: backend `prepare(method)`, `generate(job, root)`, `close()`; returns image and serializable metadata.

- [x] Test CPU-only CLI planning and notebook compilation.
- [x] Reuse notebook definitions and shared native/base weights; persist attention for refinement.
- [x] Test phase order and resumable refiner dependency; compile all sources.

### Task 3: Figures and blind ratings

Files: `snow_artifacts.py`, `analyze_snow_ratings.py`, artifact/rating tests.
Interface: `export_artifacts(root, full=False)`, `analyze(root, rating_paths)`.

- [x] Test incomplete grids, anonymous exports, stable private mapping, and invalid/partial rating data.
- [x] Implement PNG/PDF grids, anonymous pack, two-pass pages and two CSV sheets.
- [x] Implement descriptive coverage trends and disagreement summaries without inferential claims.
- [x] Render fixture grids for visual inspection; run all regression tests.

### Task 4: Review and delivery

- [x] Document Kaggle commands and limitations in README and the spec.
- [x] Obtain a fresh code/spec review, fix actionable findings, reverify.
- [ ] Commit and push under the user's already-authorized GitHub profile.
- [ ] Report verification separately from GPU validation that remains on Kaggle.

## Execution record

Ruling: execute the confirmed design inline in the existing checkout, using
the user's ongoing authorization to fix and push this project. No additional
implementation-choice approval is needed after the explicit confirmation.

Task 1: complete. Ledger tests observed red before implementation and green
afterward. Full expansion reuses 15 pilot outputs and generates 210 more.
Nonfinite metadata is rejected before mutating durable output records.

Task 2: complete. CPU planning prints 15/225 job counts without writes or GPU
imports. Attention tensor snapshots round-trip with weights-only loading.
Canonical notebook compiles 26 code cells and remains unchanged.

Task 3: complete. Figure layout visually inspected using explicitly labeled
synthetic fixtures, not presented as diffusion results. Private mapping and
ratings remain outside the shared ZIP. Two-reviewer summaries retain individual
ratings and disagreements, and flag incomplete reviews.

Final fresh reviewer: four P2 findings reproduced with failing tests and fixed:
scope shrink, entered ratings leaking into regenerated packs, missing reference
provenance, and false artifact-only completion. GPU execution is still unverified;
the user must run the pilot on Kaggle before expanding or asserting a benefit.

Final verification: 22 unittest checks passed; 11 Python source files compiled;
26 canonical notebook code cells compiled; Git whitespace check passed. The
reviewer's focused follow-up confirmed all four fixes with no remaining findings.
