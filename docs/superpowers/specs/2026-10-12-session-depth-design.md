# Deepen model sessions and benchmarking

Status: proposed; awaiting written-spec review.

## Intent and constraints

Deepen the existing model-session module so callers can generate and profile
images without knowing notebook namespaces, attention processors, or pipeline
instrumentation. Preserve legacy commands, sampling defaults, seeded behavior,
benchmark timing definitions, output schemas, and cleanup error handling.
The canonical and archived notebooks remain byte-identical. No dependencies,
new scientific objectives, GPU claims, or image-quality claims are introduced.

## Approaches

1. Recommended: deepen the existing session seam. Move profiling orchestration
   inside ModelSession and use it from benchmark workers. Keep existing adapters
   and legacy facades. This improves locality with a limited migration surface.
2. Introduce a separate profiling module with access to session internals.
   This moves code but preserves knowledge of pipelines in multiple places.
3. Replace the entire session/adapters hierarchy and all option dictionaries.
   This offers broader simplification but increases compatibility and sampling
   risk before the narrower seam is proven.

## Selected interface

Retain create_session(model, options), generate(seed), metadata(), reset(), and
close(), plus the existing context-manager behavior. Add profile(seed), returning
the profiled image and measurements. Benchmark callers use this operation
instead of reaching into pipelines or manually installing hooks.

The profile result contains the existing exclusive seconds, calls, and UNet
forward-call count. Existing benchmark serialization continues to produce the
same profile keys, generation_s convention, and pixel-equivalence flag.
Profile inference remains separate from native inference. Reset, garbage
collection, device-cache preparation, native timing, peak memory measurement,
image saving, row persistence, and repeat scheduling remain benchmark concerns.

Internally, session profiling installs and restores tokenizer/text encoder,
UNet/VAE, and preprocessing instrumentation through ExitStack. Hook restoration
must also occur after failed generation. Reporting timing remains the owner of
exclusive timing calculations. Notebook namespaces remain private implementation.

Use an internal generation context to replace the eight positional arguments
passed to adapter _generate methods. It carries the seed, prompt, words, common
sampling arguments, generator, token groups, indices, and phase measurement.
This context is not exposed to CLI callers. Adapters still delegate scientific
algorithms to their source notebooks.

Validated method-specific settings are deferred from this change. Existing
GenerationRequest validation and legacy option dictionaries remain compatible;
combining their migration with profiling would expand the interface change.

## Dependencies and tests

Profiling orchestration and result conversion are in-process. Existing local
random-weight Diffusers fixtures and lightweight fake pipelines substitute for
pretrained GPU pipelines at the internal seam. Checkpoint access remains an
external dependency and is excluded from CPU tests.

Test through session.profile and benchmark result output: timing exclusivity,
forward counts, image return, native/profile separation, seeded equivalence,
and restoration after inference failure. Retain legacy patch/reload regression
coverage. Replace tests tied only to moved benchmark hook placement with tests
at the session interface; do not accumulate duplicate implementation tests.
Run the complete CPU suite, compile notebook cells, and compare all notebook
hashes. GPU sampling and timings require separate measured Kaggle verification.

## Migration and acceptance

Modify models/session.py, adapters, experiments/benchmark.py, relevant regression
tests, and documentation. Keep the public generate_image(request) interface.
Keep profile_preprocessing as a compatibility helper while its use becomes
internal to session profiling. Preserve NotebookModel legacy construction.

Acceptance: benchmark code no longer accesses model.pipelines for profiling;
all adapters consume the same internal generation context; timing/result
contracts and exception cleanup remain covered; notebooks are unchanged; full
CPU regression tests pass. No unrelated adapters, options, or notebook changes.

## Self-review

The design preserves the existing session lifecycle rather than inventing an
additional lifecycle. It separates profiling from native timing, preserves
measurement semantics, and limits the migration to one real seam with several
existing adapters. Broader settings changes are explicitly outside this spec.
