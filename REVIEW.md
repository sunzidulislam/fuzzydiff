# Full notebook review

Scope: the entire current `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb`, as requested,
not a Git diff. Requirements: the README and the user's GitHub-to-Kaggle T4 workflow.
No repository standards documents or issue-tracker configuration were present.
Two independent reviewers examined standards and specification behavior.

## Standards

No hard documented-standard violations; these are five judgement calls:

1. **Duplicated attention processing:** base and refiner processors diverged in mask
   preparation and normalization. Repaired by sharing the standard attention path.
2. **Refused inherited contract:** custom SDXL sampling silently ignored arguments,
   returned a list, and omitted pipeline postprocessing/offload cleanup. Repaired:
   explicit supported arguments, standard pipeline output, postprocessing and cleanup.
   The supported scope remains one prompt and square images.
3. **Repeated configuration:** settings repeated in config, generation and entry point;
   attention resolution was hard-coded. Entry settings now flow through RunConfig,
   and the requested attention resolution reaches the collector. Convenience defaults
   remain in generate for notebook use.
4. **Hidden global dependencies:** model/device/CLIP globals made reruns and unloading
   fragile. Generation now accepts a pipeline and refinement accepts a refiner.
   Notebook-level CLIP resources remain explicit shared globals; model-loading cells
   must be rerun after the entry cell unloads the base pipeline.
5. **Persistence hazard:** refinement preceded saving both images. Stage one and its
   metadata now save before refinement; failures update metadata and propagate.

## Spec

1. **Inactive fuzzy guidance:** detached maps, no-grad forward and stale latent leaf
   broke the promised "differentiable graded truth values." Guidance now performs
   a gradient-enabled conditional forward on current latents and updates them before
   noise prediction at the same timestep. Frozen weights and gradient checkpointing
   limit memory; finite, nonzero gradients and update magnitudes are recorded.
2. **CFG contamination:** attention-head flattening mixed negative/positive branches;
   the refiner changed both. Batch/head dimensions are separated and only positive
   conditioning is collected or modified.
3. **Invented relations:** every tracked pair was forced left and below, violating
   "relational consistency." Only explicit left_of/right_of/above/below constraints
   apply. Depth relations such as behind are not implemented by this 2D loss.
4. **Kaggle reliability:** models competed for one GPU; efficient attention was replaced;
   a refinement error lost all images. Self-attention and high-resolution generation
   cross-attention keep efficient processors, models use CPU offload, base weights
   unload before refinement, and stage-one persistence is independent of refinement.
5. **Token mismatch:** only the first subtoken/occurrence was tracked and base indices
   were reused for the refiner. Full groups and repeated occurrences are retained,
   overlong/missing phrases are rejected, base tokenizer alignment is checked, and
   phrase masks are remapped using the refiner's tokenizer_2.
6. **Bookkeeping/reproducibility:** layer counters conflicted with manual flushes;
   refinement omitted seed and negative prompt. Explicit per-forward boundaries,
   CPU generators, propagated negative prompts, and run metadata address this.
7. **Unproven intensity/quality claims:** token presence does not calibrate "slightly"
   or "very." This limitation is now explicit. No measured image-quality improvement
   is claimed. CLIP uses direct prompts rather than unrelated attribute templates;
   old and new CLIP scores therefore should not be directly compared.

## Validation and Kaggle follow-up

CPU regression tests exercise notebook definitions and actual tiny Diffusers components
without downloading weights. GPU SDXL generation, T4 peak memory and perceived image
quality require a Kaggle run. Begin with the smoke test, inspect its diagnostics and
images, then run the full settings. Compare guidance-on/off with matched prompt,
seed, scheduler and negative prompt; compare refinement against stage one separately.

Local verification: eight regression tests pass, including a real tiny SDXL pipeline
with checkpointed gradients, guidance-on/off output differences, seeded repeatability
and VAE decoding. All 26 code cells compile, notebook schema validation passes, and
Git whitespace checks pass. The pinned Diffusers scheduler emits an upstream NumPy
deprecation warning during CPU testing; it does not fail execution.

Standards: 5 findings; worst was duplicated attention processing. Spec: 7 findings;
worst was inactive fuzzy guidance.

## Follow-up: distorted seed-142 output

The supplied metadata records a full 768px, 50-step run. Gradient RMS values around
1e-5 were normalized into update RMS values of 0.13–0.20 over 30 steps, amplifying
small gradients roughly 10,000–30,000 times. This unvalidated normalization was
introduced in the first repair and is removed. Raw gradient steps now retain their
magnitude, with an RMS ceiling of 0.01; latents stay float32 to preserve small updates
while the UNet still uses its loaded dtype. The scaling regression fails on the old
implementation and passes on the corrected implementation.

Stage one was already visually distorted. Refinement reduced full-prompt CLIP
similarity from 0.2292946 to 0.2207624. These observations motivate baseline diagnosis;
they do not prove that scaling is the only cause or that the corrected output looks
better. `--compare` saves native SDXL, custom guidance-off, and corrected guidance-on
results with identical prompt, seed, CFG, steps and dimensions, without a refiner.
The CPU tiny-model test checks custom/native latent agreement within fp32 attention
rounding tolerance, as well as connected guidance and repeatability. Pretrained GPU
output quality still requires running and inspecting that comparison in Kaggle.
Native sampling retains its normal latent precision, while custom sampling retains
float32 latents. Native/custom comparisons therefore include precision differences;
the custom guidance-off/on pair isolates the guidance correction. Small accumulated
float32 updates are preserved between steps, but individual updates may not immediately
cross the fp16 UNet input's rounding threshold.

---

# Review: 2026-10-08, against the revised abstract

Scope: the whole current `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb`, as requested,
not a Git diff. Specification: the revised paper abstract supplied by the author,
whose methodology claims are phrase-level spatial membership functions, differentiable
graded truth values, and a spatially guided attention refinement that corrects
residual omissions. No repository standards document or issue tracker exists, so the
standards axis used the Fowler smell baseline.

## Spec

1. **Membership functions were missing, and the truth values were not graded
   (fixed).** `soft_truth` pooled raw cross-attention, whose per-token scale is set
   by the 77-token context. With `alpha = 10` the softmax weighting is nearly
   uniform at that scale, so the "graded truth value" was approximately the mean
   attention, about `0.015`, for a grounded and an absent phrase alike; the
   simulated spread between them was `0.0152` against a true map maximum of
   `0.0823`. The loss `-log(score)` therefore sat near `4.2` regardless of whether
   the prompt was satisfied, which is the direct cause of the `1e-5` gradients
   recorded in the earlier follow-up. `phrase_membership` now converts attention
   into a membership function in `[0, 1]`: the phrase's share of the sharpened
   per-position distribution over the prompt's own tokens. On a CPU fixture with a
   realistic 77-token context the loss moves between `0.75` (grounded) and `4.26`
   (weak) and gradient RMS rises from `1e-5` to `1.3e-3`-`5e-2`.
2. **Presence was token-level, not phrase-level (fixed).** `compute_fuzzy_loss`
   took a flat index list and applied `min` across every subtoken, so a multi-token
   phrase became several competing constraints. It now takes phrase groups and pools
   a phrase's subtokens into one membership map (a bounded fuzzy union) before the
   Goedel t-norm combines phrases.
3. **Refinement did not target residual omissions (fixed).** Every tracked phrase
   received the same `predicate_strength`, with no notion of which phrase the base
   stage left weakly grounded. `phrase_truth_scores` now reports presence truth per
   phrase, and each phrase's attention correction is scaled by its membership
   deficit `1 - truth`, so a grounded phrase is left unchanged and an omission gets
   the full correction. Stage-one truths are printed and stored as
   `stage1_phrase_truth`.
4. **Refinement masks were built by a different rule than guidance (fixed).**
   `build_object_masks` used per-token min-max normalization and an ad hoc
   `pow(1.5)`. It is now `build_phrase_membership_masks` and reuses
   `phrase_membership`, so one definition drives both stages.
5. **Claims the abstract dropped are no longer asserted.** The previous abstract
   promised objectives for attribute intensity and spatial extent, which were never
   implemented. The revised abstract drops both, and the README no longer claims
   them. Hedge intensity remains the pretrained model's interpretation.
6. **"Throughout the denoising process" remains partial (documented, not changed).**
   Guidance applies for the first `max_iter_to_alter` steps, 30 of 50 by default.
   This is the published Attend-and-Excite schedule and is now stated in the README
   rather than implied to be every step.
7. **Update magnitude is still unvalidated on GPU (open).** The absolute `0.01`
   update-RMS ceiling was chosen when gradients were `1e-5`. Membership raised the
   gradient scale by two to three orders of magnitude, so the ceiling may now bind
   every step, and a fixed absolute bound ignores the latent scale, which varies
   with the scheduler's initial sigma. Rather than retune it blind, the diagnostics
   now record `update_ratio` (update RMS over latent RMS) and `clipped`. Read those
   from the first `--compare` run before changing the ceiling.

## Standards

No hard documented-standard violations; the repository documents no standards. Four
baseline smells, all judgement calls:

1. **Primitive Obsession (fixed).** A phrase was a bare list of integer token
   indices threaded through five call sites. The membership function is now the unit
   that moves between guidance, diagnostics and refinement.
2. **Duplicated Code (fixed).** Guidance and refinement each had their own
   attention-to-mask normalization. Both now call `phrase_membership`.
3. **Speculative Generality (fixed).** `predicate_truth`'s out-of-range guard was
   unreachable, because `compute_fuzzy_loss` validates indices first and raises, and
   `get_token_attention` had no callers. Both are removed; `phrase_truth_scores`
   replaces the latter with a value that is actually recorded.
4. **Mysterious Name (fixed).** `build_object_masks` returned token-indexed
   attention masks, not objects. It is now `build_phrase_membership_masks`.

Deliberately not changed: `refine_with_predicate`, `PredicateRefinerProcessor` and
`attach_predicate_control` keep their names, and the snow study keeps its
`predicate_strength` design key. The abstract's new wording is prose for the same
mechanism, and that key is persisted in resumable Kaggle run plans, so renaming it
would break resume for no methodological gain. Docstrings and documentation use the
abstract's "spatially guided attention refinement" wording.

Also corrected in the supplied abstract text: "inconsistant object" to "inconsistent
objects", "composi1tions" to "compositions", the duplicated "phrase-level", and the
stray space inside the final `FuzzyDiff` macro.

## Validation

Local, CPU only, with torch 2.14.1+cpu / diffusers 0.35.1 / transformers 4.56.2:
34 of 34 tests pass, among them 12 notebook regression tests including four new ones (membership is graded rather
than context-scaled, a phrase pools its subtokens into one predicate, phrase truths
and masks are read back through the store shape generate() fills, and a fully
grounded phrase is not reweighted), and all 26 notebook code cells compile
(`python run_fullpipeline.py --check`). The snow adapter tests pass and now persist
`text_span`; checkpoints written before this change resume with membership taken over
the whole context.

One consequence to plan around: the snow study's resume guard compares code
fingerprints, and the membership change alters them, so `run_snow_experiment.py` now
refuses to top up a manifest produced by the previous algorithm. That refusal is
correct, because mixing pre- and post-membership images in one study would invalidate
it. Start those runs in a new `--output` directory. A regression test pins both
halves of that behaviour: the one historical metadata repair still resumes, and a
changed generation algorithm is refused.

Not validated: everything that needs a GPU. No SDXL weights were run, so no claim is
made about image quality, T4 peak memory, or whether guidance now visibly changes the
output. Run `python run_fullpipeline.py --compare` first and read `update_ratio`,
`clipped` and `stage1_phrase_truth` before claiming any improvement.

Spec: 7 findings, 5 fixed; worst was the ungraded truth value that made the loss
nearly constant. Standards: 4 findings, all fixed; worst was the duplicated
attention-to-mask normalization between the two stages.

## Measured on GPU: seed 142, "A very fast car", 768px / 50 steps

The matched-seed comparison ran after the membership change. Native SDXL and custom
guidance-off are visually near-identical, CLIP `0.26656` against `0.26599`, which
verifies the custom sampling path against the reference implementation. The streaked
stage-one output recorded in the earlier follow-up no longer occurs.

Guidance is connected and bounded but weak at the shipped settings: across 30 updates
`clipped` is false on every step, `gradient_rms` is `3.7e-4` to `9.4e-4`, `latent_rms`
is about `1.0`, and `update_ratio` runs `6e-5` to `1.9e-4`. The `0.01` ceiling never
binds, so roughly fifty times more step size is available before it would. The guided
image is still visibly different from guidance-off, because early latent perturbations
compound through sampling.

CLIP similarity does not separate the three runs: `0.26656` native, `0.26599`
guidance-off, `0.26448` guidance-on. Two measurement problems, not method problems.
First, `A very fast car` is a prompt SDXL already satisfies, so there is no missing
object or failed binding for guidance to correct and no headroom to measure. Second,
CLIP is a coarse proxy for an objective the method defines exactly: the phrase
membership truth. `--compare` now records `phrase_truth` per run and accepts
`--prompt`, `--words`, `--seed` and a comma-separated `--lr` sweep, so guidance-off
truths can be compared against each step size on a prompt that native SDXL fails.
No improvement is claimed until that comparison exists.
