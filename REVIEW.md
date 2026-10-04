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
