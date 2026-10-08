# Dusty-car experiment handoff: seed 142

Prompt: "A slightly dusty red sports car parked on a road."

The relative-mode GPU run shows connected guidance, but does not establish visible dust improvement. The user reports that the images look the same. The supplied comparison JSON contains metrics and paths, not image pixels. A later four-way PNG was inspected separately; see the follow-up below.

## Current recommendation and implementation status

For improving the actual car image next, prototype **masked SDXL inpainting as a
second stage**. For improving FuzzyDiff's core control of "slightly", investigate
a **validated visual dust scorer and image-based guidance loss**. These are
different experiments: inpainting is the more direct proposed route to a visible
surface edit, while a calibrated dust objective requires reference data and
validation first. Neither approach has been implemented or tested in this project.

The current code includes experimental relative membership, corrected runner
settings, four-condition reporting, and a figure renderer without CLIP score
labels. CPU checks passed, but a visual dust-intensity improvement remains
unestablished. Updating this handoff does not implement either proposed extension.

## Evidence

Source: `C:/Users/sunzi/Downloads/comparison (3).json`.
Archived copy: `outputs/dust_relative_seed142_analysis/comparison.json`.
SHA-256: `90c34b663511832d4a93075052c508dc2dea69da23307ace39b10b829965dbbc`.

SDXL only, seed 142, 768 x 768, 50 steps, CFG 9.5, relative membership, product t-norm, dust-to-car binding weight 1.0. Base stage only; no refinement.

| Run | Full-prompt CLIP | Difference from fuzzy off | Dust attention truth |
| --- | ---: | ---: | ---: |
| Native SDXL | 0.298965 | +0.000006 | unavailable |
| Our SDXL pipeline, fuzzy off | 0.298959 | +0.000000 | 0.969600 |
| Our SDXL pipeline, fuzzy on, lr=0.2 | 0.301118 | +0.002159 | 0.970065 |
| Our SDXL pipeline, fuzzy on, lr=5 | 0.290086 | -0.008873 | 0.970515 |

The lr=0.2 arm has the highest full-prompt CLIP in this one pilot. That small score difference is not evidence of visible dust or of a general method win. At lr=5, CLIP is lower despite higher dust attention truth.

Both guided arms made 30 updates; none were clipped. Mean update/latent RMS ratios:

- 03_guidance_lr_0.2: 0.000038979.
- 03_guidance_lr_5: 0.001084462.

The four CLIP fields within each run are identical. The current scorer splits at " and "; this prompt has no such split, so all four fields score the entire prompt. There is no independent dust-specific score in this file. Relative membership normalizes every nonzero map by its own peak, so a dust truth around 0.97 does not establish visible dust.

## Requested four conditions

1. Our SDXL pipeline with fuzzy guidance (`sdxl_fuzzy`).
2. Our SDXL pipeline with fuzzy guidance off (`sdxl_plain`).
3. Native SD 1.5 backbone (`sd15_plain`).
4. SD 1.5 with our fuzzy pipeline (`sd15_fuzzy`).

The provided seed-142 JSON contains no SD 1.5 results. These four conditions require `--four-way`, not `--compare`. All four-way arms use the base stage only; they do not compare the complete refiner stage. The requested order is used for display. Metric rankings remain computed from the measured min-part CLIP values.

## Kaggle run

Transfer the updated repository to Kaggle before running:

```python
%run run_fullpipeline.py --four-way --prompt "A slightly dusty red sports car parked on a road." --words "slightly dusty,red sports car,road" --seed 142 --membership-mode relative --tnorm product --lr 0.2 --bind "slightly dusty>red sports car" --output /kaggle/working/dust_four_way_relative_seed142
```

Outputs: four individual PNGs, `four_way.png`, `four_way.json`, and `four_way_results.md`.

The rate 0.2 is chosen from this pilot; treat seed 142 as tuning evidence. Evaluate the frozen setting on additional preselected seeds and retain all outputs. Use blind visual ratings of dust visibility, dust degree, and scene fidelity. Neither CLIP nor the optimized attention objective supplies calibrated dust intensity.

A desired first-place ranking is a hypothesis to test. If dust remains visually absent, report the limitation and investigate a visually grounded intensity objective; reordering panels does not resolve it.

## Follow-up: similar images and figure presentation

The user subsequently supplied `C:/Users/sunzi/Downloads/four_way (7).png`.
Visual inspection showed very similar car/scene compositions within each backbone
pair. The user also observed dust in the plain outputs. This PNG is separate
evidence from `comparison (3).json`; its rounded figure scores must not be treated
as that earlier run's measurements or used to infer missing generation settings.

The requested presentation is to retain the fuzzy outputs and remove CLIP scores
from the final figure. `four_way.py::save_grid` now shows condition names without
numeric score labels. The saved JSON and results table still contain scores.
The existing local PNG was inspected, not edited; redraw the saved individual
images with the updated renderer to get a score-free figure without regeneration.
Validation after this renderer change: all 51 CPU tests passed.

The user also requested clean plain images. For the experimental comparison,
retain the actual plain outputs: both arms receive the same dusty-car prompt, so
dust in a baseline is a legitimate outcome. Any cleaned reference must be a
separate, explicitly labeled edited illustration or a separately labeled
clean-prompt condition. Selectively cleaning a baseline cannot demonstrate a
benefit of fuzzy guidance.

Why the existing outputs look similar:

- A matched seed starts both arms from the same noise, preserving comparability.
- Fuzzy guidance changes an attention-derived objective through relatively small
  latent updates; it does not directly measure dust on the rendered car.
- Relative normalization addresses the extra token-share competition between
  attribute and object, but removes absolute attention strength. Weak nonzero
  attention can therefore receive high normalized membership.
- Raising the learning rate alone has not resolved this. In the earlier seed-142
  comparison, `lr=5` raised dust attention truth while lowering full-prompt CLIP.

The desired improvement is a visible thin dust layer while retaining red paint,
car identity, geometry, and scene fidelity. A different-looking image or a higher
attention truth is insufficient evidence of that improvement.

## Proposed next experiment: masked SDXL inpainting

Purpose: test whether a targeted second-stage surface edit produces visible light
dust while preserving the existing car and scene. This is a recommendation for a
prototype, not an established best-performing method or a promised result.

1. Preserve the original base image and its generation metadata.
2. Create and inspect a mask covering the hood, roof, and relevant body panels.
   Exclude background and unrelated car details where possible. A manually
   verified mask is sufficient for an initial pilot; automatic masking needs its
   own validation.
3. Run an SDXL inpainting stage using the image, mask, and this proposed prompt:

   > A thin, visible coating of pale road dust on the red paint, faint dust
   > accumulation around panel seams and lower doors, red paint clearly visible
   > beneath the dust, no thick dirt or mud.

4. Start with a modest editing strength and assess a small, recorded pilot sweep.
   No dust-specific strength, checkpoint, or schedule has been validated here.
   Inspect car identity, panel geometry, mask boundaries, and unintended scene
   changes as well as dust visibility.
5. Save the mask, source image, edited image, original and editing prompts,
   checkpoint/version, seed, strength, steps, and other settings in a new output
   directory. Keep the original run available for comparison.

Label the new condition **FuzzyDiff + masked inpainting**. This introduces an
editing stage and a more explicit conditioning prompt; its improvement cannot be
attributed to the existing fuzzy objective alone.

To isolate contributions within a backbone, compare the same preselected seeds
under these four conditions:

| Fuzzy guidance | Inpainting | What the condition tests |
| --- | --- | --- |
| Off | Off | Base generation |
| On | Off | Existing fuzzy guidance |
| Off | On | Editing without fuzzy guidance |
| On | On | Combined method |

Keep editing prompts, checkpoint, settings, and mask-selection rules consistent
between edited arms. Use appropriately aligned masks for each source image.
This is an additional stage ablation, distinct from the existing four-way
SDXL/SD 1.5 backbone comparison. Improved edited images would support the combined
method only to the extent shown by these controls.

Technical reference: [Diffusers inpainting documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint).
The documentation supports mask-based SDXL editing; it does not demonstrate
performance on this project's dusty-car prompt.

## Longer-term method extension: visual dust guidance

Purpose: make the optimization respond to rendered dust rather than only token
attention. This component is currently missing.

1. Define clean, lightly dusty, and heavily dusty appearances using reference
   images and human judgments. "Slightly" should mean visible light dust with
   paint clearly showing through; the target range needs empirical calibration.
2. Build or select a dust scorer operating on the car region. Validate its ability
   to distinguish the levels on held-out reference images before using it for
   guidance. Check confusion with lighting, reflections, paint texture, and road
   color. A generic CLIP score is not automatically a calibrated dust scorer.
3. Prototype an optional visual loss on a decoded image estimate or a dedicated
   correction stage, targeting the light-dust range alongside existing attention
   constraints. A gradient-based implementation requires a differentiable path to
   the image/latents and measured GPU memory/runtime feasibility.
4. Compare against unchanged baselines using frozen settings and multiple
   preselected seeds. Assess visible dust, the requested degree, car/scene
   preservation, and artifacts with independent blind ratings.

The immediate prerequisite for this research path is a scorer that passes the
held-out visual calibration check. Keep its optimization signal separate from
the independent evaluation used to claim improvement. A first-place ranking is
an experimental outcome, not a requirement to encode in generation or reporting.

## Resumption guidance

Read `AGENTS.md` before implementation. Keep fuzzy logic in the canonical notebook
and update dependent loaders if cell-leading definitions change. Follow the
project's dependency and core-method approval requirements for any extension.
The user requested documenting these recommendations; do not interpret this
handoff update as approval of a particular inpainting implementation, new model
download, or calibrated-loss design. No new GPU experiment was run for this update.
