# Snow pilot findings

Evidence inspected: user-provided `manifest.json` and `alpine_seed_142 (2).png`.
This is one alpine scene, one seed (142), five prompt levels, and three methods.

## What completed

All 15 runs are marked complete. The manifest records 15 distinct image hashes;
this does not replace verification of the original image files, which were not
provided. FuzzyDiff recorded all 30 guidance updates at each prompt level.
The reported gradient RMS values are nonzero. Guidance was executing.

## What the grid shows

Native SDXL and FuzzyDiff look very similar within each row. Refinement changes
some peak details, composition and sky appearance. The very slight and slight
prompts still produce extensively snowy mountains. This pilot does not show a
clear five-level progression from little snow to heavy coverage. This is a
qualitative inspection, not completed blinded human ratings.

Across the five FuzzyDiff runs, per-step update RMS ranges from approximately
2.4e-6 to 7.8e-6. The current loss strengthens tracked token presence and has no
level-specific snow-coverage target. These findings explain why reliable graded
control should not be expected from changing only the severity wording.

## Descriptive summaries

| Method | Mean full-prompt CLIP | Mean successful job time |
| --- | ---: | ---: |
| Native SDXL | 0.299290 | 32.7 s |
| FuzzyDiff | 0.299787 | 57.9 s |
| FuzzyDiff + refiner | 0.299926 | 19.1 s for refinement only |

Refinement requires its FuzzyDiff base first. The last timing column excludes
model loading and failed attempts. These CLIP differences do not establish a
snow-coverage benefit. Native/custom also differ in latent precision and
attention implementation, so their difference does not isolate fuzzy guidance.

## Next direction

The user has chosen mountain/snow scenes in the reference's dense 9-column by
11-row layout. Keep the original pilot and its manifest as baseline evidence.
Do not fill 99 cells by duplicating or relabeling these 15 outputs. Two meaningful
grid axes and the new control mechanism must be specified before a new run;
layout changes alone cannot create graded snow control.
