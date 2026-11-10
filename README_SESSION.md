# Session notes, 2026-11-10: why Stage 1 did not change the images

Branch `fuzzysd3`. These notes record one working session: what was asked, what was found, what was
changed, and what is still open. All runs used provisional calibration, so nothing here is paper evidence.

## 1. Starting problem

Experiment iii (notebook cell 15a, Table 4) gave nearly the same image for SD3, Stage 1 and Stage 1 + SAR
on 10 size prompts ("a kettle extremely bigger than a shoe", ...).

## 2. Diagnosis of the first run (`exp3`, default settings)

Read from `all.jsonl`, the per-job `meta.json` files and the images.

| Finding | Evidence |
|---|---|
| Guidance started too late | Only steps 14-24 of 28 were guided (sigma 0.74 to 0.28); `cfgSigmaHi` was 0.75 |
| Each step was tiny | The peak limit (`cfgMaxPert` 0.2) held steps to 0.003-0.01 RMS; about 0.06 in total, against a latent RMS of about 1 |
| Pixels barely changed | Stage 1 differed from SD3 by 2.5-6 of 255 per pixel on average; bench/pillow was identical |
| Score moved without the picture | Bowl/mug: size score -0.01 to 0.56, membership 0.001 to 1.0, on a picture with no mug |
| SD3 often omits the second object | Radio (2 prompts) and mug missing; vase and mirror barely visible. The size constraint was off on 40 of 110 guided steps |

## 3. Experiments

### 3a. Guidance from the first step, 0.05 RMS per step (cell 15d, first version)

Result: flat, saturated shapes instead of photos. The latent was pushed by about 1.4 in total, more than
its own size.

### 3b. One prompt, four settings (cell 15d.0, `exp3_one`)

Prompt: "a kettle extremely bigger than a shoe". Size score = log(area ratio); target 0.95 to 1.90; SD3
scores 0.44.

| Setting | Guidance starts | Latent moved (total RMS) | Size score | Pixel change vs SD3 (0-255) | Image |
|---|---|---|---|---|---|
| `rms0.01` | first step | 0.24 | 0.93 | 53 | Kettle bigger, shoe smaller, but wall blown out to white, objects flattened |
| `rms0.02` | first step | 0.34 | 1.01 | 84 | Layout wrecked: tall boot larger than a small kettle |
| `rms0.02_hi0.9` | sigma 0.9 (step 7) | 0.30 | 0.87 | 16 | Normal photo; kettle somewhat bigger, shoe somewhat smaller |
| `rms0.03` | first step | 0.45 | 0.12 | 95 | Flat black-and-white graphic |

Conclusions:

- The damage comes from the first steps, not from the total push. At steps 0-1 the clean prediction is a
  blur, the objects are not found (areas 0.002), presence reads about -30 to -39, and the push follows
  the presence loss alone (loss 71).
- Starting at sigma 0.9 with 0.02 RMS per step is the one setting that kept the photo and moved the size
  relation the right way. The change is modest and stops short of the target.
- A score inside the target does not mean a good image (`rms0.02`). The sheet has to be looked at.
- One prompt with one seed shows the method can change the picture, not that it does so reliably.

### 3c. Ten prompts with the candidate setting (cell 15d, current version)

*Superseded: see section 10.*

Running on Kaggle at the time of writing. Settings: `--sigma-hi 0.9 --clip rms --dmax 0.02 --eta0 0.04`
and a stronger `--dmax 0.03 --eta0 0.06`. Output: `exp3_layout.zip`. Not yet analysed.

## 4. Changes made in this session

| Commit | What |
|---|---|
| `d627b60` | `code/layout_check.py` and `code/test_layout_check.py`: per setting, how far the latent moved, the size score, the pixel change to SD3, the share of saturated pixels, a first-reading verdict, `steps.csv` and a comparison sheet. Notebook cell 15d.0 (one prompt, four settings). 15d changed to 0.01 and 0.02 RMS per step |
| `8b47c3b` | 15d changed to the setting that worked on one prompt (from sigma 0.9) and a stronger one |

Not changed: the method code (`guidance.py`, `constraints.py`, `measure.py`, `calibration.py`, `sar.py`).
The decision was to follow the methodology of the paper strictly and vary only the settings it leaves
open.

Tests run on this machine (no torch, no GPU): `test_layout_check.py`, `test_reporting.py`,
`test_prompts.py`, `test_predicates.py` pass. The guidance, estimator and backend tests were not run here.

Git setup of this clone:

- `origin`: original URL, signs in with the saved `sunzidiautomation` credential, which GitHub refused
  write access to (403).
- `new-origin`: same repository, signs in as `sunzidulislam`. Pushes go here:
  `git push new-origin fuzzysd3`.
- The branch still tracks `origin`. The repo-local commit identity was set to `sunzidulislam`; commit
  `d627b60` was made before that and is authored as `sunzidiautomation`.

## 5. Code checked against the method (section E, guided sampling)

The per-step loop matches the method steps: clean prediction, frozen constraint set, Charbonnier losses per
tier, one backward pass per tier, cone projection, RMS-normalised step with a uniform limit, acceptance
(domain first, then tier-1/2 tolerances and total loss), at most two halvings, Euler update, attention
prior from the accepted latent, size-regime order.

Where the runs depart from the method (calibration, not the loop):

1. Validity flags are not fitted. Provisional calibration marks every family valid in every noise bin, so
   `--sigma-hi` is a manual substitute for the missing flags.
2. No presence model for the noisiest bin (sigma above 0.75); bin 1 silently uses bin 2's
   (`calibration.py:105`). The candidate setting guides seven steps in that bin.
3. Which quantity the step limit s_t bounds is not settled: the default bounds the largest single latent
   value, the setting that worked bounds the RMS. The manuscript's step equation is needed.
4. Size quantiles (`c68` 0.1, `c95` 0.2) and hedge centers are placeholders. With them, "barely" is
   unmeasurable (never guided) and "slightly"/"somewhat" are direction-only (any ratio above about 1.22
   satisfies them), so 4 of the 10 size prompts cannot show graded control.
5. The same hedge can be scored against two targets, depending on whether both objects ever passed their
   gates (pumpkin/vase against ">= 0.2", toaster/radio against 0.15-0.30).
6. `sigma_hi` and the negative prompt are in the code but not in the method summary.
7. The size regime is fixed at the first step where both objects pass their gates, not strictly at t0.
8. The size score is the mean of a soft segmentation map (`estimators.py:90`), so it can change when the
   mask's confidence changes without the object changing size.

## 6. Code review of the whole branch

Two reviewers read the tree at `8b47c3b`; nothing was run, and the individual findings were not all
re-verified.

### Standards (against `code/README.md` and the notebook's rules)

- "configs/ is the single source of all settings": `BINS = 4` hardcoded in `guidance.py:22` and
  `sar.py:26`; `sar.py:177` runs the fallback check with default gate values while
  `run_experiment.py:290` passes the configured ones; `sar.py:206` and `negative.py:15` duplicate
  settings; several tuned constants live in code.
- "All logic lives in code/": notebook cell 64 (15d) builds sheets inline, repeating
  `layout_check.py`; cell 28 re-implements `image_id`.
- "reporting.py refuses provisional records": `annotate.py:518`, `annotate.py:549` and `ladder.py:326`
  produce LaTeX rows without that check.
- README module list omits eight modules.
- Smells: duplicated code (`annotate.py:306-321`, `guidance.py:292-312`, two `read_jsonl`), arms as
  bare strings across seven tables, single-letter evidence keys, a 17-parameter `_log_step`, unused
  generality, `annotate.py` and `guided_step` doing several jobs.

### Spec (against the method steps A-G)

Missing or partial:

- Gradient-validity result is never enforced: the manifest is only checked for existence and `valid()`
  ignores it, so a failing family would still be guided.
- No one-step-lag SAR arm, no harm-rate metric, size regime never logged.
- No per-stratum coverage report, no admissible-domain table builder.
- SAR confidence weighting and softmax competition are coded but never switched on.
- Attention thresholds are produced for objects only, with one configuration.
- Relation types are not parsed; every pair defaults to exclusive.
- Annotations come from final images only and are joined to every noise bin.
- Fitting rejects only the test split; nothing ties a fit to G_cal or G_cons.
- Several calibration inputs have no builder.

Looks wrong:

- The widening factor gamma is scored on the training folds, not the held-out fold
  (`calibration_fit.py:214-218`).
- kappa_eff is baked in from a hand-supplied `sigma_m`, and size adds `c68` again
  (`constraints.py:305`).
- Fitted plateaus are dropped (`fit_calibration.py:172`).
- Mixtures, leakage and prototypes use OKLab; the method says CIELAB.
- Size quantiles are fitted in-sample and pooled over bins.
- Stage-2 validity is not made contiguous.
- SAR drift thresholds use the current support's size stratum, enabling uses the reference's.

Not asked for: SAR rollback and the final fallback check, a sixth prompt group `sys`, the placebo,
no-check, negative-prompt and prompt-weight arms, the sweep overrides.

Confirmed matching: SAR step order, reuse of the unconditional output, re-noising, permanent disable and
suspension; the 4-of-5 fold rule; statistics on raw maps; size-regime order.

## 7. What to do next

*Superseded: see section 10.*

1. Let 15d finish and analyse `exp3_layout.zip`: on how many of the 10 prompts does the photo survive
   and the size relation move?
2. Get the manuscript's guided-step equation to settle peak versus RMS for the step limit.
3. If 15d holds: rerun 15a with that setting for all three rows and rate the images blind (`size.html`).
4. If it does not: the next step is the calibration work, not more tuning.
5. Before anything goes into the paper: annotations and fitted calibration (method sections B and D),
   more prompts and seeds, on test prompts that were not used to choose the setting. Fix first:
   enforcement of the gradient-validity result, the in-sample gamma fit, OKLab versus CIELAB, the dropped
   plateaus.

## 8. Suggestions for later

Ordered by what they unblock.

**Decide whether the method is promising (cheap, do first)**

- Judge 15d by looking at the sheets, not the scores. Count the prompts where the image is still a photo
  and the size relation visibly moved toward the prompt.
- If only the stronger setting moves the size and it also damages images, try values between the two
  (for example 0.025 from sigma 0.9) on one prompt with cell 15d.0 before another 10-prompt run.
- Run two or three seeds on the prompts that worked. One seed cannot show reliability.
- Use prompts whose hedge has a graded target with the current values ("moderately", "considerably",
  "nearly", "extremely"). "barely", "slightly" and "somewhat" cannot show graded control until the size
  quantiles are fitted.
- Keep the SD3-with-negative-prompt control (`unguided_neg`) whenever a negative-prompt arm is reported,
  and say in the paper that only our rows use a negative prompt.

**Make the runs follow the method (needs data, not tuning)**

- Do the annotation and calibration steps (method sections B and D). Every run so far uses placeholders
  for presence, validity flags, size quantiles and hedge intervals.
- Fit a presence model and validity flags for the noisiest bin (sigma above 0.75). Without them, guiding
  from sigma 0.9 is outside what the method allows, and `--sigma-hi` stays a hand-set stand-in.
- Settle from the manuscript whether the step limit bounds the largest latent value or the RMS, then make
  that the default and remove the other option.
- Decide whether `sigma_hi` and the negative prompt belong in the paper. If they stay in the code, the
  manuscript has to describe them.

**Fix before any number goes into the paper (calibration and SAR code)**

- Enforce the gradient-validity result: a family that fails must get no degree guidance.
- Fit the widening factor gamma on the held-out fold.
- Use CIELAB for mixtures, leakage and prototypes, or change the manuscript to say OKLab.
- Keep the fitted plateaus instead of dropping them.
- Fit size quantiles out of fold and per bin; avoid adding `c68` twice in kappa_eff.
- Make Stage-2 validity contiguous.
- Log the size regime per prompt (needed for the "size-regime shares" the method asks to report).
- Add the one-step-lag SAR arm and the harm-rate metric.
- Score the same hedge against one target, whether or not both objects passed their gates.

**Measurement limit to state in the paper**

- The size score comes from a soft segmentation mask and can move without the object changing size.
  Report human ratings for size, and treat the automatic score as supporting data only.

**Housekeeping (no effect on results)**

- Read `BINS` and the step count from the config instead of hardcoding them; make the SAR fallback check
  use the configured gate values.
- Move the 15d sheet code from the notebook into `layout_check.py`.
- Add the missing modules to `code/README.md`.
- Point the `fuzzysd3` branch at `new-origin`, or give `sunzidiautomation` write access, so a plain
  `git push` works.

## 9. Questions asked in this session, with the short answers

| # | Question | Short answer |
|---|---|---|
| 1 | All images look the same; why is my method not working? Fix it. | Guidance started at sigma 0.75 and moved the latent about 0.06 in total. The `exp3` run used the default settings, not the opt-in fix. See section 2 |
| 2 | Do not change the method; follow the methodology of the paper strictly. | Agreed. Only the settings the paper leaves open were varied; the method code is untouched |
| 3 | I got these images for Stage 1 (flat orange and grey shapes). | Opposite failure: 0.05 RMS per step from the first step is far too strong. They came from `rms0.05` |
| 4 | Fix all problems; I want to see the data for one prompt first, whether it works. | Added cell 15d.0 and `layout_check.py`. See section 3b |
| 5 | Create a new origin for the `sunzidulislam` account; name it `new-origin`, keep the other as `origin`. | Done. See section 4 |
| 6 | (One-prompt results, `exp3_one`.) | One of four settings works: from sigma 0.9 at 0.02 RMS per step. The first steps do the damage |
| 7 | Can one prompt produce sufficient results? | No. It is one sample on the easiest case, tuned on that same prompt, and it fell short of the target |
| 8 | If I run 15d, what happens? | SD3 plus Stage 1 under two settings on 10 prompts, about 60 minutes per setting, one three-panel sheet per prompt, zipped as `exp3_layout.zip`. Stage 1 only, no rating |
| 9 | It is running; is the internal code okay? | The settings are applied as intended and the logged numbers are consistent. Not verified here: the torch tests and SAR |
| 10 | Check the code against the paper. | No bug in the guided step. The paper was not in the repo; four issues found from code and logs. See section 5 |
| 11 | (Method steps A-G pasted.) | Section E matches the code. The gap is sections B and D: no annotations and no fitted calibration |
| 12 | Code review of all commits. | See section 6 |
| 13 | What do I need to do now? | Wait for 15d and send `exp3_layout.zip`; send the guided-step equation. See section 7 |
| 14 | Write this file, with the suggestions and the questions. | This file |

## 10. Update: the 10-prompt result, the Table 2 rerun, and the steps to run

This section is later than sections 3c and 7 and replaces their "not yet received" and "what to do next" parts.

### 10a. Ten prompts with the candidate setting (cell 15d, `exp3_layout`)

Setting: `--sigma-hi 0.9 --clip rms --dmax 0.02 --eta0 0.04` (`rms0.02_hi0.9`), compared with SD3 on the same seed.

**Result: the setting changes the picture without breaking it, but the real gain over SD3 is modest: clearly better on about 3 of 10 prompts, the same on the rest. Nothing got worse.**

| Prompt | What changed | Better than SD3? |
|---|---|---|
| a pumpkin slightly bigger than a vase | a vase now stands behind the pumpkin; SD3 had only a tiny one in the background | yes |
| a puppy considerably smaller than a stool | stool larger and fully in view, puppy slimmer | yes |
| a kettle extremely bigger than a shoe | shoe is shorter; kettle about the same | slightly, not "extremely" |
| a bench moderately larger than a puppy | bench a little more prominent | marginal |
| a blanket somewhat bigger than a drum | drum redrawn, relation was already right | same |
| a kitten nearly larger than a mirror | small layout shift | same |
| a bowl moderately larger than a mug | still one bowl-mug hybrid, no second object | same |
| a toaster slightly bigger than a radio | still no radio | same |
| a drum extremely bigger than a radio | still no radio | same |
| a bench barely bigger than a pillow | pixel-identical; no step accepted, the target was already met | same |

Per image at this setting: 0 to 17 of 18 guided steps accepted, latent moved 0.00 to 0.28 (summed RMS), mean pixel change against SD3 0 to 22 of 255, saturated share at most 0.07.

What the numbers hide:

- **The method's own score overstates the gain.** It says 10 of 10 guided images meet the size target against 4 of 10 for SD3. It counts the toaster and the bowl/mug images as satisfied (second object "present" with membership 1.0), and neither picture shows a second object. These scores are not evidence for the paper; the blind rating has to carry the claim.
- **One prompt is scored against two targets.** For "a toaster slightly bigger than a radio" the SD3 image is judged against 0.15 to 0.30 and the guided image against "above 0.20" (the size regime resolved differently in the two runs), so the guided image passes at 3.92 where SD3 fails at 1.82.
- **Stage 1 cannot add an object SD3 left out** at this strength (toaster/radio, drum/radio, bowl/mug). This is where the missing presence model for the noisiest bin matters.

The stronger setting `rms0.03_hi0.9` (`--dmax 0.03 --eta0 0.06`) was no better on size and added artifacts: coloured patches on both drum images, distorted legs on both puppies, and the kitten prompt dropped to membership 0.39. It is dropped.

### 10b. Table 2 rerun that could not be used (`ablation_mix2`, `ablation_size2`)

Cell 14d had been run at commit `49be347`, before the fixes, with `CHOSEN = 'current'` (every `meta.json` has `overrides: {}`), which is the provisional setting: guidance from sigma 0.75 with the peak step limit.

- **All 14 arms gave the same picture on all 10 prompts.** Stage 1 and its eight ablations moved the latent 0.02 to 0.08 per size prompt (up to 0.15 for colour) and changed the image by 1.5 to 7.6 of 255 on average. The one-prompt test needed about 0.30 before the picture changed.
- **Stage 2 arms:** 7 to 15 of 255 pixel change, which is re-rendered texture from the img2img pass. SAR, placebo priors and no-consistency-check gave the same scores to two decimals on almost every prompt, so the SAR bias had no measurable effect. In 3 of 10 prompts the SAR output was discarded and the Stage 1 image returned (reason "attribute").
- **`abl_floored`** accepted 0 of 11 steps on three size prompts and returned the SD3 image.
- **`abl_attention_only`** scored exactly 0.00 with presence 0.00 on every image. Cause: the arm guided normally, but scoring the finished image has no attention maps, so its supports were empty.
- **The screened size prompts were the wrong kind of failure.** All five had an object the scorer judged absent (toaster only, vase only, a man with a drum, two suitcases, a clock-shaped pillow). A missing object gates the size constraint off, so there was nothing for an ablation to change. Of the colour prompts, two of five had both objects and a failing colour constraint.

These pages must not be rated.

### 10c. Fixes made after that

| Commit | Change |
|---|---|
| `8ab2cce` | `code/screen.py`: new `--objects-present` flag, keeps only prompts where SD3 drew every object and another constraint fails (function `keep`). `code/measure.py`: the attention-only arm scores a finished image with CLIPSeg like every other arm. Notebook 14a screens with the new flag; 14d passes the fixed setting (size: `--sigma-hi 0.9 --clip rms --dmax 0.02 --eta0 0.04`; colour: `--clip rms --dmax 0.02 --eta0 0.04`, start stays at 0.75) and writes to `ablation_mix3` / `ablation_size3`; 14e reads from them. |
| `2dfa94d` | Notebook 15a, 15b, 15c: Table 4 runs with the same size setting and writes to `exp3_v2` (the old `exp3` would have been resumed and kept its identical images). |

Both are pushed to `new-origin` (`fuzzysd3`).

Not verified: the screening rule is tested on this machine, but the `measure.py` fix and its new test (`test_attention_only_scores_a_finished_image_with_the_segmenter`) could not run here (no torch). The colour setting in 14d is untested. With the new rule, 14a may keep fewer than 5 size prompts.

### 10d. Steps to run on Kaggle

1. **Stop** the 15d cell if it is still running.
2. **Import the new notebook**: File -> Import Notebook -> GitHub, repo `sunzidulislam/fuzzydiff`, branch `fuzzysd3`, file `notebooks/fuzzydiff_kaggle.ipynb`. GPU T4 x2. Pulling the code alone is not enough, the cells changed.
3. **Run the setup cells.** The clone cell's last line should print `2dfa94d`.
4. **Run the two tests** in a new cell: `!python -I code/test_ablation.py && python -I code/test_sweep.py`. Every line should start with `ok`.
5. **Run 15a** (Table 4), about 2 to 3 hours; a rerun resumes.
6. **Run 15b** (automatic columns).
7. **Rate** `exp3_v2/size.html` from `exp3_v2.zip`. Do not open `size.key.json` first. Press "Show ratings" and copy the text.
8. **Run 15c** with the text pasted in place of `PASTE HERE`. It prints the Table 4 rows.
9. **Run 14a** (scree ning), about 1 hour. If it keeps fewer than 5 size prompts, stop and look at the output.
10. **Run 14d** (Table 2), several hours per experiment; a rerun resumes. Download `ablation3.zip`.
11. **Rate** `ablation_mix3/ablation_mix3.html` and `ablation_size3/ablation_size3.html`.
12. **Run 14e** with the two rating texts pasted into `MIX_RATINGS` and `SIZE_RATINGS`. It prints the Table 2 rows.

Notes: Table 4 comes first because it answers the main question (better than SD3 or not) and is the shorter run. Do not run 15e/15f: they add a negative prompt, which is not in the paper's method. Download each zip before the session ends.

### 10e. What to expect

Nothing so far shows a large advantage over SD3. Expect the blind rating to show a small one. If it does not, a stronger setting will not help (0.03 already adds artifacts); the next step is the calibration the method specifies, starting with a presence model and validity flags for the noisiest bin (section 8).

### 10f. Questions asked after section 9

| # | Question | Short answer |
|---|---|---|
| 15 | Read `results (3)` for Table 2. | Unusable: old code, provisional setting, all 14 arms identical. See 10b |
| 16 | Fix the issues, and what do I run to get better images than SD3? | Three fixes (10c). No guarantee of "better" yet |
| 17 | Is the partly finished 15d folder enough for testing? | Yes, the 0.02 setting was complete |
| 18 | Check `results (5).zip`. | About 3 of 10 prompts clearly better, none worse. See 10a |
| 19 | (15d printout pasted.) | Same data; it is the method's own scorer and overstates the gain |
| 20 | Step-by-step, what do I do? | See 10d |

## 11. Update: the Table 4 run, a model-made rating, the 400-prompt question, and how to reduce size violations

### 11a. Table 4 run (`exp3_v2`, cells 15a and 15b)

SD3, S1 (Stage 1) and FuzzyDiff (Stage 1 + SAR) on the 10 development size prompts, seed 0, setting `--sigma-hi 0.9 --clip rms --dmax 0.02 --eta0 0.04`, code at `2dfa94d`.

**Result: Stage 1 is visibly better than SD3 on about 4 of 10 prompts and no worse on the others. SAR adds nothing, so the FuzzyDiff row is about equal to the S1 row.**

| Prompt | S1 vs SD3 | FuzzyDiff vs S1 |
|---|---|---|
| a pumpkin slightly bigger than a vase | better: a vase now stands behind the pumpkin | texture only |
| a puppy considerably smaller than a stool | better: stool larger and in full view; two small stray blobs | identical |
| a drum extremely bigger than a radio | better: a radio-like box appears top left | texture only |
| a bench moderately larger than a puppy | slightly better: bench more prominent | identical |
| a kettle extremely bigger than a shoe | marginal: shoe a little smaller | identical |
| a kitten nearly larger than a mirror | same | identical |
| a blanket somewhat bigger than a drum | same (already right) | texture only |
| a bowl moderately larger than a mug | same: still one bowl-mug hybrid | texture only |
| a toaster slightly bigger than a radio | same: still no radio; an antenna and a blue patch appear | identical |
| a bench barely bigger than a pillow | pixel-identical | texture only |

- **SAR:** in 5 of 10 prompts its output was rejected by its own check and the Stage 1 image returned unchanged (reasons "attribute" four times, "spatial" once). In the other 5 it only re-rendered texture.
- **Automatic columns:** text-image similarity 0.317 (SD3), 0.315 (S1), 0.319 (FuzzyDiff); text-text 0.791, 0.795, 0.790; every confidence interval of a difference includes zero. CLIP-IQA 0.866, 0.894, 0.881; S1 minus SD3 is +0.028 with interval 0.002 to 0.057, so guidance did not damage the pictures.
- **Guided runs are not exactly repeatable.** The same prompt, seed and setting gave slightly different Stage 1 images here than in 15d (bowl/mug membership 1.00 there, 0.49 here; toaster score 3.92 there, 3.56 here). One seed per prompt is weak evidence.
- **Time per prompt:** SD3 39 s, Stage 1 367 s (231 to 478), SAR 37 s, about 7.4 minutes for the three rows.

### 11b. Rating made by Claude (an AI model), at the user's request

**These are not blind human ratings and cannot fill the paper's human columns.** The rater knew which image came from which method, and is not a human annotator. They could only be reported as a model-based rating, labelled as such. Use them as a preview of what a blind rating is likely to show.

| Row | Missing objects (lenient / strict) | Size violated | Fidelity (0-100) | Text-image / text-text | CLIP-IQA |
|---|---|---|---|---|---|
| SD3 | 30% / 30% | 90% | 49.0 | 0.32 / 0.79 | 0.866 |
| S1 | 30% / 30% | 60% | 56.5 | 0.31 / 0.80 | 0.894 |
| FuzzyDiff | 30% / 30% | 60% | 56.5 | 0.32 / 0.79 | 0.881 |

Paired over the 10 prompts (scored with `score_mixture`, the project's own scorer):

- Size, S1 vs SD3: 30 points fewer violations; better on 3, worse on 0, same on 7; 95% interval -60 to 0 (touches zero).
- Fidelity, S1 vs SD3: +7.5; better on 5, worse on 2, same on 3; interval 1.0 to 15.0.
- Missing objects, S1 vs SD3: no difference; one prompt gained an object (drum/radio) and one lost one (kitten/mirror).
- FuzzyDiff vs S1: zero difference on everything.

Rule used: "size" is whether the relation holds as worded, including the hedge; a missing object counts as violated.

| Prompt | SD3 | S1 and FuzzyDiff |
|---|---|---|
| pumpkin slightly bigger than a vase | vase is a tiny figurine: size no, fidelity 45 | vase beside the pumpkin, slightly smaller: yes, 80 |
| puppy considerably smaller than a stool | puppy as big as the visible stool: no, 55 | stool clearly larger: yes, 70 |
| bench moderately larger than a puppy | puppy dominates: no, 50 | bench larger: yes (borderline), 62 |
| drum extremely bigger than a radio | no radio: no, 35 | a radio-like box appears, not "extremely" smaller: no, 50 |
| kettle extremely bigger than a shoe | no, 55 | shoe a little smaller, still not "extremely": no, 60 |
| kitten nearly larger than a mirror | mirror present: no, 45 | looks like a tablet, not a mirror: no, 40 |
| toaster slightly bigger than a radio | no radio: no, 30 | no radio, small artifact: no, 28 |
| bowl moderately larger than a mug | one hybrid object: no, 30 | same: no, 30 |
| blanket somewhat bigger than a drum | blanket far bigger than "somewhat": no, 60 | same: no, 60 |
| bench barely bigger than a pillow | yes, 85 | yes, 85 |

Three ratings are borderline and move the totals: the radio-like box in the drum image, the tablet-or-mirror in the kitten image, and the bench/puppy "yes". Flipping the bench/puppy one alone changes S1 from 60% to 70% violated.

For the paper: rate `exp3_v2/size.html` by hand without opening `size.key.json`, then run 15c.

### 11c. Can this be done for 400 prompts?

Yes, with two limits: GPU time, and who rates 1,200 images.

| Prompts | GPU time (7.4 min per prompt) | Kaggle sessions (12 h limit) |
|---|---|---|
| 100 | about 12 hours | 1-2 |
| 137 | about 17 hours | 2 |
| 400 | about 49 hours | 5 or more |

- Kaggle's free GPU quota is about 30 hours a week, so 400 prompts is about two weeks on one account. Runs resume.
- The development group has only 137 distinct size prompts; the runner refuses to draw more. The test group has 557, so 400 is possible only there.
- The test group should be run once, after calibration. Running it now means the final Table 4 is on prompts already looked at.
- Nobody can hand-rate 1,200 images. At that size the size relation has to be measured automatically; the repo has an independent detector for it (OWLv2 in `code/ladder.py`), plus a hand check on about 30 random prompts.

Recommendation given: 137 development prompts now, scored with OWLv2 plus a 30-prompt hand check; 400 on the test group only after calibration.

**Status: not in the repo.** A scoring script (`code/size_check.py`), its test and notebook cells 16a/16b were written and pushed as `2d62e3b`, then reverted (`fcf6044`) at the user's request, and both commits were then removed from the history with a force push that the user approved. The branch ends at `ea85bcf` on both the machine and GitHub. The script reported, per row, the share of images with a missing object, the share with the relation violated (object missing or relation the wrong way round) and the mean log box-area ratio; it did not check the hedge word.

### 11d. How to reduce the size violations

A stronger guidance setting fixes none of the causes: 0.03 already added artifacts without better size.

Of the 6 prompts where Stage 1 still violated size (11b):

| Cause | Prompts | What happens |
|---|---|---|
| An object is missing | toaster/radio, bowl/mug, kitten/mirror | The relation cannot hold with one object. Missing objects stayed at 30%, same as SD3. |
| The scorer says "done" too early | kettle/shoe, drum/radio | Guidance stops once its own score reaches the lower edge of the target (0.99 against 0.95 to 1.90), but the picture does not yet look "extremely" bigger. |
| The target has no upper limit | blanket/drum | "Somewhat bigger" is scored as "above 0.20", so a blanket five times the drum counts as correct and nothing pulls it back. |

What to do, in order of expected gain, all inside the paper's method (its calibration sections):

1. **Fit the size calibration.** The hedge centres and the two size quantiles are hand-set placeholders. This is why 4 of 10 prompts only get a one-sided target and why the scorer is satisfied too early. Addresses the second and third causes.
2. **Add a presence model and validity flags for the noisiest bin.** Guidance now has to start at sigma 0.9, after SD3 has decided which objects exist. With a fitted presence model for the first steps it can act while the layout is open, the only point where a missing object can be added. Addresses the first cause.
3. **Fit the presence models for the later bins.** The scorer reported the radio and the mug as present when they were not; while it believes an object is there, it applies no pressure to create it.
4. **More than one seed per prompt.** This does not improve the model, but gives a fairer and more stable number.

SAR is not on the list: it changed nothing in this run and acts on attribute binding more than on size.

What it requires: annotated calibration images (the calibration group has about 9,400 size prompts to draw from; a few hundred annotated images is the usual scale), and the calibration-code fixes from the review first (the fit is evaluated on its own training data; the gradient-validity result is never enforced).

What to expect: no number can be promised. The missing-object cause is the largest block (3 of the 6 violations), so step 2 has the most room, and is also the least certain.

Proposed next step, not started: the calibration-code fixes.

### 11e. Questions asked after section 10

| # | Question | Short answer |
|---|---|---|
| 21 | Add this to the README. | Section 10 |
| 22 | Check `results (4).zip`. | Table 4 run: S1 better on about 4 of 10, SAR adds nothing. See 11a |
| 23 | You rate the images and return the result. | Done, as a model rating that is not blind. See 11b |
| 24 | Can we calculate this for 400 prompts? | Yes, about 49 GPU hours and only from the test group; 137 development prompts recommended first. See 11c |
| 25 | "Calculate it." | A scoring script and cells were added, then reverted and removed at the user's request. See 11c |
| 26 | What is in the commit? | `size_check.py`, its test, notebook section 16; the revert removed exactly those |
| 27 | Uncommit those changes. | Both commits removed locally and on GitHub (force push approved). Branch at `ea85bcf` |
| 28 | How can I get fewer size violations? | Calibration: size targets, then presence for the noisiest bin. See 11d |

## 12. Goal: size violation under 30% for the model

Question asked: "for my model it needs to approximate less than 30. What needs to be improved?"

**Under 30% means at most 2 violations in 10 prompts. Stage 1 has 6 (section 11b). At least 4 of the 6 must be fixed, and that needs both kinds of improvement; no single fix gets there.**

### 12a. The arithmetic

| Cause | Violations now | What fixes it |
|---|---|---|
| Object missing (toaster/radio, bowl/mug, kitten/mirror) | 3 | presence guidance in the first steps |
| Scorer satisfied too early (kettle/shoe, drum/radio) | 2 | fitted size targets |
| Target has no upper limit (blanket/drum) | 1 | fitted size targets |

- Fixing all three size-target cases leaves 3 violations, which is 30%, not under it.
- So at least one missing-object case must also be fixed.

### 12b. What has to be improved

1. **Size calibration (required).** Fit the hedge centres and the two size quantiles from annotated images, in place of the hand-set values. Every hedge then has a two-sided target, and guidance stops quitting at the lower edge. Best case, it removes 3 violations.
2. **Presence in the noisiest bin (required).** Fit a presence model and validity flags for the first steps, so guidance can start at step 0, before SD3 has fixed which objects exist. This is the only route to the missing-object cases. One success here takes the rate from 30% to 20%.
3. **Presence models for the later bins (supports 2).** The scorer now believes the radio and the mug are present when they are not, so it applies no pressure. Until that is fitted, step 2 cannot work reliably.

All three are the calibration the paper already specifies, so they keep to the strict-methodology rule. They need a few hundred annotated calibration images, and the calibration-code fixes from the review first.

### 12c. Cautions

- **Under 30% cannot be promised.** Step 1 is likely to help. Step 2 is the uncertain one: adding a missing object from guidance alone is hard, and it has to succeed at least once in three.
- **The 60% is a model rating on 10 prompts and one seed.** One borderline call moves it by 10 points. A blind hand rating may give a different starting number, so do that first.
- **Do not reach the number by changing the test.** Picking easier prompts, loosening the rating rule after seeing the results, or tuning settings on these same 10 prompts would make the figure meaningless to a reviewer. Fix the setting and the calibration on other prompts, and fix the rating rule before rating.

### 12d. Order of work

1. Rate `exp3_v2/size.html` blind and run 15c, to get the real baseline.
2. Fix the calibration code (the in-sample fit and the unenforced validity result).
3. Annotate the calibration images and run the fit.
4. Rerun Table 4 with the fitted calibration and guidance from the first step.

Step 2 does not depend on the rating and can start at once. Not started.
