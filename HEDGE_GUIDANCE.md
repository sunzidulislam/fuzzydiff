# Experimental fuzzy-term guidance

Enable `--hedge-guidance` to consume the canonical notebook's
`FUZZY_ATTRIBUTE_TARGETS` during generation. It is off by default.

| Term | Attention target |
|---|---:|
| barely | 0.05 |
| faint, a little | 0.10 |
| slightly | 0.15 |
| somewhat, relatively | 0.35 |
| moderately | 0.50 |
| considerably | 0.70 |
| mostly, very | 0.85 |
| nearly | 0.90 |
| almost, extremely | 0.95 |

These are chosen experimental defaults, not calibrated linguistic values.
The measured quantity is `soft_truth(phrase_membership(...))`: a soft spatial
maximum of the entire tracked attribute phrase's attention share, including its
modifier tokens. It does not measure rendered color coverage, dust, curvature,
height, distance, or size similarity. In particular, 0.85 does not mean that 85%
of an object's pixels will be blue.

## Run on Kaggle

After updating the repository files, run from `/kaggle/working/fuzzydiff`:

```python
%run run_fullpipeline.py --full --prompt "A mostly blue ceramic vase on a table." --words "mostly blue,ceramic vase,table" --bind "mostly blue>ceramic vase" --hedge-guidance --membership-mode share --sharpness 20 --tnorm product --lr 5 --predicate-strength 0.5 --seed 142 --output /kaggle/working/outputs/vase_hedge_seed142
```

For the dusty-car prompt:

```python
%run run_fullpipeline.py --full --prompt "A slightly dusty red sports car parked on a road." --words "slightly dusty,red sports car,road" --bind "slightly dusty>red sports car" --hedge-guidance --membership-mode share --sharpness 20 --tnorm product --lr 5 --predicate-strength 0.5 --seed 142 --output /kaggle/working/outputs/dust_hedge_seed142
```

The flag is forwarded by `--compare` and `--four-way`, and by `run_sd.py`
with `--fuzzy`. The separate `run_benchmark.py` runner does not expose this mode.

## Semantics and restrictions

- A supported term must start the tracked attribute phrase, case-insensitively,
  followed by whitespace and an attribute. Whole terms are matched; `almostly`
  does not match `almost`. `a little` is a single multiword term.
- Track the attribute separately from the object and supply `--bind` (or a
  `bound_to` relation). Overlapping attribute/object token groups are rejected.
  For example, use `mostly blue,ceramic vase`, not `mostly blue ceramic vase`.
- The object's ordinary presence objective remains active. The attribute's
  ordinary maximize-presence term is replaced by a target-band penalty:
  `0.5 * (relu(abs(truth - target) - 0.05) / 0.05)**2`.
  The existing spatial binding term also remains active.
- Only `share` membership is supported. `relative` normalizes every nonzero
  phrase map to a peak of 1 and is unsuitable for these absolute targets.
- During refinement, ordinary phrases use their usual presence deficits.
  Targeted phrases use signed residuals outside the same band. Positive
  residuals scale the existing spatial correction; negative residuals suppress
  attention inside the attribute mask. This is a heuristic based on base-stage
  attention, not a refiner feedback controller or guarantee of target attainment.
- Explicit hedged size relations keep their original `HEDGE_TARGETS` extent-gap
  objective. An explicitly related comparative subject is excluded from generic
  hedge detection. Use `--relate` for comparative size; this mode does not turn
  `slightly larger` into a measured 15% size increase.
- Share memberships compete for token mass. High attribute targets can conflict
  with object presence. Check output quality and clipping diagnostics; no image
  quality improvement or visual calibration is claimed.

Metadata records resolved terms, targets, token groups, tolerance, measurement
meaning, and whether hedge guidance ran. Full-pipeline output also records signed
refinement weights. Compare runs with and without the flag using identical prompts,
seeds and settings, in separate output directories. Previously generated images
were not produced with this new objective.

## Direct final-image interface

The package generation command also supports the experimental option:

```bash
python -m fuzzydiff.cli.generate --model fuzzydiff_full --prompt "A mostly blue ceramic vase on a table." --words "mostly blue,ceramic vase,table" --bind "mostly blue>ceramic vase" --hedge-guidance --seed 142 --output outputs/vase_hedge
```

It uses the same notebook objective and signed refinement masks, without benchmark
warmups or profiling. Its final image is `seed_142.png`; the legacy full command
keeps stage-one and stage-two filenames. The benchmark CLI remains unchanged in
its scientific options and does not expose generic hedge guidance.
