# FuzzyDiff — agent handoff

Research code for **FuzzyDiff**: fuzzy-logic-guided diffusion. Phrase-level cross-attention
is converted into spatial membership functions in `[0, 1]`, prompt statements are evaluated
on those maps as graded truth values, and the gradient of that objective updates the latent
during denoising. A second stage reuses the same membership maps to reweight a refiner's
cross-attention.

This is a paper artifact, not a product. **Measurement honesty matters more than green
checkmarks.** Several claims in the paper abstract are not yet supported by evidence; the
"Do not claim" section below is binding.

## The one structural rule

**The notebook `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb` is the single source of truth
for all fuzzy logic.** Nothing reimplements it. Runners load definitions through
explicit notebook policies; prefix-selected loaders match the first line of each cell:

- `fuzzydiff/cli/full_pipeline.py` executes every code cell for full SDXL runs.
- `fuzzydiff/models/sd15.py` selects backbone-independent cells with `FUZZY_DEFINITIONS`.
- `fuzzydiff/models/session.py` loads archived/canonical AST definitions for registered models.
- `fuzzydiff/experiments/snow/gpu.py` uses the guarded cell-index policy.
- `fuzzydiff/notebook_runtime.py` owns source reading/compilation and the explicit loading policies.
- `tests/test_fullpipeline.py` retains its independent prefix-selected regression fixture.

Consequence: **renaming the first definition in a cell silently drops it from those loaders.**
If you rename or reorder a cell's leading `def`/`class`, update `FUZZY_DEFINITIONS` in
`fuzzydiff/models/sd15.py` and `prefixes` in `tests/test_fullpipeline.py` in the same change.
`tests/test_sd_fuzzy.py::test_notebook_supplies_the_fuzzy_definitions` is the tripwire.

Do not copy fuzzy logic into a `.py` file "to make it easier". The two backbones must not
drift apart. Only the *sampling loop* is backbone-specific (SDXL needs added time/text
embeddings; SD 1.x does not).

## Layout

| Path | Role |
|---|---|
| `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb` | Canonical implementation: config, membership, loss, attention collection, SDXL pipeline, refinement, entry cell |
| `fuzzydiff/models/` | Model registry, adapters, resource lifecycle; scientific definitions remain in notebooks |
| `fuzzydiff/generation/` | One-image requests/results and explicit saving, no benchmark passes |
| `fuzzydiff/experiments/` | Benchmarks, matched comparisons, and resumable snow workflows |
| `fuzzydiff/reporting/` | Exclusive timing instrumentation and measured-data tables |
| `fuzzydiff/cli/` | Argument parsing and package commands |
| Root `run_*.py`, `benchmark_*.py`, `snow_*.py`, comparison files | Compatible script/import facades sharing their package owners' module state |
| `configs/` | Validated JSON examples; explicit flags override defaults |
| `REVIEW.md` | Running record of findings, including measured GPU results. **Read before changing the method.** |
| `tests/` | CPU-only regression tests; no weight downloads |

## Commands

```bash
python run_fullpipeline.py --check          # compile all 26 notebook code cells, no models
python -m unittest discover -s tests        # CPU tests, no downloads
```

GPU work runs on Kaggle (T4, Internet on). Models load sequentially with CPU offload; a T4
cannot hold SDXL and SD 1.5 at once.

```python
%run run_fullpipeline.py --full             # full FuzzyDiff: guidance + refinement
%run run_fullpipeline.py --compare --prompt "..." --words "a,b" --lr 0.2,5
%run run_fullpipeline.py --four-way --prompt "..." --words "a,b" --bind "a>b"
%run run_sd.py --prompt "..." --words "a,b" --fuzzy --sharpness 20 --lr 5
```

Shared flags: `--prompt --words --seed --output --lr --sharpness --tnorm
--membership-mode --bind --binding-weight --relate --negative --refine`.

`--refine` keeps refinement as its own condition (`sdxl_full`) rather than folding it into
`sdxl_fuzzy`. Do not merge them: `sdxl_fuzzy` must differ from `sdxl_plain` by exactly one
change, or the guidance ablation stops being readable. The chain is plain -> fuzzy -> full.

**Kaggle gotcha:** `%run` reuses the kernel. Root script entry points call
`fuzzydiff.compatibility.reload_workflow` to refresh loaded package owners after
a pull. Comparisons retain the explicit SD adapter reload. Imported legacy names
alias the owning module, so patches and `importlib.reload` reach its live globals.
Keep this behavior when editing; see README's reload example for direct Python usage.

## Method as implemented

- `phrase_membership` — a phrase's share of the per-position softmax over the prompt's own
  token span, sharpened by `membership_sharpness`. Values in `[0, 1]`.
- `soft_truth` — soft maximum over space; the graded truth that the phrase is present.
- `compute_fuzzy_loss` — combines phrase truths with a t-norm (`min` = Gödel, `product`),
  then `-log`. Relations `left_of`/`right_of`/`above`/`below` compare membership centroids;
  `<hedge>_larger_than` scores a Gaussian around the hedge's target gap;
  `larger_than`/`smaller_than` compare `soft_extent` — peak local membership density pooled
  over several window sizes — normalized by the total so the comparison is scale free. Do
  not revert this to a mass sum: a duplicate instance raises the sum as much as a larger
  one, and a measured run produced two green apples instead of one bigger one; `bound_to` is a fuzzy AND of two memberships, each rescaled by
  `relative_membership`. The verb list lives in `fuzzydiff.models.sd15.RELATION_VERBS` and is mirrored in
  `fuzzydiff.cli.full_pipeline.RELATION_VERBS` to keep `--check` off the torch import path — a test
  asserts the two stay in step.
- Guidance runs for the first `max_iter_to_alter` steps, updating the latent *before* noise
  prediction at the same timestep. Updates keep their raw gradient magnitude, capped at
  `0.01` RMS — the cap never amplifies.
- Refinement reuses the same membership maps as masks, scaling each phrase's correction by
  its deficit `1 - truth`.

### Parameters that actually matter

- `membership_sharpness` (default 100) — **100 saturates.** A saturated membership softmax
  has *exactly zero* gradient; a `bound_to` conjunct at 100 cannot be optimised at all.
  Pinned by `test_binding_gradient_reaches_both_operands`. Use 20, or 5–10 if truths still
  pin near 0/1.
- `t_norm` (default `min`) — Gödel min gives gradient **only to the weakest phrase**. In a
  measured run the whole gradient went to `road` (truth 0.001) and the attribute under study
  received none for all 30 steps. Use `product` when every phrase should be optimised.
- `attend_excite_lr` (default 0.2) — 0.2 gives `update_ratio` ~1e-4, effectively inert.
  5 gives ~3e-3, which visibly moves truths. Check `clipped` in the diagnostics: if it is
  true on most steps, the `0.01` ceiling is the limiter, not the rate.

## Verification policy

CPU tests use real Diffusers/Transformers modules with random weights — never downloads.
Keep it that way; a test that needs network or a GPU does not belong in `tests/`.

**Never claim a GPU result you did not run.** "Tests pass" means CPU regression tests. Image
quality, T4 peak memory and whether guidance helps all require a Kaggle run, and the numbers
must come from the written `*.json`, not from expectation.

## Do not claim

These are measured findings, recorded in `REVIEW.md`. Contradicting them needs new evidence.

1. **`phrase_truth` is not evidence of visual faithfulness.** Measured: `road` scored 0.00116
   while a road was plainly visible; `slightly dusty` scored 0.99 while the car rendered clean
   and glossy. Low membership ≠ absent; high membership ≠ rendered. It is also the quantity
   guidance directly optimises, so it cannot be independent evidence that guidance works.
2. **Hedge intensity is implemented only for comparatives.** `<hedge>_larger_than`
   targets a band on the extent gap, so `slightly` and `much` are distinct objectives.
   The targets (0.15 to 0.70) are chosen, not calibrated against human judgements, so do
   not call them calibrated. The opt-in `--hedge-guidance` appearance objective targets attention truth only;
   it does not measure rendered attribute intensity. Read `HEDGE_GUIDANCE.md` before
   changing or interpreting this experimental mode.
3. **No measured image-quality improvement exists.** On `"A slightly dusty red sports car
   parked on a road."`, seed 42, three configurations were tried and no arm separated;
   min-part CLIP got slightly *worse* with guidance in both backbones.
4. **Attention presence does not control attribute intensity.** Maximising attention on an
   adjective's tokens does not make the rendered attribute stronger.
   `larger_than` is the one comparative that is genuinely gradable in image space, but it
   reads attention density, not measured extent; do not report it as a size ratio.
   Note also that neither the presence term nor the size predicate constrains object
   *count*: `soft_truth` is a soft maximum, so a phrase present twice scores no worse
   than a phrase present once.
5. **CLIP scores across method versions are not comparable** when the scoring templates
   changed; `min_part` (weakest tracked phrase) is the metric to report, not `full`.

## Known open problem

Membership is a *share of one per-position distribution*, so all phrases sum to ≤ 1 at each
position. Two separate objects can occupy different regions, but **an attribute and its
object must occupy the same pixels and are therefore forced to compete.** Measured: under
guidance, `red sports car` truth fell 0.3524 → 0.2377 while the attribute was pushed up.
`bound_to` works around this inside the binding term via `relative_membership`, but the
presence terms underneath still fight.

Approved experimental implementation: a selectable membership mode — `share` (current) versus
`relative` (each phrase's attention normalised by its own spatial max), letting an attribute
and its object both read high on the same pixels. This changes a core definition of the
method; retain `share` as the default and get agreement before further method changes.
No measured image-quality improvement is established for `relative`.

## Conventions

- No new runtime dependencies without asking; `requirements.txt` is pinned for Kaggle.
- Edit the notebook programmatically (load JSON, replace a cell's `source` string, dump with
  `indent=1` and a trailing newline). That round-trips byte-identically. Assert on the old
  content before replacing so a mis-targeted cell fails loudly.
- Match surrounding style: single quotes, comments that explain *why*, no docstring padding.
- The snow study's resume guard compares code fingerprints and will refuse to top up a
  manifest produced by a different algorithm. That refusal is correct — mixing pre- and
  post-change images invalidates the study. Start a new `--output` instead of weakening it.
- `REVIEW.md` is append-only history. Add a dated section; do not rewrite past findings.
