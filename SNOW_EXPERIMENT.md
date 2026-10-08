# FuzzyDiff snow coverage experiment

Confirmed design (2026-10-05): evaluate the existing FuzzyDiff algorithm,
using a WaterGen-inspired comparison layout. This is not a reproduction of
WaterGen's underwater physics or a claim of equal scene geometry.

## Fixed study

- Five ordinal prompt levels: very slight, slight, moderate, heavy, very heavy snow cover.
- Three scenes: wide alpine mountains, close rocky mountain peaks, mountains beside a lake.
- Seeds: 42, 84, 142, 202, 314.
- Methods: native SDXL, current FuzzyDiff, current FuzzyDiff plus the spatially guided attention refiner.
- 768 x 768, 50 base steps, CFG 9.5; FuzzyDiff uses the existing raw gradient,
  learning rate 0.2, 30 alteration steps, alpha 10, spatial weight 0.5,
  no relations, and tracks `mountains` and `snow`.
- Refiner: 30 steps, strength 0.4, CFG 9.5, refinement strength 2.5 scaled by each phrase's membership deficit.
- Use the existing notebook's checkpoints, fixed negative prompt, and CPU seeded generators.
- Same scene/camera prompt and seed across each five-level/three-method block.
  Independent text-to-image sampling can still change geometry.
- Native latents use fp16, custom latents float32. The native/custom comparison
  therefore includes this precision difference as well as the guidance changes.

The pilot is the alpine scene, seed 142: 15 images, requiring 10 base generations
and 5 refinements. Inspect it before expanding to all 225 images. The runner
must persist each result before moving on, retain errors, resume completed jobs,
and reject incompatible settings/code when resuming.

## Evaluation

Export a labeled five-row/three-column grid per scene/seed (PNG and PDF),
metadata, and a separate anonymous reviewer pack. At least two independent
human reviewers rate each image: observed snow coverage, prompt adherence,
and scene preservation. Coverage is judged before seeing the prompt. Scene
preservation uses the same native moderate-snow reference for all methods and
levels in that scene/seed block. This reference choice can favor the native
baseline and must be reported. It is not pixel ground truth.

CLIP similarity is supplemental, not a snow severity or geometry measurement.
Keep disagreements, incomplete ratings, missing images, and failed generations
visible. Do not select only attractive samples or describe a guidance benefit
before measuring it. These five-seed summaries are descriptive, not evidence
of statistical significance. The full study has 15 scene/seed blocks.

Reviewer identities, method/level mappings, and manifests must stay outside
the shared anonymous pack. Anonymous images are RGB exports with metadata
removed. Reviewers get separate blank CSV files; analysis joins the private
key only after ratings are collected. Do not provide the labeled grids or
private key during blind review.

## Running on Kaggle

Enable GPU T4 and Internet, then run:

```python
%cd /kaggle/working
!git clone https://github.com/sunzidulislam/fuzzydiff.git
%cd /kaggle/working/fuzzydiff
!git pull --ff-only
%pip install -r requirements.txt
!python run_fullpipeline.py --check
!python run_snow_experiment.py --plan
!python run_snow_experiment.py
```

If already cloned, omit the clone line. Do not install a different torch build
over Kaggle's CUDA environment. The pilot uses full study image settings;
it is a smaller sample, not a reduced-quality smoke test.

After inspecting the pilot images, diagnostics, failures and runtime:

```python
!python run_snow_experiment.py --full
```

Run the same command after a session interruption to resume. Keep the entire
`/kaggle/working/outputs/snow_experiment` directory, including attention files
and manifest, as a Kaggle output/dataset across sessions. Copy it back to the
same output path before resuming. Git pull must match the saved code/settings;
a changed implementation requires a new `--output` directory.

The original `e9fad45` runner failed after sampling because its checkpoint
metadata assumed mapping configs. The metadata repair also serializes the
scheduler's legitimate `-inf` bound as a string. It can automatically upgrade
that original run when settings are identical and no images were marked
complete, retaining failure history. Failed samples were not saved as PNGs and
must be generated again. Other code/settings changes still require a new output directory.
Once expanded, resume with `--full`; the default pilot command rejects a full
study directory so its other images cannot be hidden accidentally.

Artifacts can be rebuilt without a GPU:

```python
!python run_snow_experiment.py --artifacts
```

Find grids in `grids/`, private metadata in `manifest.json` and
`review_key.json`, and the shareable pack in `review_pack.zip`. Read
`review_pack/README.md` for the two-pass review procedure. Submit the completed
CSV sheets and analyze with:

```python
!python analyze_snow_ratings.py --output /kaggle/working/outputs/snow_experiment reviewer_1.csv reviewer_2.csv
```

Local CPU tests validate orchestration, artifact creation and rating analysis.
They do not establish T4 memory use, runtime, or generated image quality.
