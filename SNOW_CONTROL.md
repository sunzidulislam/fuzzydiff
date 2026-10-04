# Fixed-scene snow control

Confirmed: mountain/snow subject; one fixed scene; 11 coverage levels by 9
relative snow-thickness settings; reference-style dense grid with axes and a
clean reference beneath it. Use seed 142 and fixed daylight by default.

This is a **FuzzyDiff base scene plus an explicit image-space snow renderer**.
It is a separate experiment from the existing native/FuzzyDiff/refiner pilot.
It does not demonstrate that fuzzy latent guidance learned snow intensity.
The renderer is a procedural appearance model, not WaterGen's decoder, a
trained snow generator, a 3D simulation, or an estimate of snow depth in cm.

## Controls and definitions

- Coverage C: 0.0, 0.1, ..., 1.0. The fraction of pixels selected inside the
  fixed mountain mask; not an independently measured fraction of real snow.
- Relative thickness D: nine settings from 0.1 to 1.0. Controls opacity and
  how much rock texture remains visible in the snow material. It is an
  appearance proxy for accumulation, with no physical depth calibration.
- A deterministic accumulation field prioritizes higher image regions plus
  smooth spatial variation. It is not an elevation or depth estimate.
- Coverage masks are nested and fixed across thickness columns. Feathering
  blends boundaries inside the mountain mask. Pixels outside it stay exact.
- Source camera geometry and illumination cues are reused from the same image;
  no image is independently resampled for a grid cell. Changing reflectance
  still changes brightness, as snow is brighter than bare rock.

FuzzyDiff generates a bare-rock summer mountain reference once using the
existing guidance algorithm and tracking `mountains`. SDXL may still create
snow or undesirable content; inspect the source rather than assuming it is
snow-free. Alternatively, provide a genuine snow-free image with `--source`.
Do not use the heavily snowy original pilot as a clean reference.

CLIPSeg estimates a mountain mask on CPU, with sky and existing-snow queries
for diagnostics. The mask is approximate and must be inspected. A supplied
black/white mask (`--mask`) can replace it: white denotes mountain terrain.
The saved overlay makes sky leakage or excluded terrain visible. There is no
silent fallback to an arbitrary lower-image rectangle.

## Outputs and reproducibility

Default directory: `/kaggle/working/outputs/snow_control`. Keep it separate from
`snow_experiment`. Save source image, mask, accumulation field, all 99 renderings,
per-cell target/selected mask coverage, relative thickness, original image/mask
hashes, source-generation diagnostics, versions, and the renderer code hash.
The zero-coverage row equals the source for every thickness; duplicates there
are expected. Exports: PNG/PDF grid with heavy coverage at the top, green
horizontal arrow, black vertical arrow, and source thumbnail beneath.

Completed source and mask are reused. Rendered cells resume only when their
file hashes match. Changed settings or a changed renderer require a new output
directory; missing/corrupted source or mask is an error, never a silent new
scene. Failed attempts stay in the manifest. Artifact regeneration loads saved
results without loading SDXL or CLIPSeg.

## Kaggle

Enable Internet and T4 GPU. In the existing checkout:

```python
%cd /kaggle/working/fuzzydiff
!git pull --ff-only
%pip install -r requirements.txt
!python run_snow_control.py --plan
!python run_snow_control.py
```

This performs one FuzzyDiff generation and then CPU segmentation/rendering;
the 99 grid cells do not each require an SDXL run. Inspect `source.png`,
`mountain_mask.png`, `mask_overlay.png`, and the grid before making quality claims.

For an existing clean mountain photo and a matching terrain mask:

```python
!python run_snow_control.py --source /kaggle/input/my-scene/mountain.png --mask /kaggle/input/my-scene/mask.png --output /kaggle/working/outputs/snow_control_photo
```

That path works on CPU without model downloads. A mask must match source
dimensions; it is thresholded at 128. For saved outputs, rerun the same command
to resume or use `--artifacts` to rebuild the figure. Keep originals as evidence.

The existing pilot showed nearly identical native/custom images and no reliable
coverage progression; see `SNOW_PILOT_FINDINGS.md`. This controlled rendering
study answers a different question: whether explicit appearance controls can
produce aligned parameter sweeps. Visual realism, snow-mask correctness, and
perceived depth still require independent human evaluation.

CLIPSeg's text-conditioned segmentation interface is documented by
[Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/clipseg).
