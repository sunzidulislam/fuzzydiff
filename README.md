# FuzzyDiff: Fuzzy Logical Diffusion for Structured and Graded Image Generation

FuzzyDiff introduces **fuzzy-logic-guided diffusion** to improve text-to-image generation when prompts contain **graded, uncertain, or softly constrained semantics** such as *slightly*, *moderately*, or *roughly*. The method integrates **fuzzy reasoning directly into the denoising process**, enabling diffusion models to better interpret linguistic uncertainty and produce more semantically faithful images.

---

# Abstract

Diffusion models have demonstrated impressive text-to-image generation capabilities, producing high-fidelity images from natural-language prompts. However, they often struggle when prompts contain **graded or uncertain semantics**, frequently yielding **missing objects, weak attributes, or inconsistent relations**. To address this limitation, we propose **FuzzyDiff**, a fuzzy-logic-guided diffusion framework that explicitly represents linguistic uncertainty and propagates it throughout the denoising process to improve prompt-faithful image synthesis. FuzzyDiff converts **phrase-level cross-attention maps into grounded fuzzy predicates**, enabling prompt statements to be evaluated as **differentiable graded truth values rather than binary constraints**. These fuzzy truth scores define guidance objectives that softly enforce **attribute intensity, object existence, spatial extent, and relational consistency**, while preserving the flexibility of diffusion sampling. In addition, we introduce a **predicate-based image-to-image refiner** that further strengthens attribute completeness by correcting residual omissions and refining weakly grounded regions. Extensive experiments on prompts containing **hedges, graded attributes, and multi-object relations** show that FuzzyDiff improves semantic faithfulness and reduces **missing-object and attribute-leakage failures**, while maintaining strong visual quality. These results demonstrate the benefit of integrating **fuzzy reasoning into diffusion-based image generation**.

---

# Influence of Fuzzy Attributes on Image Generation

![Fuzzy Attribute Impact](https://github.com/user-attachments/assets/817ae9df-f509-4fe7-bb39-b87c069de08e)

FuzzyDiff enables diffusion models to interpret **graded linguistic modifiers** and translate them into consistent visual changes.

Examples include:

* **Vehicle type**: A *moderately fast car* appears as a regular vehicle, while a *very fast car* becomes a high-performance sports car.
* **Snow intensity**: Mountain scenes transition from *slightly snow-covered* to *heavily snow-covered* terrain.
* **Lighting strength**: Indoor scenes evolve from *dimly lit* to *brightly illuminated* environments.
* **Object quantity**: Underwater scenes gradually increase the number of fish from *few* to *many*.

These examples demonstrate that **FuzzyDiff captures gradual semantic changes and translates them into coherent visual transformations.**

---

# Architecture

![FuzzyDiff Architecture](https://github.com/user-attachments/assets/ecbe6c83-c0de-49cb-b565-19bcdc25d666)

---

# Highlights

![FuzzyDiff Examples](https://github.com/user-attachments/assets/857708e0-7222-4097-9996-29c6c9c2aef2)

---

# Installation

## System Requirements

* **GPU:** NVIDIA GPU; the main notebook targets a Kaggle T4 with CPU offload.
* **Memory:** Models run sequentially on GPU 0; T4 ×2 does not pool VRAM. Actual
  T4 peak memory must be checked with the smoke test before a full run.
* **Python:** 3.10+ with CUDA-enabled PyTorch (keep Kaggle's preinstalled PyTorch).
* **Internet:** Required to clone the repository and download model weights.

---

### Install CLIP (for evaluation)

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

# Usage

The maintained entry point is `pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb`.
The other pipeline notebook is an older experiment and has not received these fixes.

In Kaggle, enable **Internet** and select **GPU T4 ×2**, then run:

```python
!git clone https://github.com/sunzidulislam/fuzzydiff.git /kaggle/working/fuzzydiff
%cd /kaggle/working/fuzzydiff
%pip install -r requirements.txt
!nvidia-smi
%run run_fullpipeline.py
```

For an existing checkout, use `!git pull --ff-only` inside that directory instead
of cloning again. These changes must be committed and pushed to GitHub before
Kaggle can retrieve them.

The default smoke test generates a 512 × 512 image with four base steps and
eight refiner steps (strength 0.4). It checks execution, not final image quality.
After it succeeds, run the full 768 × 768 / 50 base-step configuration:

```python
%run run_fullpipeline.py --full
```

Each seed writes `seed_<seed>_stage1.png`, `seed_<seed>_stage2_refined.png` and
`seed_<seed>_metadata.json` to `/kaggle/working/outputs`. Stage one saves before
refinement starts, so its image survives a refiner error. Metadata records scores,
configuration, versions and actual fuzzy gradient/update magnitudes. Smoke and full
runs with the same seed overwrite those filenames; download smoke outputs first
if you want to keep them.

To run interactively, import the main notebook into Kaggle and run top to bottom.
Set `SMOKE_TEST = False` in its final cell for a full run. Change prompt, tracked
phrases, seeds, negative prompt and optional relations there. Supported explicit
relations are `left_of`, `right_of`, `above`, `below`; no spatial constraint is
added automatically. Model-loading cells must be rerun after the final cell unloads
the base pipeline.

The loss guides token presence and configured 2D relations. It does not calibrate
hedges such as "slightly" or "very", or depth relations such as "behind". Better
visual results require matched-seed comparison on GPU; CLIP scores alone do not
establish improvement. See [the full two-axis review](REVIEW.md).

If an image is distorted, diagnose stage one with a matched-seed comparison:

```python
%run run_fullpipeline.py --compare
```

This runs native SDXL, custom sampling with guidance disabled, and corrected
raw-gradient guidance using the original prompt `A very fast car`, seed 142,
768 × 768 pixels, 50 steps and CFG 9.5. It skips refinement. Images and scores
save to `/kaggle/working/outputs/comparison_seed_142`. Inspect all three before
changing prompts or claiming a visual improvement. Small gradients are no longer
normalized into fixed-size updates; updates preserve their magnitude, cap RMS at
0.01, and retain float32 latent precision.
Native SDXL uses its usual latent precision; the custom runs use float32 latents.
Compare the two custom images to isolate the effect of guidance. Native-versus-custom
differences also include latent precision and attention implementation differences.

CPU regression checks (no model downloads):

```bash
python -m unittest discover -s tests -v
python run_fullpipeline.py --check
```

---

# Stage 1: Fuzzy-Guided Generation

```python
image, scores, attn_store = generate(
    prompt=prompt,
    words_to_track=words_to_track,
    seed=seed,
    num_steps=50,
    guidance=9.5,
    height=768,
    width=768,
    max_iter_to_alter=30,
    attend_excite_lr=0.2,
    alpha=10.0,
    spatial_loss_weight=0.5,
)
```

Parameters such as **seed**, **negative prompts**, and **tracked tokens** can be adjusted depending on the prompt structure.

---

# Stage 2: Predicate-Based Refinement

```python
refined = refine_with_predicate(image, prompts, attn_store, token_indices)
```

The refinement stage improves **attribute grounding and object completeness**, particularly for prompts with **weak or uncertain semantic constraints**.

# Baselines and Comparisons

We compare FuzzyDiff with several diffusion-based generation methods:

* **Stable Diffusion**
  [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

* **Attend-and-Excite**
  [https://github.com/yuval-alaluf/Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite)

* **Predicated Diffusion**
  [https://github.com/tksmatsubara/PredicatedDiffusion](https://github.com/tksmatsubara/PredicatedDiffusion)

* **SPO**
  [https://github.com/RockeyCoss/SPO](https://github.com/RockeyCoss/SPO)

* **C3**
  [https://github.com/daheekwon/C3](https://github.com/daheekwon/C3)

---
