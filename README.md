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

![FuzzyDiff Architecture](<img width="2814" height="1391" alt="diagram1" src="https://github.com/user-attachments/assets/ecbe6c83-c0de-49cb-b565-19bcdc25d666" />)

---

# Highlights

![FuzzyDiff Examples](<img width="1657" height="1170" alt="f2" src="https://github.com/user-attachments/assets/857708e0-7222-4097-9996-29c6c9c2aef2" />
)

---

# Installation

## System Requirements

* **GPU:** NVIDIA GPU (T4 ×2 tested)
* **Memory:** Minimum **24GB VRAM** (48GB recommended for 768×768 generation)
* **Python:** 3.8+
* **CUDA:** 11.8+

---

### Install CLIP (for evaluation)

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

# Usage

Follow the notebook or script workflow:

1. Install dependencies
2. Load configuration
3. Define fuzzy logic operators
4. Load CLIP model
5. Set up attention control mechanisms
6. Load the SDXL base model
7. Generate images with fuzzy guidance
8. Refine results with predicate-aware refinement
9. Evaluate using semantic similarity and CLIP-based metrics

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
