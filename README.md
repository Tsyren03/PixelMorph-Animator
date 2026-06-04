# Pixel Shuffler — Image Translation Desktop Application

Implementation inspired by the ICIP 2025 paper [*PixelShuffler: A Simple Image Translation through Pixel Rearrangement*](https://ieeexplore.ieee.org/document/11084515).
PyQt6 desktop app: load **content** (structure) + **style** (appearance) → optimize a deformation field → preview result and **morph animation**.

---

## How the project works

### Idea (from the paper)

Classic **style transfer** asks: *how can we combine the layout of one image with the look of another?*
The PixelShuffler paper proposes doing this by **rearranging pixels** of the **style** image instead of generating new pixels from scratch. Each output pixel still comes from the style image’s color information, but it is **read from a warped location** chosen so that the final image:

- **Matches the content image’s structure** (edges, object layout)
- **Keeps the style image’s appearance** (color distribution, texture statistics)

This project implements that idea with a **learned deformation field**: a small neural network predicts *where* to sample each pixel from the style image, and gradient descent adjusts that field until perceptual losses are satisfied.

### Algorithm pipeline

```mermaid
flowchart LR
A[Content image] --> C[Deformation U-Net]
B[Style image] --> C
C --> D[2D flow field Δ]
B --> E[grid_sample warp]
D --> E
E --> F[Warped style = output]
G[Content loss LPIPS] --> H[Total loss]
I[Style loss VGG Gram] --> H
J[TV loss on Δ] --> H
F --> G
F --> I
D --> J
H --> C
```

**Step-by-step (each training iteration):**

1. **Preprocess** — Both images are resized and center-cropped to 256×256, converted to tensors, and normalized to roughly $[-1, 1]$.

2. **Predict deformation** — A U-Net takes the **concatenated** content and style (6 channels) and outputs a **2-channel displacement field** $\Delta(x, y)$, bounded with `tanh` so warps stay moderate.

3. **Warp the style image** — Differentiable **`grid_sample`** builds a sampling grid: each output coordinate looks up a location in the style image offset by $\alpha \cdot \Delta$. At $\alpha = 0$ you get the original style; at $\alpha = 1$ you get the fully warped result. This is the **pixel rearrangement** step.

4. **Compute losses** on the warped image:
- **Content (LPIPS)** — Penalizes perceptual distance to the content image so structure aligns.
- **Style (VGG Gram)** — Compares Gram matrices of VGG19 feature maps to the **unwarped** style, preserving global color/texture statistics.
- **Total variation (TV)** — Penalizes sharp jumps in $\Delta$ to reduce tearing and blocky artifacts.

5. **Optimize** — Adam updates **only the U-Net weights** (VGG and LPIPS are frozen). After hundreds–thousands of steps, the warped style is the final stylized image.

6. **Morph animation** — For visualization, the same field is applied with $\alpha$ smoothly increasing from 0 to 1 (smoothstep easing), producing a GIF of pixels “sliding” into place.

**Combined loss:**

$$
\mathcal{L} = \lambda_c \, \mathcal{L}_{\mathrm{LPIPS}} + \lambda_s \, \mathcal{L}_{\mathrm{Gram}} + \lambda_{\mathrm{tv}} \, \mathcal{L}_{\mathrm{TV}}
$$

**Default weights:** $\lambda_c = 15$, $\lambda_s = 800$, $\lambda_{\mathrm{tv}} = 5$, learning rate $3 \times 10^{-3}$.

### What each input means

| Input | Role in the app | Used for |
|-------|-----------------|----------|
| **Content** | Structure target | LPIPS pulls the warped style toward this layout |
| **Style** | Appearance source | Warped with $\Delta$; Gram loss keeps its look |

The network never paints new colors—it **only moves** style pixels. That is why pairing matters: if content is a face and style is a waterfall, no smooth warp can produce a sensible face.

### Application architecture

The **desktop app** (`main.py`) wraps the training loop in a background thread so the UI stays responsive:

```
User loads images → PyQt6 previews
↓
"Run Pixel Shuffler" → PixelShufflerTrainer (pixel_shuffler/engine.py)
↓
Live preview every N iterations → main window
↓
Training done → result PNG + morph frames (pixel_shuffler/morph.py)
↓
User plays morph slider / exports GIF
```

| Module | Responsibility |
|--------|----------------|
| `pixel_shuffler/model.py` | U-Net + `grid_sample` warping + TV loss |
| `pixel_shuffler/losses.py` | VGG features and Gram matrices |
| `pixel_shuffler/engine.py` | Full optimization loop, snapshots |
| `pixel_shuffler/morph.py` | Frame sequence for animation |
| `pixel_shuffler/io_utils.py` | Load/save images and tensors |
| `main.py` | PyQt6 GUI, presets, progress, export |
| `cli.py` | Same training without GUI |

Training runs on **CPU or CUDA** automatically. Checkpoints are not saved; each run optimizes a **fresh** U-Net for the current image pair (test-time optimization / image-specific fitting).

### Relation to the published method

The IEEE paper describes maximizing **mutual information** between the shuffled style and content via a simple pixel-shuffle formulation. The [official implementation](https://github.com/OmarSZamzam/PixelShuffler) uses **MONAI** (UNet + Warp) with MI, LPIPS, and a VGG mean/std style term.

This coursework project pursues the **same high-level goal** (structure from content, appearance from style) but uses a **different pipeline**: a custom PyTorch U-Net, `grid_sample` warping, **LPIPS + multi-layer Gram loss + TV**, and a **PyQt6** desktop app. No source files were copied verbatim from the official repository (see [References](#references)).

---

## Demo & Screenshots


### 1. Application UI

Main window (load Content / Style, parameters, Run, play morph).


<img src="docs/screenshots/app_main.jpeg"  alt="Application main window" />


*Caption: Pixel Shuffler desktop UI — content & style inputs, live preview, morph controls.*

---

### 2. Successful result (works well)

**When this works:** Content and style have **similar composition and main shapes**, with clear subjects (e.g., architecture ↔ architecture, portrait ↔ portrait).

<table>
<tr>
<td align="center"><b>Content</b><br>
<img src="docs/examples/good/content.jpeg" width="320" alt="Good — content" />
</td>
<td align="center"><b>Style</b><br>
<img src="docs/examples/good/style.jpeg" width="320" alt="Good — style" />
</td>
</tr>
</table>

**Result**

<p align="center">
<img src="docs/examples/good/result.png" alt="Good — result" />
</p>

**Morph animation (GIF)**

<p align="center">
<img src="docs/examples/good/morph.gif" alt="Good — morph animation" />
</p>


*Caption: Style pixels rearrange toward content structure; colors and textures from the style remain plausible.*

---

### 3. Failure case (does not work well)

**When this fails:** Image pairs with **different structure** (e.g., face as content + landscape as style), **high-frequency or random** style textures, or TV weight set too low.


<table>
<tr>
<td align="center"><b>Content</b><br>
<img src="docs/examples/bad/content2.jpg" width="320" alt="Bad — content" />
</td>
<td align="center"><b>Style</b><br>
<img src="docs/examples/bad/style2.jpeg" width="320" alt="Bad — style" />
</td>
</tr>
</table>

**Result**

<p align="center">
<img src="docs/examples/bad/result.png"  alt="Bad — result" />
</p>

**Morph animation (GIF)**

<p align="center">
<img src="docs/examples/bad/morph.gif"  alt="Bad — morph animation" />
</p>


*Caption: Tearing, ghosting, or loss of recognizable structure — method limits are visible.*

---

## Conclusion — When It Works vs. When It Fails

### Strengths (works well)

1. **Aligned structure** — Content and style share a similar layout (e.g., both frontal portraits, both skyline/architecture). The deformation field can map regions without extreme stretching.
2. **Clear subjects** — Distinct foreground vs. background; not extremely cluttered. LPIPS can match structure; Gram loss preserves style statistics.
3. **Moderate resolution (256×256)** — Training is stable with the default crop; TV regularization keeps the field smooth.
4. **Balanced loss weights** — Defaults (content 15, style 800, TV 5) balance structure vs. appearance; morph interpolation ($\alpha$: 0→1) gives a smooth pixel-shuffle visualization.
5. **Desktop workflow** — PyQt6 GUI supports interactive tuning, live preview, and GIF export without a browser.

### Weaknesses (works poorly)

1. **Structural mismatch** — Face + landscape, object + texture-only style: pixels cannot rearrange into a coherent semantic layout; results show tearing or “melted” regions.
2. **Heavy clutter / fine detail** — Crowds, dense foliage, or high-frequency style (food, noise) fight the smooth deformation prior; style Gram dominates structure incorrectly.
3. **Low TV weight** — Reducing TV causes visible grid artifacts and discontinuities in the warp.
4. **Compute cost** — Hundreds–thousands of iterations per pair; CPU-only runs are slow; the first run downloads LPIPS/VGG weights.
5. **Fixed crop size** — Center crop to 256×256 drops context; off-center subjects may fail.


### Summary

| Aspect | Works well | Works poorly |
|--------|------------|--------------|
| Image pairing | Similar pose / scene type | Unrelated semantics |
| Style image | Painterly or coherent texture | Random high-frequency texture |
| Parameters | Default or higher TV | Very low TV, extreme style weight |
| Hardware | GPU, enough iterations | Very few iterations on CPU |

---

## Reference

**O. Zamzam**, “PixelShuffler: A Simple Image Translation through Pixel Rearrangement,” in *2025 IEEE International Conference on Image Processing (ICIP)*, Anchorage, AK, USA, 2025, pp. 1360–1365.

- **IEEE Xplore:** https://ieeexplore.ieee.org/document/11084515
- **Preprint:** https://arxiv.org/abs/2410.03021

```bibtex
@inproceedings{zamzam2025pixelshuffler,
author = {Zamzam, Omar},
title = {{PixelShuffler}: A Simple Image Translation through Pixel Rearrangement},
booktitle = {2025 IEEE International Conference on Image Processing (ICIP)},
pages = {1360--1365},
year = {2025},
organization = {IEEE},
url = {https://ieeexplore.ieee.org/document/11084515}
}
```


---

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)
- See `requirements.txt`

---

## Installation & Run

```bash
cd "Computer Vision Project"
python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

1. Click **Content** and **Style** to load images.
2. Choose a quality preset or adjust weights / iterations.
3. **Run Pixel Shuffler** → watch preview.
4. **Play morph** or export GIF / jpeg.

**CLI (optional):**

```bash
python cli.py path/to/content.jpg path/to/style.jpg --iterations 800
```

Outputs: `output/` (gitignored).

**Quick test images:** `assets/sample_content.jpeg`, `assets/sample_style.jpeg`

---

## Project structure

```
├── main.py # PyQt6 desktop app
├── cli.py # Headless training
├── requirements.txt
├── assets/ # Sample inputs
├── docs/
│ ├── screenshots/ # ← UI screenshot (app_main.jpeg)
│ └── examples/ # ← good/, bad/, … (content, style, result, morph.gif)
└── pixel_shuffler/
├── model.py
├── losses.py
├── engine.py
├── morph.py
└── io_utils.py
```
### P.S: Used Gemini and Cursor
