# render-model

Differentiable tag rendering and camera response model (JAX). Lives separately from the TypeScript/WebGPU app in this repo.

## Setup

```bash
cd python
uv sync
```

Requires Python 3.11+ (macOS x86_64: JAX is pinned to 0.4.38 for wheel availability).
The blob detector uses PyTorch MPS on Apple Silicon.

## Notebooks

| Notebook | What it does |
|----------|----------------|
| `03_render_model.py` | Camera pipeline demo (PSF, sharpen, gamma) |
| `04_optimize_render_model.py` | Synthetic joint H + camera fit (LM) |
| `06_annotate_tag_corners.py` | Annotate tag corners on test photos (`.corners.json`) |
| `08_lm_optimizer.py` | LM fit on real photos using saved corners |
| `11_blob_detector_lab.py` | Full-target perspective renders, single-pixel center heatmaps, tiny U-Net, separate peak detector |

```bash
uv run marimo edit notebooks/04_optimize_render_model.py
```

## Retina test images (`*_2x` → `*_1x`)

Drop `image_502_2x.png` etc. into a folder, then halve and remove the 2x files:

```bash
uv run python scripts/downscale_2x_to_1x.py notebooks/test_images
uv run python scripts/downscale_2x_to_1x.py --dry-run path/to/photos
uv run python scripts/downscale_2x_to_1x.py --replace notebooks/test_images
```

Writes `image_502_1x.png` by decimating every other pixel (no Lanczos), scales `.corners.json` if present, deletes `*_2x` sources.

## Package layout

| Module | Role |
|--------|------|
| `render_model.tag_data` | Tag36h11-style patterns |
| `render_model.homography` | DLT, apply, corner metrics |
| `render_model.renderer` | Inverse-warp + analytic AA render |
| `render_model.pipeline` | Supersample → PSF → bin → sharpen → gamma |
| `render_model.optimizer` | Joint H + camera Levenberg–Marquardt |
| `blob_lab.target_layout` | TS-parity hex target layout from params or export JSON |
| `blob_lab.render` | Differentiable JAX full-target perspective renderer |
| `blob_lab.heatmap_dataset` | Full-target renders with single-pixel square-center targets |
| `blob_lab.roi_dataset` | Office composites with the target floating on a background and exact target-area masks |
| `blob_lab.roi_net` | Low-resolution target-area (ROI) segmenter + presence head |
| `blob_lab.unet` | JAX U-Net reference |
| `blob_lab.unet_torch` | Deeper U-Net with center, target-area, and junction heads using PyTorch MPS |
| `blob_lab.peaks` | Separate local-maximum peak detector and keypoint metrics |
| `blob_lab.dataset` | Legacy full-target patch extraction |
| `blob_lab.cnn` | Legacy patch classifier |
| `blob_lab.baselines` | DoH and center/ring patch baselines |

## CPU note

`jax` runs on CPU (JIT). On Apple Silicon you can try newer JAX; Intel Mac x86_64 needs jax/jaxlib ≤ 0.4.38.

## Blob detector lab

The blob lab renders the **full hex target** from the same layout as the
TypeScript target page under random perspective. The U-Net predicts a
single-pixel heatmap at every projected square center; peaks are extracted by
a separate detector. Junctions are not segmented or predicted.

Training uses class-balanced focal BCE: positive and negative pixels are
normalized separately. This is important because the single-pixel targets are
extremely sparse; a global pixel mean or Dice term would be dominated by the
negative background. The dataset also includes **completely negative
office/workspace photos** from `notebooks/natural_images/`; their target
heatmaps are all zero. See `notebooks/natural_images/SOURCES.md` for
attribution. Training always runs the full configured epoch count; validation
loss is logged but does not trigger early stopping.

```bash
uv run marimo edit notebooks/11_blob_detector_lab.py
uv run python scripts/smoke_blob_lab.py
uv run python -m unittest discover -s tests -v
```

The renderer is a differentiable JAX path:

```text
target texture → inverse homography → bilinear sample → PSF
```

The homography remains a differentiable input. The target layout can be built
from the exported target JSON or from the same columns / points-per-edge /
marker-scale / color-seed parameters used by the target page.

Training heatmaps use a small Gaussian around each projected square center
(and each degree-3 junction). Exact projected center coordinates are retained
in metadata for evaluation. They do not cover the square interior. Square
colors are re-randomized for every rendered scene, so the detector cannot rely
on a fixed binary color pattern.

## Target-area (ROI) lab

The ROI lab separates the two scales of the problem. A tiny low-resolution
network segments the **general target area**; the existing full-resolution
peak finder then runs inside that mask.

The target is rendered with the full browser canvas, including the same 10%
blank padding as the Hex Target page, and composited over an office photograph
as a floating card with a soft drop shadow. The board gray level is varied
slightly for imperfect office lighting. The mask can be the full canvas, the
convex hull of the projected markers, or the union of projected honeycomb
cells.

```bash
uv run python scripts/fetch_office_images.py --count 24
uv run python scripts/preview_roi_dataset.py --samples 6
uv run python scripts/train_roi.py --samples 500 --epochs 20
uv run python scripts/roi_ablation.py --samples 500 --epochs 20
```

Training uses horizontal flips and a false-negative-weighted BCE + Dice +
Tversky loss: missing target pixels is much worse than including a little
extra background. The ablation script compares `(2,4,8,16)`,
`(3,6,12,24)`, `(4,8,16,32)`, and `(8,16,32,64)` models at 96×72, 128×96, and
192×144, and reports parameters, GFLOPs, latency, mask IoU/Dice, false-negative
rate, false-positive rate, center containment, and area ratio.

### Ablation result

500 composites from 31 office backgrounds, horizontal flips, 20 epochs,
input 192×144, output stride 2. All models use false-negative-weighted BCE +
Dice + Tversky (`fn=10`, `tversky=1.0`). Metrics are on the held-out test
split; `containment` is the fraction of true square centers inside the raw
predicted mask (before the extra inference dilation).

| Model | Params | GFLOPs | Containment | FN rate | Area ratio |
|---|---:|---:|---:|---:|---:|
| `micro` (2,4,8,16) @640 | 12,332 | 0.029 | 0.969 | 4.87% | 1.014 |
| **`tiny` (3,6,12,24) @640** | **27,542** | **0.064** | **0.991** | **2.60%** | **1.026** |
| `small` (4,8,16,32) @640 | 48,782 | 0.114 | 0.892 | 12.39% | 1.163 |
| `medium` (8,16,32,64) @320 | 194,042 | 0.450 | 0.998 | 1.36% | 1.036 |

The 49k model overfit at 640×480 with this data; the 27.5k `tiny` model is the
sweet spot. The 12k model is close but misses ~3% of centers. On 320×240 data,
the same `tiny` model reached 0.991 containment, so the recommendation is not
resolution-specific. Input resolution does matter: the same 27.5k model at
128×96 reached only 0.963 containment, so 192×144 is the minimum useful input
for the 99% target.

### Real-world positives

Unannotated camera frames with the target present go in a single folder,
`python/notebooks/real_positives/`, with a sibling mask:

```text
<stem>.jpg
<stem>_mask.png   # white = target, same size as the image
```

The loader cover-resizes/center-crops each pair to 640x480, adds it to the
training split only, and repeats it `--real-positive-repeat` times (default 4).
Annotate with the built-in polygon tool:

```bash
uv run python scripts/annotate_real_positive.py \
  --image notebooks/real_positives/real_positive_01.jpg
```

See [real_positives/README.md](notebooks/real_positives/README.md). The first
frame is `real_positive_01.jpg` + `real_positive_01_mask.png`.

### Negative backgrounds: walls and abstract art

Presence negatives are no longer limited to office photographs. Fetch 24 wall
textures and 24 abstract-art images from Wikimedia Commons:

```bash
uv run python scripts/fetch_negative_images.py
```

They are stored in `python/notebooks/natural_images/negatives/` (49 images
currently, including `wall_user_01.jpg`), with title, artist, license, and
source recorded in that directory's `SOURCES.md`. The two additional user
frames (`wall_user_02.jpg`, `wall_user_03.jpg`) are treated exactly like office
backgrounds and live in the office-background directory, so target composites
are rendered on them as positives.

There are also 48 **procedural synthetic negatives** — spot clutter, fractal
noise, grain, scratches, grids, and paper/whiteboard marks — generated with:

```bash
uv run python scripts/generate_synthetic_negatives.py
```

They are pure negatives with no hex target. The combined negative pool is
49 photos + 48 synthetic images.

The
training defaults now use `--natural-fraction 0.45` and a separate
`--negative-dir`; positives still composite onto the office backgrounds.

The larger `(16,24,24,24)` model was trained at 256×192 with far targets
(15–75% width), reduced scale-aware blur, 2,200 samples, 100 epochs, batch size
32, plus the real-positive frames repeated 8×. On the mixed test split (174
positives / 156 negatives) it reached 0.997 center containment, 0.0000 negative
mask area, and 100% presence accuracy. Pure negatives were drawn from all photo pools (offices,
user backgrounds, walls, art, and synthetic noise). On 120
held-out wall/art negatives the previous checkpoint produced mask area 0.0319
and 68.3% presence accuracy; the retrained family reduced mask area to
~0.00006 and raised presence accuracy to ~89% before the final run.

### Close-up / blur / rotation augmentation

The current ROI dataset also samples:

- positive target widths from 15% to 90% of the frame, including far targets
  at 15–30% width,
- 30% close-ups at 70–90% of the frame width,
- 30% cropped targets: one side is pushed past the frame edge, with a
  minimum 8% of the target still visible,
- heavy defocus blur (`sigma` 0.6–1.8) on 20% of positives,
- full-frame Gaussian blur on 15% of positives,
- blur sigma scaled by target width, so far targets receive much less blur,
- canvas aspects 4:3, 16:9, and 1:1,
- in-plane roll from −180° to +180°.

For cropped scenes the training mask is clipped at the image border and
touches the edge, so the model cannot learn a background margin there.

This is deliberately harder than the earlier easy set. On a mixed hard test
with cropped targets, the current 27.5k checkpoint reached 0.932 overall
containment; cropped targets 0.965 vs 0.817 for the previous checkpoint,
close-ups 0.986 vs 0.881, heavy blur 0.891 vs 0.838, and non-4:3 canvases
0.901 vs 0.878. On the real phone photo the target mask remains bright.

False-negative weighting matters: at 320×240 the 49k model went from 0.975
containment at `fn=1` to 0.990 at `fn=5` and 0.991 at `fn=10`; the 194k model
reached 0.998 at `fn=10`. The cost is a slightly larger mask (area ratio
1.0–1.07), which is acceptable for an ROI that feeds a peak finder.

A single-frame MPS timing is ~1–2 ms for all variants and is dispatch-bound,
so GFLOPs is the useful scaling metric. At 0.064 GFLOPs/frame, the `tiny`
model is ~3.9 GFLOPs/s at 60 fps.

The recommended checkpoint is `notebooks/blob_lab_data/roi_net_best.pt`
(`tiny`, 27,542 parameters, 192×144 input, fn=10). The ablation JSONs are
`roi_ablation.json`, `roi_fn_ablation.json`, `roi_size_ablation.json`, and
`roi_size_640.json`.

## Browser ROI lab

`/roi-lab` runs the exported ONNX model on the webcam feed. The mask is
composited over the video and is ready to be fed into the full-resolution peak
finder. `/?demo=1#/roi-lab` streams a static target-on-office composite through
a canvas, so the page can be exercised without a camera.

Export the browser model after training:

```bash
uv run python scripts/export_roi_noconcat_onnx.py
```

This writes `public/models/roi_tiny_no_concat.onnx` (~297 KiB). It is a
GPU-compatible version of the larger backbone (74.0k parameters): the decoder's channel
concatenations are algebraically folded into separate convolutions plus `Add`,
and the two 6-channel conv inputs are zero-padded to 8 channels. The output
matches PyTorch to ~1e-7.

The ROI Lab is **WebGPU-only**. It shares one `GPUDevice` with TypeGPU and uses
`Tensor.fromGpuBuffer` for both input and output, so there is no CPU inference
fallback and no CPU round trip. ORT still needs its wasm bootstrap runtime to
initialize the WebGPU EP; those files are not an inference fallback.
The input is 256 px wide and the height follows the camera aspect (256×192 for
4:3, 256×144 for 16:9); the full frame is resized, not center-cropped.

ORT WebGPU EP findings on this model:

- `Concat` produced wrong output, including with `preferredLayout: 'NHWC'`.
- Conv blocks whose **input** had 6 channels produced wrong output; 3, 4, 8,
  12, 16, and 24 input channels worked.
- With both worked around, ORT WebGPU matches WASM exactly at every tested
  height. On the demo composite it runs in ~4.8 ms with ~7.5 ms total frame
  time (~28 FPS in the lab page).

The TypeGPU examples that informed the shared-device and mask-texture path are:

- <https://github.com/software-mansion/TypeGPU/tree/main/apps/typegpu-docs/src/examples/image-processing/background-segmentation>
- <https://typegpu.com/examples/#example=image-processing--selfie-segmentation>

The ORT wasm bootstrap files are copied to `public/ort/` and loaded with the
ROI page. They are about 26 MiB. They initialize the ORT runtime/WebGPU EP;
model inference does not fall back to WASM.
