# Hex Blob Lab — SOTA Review and Recommendations

Status: advisory note. No production code was changed.

Scope: the Python `blob_lab` work (`python/src/blob_lab`, notebooks 11,
`run_*_experiment.py`, `test_blob_lab.py`). The AprilTag path is explicitly
out of scope.

## TL;DR

1. **Do not train a 3-way-junction heatmap.** The junction is a deterministic
   function of the target layout and the plane homography. Detect square
   centers, fit `H` (8 parameters) with RANSAC, then *project* the junctions.
   That removes the hardest head, the negative transfer, and a large class of
   false positives in one step.
2. **The current bottleneck is not U-Net capacity; it is the representation.**
   At 640×480 the closest center/junction pair is ~2.6 px apart, the marker
   side is ~5 px, and the U-Net bottleneck is 8× downsampled. A one-pixel
   Gaussian at a junction is asking the network to label a *gap between three
   marker chains* using features that no longer resolve the chains.
3. **Add geometric verification before adding a bigger model.** A homography
   fit plus a photometric check rejects office clutter that happens to look
   locally square-like. The current checkpoint emits hundreds of peaks on 12
   office negatives; those peaks are almost all geometrically inconsistent.
4. **Add subpixel output and evaluate the right metric.** The current center
   error is ~0.8 px, while `docs/hex-blob-detection.md` targets <0.2 px.
   Quadratic peak refinement, an offset head, DSNT/soft-argmax, or a marker
   mask centroid will close most of that gap without new parameters.
5. **Fix the data distribution.** The model trains on full-frame synthetic
   targets and 12 historical black-and-white office photos. Deployment is a
   phone/tablet screen in a modern office, with clutter, glare, moiré, motion
   blur, and a target that occupies only part of the frame. Composite
   targets into office scenes, add hard negatives, and use the existing
   `render_model` camera pipeline.

## What the code currently does

`test_blob_lab.py` is only a layout-parity test:

- `build_hex_target_layout(640, 480, columns=8, points_per_edge=4, ...)`
  produces 476 quads, `side_px=4`, `spacing_px=10`, and the first quad is
  `(0, 1, 88, 78, 92, 82)`.
- `random_binary_colors(8, 1)` equals `(1, 0, 1, 1, 1, 0, 1, 1)`.

There are **no tests for `detect_peaks`, `match_peaks`, `evaluate_heatmaps`,
the U-Net, or the junction/area heads**. The name suggests detector coverage
that does not exist.

The lab itself:

- Renders the full hex target with a JAX homography, PSF, noise, and contrast.
- Builds a 4-channel input: gray, gradient magnitude, local variance, DoH.
- Trains a tiny PyTorch U-Net (`filters=8`, 122,203 parameters) with
  class-balanced focal BCE.
- Adds an area head (union of projected hex cells) and a junction head
  (Gaussian at degree-3 vertices).
- Extracts integer peaks with threshold + local maximum + NMS.

Current reported results:

| Variant | Center F1 | Junction F1 | Center mean error |
|---|---:|---:|---:|
| `center_features` | 0.966 | — | 0.83 px |
| `gaussian_center` | 0.975 | — | 0.85 px |
| `gaussian_center_area_junction` | 0.976 | 0.215 | 0.80 px |
| `center_area_junction_features` | 0.863 | 0.143 | 0.83 px |
| `junction_only` | — | 0.297 | 1.04 px |

The junction head is not learning the geometry. Joint training is *worse*
than junction-only (0.14–0.22 vs 0.30), which is the signature of negative
transfer from the shared encoder/decoder and the dominant center loss.

With the checkpoint currently in
`python/notebooks/blob_lab_data/unet_checkpoint.pt`, the center head on the
12 office negatives produces:

| Peak threshold | Peaks over 12 images | Peaks/image |
|---:|---:|---:|
| 0.2 | 545 | 45 |
| 0.3 | 487 | 41 |
| 0.5 | 366 | 31 |
| 0.7 | 172 | 14 |
| 0.9 | 45 | 4 |

Some individual negatives reach 0.99 confidence and 80+ peaks. This is the
real false-positive problem, and it is not solved by architecture alone.

## Measured diagnosis

For the default 7-column, 4-points-per-edge target at 1280×960:

- 420 square centers.
- 54 interior degree-3 junctions (12.9% of the center count).
- Marker side = 10 px, center spacing = 23 px.
- Junction spacing = 93 px, but the **nearest center/junction distance is
  only ~5.3 px** at target resolution and **~2.6 px** at 640×480.
- At 640×480, marker side is ~5 px.
- After the U-Net's three 2× pools (1/8 resolution), a marker is ~0.6 px and
  the center/junction separation is smaller than one bottleneck pixel.

The evaluation tolerance in the lab is 2 px. The closest center/junction pair
is at or below that tolerance after moderate perspective. A junction and its
nearest square center are not separable at the resolution where the network
makes its deepest decisions.

## Why 3-way junctions are surprisingly hard

The problem is not that junctions are "geometric"; it is that the chosen
representation is wrong for the geometry:

1. **A junction is a gap, not a blob.** It sits in the gray padding between
   three marker chains. Its evidence is distributed over three arms, each
   ~20–50 px long. A one-pixel Gaussian label at the gap gives the network no
   supervision for the arms.
2. **The local appearance is nearly ambiguous.** At 640×480 the nearest
   square center is ~2.6 px away. The junction patch contains marker edges,
   but the center itself is background. The network is asked to distinguish
   "background next to a marker" from "marker center" at sub-pixel separation.
3. **The resolution is below the information limit.** A 5 px marker at 1/8
   resolution is sub-pixel. No amount of decoder capacity recovers the
   three-fold arrangement from a 0.6 px feature.
4. **Negative transfer.** The center head has ~330 positives per scene, the
   junction head ~43; the shared decoder is dominated by centers. Junction-only
   training is better, which confirms the interference.
5. **Random colors remove the local pattern prior.** Re-randomizing colors
   for every scene means the junction has no consistent local template. The
   network must learn a color-invariant 3-fold geometry from very few
   positives. The deployed target, by contrast, uses a fixed seed (or a small
   set of seeds). This is both a learnability problem and a domain gap.
6. **The label is sparse and the loss is point-wise.** Only the gap pixel is
   positive. There is no loss that says "three arms must be present at 120°."
   The model can minimize the loss by memorizing a generic "gray gap"
   detector, which is not discriminative in office scenes.
7. **The evaluation is optimistic.** `best_metrics` selects the threshold
   that maximizes F1 *on the test split*, and `mean_error_px` is averaged only
   over matched peaks. The junction F1 of 0.14–0.21 at a hand-picked threshold
   is not a stable estimate.

## The reframing: centers → homography → junctions

The target is a known planar pattern. The square centers lie on the honeycomb
edges; the 3-way junctions are the honeycomb vertices. Both are deterministic
functions of the same plane homography:

```text
image
  → marker proposals (centers)
  → correspondences (color fingerprint, chain graph, or lattice fit)
  → H (8 parameters) with RANSAC
  → project all square centers and all 3-way junctions
  → photometric refinement of H
```

Consequences:

- **No junction head is needed.** The junction coordinates are exact
  projections, not detections. There are no junction false positives.
- **You do not need to detect all 420 centers.** Four to twenty good
  correspondences determine `H`; RANSAC then recovers the rest, including
  centers that were missed or occluded.
- **Office false positives become outliers.** A random set of office peaks
  will not fit the target's specific lattice. Even a handful of spurious
  points are rejected by the inlier/residual test.
- **Subpixel comes from fitting, not from a heatmap argmax.** Once `H` is fit,
  the projected centers are subpixel by construction; a photometric refinement
  can push them below 0.2 px.

This is also what `docs/hex-blob-detection.md` already specifies:

```text
blob list → hex lattice fit → temporal tracking
```

The U-Net lab skipped the last two stages. They are exactly the stages that
remove clutter and make junction detection unnecessary.

## Recommended pipeline

```text
frame (prefer 1280×720 or 1280×960)
  → grayscale / local contrast normalization
  → marker proposals:
       tiny full-resolution CNN (1–5k params) or multi-scale DoG/DoH
       (XFeat/SiLK optional if classical proposals are not robust enough)
  → subpixel centroid + color/contrast
  → context gate: uniform core, gray ring, expected area/elongation
  → chain/graph building:
       nearest neighbors, 3 lattice directions, degree/angle histogram
  → homography RANSAC to the known layout:
       inlier count, residual, frame-level accept/reject
  → project all centers + junctions
  → photometric refinement with the existing JAX renderer
  → temporal tracking / multi-frame bundle adjustment
```

Model output should be **sparse**: a set of centers plus an `H`, not a dense
heatmap for every pixel. The target has ~420 points; a 640×480 heatmap has
307,200 outputs. The geometry has ~10 meaningful parameters.

### If you keep a learned detector

Use a high-resolution, low-parameter design:

```text
input 960×720 grayscale
  3×3 conv 8, stride 1
  3×3 conv 8, dilation 2
  3×3 conv 16, dilation 4
  heads:
    center logit (1)
    subpixel offset (2)
    marker mask (1)
    color logit (1) [optional]
loss:
  focal BCE + L1 offset + Dice/BCE marker mask + color CE
```

This is roughly 2–10k parameters, not 122k. The dilated convolutions provide
context without destroying the 5–10 px markers. If you need more capacity,
add channels before adding pooling.

### If you keep a heatmap

Use the modern coordinate-representation tricks rather than a plain U-Net:

- **Subpixel offset head** (CenterNet-style) or **DSNT** / integral regression.
- **DARK / distribution-aware decoding** instead of integer argmax.
- **SimCC**-style 1D coordinate classification if you want to avoid heatmap
  quantization entirely.
- **HRNet/HigherHRNet**-style high-resolution fusion if you need multi-scale
  context without losing the marker scale.

### If you must detect junctions from pixels

Do not use a one-pixel heatmap. Three better options:

1. **Derive them from the homography** (preferred; zero parameters).
2. **Skeleton/Voronoi of the marker mask.** Junctions are branch points of
   the background medial axis, or degree-3 vertices of the Voronoi diagram of
   the marker centers. This matches the "gap between three chains" geometry.
3. **Star-convex / ray prediction** (StarDist-style): at each candidate,
   predict three ray directions/distances. The junction is verified by three
   arms at ~120°, not by a local blob.
4. **Line/wireframe parsing**: fit the three marker chains (RANSAC lines) and
   intersect them; or use a wireframe parser such as HAWP, L-CNN, WETR, or
   DeepLSD if you need junctions in unconstrained scenes. For a known dashed
   target, chain fitting is simpler and more reliable.

## Method landscape

| Need | Method | Why it matters here | Caveat |
|---|---|---|---|
| Lightweight proposals | [XFeat](https://arxiv.org/abs/2404.19174) (CVPR 2024), [SiLK](https://arxiv.org/abs/2304.06194) (ICCV 2023), [SuperPoint](https://arxiv.org/abs/1712.07629) | Robust detectors at high resolution; sub-million to ~1.3M params; ONNX/WebGPU friendly | Generic; still needs geometric verification |
| Homography-aware keypoint selection | [BoNeSS-ST](https://arxiv.org/abs/2503.18767) (2025) | Selects repeatable points with low measurement error for homography, not just high repeatability | Research code; adapt ideas rather than drop in |
| High-resolution heatmaps | [HRNet](https://arxiv.org/abs/1902.09212), [HigherHRNet](https://arxiv.org/abs/1902.00593) | Keeps high-resolution features for small targets | More parameters; use a tiny variant |
| Subpixel / coordinate decoding | [DSNT](https://arxiv.org/abs/1801.07372), [DARK](https://arxiv.org/abs/1910.06278), [SimCC](https://arxiv.org/abs/2107.03332) | Removes integer-peak quantization; current 0.8 px error is far from the 0.2 px target | Some methods assume a single instance per heatmap |
| Junction/wireframe parsing | [HAWP](https://arxiv.org/abs/2003.01663), [L-CNN](https://arxiv.org/abs/2007.07527), [WETR](https://doi.org/10.1016/j.cag.2025.104323), [DeepLSD](https://openaccess.thecvf.com/content/CVPR2023/html/Pautrat_DeepLSD_Line_Segment_Detection_and_Refinement_With_Deep_Image_Gradients_CVPR_2023_paper.html) | Explicit lines/junctions, better inductive bias than a point heatmap | Trained on continuous wireframes; the target is dashed and repetitive |
| Star-convex shapes | [StarDist](https://arxiv.org/abs/1806.03535) | Predicts radial distances; natural fit for 3-arm/4-arm structures | Designed for nuclei; needs adaptation |
| Geometric equivariance | [E(2)-steerable CNNs](https://arxiv.org/abs/1911.08251) | 3-fold/4-fold geometry by construction; fewer parameters, better sample efficiency | More complex to implement/export |
| Tiny line detector | [MiLSD](https://arxiv.org/abs/2607.06600) (2026) | ~25k-parameter line detector with subpixel decoding | Line detector, not a junction detector |
| Motion-blur keypoints | [SSMB](https://arxiv.org/abs/2608.27181) (2026) | Webcam frames are often blurred; blur shifts/destroys tiny markers | Newer; needs evaluation on this target |
| Detector-free matching | [RoMa](https://arxiv.org/abs/2305.15404), [MASt3R](https://arxiv.org/abs/2406.09756), [VGGT](https://arxiv.org/abs/2503.11651) | Match a rendered target to the image, or infer pose directly; avoids keypoint training | Heavy; likely overkill for a known target; domain gap to a screen |
| General keypoint foundation model | [GKDT](https://arxiv.org/abs/2607.00752) (ECCV 2026, DINOv3) | Zero-shot/visual-prompt keypoints; useful as an upper bound | Large; not a browser deployment candidate |
| Camera-target centroids | [Circular-pattern calibration](https://arxiv.org/abs/2506.16842) (2025) | Shows centroid measurements can be more precise than corners and models their uncertainty | For circles; the centroid argument transfers to squares |
| Robust pose | [DSAC*](https://arxiv.org/abs/2002.12324), RANSAC/MAGSAC, differentiable rendering | Optimizes pose/H directly and handles outliers | Needs a good initialization |

The newest generic detectors (GKDT, MoD, SSMB) are impressive, but they solve
a different problem: general keypoints in unconstrained images. For a known,
highly structured target, the larger win is **model-based geometry**, not a
larger generic network.

## Prioritized experiment plan

| # | Experiment | Effort | Expected effect |
|---|---|---|---|
| 1 | Add subpixel refinement (quadratic fit / soft-argmax) to current peaks | 0.5 day | Center error 0.8 → ~0.3 px; no retraining |
| 2 | Add homography RANSAC + inlier/residual gate after peaks | 1–2 days | Office FP/image 31 → near 0 at threshold 0.5; frame-level accept/reject |
| 3 | Remove junction head; project junctions from `H` | 0.5 day | Exact junctions; removes negative transfer; one fewer head |
| 4 | Replace area head with marker mask; centroid markers | 1–2 days | Subpixel centers, context gate cues, better auxiliary signal |
| 5 | Train/evaluate at 1280×720 or reduce U-Net stride 8 → 4/2 | 1 day | Marker side 5 → 10 px; center/junction separation 2.6 → 5.3 px |
| 6 | Tiny full-res CNN (1–5k params) vs U-Net; DoG/DoH baseline | 2–3 days | Same or better F1 with ~10–100× fewer parameters |
| 7 | Composite targets into office scenes; add hard negatives | 2–3 days | Closes the synthetic-to-real gap; reduces FP |
| 8 | Fixed/few color seeds vs i.i.d. random colors | 1 day | Tests whether random colors are hurting learnability |
| 9 | Photometric refinement of `H` with the JAX renderer | 3–5 days | Subpixel, target-presence verification, calibration-grade pose |
| 10 | Temporal tracking / multi-frame bundle adjustment | 3–5 days | Kills transient FPs; improves calibration RMS |
| 11 | Steerable/equivariant junction detector (only if needed) | 1 week | Better geometry prior than a U-Net; still likely worse than deriving from `H` |

Do 1–3 first. They are cheap, do not require retraining, and address both the
junction difficulty and the office false positives.

## Metrics and evaluation fixes

The current harness optimizes keypoint F1 on synthetic renders. For
calibration, that is not the end metric. Add:

- **False positives per image** on a dedicated office-negative set at a fixed
  recall (e.g., 99%). Report median/p95, not only the pooled total.
- **Center error p50/p95**, not only mean over matched peaks.
- **Homography inlier ratio and residual** after RANSAC.
- **Target-presence AUC** at the frame level (target vs no target).
- **Reprojection RMS** on real target photos, and end-to-end calibration RMS.
- **Parameters and latency** on the target device (60 fps at 1280×720 is the
  stated target).
- **Per-image** precision/recall, not pooled over scenes.

Methodological fixes:

- Select the peak threshold on validation, never on test. The current
  `best_metrics` tunes on the test split.
- Separate the office-negative set from the synthetic test set; report them
  independently.
- Use a one-to-one assignment (Hungarian or greedy with a hard distance
  threshold) and report unmatched predictions explicitly.
- Keep a fixed synthetic regression scene with known centers and junctions;
  assert exact projections after `H`.

## Test plan for `test_blob_lab.py`

The current file should stay as a layout-parity test, but add:

1. **Layout invariants**
   - Every `hex.three_way_junction_points` is incident to exactly three
     guides (interior), and every square center lies on a guide.
   - `len(layout.square_centers) == len(layout.quads)`.
   - Projecting the layout through an identity `H` reproduces the centers.
2. **Peak detector**
   - Empty heatmap → no peaks.
   - Single Gaussian with subpixel center → refined coordinate within 0.1 px.
   - Two peaks inside NMS radius → one survives; outside → two.
   - Plateau / flat maximum → one representative.
   - Border peaks and negative values.
3. **Matching**
   - One-to-one matching, tolerance boundary, duplicate predictions, empty
     truth, empty predictions.
4. **Geometry**
   - Exact `H` recovery from 4+ correspondences; outlier rejection;
     degenerate/collinear points rejected.
   - Projected junctions match the analytic layout under a known `H`.
5. **Regression**
   - A fixed 320×240 synthetic scene with a known pose: assert center count,
     first/last center, and a small set of junction coordinates.

## Risks and trade-offs

- **Homography fitting assumes a planar target and a reasonable
  initialization.** If the target is heavily occluded or only a tiny corner
  is visible, RANSAC may fail. Fall back to per-frame detection and temporal
  accumulation; do not accept a low-inlier fit.
- **A single global `H` does not model lens distortion.** Either undistort
  before detection (using the current calibration estimate) or include
  distortion in the refinement. Near the image edges, distortion can bias
  centers by more than the subpixel target.
- **Photometric refinement can be trapped by local minima.** Use a
  coarse-to-fine pyramid and initialize from RANSAC; use a robust loss
  (Huber/Tukey) for glare/occlusion.
- **A fixed color seed is a deployment assumption.** If users can shuffle
  colors, train on the actual seed distribution or use color-invariant
  features. Do not train on i.i.d. random colors and deploy on one seed
  without measuring the gap.
- **Foundation models are not free.** VGGT/SAM/GKDT are attractive upper
  bounds but are heavy and may not export to the browser target. Keep the
  deployed path small; use them offline for labels/verification if useful.

## References

- XFeat: <https://arxiv.org/abs/2404.19174>
- SiLK: <https://arxiv.org/abs/2304.06194>
- SuperPoint: <https://arxiv.org/abs/1712.07629>
- BoNeSS-ST (homography-aware keypoints): <https://arxiv.org/abs/2503.18767>
- HRNet: <https://arxiv.org/abs/1902.09212>
- HigherHRNet: <https://arxiv.org/abs/1902.00593>
- DSNT: <https://arxiv.org/abs/1801.07372>
- DARK: <https://arxiv.org/abs/1910.06278>
- SimCC: <https://arxiv.org/abs/2107.03332>
- HAWP: <https://arxiv.org/abs/2003.01663>
- L-CNN / wireframe parsing: <https://arxiv.org/abs/2007.07527>
- WETR: <https://doi.org/10.1016/j.cag.2025.104323>
- DeepLSD: <https://openaccess.thecvf.com/content/CVPR2023/html/Pautrat_DeepLSD_Line_Segment_Detection_and_Refinement_With_Deep_Image_Gradients_CVPR_2023_paper.html>
- StarDist: <https://arxiv.org/abs/1806.03535>
- E(2)-steerable CNNs: <https://arxiv.org/abs/1911.08251>
- MiLSD: <https://arxiv.org/abs/2607.06600>
- SSMB: <https://arxiv.org/abs/2608.27181>
- RoMa: <https://arxiv.org/abs/2305.15404>
- MASt3R: <https://arxiv.org/abs/2406.09756>
- VGGT: <https://arxiv.org/abs/2503.11651>
- GKDT: <https://arxiv.org/abs/2607.00752>
- Circular-pattern calibration: <https://arxiv.org/abs/2506.16842>
- DSAC*: <https://arxiv.org/abs/2002.12324>
