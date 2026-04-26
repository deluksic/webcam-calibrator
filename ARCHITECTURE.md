# Architecture

## Overview

WebGPU runs the vision pipeline: grayscale, Sobel, histogram-driven threshold, NMS, connected components, compact labels, and extent boxes. The **grid** path adds async CPU work (`readDetection`) for regions, line-based corners, homography, and tag36h11 decode. A separate lens **solver** (intrinsics, distortion, bundle adjustment) is out of band for now.

**Calibrate** maintains a top‑K pool of per-tag scores from decoded frames. **Results** is the UI slot for future solved parameters and export.

## App shell (Solid)

- **Views** ([`App.tsx`](src/components/App.tsx)): **Target** (printable SVG), **Calibrate** ([`CalibrationView.tsx`](src/components/CalibrationView.tsx) — collection controls, top‑K pool, stats; adaptive threshold uses the same histogram as **Debug** but the histogram is not shown on this page), **Results** (solver output TBD), **Debug** ([`DebugView.tsx`](src/components/DebugView.tsx) — mode switcher, histogram, optional bbox-style overlay, logs).
- **Camera** — [`CameraStreamProvider`](src/components/camera/CameraStreamContext.tsx) at the app root; stream acquisition and device constraints in [`cameraStreamAcquire.ts`](src/components/camera/cameraStreamAcquire.ts).
- **Live WebGPU path** — [`LiveCameraPipeline.tsx`](src/components/camera/LiveCameraPipeline.tsx) for both Calibrate and Debug.

Product summary and roadmap: [`docs/plan.md`](docs/plan.md).

## Coordinate spaces

- **Frame size** — up to 1280×720
- **Raw label values** (pointer-jump) — per-pixel index into the labeling union-find structure (0 … area−1)
- **Compact label values** — 0 … N−1 after canonical remapping; used downstream
- **Extent buffer keys** — compact IDs &lt; `MAX_EXTENT_COMPONENTS` (16384)

## Pipeline (per frame)

```
Frame
  ↓
Sobel → histogram (drives edge threshold in CPU)
  ↓
NMS + edge filter
  ↓
Pointer-jump labeling (raw per-pixel labels)
  ↓
Canonical labeling (compact 0..N-1)
  ↓
Extent tracking
  ↓
Render (mode-specific)
```

### Pointer-jump labeling

GPU, ~10 iterations: pointer doubling plus atomic parent tightening.

- **Buffers:** `pointerJumpBuffer0/1` (ping-pong), `pointerJumpAtomicBuffer`
- **Out:** raw labels → canonical pass only

### Canonical labeling

GPU, three passes: reset roots → roots claim compact IDs → pixels remap to compact `compactLabelBuffer`.

### Extent tracking

GPU: atomic min/max per component into `extentBuffer` (at most `MAX_EXTENT_COMPONENTS` components).

## Display modes

| Mode | GPU work | View | CPU readback |
| ---- | -------- | ---- | ------------ |
| `grayscale` | Gray | Luma | Histogram |
| `edges` | Sobel | Edges | Histogram |
| `nms` | Sobel + NMS | Edges | Histogram |
| `labels` | Full chain through compact | False-color labels | — |
| `debug` | + extent | Labels + extent overlay | Extent |
| `grid` | + extent | Grayscale + homography grid | `readDetection` when a [frame slot](src/gpu/frameSlotPool.ts) is free (default 3 slots; busy pool skips the frame) |

## CPU readbacks

| API | When | Data |
| --- | ---- | ---- |
| Extent read in `debug` | Each `debug` frame | Extent table (~320 KB for max components) |
| `readDetection` | Each **grid** attempt with a free slot | Compact labels + NMS `filtered` buffer (~11 MB) → regions, corners, decode |

## Corner pipeline (grid, CPU)

Flow: `readDetection` → `validateAndFilterQuads` in [`contour.ts`](src/gpu/contour.ts) → for each region, `findCornersFromEdgesWithDebug` in [`corners.ts`](src/lib/corners.ts) (after area / aspect / edge-density filters).

**GPU (grid submit):** grayscale → Sobel → threshold from histogram → NMS → pointer-jump → compact → extent. **Readback** supplies dense compact labels and filtered `(gx, gy)`.

**Per region (CPU), order is fixed.** Failures in steps 1–5 mean the intersection set never gets four clean points, so a failure on “intersections” can still be caused upstream.

| Step | Work | Typical failure code |
| ---- | ---- | -------------------- |
| 1 | Pixels in the region with matching compact label; **NMS-filtered** `(gx, gy)` from the same buffer as decode | `FAIL_INSUFFICIENT_EDGES` (0) if count &lt; `minEdgePixels` (12) |
| 2 | K-means k=4 on gradient directions, cosine dissimilarity | (no bit — weak lines hurt later) |
| 3 | RANSAC + PCA line per cluster | `FAIL_LINE_FIT_FAILED` (2) if a line is missing |
| 4 | All line–line intersections, clipped to extent ± `extentBBoxSlack` | `FAIL_NO_INTERSECTIONS` (4) if &lt;4 raw hits after clip |
| 5 | Dedupe within 5 px | (4) if &lt;4 points remain |
| 6 | Convex CCW order + plausibility (`R²`, bbox slack, edge ratios) → **`[TL, TR, BL, BR]`** | `FAIL_PLAUSIBILITY` (3) |

If four refined corners are not found, a bbox quad is still used for homography; `cornerDebug` records the CPU attempt.

## AprilTag grid and decode

After corners (fitted or bbox), `validateAndFilterQuads` runs grid + dictionary decode in [`contour.ts`](src/gpu/contour.ts).

1. **Grid** — `buildTagGrid` in [`grid.ts`](src/lib/grid.ts) uses corners **TL → TR → BL → BR** (same as `computeHomography` and `DetectedQuad.corners`).

2. **Pattern** — `decodeTagPattern` scans the quad AABB, maps pixels through the inverse homography, accumulates half-space votes into an 8×8 module grid from **filtered** (`filteredBuffer`) gradients. Inner **6×6** bits go to the dictionary. The decode path is homography + bbox scan; `decodeCell` exists for **unit tests and tooling** in the same module.

3. **Dictionary** — `decodeTag36h11AnyRotation(pattern, maxError)` with `maxError = ALLOWED_ERROR_COUNT` (**3**, [`contour.ts`](src/gpu/contour.ts)) over 587 tag36h11 words in [`tag36h11.ts`](src/lib/tag36h11.ts).

4. **Outputs** — `DetectedQuad` carries `pattern`, optional `decodedTagId` / `decodedRotation`. UI shows the id or **`?`**. `updateQuadCornersBuffer` sends `vizTagId` to the instanced `decodedTagId` (`0xFFFFFFFF` = unknown, black fill in the shader); known IDs are tinted with `stableHashToRgb01`.

Tuning is primarily GPU NMS and corner geometry; an optional `edgeMask` is available in code but the live path passes none.

**Failure bitmask** (see [`corners.ts`](src/lib/corners.ts)): bits 0–4 defined; bit 1 reserved. Bit 4 covers both “too few intersection hits” and “dedupe &lt;4”. Bit 3 covers ordering and plausibility after four points exist.

## Homography

Eight-parameter homography, Gaussian elimination with partial pivot. Shader uses `w` in `outPos.w` for perspective-correct varyings.

## GPU buffers (summary)

| Buffer | Role |
| ------ | ---- |
| `sobelBuffer` | Raw gradients |
| `filteredBuffer` | After NMS |
| `pointerJumpBuffer0/1`, `pointerJumpAtomicBuffer` | Labeling |
| `compactLabelBuffer` | Final labels |
| `canonicalRootBuffer` | Canonical id map |
| `histogramBuffer` | Edge histogram |
| `extentBuffer` | Per-component bounds |
| `quadCornersBuffer` | [`GridDataSchema`](src/gpu/pipelines/gridVizPipeline.ts): homography, debug fields, `decodedTagId` (1024 instances max) |
