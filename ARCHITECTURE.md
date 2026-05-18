# Architecture

## Overview

WebGPU runs the vision pipeline: ingest to luma, Sobel, histogram-driven threshold, NMS, pointer-jump connected components, and compact labels. The **grid** path (Calibrate) runs GPU edge clustering, line fit, quad homography, and tag36h11 decode, then reads back only the quad buffer for host callbacks ([`readGpuDetection`](src/gpu/gpuQuadReadback.ts)).

**Calibration** runs in a dedicated worker ([`calibration.worker.ts`](src/workers/calibration.worker.ts)) using `@deluksic/opencv-calibration-wasm`: intrinsics `K`, OpenCV **rational** distortion (`RationalDistortion8` in [`cameraModel.ts`](src/lib/cameraModel.ts)), and per-frame extrinsics. [`CalibrationRunContext`](src/components/calibration/CalibrationRunContext.tsx) owns the live session, worker solves, and **`latestCalibration` / metadata** mirrored for **Results**. **Calibrate** uses an `ok` model for a live **reprojection overlay** on the grid ([`reprojectionOverlayPipeline.ts`](src/gpu/pipelines/reprojectionOverlayPipeline.ts), wired from [`LiveCameraPipeline.tsx`](src/components/camera/LiveCameraPipeline.tsx)). **Results** reads the same context and renders a 3D summary plus JSON export ([`ResultsView.tsx`](src/components/results/ResultsView.tsx), [`exportCalibrationJson.ts`](src/components/results/exportCalibrationJson.ts)).

## App shell (Solid)

- **Views** ([`App.tsx`](src/components/App.tsx)): **Home** ([`Home.tsx`](src/components/Home.tsx)), **Target** (printable SVG), **Calibrate** ([`CalibrationView.tsx`](src/components/CalibrationView.tsx) — collection controls, top‑K pool, stats, live solve + reprojection when `CalibrationResult` is `ok`; adaptive threshold uses the same histogram as **Debug** but the histogram is not shown on this page), **Results** ([`ResultsView.tsx`](src/components/results/ResultsView.tsx) — 3D orbit scene + export when latest result is `ok`), **Debug** ([`GradientProfilesView.tsx`](src/components/gradientProfiles/GradientProfilesView.tsx) at `/debug` — GPU pipeline modes, histograms, edge profiles, quad grid + tag decode, undistort preview).
- **Camera** — [`CameraStreamProvider`](src/components/camera/CameraStreamContext.tsx) at the app root; stream acquisition and device constraints in [`cameraStreamAcquire.ts`](src/components/camera/cameraStreamAcquire.ts).
- **Live WebGPU** — **Calibrate:** [`LiveCameraPipeline.tsx`](src/components/camera/LiveCameraPipeline.tsx) + GPU tag path on **grid** ([`gpuQuadReadback.ts`](src/gpu/gpuQuadReadback.ts)). **Debug:** [`GradientProfilesPipeline.tsx`](src/components/gradientProfiles/GradientProfilesPipeline.tsx) (same GPU stages plus profiles/histograms; see [`docs/gradient-profile-pipeline.md`](docs/gradient-profile-pipeline.md)).

Product summary and roadmap: [`docs/plan.md`](docs/plan.md).

## Coordinate spaces

- **Frame size** — up to 1280×720
- **Raw label values** (pointer-jump) — per-pixel index into the labeling union-find structure (0 … area−1)
- **Compact label values** — 0 … N−1 after canonical remapping; used downstream
- **Compact label cap** — `MAX_EXTENT_COMPONENTS` (4096) limits remapped ids and downstream cluster tables

## Pipeline (per frame)

Compute order matches [`encodeCameraCompute`](src/gpu/cameraComputeEncoding.ts): one command encoder submits **ingest → gray → Sobel → histogram accumulate → NMS → labeling chain** in a single compute pass (then optional slot copies for **grid**).

```
Video frame → ingest (external texture → luma)
  ↓
Grayscale → Sobel → histogram (GPU accumulate; CPU reads bins and sets threshold)
  ↓
NMS + edge filter (threshold uniform written from CPU each frame)
  ↓
Pointer-jump labeling (raw per-pixel labels)
  ↓
Canonical labeling (compact 0..N-1)
  ↓
Render / readback (mode-specific; grid adds tag chain)
```

Labeling and tag detection use the **NMS `filteredBuffer`** directly (Sobel → NMS → pointer-jump).

### Pointer-jump labeling

GPU, ~10 iterations: pointer doubling plus atomic parent tightening.

- **Buffers:** `pointerJumpBuffer0/1` (ping-pong), `pointerJumpAtomicBuffer`
- **Out:** raw labels → canonical pass only

### Canonical labeling

GPU, three passes: reset roots → roots claim compact IDs → pixels remap to compact `compactLabelBuffer`.

## Display modes

| Mode        | GPU work                   | View                        | CPU readback                                                                                                       |
| ----------- | -------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `grayscale` | Gray                       | Luma                        | Histogram                                                                                                          |
| `edges`     | Sobel                      | Edges                       | Histogram                                                                                                          |
| `nms`       | Sobel + NMS                | Edges                       | Histogram                                                                                                          |
| `labels`    | Full chain through compact | False-color labels          | —                                                                                                                  |
| `debug`     | Same as `labels`           | False-color labels          | —                                                                                                                  |
| `grid`      | + GPU tag chain            | Live gray + homography grid (single submit) | Async `readGpuDetection` for HTML overlay / calibration; [frame slot pool](src/gpu/frameSlotPool.ts) tokens only (default 3; busy pool skips the frame) |

## CPU readbacks

| API                    | When                                   | Data                                                                       |
| ---------------------- | -------------------------------------- | -------------------------------------------------------------------------- |
| `readGpuDetection`     | Each **grid** attempt with a free slot | Quad buffer + count (~few KB) via [`detectedQuad.ts`](src/gpu/detectedQuad.ts) |

## Corner and decode pipeline (grid, GPU)

**GPU (grid submit):** grayscale → Sobel → threshold → NMS → pointer-jump → compact → oriented edge histogram → line fit → quad homography ([`quadCornerOrder.ts`](src/gpu/shaders/quadCornerOrder.ts)) → tag decode ([`tagDecodePipeline.ts`](src/gpu/pipelines/tagDecodePipeline.ts)). **Readback:** [`readGpuDetection`](src/gpu/gpuQuadReadback.ts) → [`DetectedQuad`](src/gpu/detectedQuad.ts).

**Failure bitmask** (grid viz tints): same order as [`quadCornerOrder.ts`](src/gpu/shaders/quadCornerOrder.ts) / [`gridVizPipeline.ts`](src/gpu/pipelines/gridVizPipeline.ts) — insufficient edges (0), line fit (2), plausibility (3), no intersections (4).

## AprilTag overlay and host types

1. **Corners** — GPU writes **TL, TR, BL, BR** into `quadCornersBuffer`; host maps via `readGpuDetection`.
2. **Decode** — GPU tag36h11 in `tagDecodePipeline` (`maxError` 3). Dictionary-miss quads get `DECODED_TAG_ID_DICT_MISS` for **?** tint.
3. **Outputs** — [`DetectedQuad`](src/gpu/detectedQuad.ts): `decodedTagId`, `decodedRotation`, optional `vizTagId`. **Custom** (negative) tag ids from layout are a calibration/session concept; live decode is tag36h11 on GPU. Calibrate overlay: plain numbers for dictionary tags; **`*0`, `*1`, …** for session custom tags when configured in [`CalibrationRunContext`](src/components/calibration/CalibrationRunContext.tsx).
4. **Tooling** — [`grid.ts`](src/lib/grid.ts) builds perspective tag grids for geometry tests; [`tag36h11.ts`](src/lib/tag36h11.ts) dictionary helpers remain for unit tests only (live decode is GPU-only).

## Homography

Eight-parameter homography, Gaussian elimination with partial pivot. Shader uses `w` in `outPos.w` for perspective-correct varyings.

## GPU buffers (summary)

| Buffer                                            | Role                                                                                                                    |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `sobelBuffer`                                     | Raw gradients                                                                                                           |
| `filteredBuffer`                                  | After NMS                                                                                                               |
| `pointerJumpBuffer0/1`, `pointerJumpAtomicBuffer` | Labeling                                                                                                                |
| `compactLabelBuffer`                              | Final labels                                                                                                            |
| `canonicalRootBuffer`                             | Canonical id map                                                                                                        |
| `histogramBuffer`                                 | Edge histogram                                                                                                          |
| `quadCornersBuffer`                               | [`GridDataSchema`](src/gpu/pipelines/gridVizPipeline.ts): homography, debug fields, `decodedTagId` (1024 instances max) |
