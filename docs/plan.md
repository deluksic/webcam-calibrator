# Webcam calibrator — product notes

## Overview

In-browser AprilTag 6×6 target capture; no server. All capture, GPU stages, and CPU decode run client-side.

**Stack:** SolidJS 2.0, TypeGPU (WebGPU), CSS modules.

---

## UI

| View          | Role                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Home**      | Landing copy and links to Target / Calibrate                                                                                         |
| **Target**    | SVG AprilTag36h11 sheet for printing                                                                                                 |
| **Calibrate** | Live **grid**; **Start / Pause / Reset**; top‑K pool and stats; async worker solve + **reprojection overlay** when `CalibrationResult` is `ok` |
| **Results**   | After an **`ok`** solve: WebGPU orbit view (refined geometry) and **Export JSON**; empty state points users to Calibrate (**Reset** clears shared latest result) |
| **Debug**     | Full `DisplayMode` set, edge histogram, optional bbox overlay, `Fallbk` (show quads that did not pass dictionary decode), log tail   |

**Camera** — shared [`CameraStreamContext`](../src/components/camera/CameraStreamContext.tsx) at the root: `MediaStream`, device selection, `devicechange` refresh, resolution ladder, and `applyConstraints` where supported ([`cameraStreamAcquire.ts`](../src/components/camera/cameraStreamAcquire.ts)). **Live** preview and GPU work: [`LiveCameraPipeline`](../src/components/camera/LiveCameraPipeline.tsx).

---

## Detection pipeline

**GPU (modes that need the full label chain: `labels`, `debug`, `grid`):**

1. Grayscale
2. Sobel
3. Histogram; adaptive edge threshold (95th percentile, `THRESHOLD_PERCENTILE` in [`histogramPipelines.ts`](../src/gpu/pipelines/histogramPipelines.ts))
4. NMS and edge filter
5. Pointer-jump CCL
6. Compact remap to 0…N−1
7. Extent (axis-aligned bounds per component)

**CPU-on-grid** — after each submitted grid pass, if a [frame slot](../src/gpu/frameSlotPool.ts) is available, `readDetection` maps staging buffers, builds regions, runs `validateAndFilterQuads`, and completes homography + tag36h11 decode. Frame slots (default: 3) provide backpressure: if all slots are busy, the incoming frame is skipped.

Order for one region: labeled edge samples → k-means (k=4) on NMS `(gx, gy)` → RANSAC+PCA line per cluster → line intersections (with slack) → dedupe → convex order + plausibility → **TL, TR, BL, BR** → `buildTagGrid` / `decodeTagPattern` → `decodeTag36h11AnyRotation(..., ALLOWED_ERROR_COUNT)` with `maxError = 3` → `DetectedQuad` fields. See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the corner table and decode notes.

**Grid draw** — [`gridVizPipeline`](../src/gpu/pipelines/gridVizPipeline.ts) warps a unit square with **`GRID_DIVISIONS`** (8) UV subdivisions using the CPU homography; `decodedTagId` drives tint via `stableHashToRgb01` when known.

---

## What ships in this build

- WebGPU frame pipeline: ingest → gray → Sobel → histogram → threshold → NMS → labeling → extent (see [`cameraComputeEncoding.ts`](../src/gpu/cameraComputeEncoding.ts))
- `grid` + async `readDetection` with [frame slot pool](../src/gpu/frameSlotPool.ts) (default 3 slots)
- Per-quad homography, bounding-box fallback, grid visualization, optional `Fallbk` in Debug
- tag36h11 decode (587 codewords, Hamming `maxError` 3 from constants)
- **Calibrate:** top‑K tag observations (id, rotation, score) with merge/eviction; duplicate tag IDs in one frame are rejected; stats panel; **OpenCV WASM** solve ([`calibration.worker.ts`](../src/workers/calibration.worker.ts)) and live **reprojection** when solve is `ok`
- **Target** sheet generator (layout, spacing, optional checker, fullscreen)
- **Results:** 3D WebGPU scene ([`resultsCanvasPipeline.ts`](../src/gpu/resultsCanvasPipeline.ts)) + **Export JSON** for `CalibrationOk` ([`exportCalibrationJson.ts`](../src/components/results/exportCalibrationJson.ts)); latest result shared via [`CalibrationLatestContext`](../src/components/calibration/CalibrationLatestContext.tsx)

---

## Roadmap (not in this build)

- Subpixel corner refinement (e.g. parabolic fit on gradient magnitude)
- Richer **Results** UI: inline numeric intrinsics / distortion tables, inlier breakdown, motion / capture hints (export and 3D summary already exist)
- Refined UX: stronger error states, session persistence beyond in-memory latest calibration

**Camera / solver model (in use):** pinhole `K` and OpenCV **rational** distortion (`k1`…`k6` as `RationalDistortion8`); types in [`cameraModel.ts`](../src/lib/cameraModel.ts). The WASM worker returns these in `CalibrationOk`.

**Capture-quality heuristics (informal):** on the order of many visible tags, stable focus, and several diverse views improve robustness; the app enforces a minimum view count before reporting `ok` (see **Calibrate** solve path).

Grid overlay design notes: [`PLAN.md`](../PLAN.md) (homography, buffer layout, corner order **TL, TR, BL, BR** in [`geometry.ts`](../src/lib/geometry.ts)).
