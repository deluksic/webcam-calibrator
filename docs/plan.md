# Webcam calibrator — product notes

## Overview

In-browser AprilTag 6×6 target capture; no server. All capture, GPU stages, and CPU decode run client-side.

**Stack:** SolidJS 2.0, TypeGPU (WebGPU), CSS modules.

---

## UI

| View          | Role                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Home**      | Two-role landing copy; links to Target / Calibrate; dismissible tips (`userGuidancePrefs`)                                         |
| **Target**    | SVG AprilTag36h11 sheet for printing                                                                                                 |
| **Calibrate** | Live **grid**; **Start** / **Snapshot** / **Reset** (no Pause/Stop); persistent **guidance** panel; collapsed **Advanced metrics**; advisory **focus guide** on the feed (hidden after the first pooled frame); **Start** only when ≥2 **decoded** tag IDs appear in frame; top‑K pool; async worker solve + **reprojection** when calibration yields a valid result |
| **Results**   | WebGPU orbit view, **Export JSON**, **Save to library** & **Continue Calibration** when viewing **latest** (not a saved row); **Saved calibrations** list + compare + built-in **Demo calibration** row; displayed calibration = selected library entry or **`latestCalibration`** |
| **Debug**     | GPU gradient-profile pipeline ([`GradientProfilesView`](../src/components/gradientProfiles/GradientProfilesView.tsx) at `/debug`): gray / NMS / labels / quads / line debug / **quad grid** + tag decode, edge threshold histogram, orientation + tag histograms, edge profiles, **undistort** preview from latest calibration |

**Camera** — shared [`CameraStreamContext`](../src/components/camera/CameraStreamContext.tsx) at the root: `MediaStream`, device selection, `devicechange` refresh, resolution ladder, and `applyConstraints` where supported ([`cameraStreamAcquire.ts`](../src/components/camera/cameraStreamAcquire.ts)). **Calibrate** live preview: [`LiveCameraPipeline`](../src/components/camera/LiveCameraPipeline.tsx). **Debug** live preview: [`GradientProfilesPipeline`](../src/components/gradientProfiles/GradientProfilesPipeline.tsx).

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

**GPU-on-grid (Calibrate)** — each rAF with a free [frame slot token](../src/gpu/frameSlotPool.ts) runs ingest, edge clustering, line fit, homography, tag decode, and presents live gray + GPU quad grid in one submit. [`readGpuDetection`](../src/gpu/gpuQuadReadback.ts) reads active quads only for HTML overlay and calibration (no gray snapshot, no corner write-back). Frame slot tokens (default: 3) provide backpressure.

See [`docs/gradient-profile-pipeline.md`](../docs/gradient-profile-pipeline.md) and [`ARCHITECTURE.md`](../ARCHITECTURE.md) for stage order and corner conventions (**TL, TR, BL, BR**).

**Grid draw** — [`gridVizPipeline`](../src/gpu/pipelines/gridVizPipeline.ts) warps a unit square with **`GRID_DIVISIONS`** (8) UV subdivisions using the CPU homography; `decodedTagId` drives tint via `stableHashToRgb01` when known.

---

## What ships in this build

- WebGPU frame pipeline: ingest → gray → Sobel → histogram → threshold → NMS → labeling (see [`cameraComputeEncoding.ts`](../src/gpu/cameraComputeEncoding.ts))
- `grid` + GPU tag decode + async `readGpuDetection` with [frame slot pool](../src/gpu/frameSlotPool.ts) (default 3 slots)
- Per-quad homography, bounding-box fallback, grid visualization on **Calibrate** (`Fallbk` off by default)
- GPU tag36h11 decode (587 codewords, Hamming `maxError` 3 in [`tagDecodePipeline.ts`](../src/gpu/pipelines/tagDecodePipeline.ts))
- **Calibrate:** [`CalibrationRunContext`](../src/components/calibration/CalibrationRunContext.tsx) (session survives route changes); top‑K tag observations with merge/eviction; **OpenCV WASM** solve ([`calibration.worker.ts`](../src/workers/calibration.worker.ts)) and live **reprojection** when solve is `ok`; live grid uses `*0`, `*1`, … for **custom** (negative) tag ids after the first running frame that sees them (`*?` before **Start** / earlier), with blue-on-blue overlay styling ([`LiveCameraPipelineOverlays.tsx`](../src/components/camera/LiveCameraPipelineOverlays.tsx))
- **Target** sheet generator (layout, spacing, optional checker, fullscreen)
- **Results:** 3D WebGPU scene ([`resultsCanvasPipeline.ts`](../src/gpu/resultsCanvasPipeline.ts)) + **Export JSON** ([`exportCalibrationJson.ts`](../src/components/results/exportCalibrationJson.ts)); **`latestCalibration`** and metadata on [`CalibrationRunContext`](../src/components/calibration/CalibrationRunContext.tsx); [`CalibrationLibraryContext`](../src/components/calibration/CalibrationLibraryContext.tsx) for saved runs + default **Demo calibration** ([`demoCalibrationExample.ts`](../src/lib/demoCalibrationExample.ts))

---

## Roadmap (not in this build)

- Subpixel corner refinement (e.g. parabolic fit on gradient magnitude)
- Richer **Results** UI: inline numeric intrinsics / distortion tables, inlier breakdown, motion / capture hints (export and 3D summary already exist)
- **Saved calibrations:** persist library to `localStorage` (entries are in-memory only today; refresh clears user-added rows but restores the bundled demo entry)

**Camera / solver model (in use):** pinhole `K` and OpenCV **rational** distortion (`k1`…`k6` as `RationalDistortion8`); types in [`cameraModel.ts`](../src/lib/cameraModel.ts). The WASM worker returns these in `CalibrationOk`.

**Capture-quality heuristics (informal):** on the order of many visible tags, stable focus, and several diverse views improve robustness; the app enforces a minimum view count before reporting `ok` (see **Calibrate** solve path).

Grid overlay design notes: [`PLAN.md`](../PLAN.md) (homography, buffer layout, corner order **TL, TR, BL, BR** in [`geometry.ts`](../src/lib/geometry.ts)).
