# Webcam calibrator — product notes

## Overview

In-browser AprilTag 6×6 target capture; no server. All capture, GPU stages, and CPU decode run client-side.

**Stack:** SolidJS 2.0, TypeGPU (WebGPU), CSS modules.

---

## UI

| View | Role |
| ---- | ---- |
| **Target** | SVG AprilTag36h11 sheet for printing |
| **Calibrate** | Live **grid** view; **Start / Pause / Reset**; top‑K observation pool and session stats |
| **Results** | Camera intrinsics, distortion, and export (solver and wiring not in this build) |
| **Debug** | Full `DisplayMode` set, edge histogram, optional bbox overlay, `Fallbk` (show quads that did not pass dictionary decode), log tail |

**Camera** — shared [`CameraStreamContext`](../src/components/camera/CameraStreamContext.tsx) at the root: `MediaStream`, device selection, `devicechange` refresh, resolution ladder, and `applyConstraints` where supported ([`cameraStreamAcquire.ts`](../src/components/camera/cameraStreamAcquire.ts)). **Live** preview and GPU work: [`LiveCameraPipeline`](../src/components/camera/LiveCameraPipeline.tsx).

---

## Detection pipeline

**GPU (modes that need the full label chain: `labels`, `debug`, `grid`):**

1. Grayscale
2. Sobel
3. Histogram; adaptive edge threshold (95th percentile, `THRESHOLD_PERCENTILE` in [`constants.ts`](../src/gpu/pipelines/constants.ts))
4. NMS and edge filter
5. Pointer-jump CCL
6. Compact remap to 0…N−1
7. Extent (axis-aligned bounds per component)

**CPU-on-grid** — after each submitted grid pass, if a [frame slot](../src/gpu/frameSlotPool.ts) is available, `readDetection` maps staging buffers, builds regions, runs `validateAndFilterQuads`, and completes homography + tag36h11 decode. Frame slots (default: 3) provide backpressure: if all slots are busy, the incoming frame is skipped.

Order for one region: labeled edge samples → k-means (k=4) on NMS `(gx, gy)` → RANSAC+PCA line per cluster → line intersections (with slack) → dedupe → convex order + plausibility → **TL, TR, BL, BR** → `buildTagGrid` / `decodeTagPattern` → `decodeTag36h11AnyRotation(..., ALLOWED_ERROR_COUNT)` with `maxError = 3` → `DetectedQuad` fields. See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the corner table and decode notes.

**Grid draw** — [`gridVizPipeline`](../src/gpu/pipelines/gridVizPipeline.ts) warps a unit grid with the CPU homography; `decodedTagId` drives tint via `stableHashToRgb01` when known.

---

## What ships in this build

- WebGPU frame pipeline: gray → Sobel → threshold → NMS → labeling → extent
- `grid` + async `readDetection` with slot pool
- Per-quad homography, bounding-box fallback, grid visualization, optional `Fallbk` in Debug
- tag36h11 decode (587 codewords, Hamming `maxError` 3 from constants)
- **Calibrate:** top‑K tag observations (id, rotation, score) with merge/eviction; duplicate tag IDs in one frame are rejected; stats panel
- **Target** sheet generator (layout, spacing, optional checker, fullscreen)
- **Results** route: UI shell only; no solver data or export yet

---

## Roadmap (not in this build)

- Subpixel corner refinement (e.g. parabolic fit on gradient magnitude)
- Camera pose: EPnP + RANSAC, then bundle adjustment (e.g. Levenberg–Marquardt)
- **Results:** show intrinsics, rational / OpenCV-style distortion, export
- Refined UX: motion hints, reprojection and inlier stats once a solver exists, stronger error states

**Camera / solver model (planned):** pinhole with OpenCV **rational** distortion (`k1`…`k6`); types in [`cameraModel.ts`](../src/lib/cameraModel.ts). Earlier five-parameter distortion (k1, k2, k3, p1, p2) appears in some OpenCV examples; the solver is expected to use the full rational model.

**Good-view heuristics (for a future BA):** on the order of 30+ tags visible, high RANSAC inlier rate, reprojection under ~5 px, and 10–15+ diverse views before a global solve.

Grid overlay design notes: [`PLAN.md`](../PLAN.md) (homography, buffer layout, corner order **TL, TR, BL, BR** in [`geometry.ts`](../src/lib/geometry.ts)).
