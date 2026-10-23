# Architecture

## Overview

The browser app calibrates a camera against an editable hex target. All vision work is client-side. The ONNX model is fetched from a GitHub release asset using [`model-manifest.json`](../model-manifest.json) before dev/build; see [`model-weights.md`](model-weights.md).

```text
camera frame
  → ROI ONNX mask (256 px wide, aspect-following height)
  → histogram segmentation: black / gray / white
  → 2×2 color-aware opening, connected components, centroid NMS
  → observed proximity graph + binary-color fingerprints
  → identity match in a worker: hash first, local homography/ICP recovery for ambiguous markers
  → accepted point observations { designId, x, y }
  → per-session frame pool
  → OpenCV calibrateCameraRO in a worker
  → K, rational distortion, per-frame poses, refined target points
```

The target is configured on the Target page: columns, rows, points per cell edge, marker scale, color seed, and resolution; identity depth remains an advanced URL/default value. [`config.ts`](../src/lib/hexTarget/config.ts) parses and serializes these values; the sidebar builds a Calibrate URL containing the full configuration and renders it as a QR code. The Calibrate page reads those URL parameters and builds its target model and identity indexes from them. The defaults are 8 columns, 4 rows, 4 points per edge, 40% marker scale, seed 1, depth 12.

## App shell (Solid)

- **Views** ([`App.tsx`](../src/components/App.tsx)): **Home**, **Target** ([`HexTargetView.tsx`](../src/components/HexTargetView.tsx) — editable target, QR Calibrate link, fullscreen, JSON export), **Calibrate** ([`CalibrationView.tsx`](../src/components/CalibrationView.tsx)), **Results** ([`ResultsView.tsx`](../src/components/results/ResultsView.tsx)), **Point Lab**, **ROI Lab**. Removed routes (`/hex-target`, `/debug`, `/gradient-profiles`) are not redirected.
- **Camera** — [`CameraStreamProvider`](../src/components/camera/CameraStreamContext.tsx) at the app root; stream acquisition and device constraints in [`cameraStreamAcquire.ts`](../src/components/camera/cameraStreamAcquire.ts).
- **Live capture** — [`HexCameraPipeline.tsx`](../src/components/camera/HexCameraPipeline.tsx) wraps [`roiLabPipeline.ts`](../src/gpu/roiLab/roiLabPipeline.ts): ONNX mask, component readback, identity graph overlay, snapshot capture, and a lightweight reprojection overlay. The pipeline keeps two frame slots (encode N+1 while presenting N) and `HexCameraPipeline` hands identity to [`hexMatch.worker.ts`](../src/workers/hexMatch.worker.ts), so marker matching overlaps the next frame's GPU work instead of blocking the camera loop.
- **Calibration session** — [`CalibrationRunContext`](../src/components/calibration/CalibrationRunContext.tsx) owns the frame pool, the background worker solve, and the latest result shared with Results.

## Target geometry and identity

- [`config.ts`](../src/lib/hexTarget/config.ts) — editable target type, defaults, ranges, URL serialization/parsing, and Calibrate link building.
- [`target.ts`](../src/lib/hexTarget/target.ts) — pattern/color generation from a config, proximity graph, depth indexes, a cached design-side ICP bundle, and normalized object points.
- [`hexBlobTarget.ts`](../src/lib/hexBlobTarget.ts) — display layout: integer pixel squares, padding, resolution-independent design points.
- [`match.ts`](../src/lib/hexTarget/match.ts) — observed graph + hash match + local-ICP recovery. Returns matched identities, observed edges, and counts. Calibrate runs it in [`hexMatch.worker.ts`](../src/workers/hexMatch.worker.ts), which caches the target model across frames.
- [`roiBackgroundPipeline.ts`](../src/gpu/roiLab/roiBackgroundPipeline.ts) — mask-aware low-frequency illumination estimate from the ONNX downsample.
- [`calibrationPoints.ts`](../src/lib/hexTarget/calibrationPoints.ts) — accepts resolved markers and safe recovered markers; rejects duplicate design IDs for the frame.

Object coordinates are the design-space marker centers divided by the median lattice spacing, with `z = 0`. Calibration is therefore scale-free; extrinsics are in “design-spacing” units. The export names this unit explicitly.

## Calibration

- [`calibrationInput.ts`](../src/lib/calibrationInput.ts) — pure builder. Requires ≥3 usable views, ≥8 identified points per frame, and ≥8 design IDs shared by every view; builds the common object-point template required by object-release mode.
- [`calibration.worker.ts`](../src/workers/calibration.worker.ts) — `calibrateCameraRO` with OpenCV’s rational distortion model; returns K, distortion, per-frame extrinsics, RMS, and the refined shared target points.
- [`calibrationValidFrames.ts`](../src/lib/calibrationValidFrames.ts), [`calibrationTopK.ts`](../src/lib/calibrationTopK.ts) — pool filtering and frame eviction.
- [`hexReprojection.ts`](../src/lib/hexReprojection.ts) — live `solvePnP`/`projectPoints` overlay while a calibration is available.

`CalibrationOk` in memory contains `K`, `distortion`, `extrinsics`, `updatedTargetPoints`, `rmsPx`, `perFrameRmsPx`, `imageSize`, and `cameraId`.

## Results and export

[`resultsSceneCpu.ts`](../src/gpu/pipelines/resultsSceneCpu.ts) and [`resultsMarkerPipeline.ts`](../src/gpu/pipelines/resultsMarkerPipeline.ts) render the refined points in the orbit view. The tag-quad renderer was removed.

[`exportCalibrationJson.ts`](../src/components/results/exportCalibrationJson.ts) writes **schema v2**:

```jsonc
{
  "schemaVersion": 2,
  "kind": "ok",
  "target": {
    "kind": "hex",
    "columns": 8,
    "rows": 4,
    "pointsPerEdge": 4,
    "colorSeed": 1,
    "markerScale": 0.4,
    "identityDepth": 12,
    "widthPx": 1280,
    "heightPx": 960,
    "objectUnits": "median-design-spacing",
    "colors": [],
    "markers": [{ "designId": 0, "x": 0, "y": 0, "z": 0 }],
  },
  "K": { "fx": 0, "fy": 0, "cx": 0, "cy": 0 },
  "distortion": [0, 0, 0, 0, 0, 0, 0, 0],
  "extrinsics": [{ "frameId": 1, "R": [], "t": {} }],
  "rmsPx": 0,
  "perFrameRmsPx": [[1, 0]],
  "imageSize": { "width": 640, "height": 480 },
  "cameraId": "…",
  "frameImages": { "1": "frame_1.png" },
  "observations": [{ "frameId": 1, "imageFile": "frame_1.png", "points": [{ "designId": 0, "x": 0, "y": 0 }] }],
}
```

AprilTag exports (schema v1) remain valid files; the current differential renderer consumes them. New hex exports require a hex-aware differential renderer, which can use the camera model, per-frame poses, refined markers, marker colors/scale, observations, and PNGs above.

## Labs

- **Point Lab** ([`PointLabView.tsx`](../src/components/pointLab/PointLabView.tsx)) — synthetic projection, fingerprint matching, local homography recovery, and stress metrics.
- **ROI Lab** ([`RoiLabView.tsx`](../src/components/roiLab/RoiLabView.tsx)) — configurable detector/identity diagnostics, seed scan, component and label views.

## Deleted legacy stack

The AprilTag browser implementation was removed: printable tag sheet, Sobel/edge/quad/homography chain, tag36h11 GPU decode, grid/tag overlays, Debug page, live tag reprojection, and their tests. Point-identity experiments and Python notebooks remain.
