# Architecture

## Overview

WebGPU runs the vision pipeline: ingest to luma, Sobel, histogram-driven threshold, NMS, pointer-jump connected components, and compact labels. The **grid** path (Calibrate) runs GPU edge clustering, line fit, quad homography, and tag36h11 decode, then reads back only the quad buffer for host callbacks ([`readGpuDetection`](src/gpu/gpuQuadReadback.ts)).

**Calibration** runs in a dedicated worker ([`calibration.worker.ts`](src/workers/calibration.worker.ts)) using `@deluksic/opencv-calibration-wasm`: intrinsics `K`, OpenCV **rational** distortion (`RationalDistortion8` in [`cameraModel.ts`](src/lib/cameraModel.ts)), and per-frame extrinsics. [`CalibrationRunContext`](src/components/calibration/CalibrationRunContext.tsx) owns the live session, worker solves, and **`latestCalibration` / metadata** mirrored for **Results**. **Calibrate** uses an `ok` model for a live **reprojection overlay** on the grid ([`reprojectionOverlayPipeline.ts`](src/gpu/pipelines/reprojectionOverlayPipeline.ts), wired from [`LiveCameraPipeline.tsx`](src/components/camera/LiveCameraPipeline.tsx)). **Results** reads the same context and renders a 3D summary plus JSON export ([`ResultsView.tsx`](src/components/results/ResultsView.tsx), [`exportCalibrationJson.ts`](src/components/results/exportCalibrationJson.ts)).

## App shell (Solid)

- **Views** ([`App.tsx`](src/components/App.tsx)): **Home** ([`Home.tsx`](src/components/Home.tsx)), **Target** (printable SVG), **Calibrate** ([`CalibrationView.tsx`](src/components/CalibrationView.tsx) — collection controls, top‑K pool, stats, live solve + reprojection when `CalibrationResult` is `ok`; adaptive threshold uses the same histogram as **Debug** but the histogram is not shown on this page), **Results** ([`ResultsView.tsx`](src/components/results/ResultsView.tsx) — 3D orbit scene + export when latest result is `ok`), **Debug** ([`GradientProfilesView.tsx`](src/components/gradientProfiles/GradientProfilesView.tsx) at `/debug` — GPU pipeline modes, histograms, edge profiles, quad grid + tag decode, undistort preview).
- **Camera** — [`CameraStreamProvider`](src/components/camera/CameraStreamContext.tsx) at the app root; stream acquisition and device constraints in [`cameraStreamAcquire.ts`](src/components/camera/cameraStreamAcquire.ts).
- **Live WebGPU** — **Calibrate:** [`LiveCameraPipeline.tsx`](src/components/camera/LiveCameraPipeline.tsx) + GPU tag path on **grid** ([`gpuQuadReadback.ts`](src/gpu/gpuQuadReadback.ts)). **Debug:** [`GradientProfilesPipeline.tsx`](src/components/gradientProfiles/GradientProfilesPipeline.tsx) at `/debug` (same GPU tag chain plus orientation/tag histograms, edge profiles, line-fit debug, undistort preview; `/gradient-profiles` redirects here).

Product summary and roadmap: [`docs/plan.md`](docs/plan.md).

## Coordinate spaces

- **Frame size** — up to 1280×720
- **Raw label values** (pointer-jump) — per-pixel index into the labeling union-find structure (0 … area−1)
- **Compact label values** — 0 … N−1 after canonical remapping; used downstream
- **Compact label cap** — `MAX_EXTENT_COMPONENTS` (4096) limits remapped ids and downstream cluster tables

## Pipeline (per frame)

Labeling and tag detection use the **NMS `filteredBuffer`** (Sobel → NMS → pointer-jump). Adaptive threshold: GPU histogram accumulate, CPU reads bins and writes threshold each frame ([`histogramPipelines.ts`](src/gpu/pipelines/histogramPipelines.ts), 95th percentile).

### Base chain (all modes that need labels)

```
Video → ingest (external texture → luma)
  → grayscale → Sobel → histogram accumulate
  → NMS + edge filter
  → pointer-jump labeling (~10 iterations, ping-pong)
  → canonical compact labels (0 … N−1)
```

**Pointer-jump:** `pointerJumpBuffer0/1`, `pointerJumpAtomicBuffer` → raw labels. **Canonical:** reset roots → claim compact IDs → `compactLabelBuffer`. **Cap:** `MAX_EXTENT_COMPONENTS` (4096) on compact ids and downstream cluster tables.

### Quad + tag chain (Calibrate **grid** and Debug)

When a frame slot is active ([`encodeCameraCompute`](src/gpu/cameraComputeEncoding.ts)) or on every Debug frame ([`encodeGradientProfileCompute`](src/gpu/gradientProfileComputeEncoding.ts)):

```
  → edge histogram cluster (per label: orient hist → peaks → TLS line fit → register quads)
  → [Debug only] line-fit debug classify, orient-hist pack
  → scatter flat lineOut (quadCount × 4 edges)
  → quad corner homography (intersections + DLT → grid.quadCornersBuffer)
  → [Debug only] edge profiles along normals (quadGrid / fittedLines / lineFitDebug)
  → tag decode vote passes (32-bin hist → peaks → module votes; quadCount from prior readback)
  → tag decode (classify → 587×quadCount dict → canonicalize corners + H)
  → [Calibrate] host quad pack into readback buffer
```

| Step | Stage | Notes |
|------|--------|--------|
| 1–7 | ingest … compact | Same as base chain |
| 8 | `edgeHistogram.encodeCompute` | Hist reset/accum/find peaks; label line fit; compact quads (`MAX_QUADS` = 512); quad label map; assign `packedEdgeLabels` |
| 9 | `lineFitDebug` | **Debug**, display `lineFitDebug` only |
| 10 | `orientHistViz` | **Debug**, optional side canvas |
| 11 | `lineFit` | Scatter label fits → `lineOut[MAX_FLAT_EDGES]` |
| 12 | `quadHomography` | CCW corners, DLT `H`, `decodedTagId = UNKNOWN` |
| 13 | `profile` | **Debug**, when display is `quadGrid` / `fittedLines` / `lineFitDebug` |
| 14 | `tagDecode.encodeVotePasses` | Plain-u32 GPU clears for hist/votes; render + compute |
| 15 | `tagDecode.encodeDecode` | Classify → dict → `rotateStripCorners` + DLT; no second peak pass |

Wiring: [`cameraPipeline.ts`](src/gpu/cameraPipeline.ts) (Calibrate), [`gradientProfilePipeline.ts`](src/gpu/gradientProfilePipeline.ts) (Debug). Thresholds: [`lineFitThresholds.ts`](src/gpu/lineFitThresholds.ts), [`tagDecodeThresholds.ts`](src/gpu/tagDecodeThresholds.ts).

### Calibrate vs Debug

| | **Calibrate** (`grid`) | **Debug** (`/debug`) |
|--|------------------------|----------------------|
| Encode | [`encodeCameraCompute`](src/gpu/cameraComputeEncoding.ts) + present gray + grid | [`encodeGradientProfileCompute`](src/gpu/gradientProfileComputeEncoding.ts) + mode-specific present |
| Frame slots | [Pool](src/gpu/frameSlotPool.ts) (default 3); busy pool skips frame | Every rAF |
| Extra GPU | Reprojection overlay when calibration `ok` | Profiles, orient/tag histogram canvases, fitted-line overlay, undistort |
| Readback | [`readGpuDetection`](src/gpu/gpuQuadReadback.ts) per slot | Same API when inspecting quads |
| Grid draw | Decoded quads only (`hideNonDecoded`) | Full instance buffer; hash / amber / failure tints |

### Oriented edge clustering

[`edgeHistogramClusterPipeline.ts`](src/gpu/pipelines/edgeHistogramClusterPipeline.ts): 64-bin orientation histogram per compact label; up to **4** peaks ≥ `MIN_PEAK_BIN_SEPARATION` (8 bins). [`labelLineFitPipeline.ts`](src/gpu/pipelines/labelLineFitPipeline.ts): TLS + inlier refine per `labelId × peak`. A label registers as a quad only with four valid peaks, four sides with enough inliers, and orientation spread (rejects two parallel pairs). `quadCount` is atomic per frame; `labelToQuadId` / `quadPeakEdge` index flat edges for profiles and homography.

### Quad corners and homography

[`quadCornerHomographyPipeline.ts`](src/gpu/pipelines/quadCornerHomographyPipeline.ts): intersect adjacent TLS lines, sort CCW, try four polar-ring homography starts ([`homographyDlt.ts`](src/gpu/shaders/homographyDlt.ts)). Corners are stored in **triangle-strip order** **TL, TR, BL, BR** for rendering and DLT; shoelace / edge checks use cyclic perimeter **TL, TR, BR, BL** via `cyclicIdx` in [`quadCornerOrder.ts`](src/gpu/shaders/quadCornerOrder.ts). After decode, **canonicalize** applies `rotateStripCorners` (strip quarter-turns, not polar ring) and recomputes `H`; `decodedRotation` is cleared on GPU.

**Failure bitmask** (grid viz): insufficient edges (0), line fit (2), plausibility (3), no intersections (4) — [`gridVizPipeline.ts`](src/gpu/pipelines/gridVizPipeline.ts).

### GPU tag decode (tag36h11)

[`tagDecodePipeline.ts`](src/gpu/pipelines/tagDecodePipeline.ts): per-quad **32-bin linear** pixel histogram → black/white peaks with `TAG_DECODE_PEAK_GAP_FRAC` (0.25) deadband → per-module votes → classify weak `-1` / tie `-2` / `0`/`1` → parallel dictionary (`587 × quadCount` threads, Hamming `maxError` 3, up to 6 weak wildcards) → canonicalize. Sentinels on `decodedTagId`: `UNKNOWN` (`0xFFFFFFFF`, black grid), `DICT_MISS` (`0xFFFFFFFE`, amber), `0…586` (stable hash via [`hashStableColor.ts`](src/lib/hashStableColor.ts)). Dictionary on GPU only; [`tag36h11.ts`](src/lib/tag36h11.ts) for unit tests.

## Display modes

### Calibrate ([`cameraPipeline.ts`](src/gpu/cameraPipeline.ts))

| Mode | View | CPU readback |
|------|------|----------------|
| `grayscale` / `edges` / `nms` | Luma / Sobel / filtered edges | Histogram |
| `labels` / `debug` | False-color compact labels | — |
| `grid` | Live gray + instanced 8×8 grid overlay | Async `readGpuDetection` when a frame slot is free |
| `undistort` | Calibrated undistort preview | — |

### Debug ([`gradientProfilePresentEncoding.ts`](src/gpu/gradientProfilePresentEncoding.ts))

| Mode | Camera | Overlay / side |
|------|--------|----------------|
| `grayscale` / `edges` / `nms` | Luma / Sobel / NMS | — |
| `labels` | Compact labels | — |
| `quads` | `quadLabelBuffer` | — |
| `edgeLabels` | `packedEdgeLabels` | — |
| `fittedLines` | Dim gray | Green TLS segments (registered quads only) |
| `quadGrid` | Full gray | 8×8 perspective grid per quad |
| `lineFitDebug` | Dim gray + failure colors | Fitted-line overlay |

Side canvases: orientation histograms, per-quad tag grayscale histogram (32 bins), edge gradient profile plot.

## CPU readbacks

| API | When | Data |
|-----|------|------|
| `readGpuDetection` | **grid** / Debug with slot or inspection | [`HostQuadReadback`](src/gpu/pipelines/hostQuadReadbackPipeline.ts) + pattern prefix via [`gpuQuadReadback.ts`](src/gpu/gpuQuadReadback.ts) → [`DetectedQuad`](src/gpu/detectedQuad.ts) |

## AprilTag overlay and host types

1. **Corners** — GPU writes **TL, TR, BL, BR** strip order into `quadCornersBuffer`; host maps via `readGpuDetection` (already canonicalized on GPU).
2. **Decode** — GPU tag36h11; dictionary-miss → `DECODED_TAG_ID_DICT_MISS` (**?**). Calibrate **Start** requires ≥2 **decoded** numeric ids ([`acceptQuadForTagUse.ts`](src/lib/acceptQuadForTagUse.ts)).
3. **Outputs** — `decodedTagId`, optional session **custom** ids (negative) mapped in [`CalibrationRunContext`](src/components/calibration/CalibrationRunContext.tsx); overlay `*0`, `*1`, … for custom tags.
4. **Tooling** — [`grid.ts`](src/lib/grid.ts) builds perspective tag grids for tests.

## Homography

Eight-parameter DLT, Gaussian elimination with partial pivot ([`homographyDlt.ts`](src/gpu/shaders/homographyDlt.ts)). Grid vertex shader uses `w` in clip space for perspective-correct UVs; degenerate `H` falls back to affine `screenCorners`.

## GPU buffers (summary)

| Buffer | Role |
|--------|------|
| `sobelBuffer` | Raw gradients |
| `filteredBuffer` | After NMS |
| `pointerJumpBuffer0/1`, `pointerJumpAtomicBuffer` | Labeling |
| `compactLabelBuffer` | Final compact labels (W×H) |
| `labelClusters[labelId]` | 64-bin orient hist + 4 peaks (`MAX_EXTENT_COMPONENTS`) |
| `labelLineOut`, `labelToQuadId`, `quadPeakEdge`, `quadCount` | Per-label fits and quad registration |
| `packedEdgeLabels` | `quadId×4+edgeId` per edge pixel |
| `lineOut[flatSlot]` | `MAX_FLAT_EDGES` = `MAX_QUADS×4` scattered fits |
| `profileAvg` | **Debug:** 64-bin profiles per flat edge |
| `quadCornersBuffer` | [`GridDataSchema`](src/gpu/pipelines/gridVizPipeline.ts): `MAX_INSTANCES` = `MAX_QUADS` (512) — `homography`, `screenCorners`, `debug`, `decodedTagId`, `decodedRotation` |
| Tag decode atomics / pattern | Per-quad hist, module votes, pattern cells, codewords (587) |

## Implementation map

| Concern | File |
|---------|------|
| Calibrate encode | [`cameraComputeEncoding.ts`](src/gpu/cameraComputeEncoding.ts) |
| Debug encode | [`gradientProfileComputeEncoding.ts`](src/gpu/gradientProfileComputeEncoding.ts) |
| Hist + quad registration | [`edgeHistogramClusterPipeline.ts`](src/gpu/pipelines/edgeHistogramClusterPipeline.ts) |
| TLS per label×peak | [`labelLineFitPipeline.ts`](src/gpu/pipelines/labelLineFitPipeline.ts) |
| Corners + homography | [`quadCornerHomographyPipeline.ts`](src/gpu/pipelines/quadCornerHomographyPipeline.ts) |
| Tag decode | [`tagDecodePipeline.ts`](src/gpu/pipelines/tagDecodePipeline.ts) |
| Grid overlay | [`gridVizPipeline.ts`](src/gpu/pipelines/gridVizPipeline.ts) |
| Host readback pack | [`hostQuadReadbackPipeline.ts`](src/gpu/pipelines/hostQuadReadbackPipeline.ts) |
