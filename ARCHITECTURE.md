# Architecture

## Overview

The webcam calibrator uses WebGPU compute shaders to perform camera calibration via AprilTag detection. The pipeline runs entirely on the GPU except for minimal CPU readbacks used for debug visualization and grid rendering.

## Coordinate Spaces

- **Frame size**: 1280×720 (HD)
- **Raw label values**: pixel indices (0 to 921,599 for HD) — output of pointer-jump labeling
- **Compact label values**: 0 to N-1 — remapped by canonical labeling, used everywhere downstream
- **Extent buffer keys**: compact label values (must be < MAX_EXTENT_COMPONENTS = 16384)

## Pipeline Stages (per frame)

Each `processFrame()` call executes these stages:

```
Frame Input
  ↓
Sobel Gradient → Histogram (for adaptive threshold)
  ↓
NMS + Edge Filter
  ↓
Pointer-Jump Labeling (iterative connected components → raw pixel-index labels)
  ↓
Canonical Labeling (3-pass remap → compact IDs 0..N-1)
  ↓
Extent Tracking (atomic min/max per compact label)
  ↓
Render (mode-dependent visualization)
```

### Pointer-Jump Labeling

GPU-only. Runs ~10 iterations per frame. Each pixel initially labels itself; iterations resolve root labels via atomic operations. No CPU readback.

- **Bufs**: `pointerJumpBuffer0`, `pointerJumpBuffer1` (ping-pong)
- **Output**: raw pixel-index labels (0..921,599) — used only as input to canonical labeling
- **Cost**: ~10 compute dispatches per frame, ~2ms on typical GPU

### Canonical Labeling

GPU-only. 3-pass algorithm that remaps raw pixel-index labels to compact IDs (0..N-1). Always runs after pointer-jump — its output (`compactLabelBuffer`) is used everywhere downstream (labelViz render, extent tracking, detectContours). Without this step, labels exceed MAX_EXTENT_COMPONENTS and are silently dropped from extent tracking.

- **Pass 1**: Reset `canonicalRootBuffer[i] = INVALID`
- **Pass 2**: Each root pixel atomically claims `compactCounter++` as its canonical ID
- **Pass 3**: Every pixel reads `L[i] = canonicalRootBuffer[label[i]]` (compact ID)
- **Buf**: `canonicalRootBuffer` (read/write), `compactLabelBuffer` (output)
- **Cost**: 3 dispatchWorkgroups calls, negligible

### Extent Tracking

GPU-only. One reset dispatch + one track dispatch per frame. Atomically tracks (minX, minY, maxX, maxY) per component.

- **Buf**: `extentBuffer` — sized for `MAX_EXTENT_COMPONENTS` entries
- **Cost**: negligible (<0.1ms)

### CPU Readbacks

Only two readback operations exist, both initiated from the JS render loop:

| Function | When | What | Size |
|---|---|---|---|
| `readExtentBuffer()` | Every frame in debug mode | Extent entries | 320 KB |
| `detectContours()` | Every 30 frames in grid mode | Full label + sobel buffers | ~2.5 MB |

## Extent Buffer

The extent buffer stores bounding boxes for each labeled component. Components are keyed by their root pixel index (label value). The buffer must be large enough to hold all components.

**Format** (4× u32 per entry):

```
entry[i*4 + 0] = minX (or 0xFFFFFFFF if uninitialized)
entry[i*4 + 1] = minY
entry[i*4 + 2] = maxX
entry[i*4 + 3] = maxY
```

Unused entries: `minX = 0xFFFF`

**Sizing**: `MAX_EXTENT_COMPONENTS = 16384` → buffer is `16384 * 5 * 4 = 320 KB`

Components with label ≥ MAX_EXTENT_COMPONENTS are not tracked (acceptable: large single-component noise regions are not calibration targets).

## Display Modes

| Mode | GPU Compute | Render | CPU Readback |
|---|---|---|---|
| `grayscale` | Sobel + edge | grayscale | histogram |
| `edges` | Sobel + edge | edges | histogram |
| `nms` | Sobel + NMS | edges | histogram |
| `edgesDilated` | *(removed — NMS alone is sufficient)* | — | — |
| `labels` | Sobel + pointer-jump + compact | labels | none |
| `debug` | Sobel + pointer-jump + extent | labels + bbox overlay | extent buffer (every frame) |
| `grid` | Sobel + pointer-jump + extent | labels + grid overlay | extent buffer (every frame) + detectContours (every 30 frames) |

## detectContours (Grid Mode)

Reads compact label buffer + sobel buffer, then performs CPU-side corner detection via edge orientation clustering. Uses the same compact labels produced by the per-frame pipeline (no separate pointer-jump needed).

```
detectContours():
  GPU→CPU: read compactLabelBuffer + dilatedEdgeBuffer (~2.5 MB)
  CPU: extract regions from labels
  CPU: for each region, extract edge pixels
  CPU: find corner candidates (pixels where neighbors have differing orientations)
  CPU: cluster into 4 quadrants → corner points
  CPU: perspective-correct grid fitting
  GPU→CPU: read extentBuffer (debug overlay, ~320 KB)
  return { quads, extentData, dilatedGradients, labelData }
```

This is the expensive operation (full buffer readback + CPU corner detection) and is why it's gated to grid mode only.

## GPU Buffers

| Buffer | Size | Type | Access |
|---|---|---|---|
| `sobelBuffer` | 1280×720×vec2f | gradient | read |
| `filteredBuffer` | 1280×720×vec2f | NMS-suppressed gradient | read/write |
| `pointerJumpBuffer0/1` | 1280×720×u32 | labels | read/write |
| `pointerJumpAtomicBuffer` | 1280×720×atomic u32 | atomic labels | read/write |
| `compactLabelBuffer` | 1280×720×u32 | compact labels (0..N-1) | read/write |
| `canonicalRootBuffer` | area×atomic u32 | canonical root IDs | read/write |
| `histogramBuffer` | 256×atomic u32 | edge histogram | read/write |
| `extentBuffer` | MAX_EXTENT_COMPONENTS×ExtentEntry | bounding boxes | read/write |
| `quadCornersBuffer` | MAX_DETECTED_TAGS×8×f32 | grid corners | write |

All gradient buffers (`sobelBuffer`, `filteredBuffer`, `dilatedEdgeBuffer`) store `vec2f` (gx, gy) per pixel.
Magnitude is computed on-the-fly via `length()` in rendering kernels — no separate scalar storage.
