# Architecture

## Overview

The webcam calibrator uses WebGPU compute shaders to perform camera calibration via AprilTag detection. The pipeline runs entirely on the GPU except for minimal CPU readbacks used for debug visualization and grid rendering.

## Coordinate Spaces

- **Frame size**: up to 1280×720 (HD)
- **Raw label values**: pixel indices (0 to area-1) — output of pointer-jump labeling
- **Compact label values**: 0 to N-1 — remapped by canonical labeling, used everywhere downstream
- **Extent buffer keys**: compact label values (must be < MAX_EXTENT_COMPONENTS = 16384)

## Pipeline Stages (per frame)

```
Frame Input
  ↓
Sobel Gradient → Histogram (adaptive threshold display)
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

GPU-only. ~10 iterations per frame via pointer doubling + atomic parent-tightening.

- **Bufs**: `pointerJumpBuffer0`, `pointerJumpBuffer1` (ping-pong), `pointerJumpAtomicBuffer`
- **Output**: raw pixel-index labels — used only as input to canonical labeling

### Canonical Labeling

GPU-only. 3-pass remap: reset → atomic root claiming → canonical read.

- **Pass 1**: Reset `canonicalRootBuffer[i] = INVALID`
- **Pass 2**: Each root atomically claims `compactCounter++` as its canonical ID
- **Pass 3**: Every pixel reads `L[i] = canonicalRootBuffer[label[i]]`
- **Output**: `compactLabelBuffer` — used by all downstream stages

### Extent Tracking

GPU-only. One reset + one track dispatch. Atomically tracks (minX, minY, maxX, maxY) per component.

- **Buf**: `extentBuffer` — sized for MAX_EXTENT_COMPONENTS entries

## Display Modes

| Mode | GPU Compute | Render | CPU Readback |
|------|-------------|--------|--------------|
| `grayscale` | Gray | grayscale | histogram |
| `edges` | Sobel | edges | histogram |
| `nms` | Sobel + NMS | edges | histogram |
| `labels` | Sobel + pointer-jump + compact | labels (hash-based colors) | none |
| `debug` | Sobel + pointer-jump + extent | labels + bbox overlay | extent buffer |
| `grid` | Sobel + pointer-jump + extent | grayscale + homography-warped quad grid | detectContours every ~30 frames |

## CPU Readbacks

| Function | When | What |
|---------|------|------|
| `readExtentBuffer()` | Every frame in debug mode | Extent entries (320 KB) |
| `detectContours()` | Every ~30 frames in grid mode | Full label + gradient buffers (~11 MB) |

## Corner Detection (grid mode, CPU)

Only runs in grid mode, every ~30 frames:

1. Extract regions from compact labels
2. For each region: extract edge pixels within bounding box
3. Compute tangent per pixel: `atan2(gy, gx) + π/2`
4. K-means (k=4) in circular tangent space — 3 random restarts, seeded LCG RNG
5. RANSAC line fitting per cluster (50 iterations)
6. Pair perpendicular lines → line intersection → 4 corner points
7. Plausibility checks: R² ≥ 0.80, convex quad, edge length ratios
8. Fall back to bounding box corners if corner detection fails

## Homography

Per-quad 8-parameter homography solved via Gaussian elimination with partial pivoting.

```
w = h7*u + h8*v + 1
x = (h1*u + h2*v + h3) / w
y = (h4*u + h5*v + h6) / w
```

Vertex shader passes `w` in `outPos.w` for automatic perspective-correct interpolation.

## GPU Buffers

| Buffer | Size | Type |
|--------|------|------|
| `sobelBuffer` | W×H×vec2f | gradient (gx, gy) |
| `filteredBuffer` | W×H×vec2f | NMS-suppressed gradient |
| `pointerJumpBuffer0/1` | W×H×u32 | labels (ping-pong) |
| `pointerJumpAtomicBuffer` | W×H×atomic u32 | atomic labels |
| `compactLabelBuffer` | W×H×u32 | compact labels |
| `canonicalRootBuffer` | area×atomic u32 | canonical root IDs |
| `histogramBuffer` | 256×atomic u32 | edge histogram |
| `extentBuffer` | MAX_EXTENT_COMPONENTS×ExtentEntry | bounding boxes |
| `quadCornersBuffer` | MAX_INSTANCES×3×vec4f | homography params + hasCorners flag |
