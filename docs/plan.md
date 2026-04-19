# Webcam Calibration App — Architecture Plan

## Overview

Browser-based camera calibration using AprilTag 6×6 grid. Client-side only — no backend.

**Stack:** SolidJS 2.0, TypeGPU (WebGPU), CSS Modules

---

## Pages

| View | Description |
|------|-------------|
| **Target** | SVG AprilTag36h11 grid generator for printing |
| **Calibrate** | Live camera feed, GPU pipeline, display modes |
| **Results** | Calibration output + export (stub) |

---

## Detection Pipeline

**GPU (every frame in grid / labels / debug paths):**

1. Grayscale conversion  
2. Sobel edge detection  
3. Histogram + adaptive threshold (90th percentile)  
4. Non-maximum suppression (NMS) + edge filter  
5. Connected component labeling (GPU pointer-jump)  
6. Compact labeling (atomic remap to 0..N-1)  
7. Extent tracking (atomic bounding boxes)  

**CPU (grid mode, throttled readback — `detectContours` / `validateAndFilterQuads`):**

8. Regions from compact labels → per region **corner pipeline** in order: labeled edge pixels → k-means on gradients → **four** line fits (RANSAC+PCA) → all-pairs line intersections (clip to extent bbox ± shared slack) → dedupe → strict convex CCW ordering + TL/TR/BR/BL rotation + plausibility (same slack as intersection clip)  
9. Homography solve per quad (CPU)  
10. Grid render (GPU, instanced homography warp)  

Step 8 detail and failure ordering: **`ARCHITECTURE.md` → Corner Detection**.

---

## Implementation Phases

### Phase 1 — Infrastructure
- [x] Vite + SolidJS + TypeGPU setup
- [x] CSS design system

### Phase 2 — UI Shell
- [x] App with view switching
- [x] CalibrationView with display modes

### Phase 3 — Camera + Grayscale
- [x] Camera access + video display
- [x] Grayscale conversion
- [x] Sobel edge detection
- [x] Histogram + adaptive threshold
- [x] Edge filtering

### Phase 4 — AprilTag Detection
- [x] Edge detection (Sobel + threshold)
- [x] NMS (non-max suppression)
- [x] Connected components (GPU pointer-jump)
- [x] Compact labeling (atomic counter)
- [x] Extent tracking (atomic bounding boxes)

### Phase 4.1 — Quad Fitting
- [x] Region extraction from labels
- [x] K-means clustering on **raw Sobel gradients** (cosine dissimilarity, k=4, restarts)
- [x] **RANSAC + PCA** line fit per cluster (`src/lib/corners.ts`)
- [x] All line-pair intersections → dedupe → strict convex CCW cycle + rotation to TL/TR/BL/BR + plausibility
- [x] Homography solve via Gaussian elimination
- [x] Corner plausibility checks
- [x] Bounding box fallback for failed quads

See **`ARCHITECTURE.md` → Corner Detection** for the ordered CPU stages (what runs *before* line intersection and how that relates to failure codes).

### Phase 4.2 — Tag Decode
- [x] tag36h11 dictionary (587 codings, Hamming distance ≥ 11)
- [x] Hamming distance matching
- [x] Rotation-invariant decoding (all 4 orientations)
- [x] Pattern validation

### Phase 4.3 — Subpixel Refinement
- [ ] Parabolic surface fit on gradient magnitude

### Phase 5 — Pose + Solver
- [ ] EPnP + RANSAC
- [ ] Bundle adjustment (LM)
- [ ] Target point relaxation

### Phase 6 — Full Pipeline
- [ ] Connect all stages
- [ ] Real-time corner overlay
- [ ] View collection + BA trigger
- [ ] Results display + export

### Phase 7 — Target Display
- [x] SVG AprilTag grid generation
- [x] Configurable N×M layout
- [x] Spacing control (as ratio of tag size)
- [x] Optional checkerboard between tags
- [x] Fullscreen mode

### Phase 8 — Polish
- [ ] Motion blur feedback
- [ ] Quality indicators
- [ ] Error states

---

## Camera Model

Brown-Conrady rational tangential (9 params):

```
Intrinsics: fx, fy, cx, cy (4)
Distortion: k1, k2, k3, p1, p2 (5)
```

---

## Good View Criteria

- ≥ 30 tags detected
- RANSAC inlier ratio > 80%
- Reprojection error < 5 px
- Minimum 10–15 views before BA
