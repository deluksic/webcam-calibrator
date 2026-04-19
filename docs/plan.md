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

8. Regions from compact labels → per region **corner pipeline** in order: labeled edge pixels → k-means on gradients → **four** line fits (RANSAC+PCA) → all-pairs line intersections (clip to extent bbox ± shared slack) → dedupe → strict convex CCW ordering + plausibility; emit corners **TL, TR, BL, BR** for homography (see **`ARCHITECTURE.md` → Corner Detection**).  
9. **AprilTag decode (same pass)** — remap corners to **TL, TR, BR, BL** for `buildTagGrid` → `decodeTagPattern` (quad bbox pixel loop, inverse homography → tag UV, **8×8** center+diagonal half-spaces → two adjacent bins, **unweighted** ±1 **`decodeVoteBinRadialDot`** per bin on **NMS-filtered** `(gx, gy)` — same readback family as corner Sobel inputs) → `decodeTag36h11AnyRotation` → optional `decodedTagId` / `decodedRotation` on `DetectedQuad`.  
10. Homography solve per quad (CPU) → GPU `quadCornersBuffer` for instanced grid viz.  
11. Grid render (GPU, `gridVizPipeline`: homography warp + fragment **8×8** grid lines; successful quads with a CPU-supplied id get **`stableHashToRgb01`** tint from the buffer’s **`decodedTagId`** field, populated from **`vizTagId`** when the dictionary match exists).  

Step 8 failure ordering: **`ARCHITECTURE.md` → Corner Detection**. Grid + decode detail: **`ARCHITECTURE.md` → AprilTag grid + decode**.

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
- [x] tag36h11 dictionary (587 codings; Hamming match in `decodeTag36h11`, configurable `maxError`)
- [x] Hamming distance matching + rotation-invariant decode (`decodeTag36h11AnyRotation`)
- [x] End-to-end CPU wire: `validateAndFilterQuads` → `buildTagGrid` (with TL/TR/BR/BL corner order) → `decodeTagPattern` → dictionary decode; `pattern`, `decodedTagId`, `decodedRotation` on `DetectedQuad`
- [x] UI / GPU tint: prefer decoded id; show **`?`** when no dictionary match (see `CalibrationView`)
- [x] **Robust decode (homography + 8×8)** — `decodeTagPattern`: bbox scan, inverse **H** → `(u,v)`, NMS-filtered Sobel, tag-UV gradient, half-space triangle per sample → **two** adjacent module bins; **unweighted** ±1 votes use **`decodeVoteBinRadialDot`** per bin (**`gᵤ(u−cu)+gᵥ(v−cv)`**); proximity **`max(τ, 2/L_min, 0.5/8)`** in UV (`L_min` = shortest quad edge px). No bbox quantile **`magCut`**, no **`mag²`** weighting—edge filtering is **GPU NMS**. Inner **6×6** pattern with **`-1`** / **`-2`** + **`fillUnknownNeighbors6`** (**`-1`** only). Optional **`buildDecodeEdgeMask`**; live path passes **`undefined`**. Per-cell **`decodeCell`** remains for unit tests.

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
