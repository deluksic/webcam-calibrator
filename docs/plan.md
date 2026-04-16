# Webcam Calibration App — Architecture Plan

## Overview

Browser-based camera calibration using AprilTag 6×6 grid. Client-side only — no backend.

**Stack:** SolidJS 2.0, TypeGPU (WebGPU), CSS Modules

---

## Pages

### Page 1 — Home / Target Display

Display the calibration target (AprilTag 6×6 grid) full-screen for printing or screen display.

- SVG rendering for crisp output at any DPI
- Size/spacing controls
- Print/export

### Page 2 — Calibration / Live Feed

Stream from camera, detect AprilTags, collect observations, run solver.

**Detection stages:**
1. Grayscale conversion
2. Sobel edge detection
3. Adaptive threshold (85th percentile)
4. Non-maximum suppression (NMS)
5. Edge dilation (tangent-only fill)
6. Connected component labeling (pointer-jump)
7. Quad fitting + tag decode
8. Subpixel corner refinement

**Pose estimation:**
1. EPnP + RANSAC
2. Bundle adjustment (LM)
3. Target point relaxation

### Page 3 — Results

Display intrinsics, distortion, reprojection error stats. Export OpenCV YAML or JSON.

---

## Module Map

```
src/
├── lib/
│   ├── april-tag-gen.ts       — Generate 6×6 tag grid as SVG
│   ├── geometry.ts            — 2D geometry utilities (line intersection, fitLine)
│   ├── corners.ts             — Corner detection via tangent turns
│   ├── grid.ts                — Perspective-correct grid subdivision
│   ├── tag36h11.ts            — tag36h11 dictionary + decode
│   └── contour.ts            — Region extraction + quad fitting
│
├── gpu/
│   ├── camera.ts             — Pipeline factory + per-frame processing
│   ├── pipelines/            — GPU compute shaders
│   └── contour.ts           — Connected components + quad fit
│
├── solver/
│   ├── pnp.ts                — EPnP + RANSAC
│   ├── ba.ts                 — Bundle adjustment + LM
│   └── camera-model.ts       — Intrinsics, distortion, projection
│
├── store/
│   └── calibration.ts        — SolidJS reactive store
│
└── components/
    ├── CalibrationView.tsx   — Live feed + detection
    ├── TargetDisplay.tsx     — AprilTag SVG grid
    ├── ResultsPanel.tsx      — Parameters + export
    └── App.tsx               — View switching
```

---

## Implementation Phases

### Phase 1 — Infrastructure ✅
- [x] Vite + SolidJS + TypeGPU setup
- [x] CSS design system

### Phase 2 — UI Shell ✅
- [x] App with view switching
- [x] CalibrationView with display modes

### Phase 3 — Camera + Grayscale ✅
- [x] Camera access + video display
- [x] Grayscale conversion
- [x] Sobel edge detection
- [x] Histogram + adaptive threshold
- [x] Edge filtering

### Phase 4 — AprilTag Detection ✅
- [x] Edge detection (Sobel + threshold)
- [x] NMS (non-max suppression)
- [x] Edge dilation (tangent-only)
- [x] Connected components (pointer-jump)
- [x] Compact labeling (atomic counter)

### Phase 4.1 — Quad Fitting ✅
- [x] Sort components by area (descending), largest = tag boundary
- [x] Extract contour pixels with edge directions (Sobel tangent)
- [x] Detect 4 corners (sharp turns in tangent, weighted by magnitude)
- [x] Build perspective-correct grid using line intersection + proportional subdivision

**Grid Construction:**
- [x] Subdivide each edge into 6 equal segments
- [x] Connect corresponding division points on opposite edges → grid lines
- [x] Sample multiple pixels per cell for decode

### Phase 4.2 — Tag Decode ✅
- [x] Define tag36h11 dictionary (587 codings, Hamming distance ≥ 11)
- [x] Match 6×6 pattern against dictionary
- [x] Validate using grid regularity checks (cell sizes, pattern consistency)
- [x] Reject invalid patterns (excessive -1 cells or bad orientation consensus)
- [x] Support rotation-invariant decoding (try all 4 orientations)

### Phase 4.3 — Subpixel Refinement
- [ ] Parabolic surface fit on gradient magnitude in 5×5 window
- [ ] Solve analytically for subpixel corner position
- [ ] Iterative refinement for accuracy

### Phase 4.3 — Subpixel Refinement
- [ ] Parabolic surface fit on gradient magnitude in 5×5 window
- [ ] Solve analytically for subpixel corner position

### Phase 5 — Pose + Solver
- [ ] EPnP implementation
- [ ] RANSAC loop
- [ ] Bundle adjustment (LM)
- [ ] Target point relaxation

### Phase 6 — Full Pipeline
- [ ] Connect all stages
- [ ] Real-time corner overlay
- [ ] View collection + BA trigger
- [ ] Results display + export

### Phase 7 — Target Display ✅
- [x] SVG AprilTag grid generation
- [x] Configurable NxM layout
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

- ≥30 tags detected
- RANSAC inlier ratio > 80%
- Reprojection error < 5px
- Minimum 10–15 views before BA