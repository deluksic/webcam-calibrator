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
│   └── tag36h11.ts           — tag36h11 dictionary + decode
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

### Phase 4.1 — Quad Fitting
- [ ] Sort components by area (descending), largest = tag boundary
- [ ] Extract contour pixels with edge directions (Sobel tangent)
- [ ] Detect 4 corners (sharp turns in tangent, weighted by magnitude)
- [ ] Validate corners via projective geometry before proceeding

**Grid Construction (perspective-correct):**
- Use line intersection + proportional subdivision
- Assume square tag with equal sides
- Divide each edge into 6 equal segments (1/6, 2/6, ..., 5/6)
- Connect corresponding division points on opposite edges → grid lines
- At each cell corner, sample multiple pixels for decode

### Phase 4.2 — Tag Decode
- [ ] Group samples by cell, use gradient direction consensus for black/white
- [ ] Validate using projective checks (cell sizes, grid regularity in (u,v) space)
- [ ] Match 6×6 pattern against tag36h11 dictionary
- [ ] Reject invalid patterns
- [ ] Accept successful decode

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

### Phase 7 — Target Display
- [ ] SVG AprilTag grid generation
- [ ] Size/spacing controls
- [ ] Print/export

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