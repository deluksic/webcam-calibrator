# Webcam Calibration App — Architecture Plan

## Overview

A browser-based camera calibration tool. Everything runs client-side in the browser — no backend, no persistence, no Python.

**Stack:**
- **UI**: Solid.js 2.0 (beta)
- **GPU compute**: TypeGPU (WebGPU)
- **Styling**: CSS Modules + CSS custom properties
- **Target**: AprilTag 6×6 grid (tag36h11)

**Goal**: Calibrate a camera using a printed or screen-displayed AprilTag grid, producing OpenCV-compatible intrinsics and distortion coefficients.

---

## Pages

### Page 1 — Home / Target Display

**Purpose:** Display the calibration target (AprilTag 6×6 grid) full-screen so the user can print it or point another camera at the screen.

**Behavior:**
- Renders the 6×6 AprilTag grid via SVG (crisp at any zoom)
- Size control: user can adjust physical target dimensions (mm) and grid spacing
- Toggle: fullscreen / split-screen (target + camera feed)
- Shows the generated target for download/print as a single SVG/PNG

**Target specifications:**
- Family: tag36h11 (11-bit Hamming distance — maximum robustness)
- Grid: 6×6 = 36 unique tags
- Layout: tags spaced with configurable gap between them
- Total size: user-configurable, default 200×200mm

**Technical notes:**
- SVG rendering for crisp output at any DPI
- Target rendered from a generated bit pattern — no image assets
- Each tag: black outer border, inner binary pattern (6×6 bits per tag)
- Target point relaxation: BA-refined world points are re-projected for display when available

---

### Page 2 — Calibration / Live Feed

**Purpose:** Stream from the camera, detect AprilTags in real-time, collect valid observations, and run the solver.

**Sections:**

#### 2.1 Camera Input
- `getUserMedia` → `<video>` element
- Frame sync via `requestVideoFrameCallback`
- Format: request 1280×720 (configurable)

#### 2.2 GPU Detection Pipeline (TypeGPU Compute)

All detection runs on the GPU via TypeGPU compute shaders.

**Stage 2.2.1 — Camera Copy**
- Copy from `importExternalTexture` to intermediate RGBA texture

**Stage 2.2.2 — Grayscale Conversion**
- Convert from camera format (RGBA) to grayscale float buffer
- One compute dispatch per frame

**Stage 2.2.3 — Edge Detection**
- Sobel gradient (X + Y) in one pass
- Gradient magnitude via `sqrt(gx² + gy²)`
- Same-padding to avoid edge artifacts

**Stage 2.2.4 — Histogram + Adaptive Threshold**
- 256-bin histogram of edge magnitudes
- Threshold computed at Nth percentile (strongest edges)
- Filter: zero out edges below threshold

**Stage 2.2.5 — Contour Extraction**
- Jump Flood Algorithm (JFA) for connected component labeling (TypeGPU has a JFA example)
- Fit connected edge chains to quadrilaterals
- RANSAC-based quad validation

**Stage 2.2.5 — AprilTag Decoding**
- Compute homography from detected quad to canonical tag coordinates
- Decode the 6×6 binary pattern inside the tag border
- Validate using Hamming distance against tag36h11 dictionary
- Discard tags with decode errors

**Stage 2.2.6 — Subpixel Corner Refinement**
- For each of the 4 corners of each detected tag: fit a 2D parabolic surface to gradient magnitude in a 5×5 window
- Solve analytically for the subpixel peak
- Target: ~0.1 pixel accuracy

#### 2.3 CPU Processing

**Blurry Frame Detection:**
- [Deferred] CPU-side: compute Laplacian variance per frame
- If below threshold → show "hold steady" feedback, do not capture

**Pose Estimation (EPnP + RANSAC):**
- EPnP for initial pose from detected tag corners (O(n), planar target)
- RANSAC loop: sample 4 correspondences, solve, count inliers
- Inlier threshold: 3–5 px reprojection error
- Accept view if: ≥30 tags detected, inlier ratio > 80%

**Observation Collection:**
- Each "good" view contributes: detected image points + known world points
- World points defined in tag coordinate frame (Z=0 plane, meters)
- 6×6 grid → 36 tags × 4 corners = 144 raw world points (corners shared at grid intersections)
- Display: real-time corner overlay on video feed

**Target Point Relaxation:**
- After each BA iteration, re-project the 3D world points using updated extrinsics
- The refined projections become the "target" for the next iteration
- Handles perspective + lens distortion interaction

#### 2.4 Bundle Adjustment + Levenberg-Marquardt

**Camera Model — Brown-Conrady Rational Tangential (9 parameters):**

Intrinsics: `fx, fy, cx, cy` (4 params)
Distortion: `k1, k2, k3, p1, p2` (5 params)

Projection:
```
r² = (x² + y²) / z²
x' = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
y' = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
u  = fx * x' + cx
v  = fy * y' + cy
```

**Bundle Adjustment:**
- Optimize: 9 intrinsic + (num_views × 6) extrinsic parameters
- Levenberg-Marquardt with damping adaptation
- Numerical Jacobian (forward differences)
- Convergence: parameter change < 1e-6 AND residual change < 1e-6
- Iterations: max 100, expected convergence ~20-40 iterations

**Good View Criteria:**
- ≥30 tags detected per frame
- RANSAC inlier ratio > 80%
- Reprojection error mean < 5px
- Minimum 10–15 views before running BA

---

### Page 3 — Results

**Purpose:** Display computed calibration parameters with export options.

**Displays:**
- Intrinsic matrix (3×3)
- Distortion coefficients (k1, k2, k3, p1, p2)
- Principal point (cx, cy)
- Focal length (fx, fy)
- Resolution used
- Reprojection error statistics (mean, RMS, max)
- Quality indicator (Good / Fair / Poor based on RMS error)

**Export options:**
- OpenCV YAML (camera matrix + distortion, compatible with `cv2.FileStorage`)
- JSON (all parameters, typed)
- Copy to clipboard

---

## Module Map

```
src/
├── lib/
│   ├── april-tag-gen.ts       — Generate 6×6 tag grid as SVG bitmaps
│   └── tag36h11.ts           — tag36h11 dictionary (4096 codes + decode)
│   │
├── gpu/
│   ├── camera.ts             — Pipeline factory + per-frame processing
│   ├── pipelines/
│   │   ├── copyPipeline.ts   — Copy external texture to usable format
│   │   ├── grayPipeline.ts   — RGBA → grayscale float buffer
│   │   ├── sobelPipeline.ts  — Sobel edge detection
│   │   ├── histogramPipelines.ts — Histogram reset + accumulate
│   │   ├── histogramRenderPipeline.ts — Histogram visualization
│   │   ├── edgeFilterPipeline.ts — Threshold-based edge filtering
│   │   ├── edgesPipeline.ts   — Edge render to canvas
│   │   ├── layouts.ts        — All bind group layouts
│   │   └── constants.ts      — Shared constants + threshold computation
│   ├── clahe.ts              — Contrast-Limited Adaptive Histogram Eq.
│   ├── contour.ts            — JFA connected components + quad fit
│   ├── decode.ts             — Homography decode + tag validation
│   ├── subpixel.ts           — Parabolic surface corner refinement
│   └── [fft.ts]              — Radix-2 Cooley-Tukey 2D FFT (deferred)
│   │
├── solver/
│   ├── pnp.ts                — EPnP + RANSAC pose estimation
│   ├── ba.ts                 — Bundle adjustment + LM optimizer
│   └── camera-model.ts       — Intrinsics, distortion, projection
│   │
├── store/
│   └── calibration.ts        — Solid.js 2.0 reactive store
│   │
├── components/
│   ├── CameraFeed.tsx         — Live video + detection overlay
│   ├── TargetDisplay.tsx      — AprilTag SVG grid renderer
│   ├── ResultsPanel.tsx      — Parameter display + export
│   └── App.tsx               — Root component with view switching
│   │
├── styles/
│   ├── design-system.css     — CSS custom properties (all design tokens)
│   ├── reset.css             — Minimal CSS reset
│   └── [component].module.css — Per-component CSS modules
│   │
└── main.tsx                  — Entry point
```

---

## Data Flow

```
Webcam frame (VideoFrame / ImageBitmap)
  → TypeGPU importExternalTexture
  → Copy to intermediate RGBA texture
  → Grayscale conversion (GPU compute)
  → Sobel edge detection (GPU compute)
  → Histogram + adaptive threshold (GPU compute)
  → Edge filtering (GPU compute)
  → Display filtered edges
  → [Future: JFA + quad fit + tag decode]
  → Read back corner positions (CPU)
  → EPnP + RANSAC pose (CPU)
  → If good view → add to observation set
  → After N views → BA + LM optimization (CPU)
  → Updated intrinsics → display overlay + results
```

---

## Implementation Phases

### Phase 1 — Infrastructure (current)
- [x] Vite + Solid.js 2.0 + TypeGPU project setup
- [x] CSS design system (variables + modules)
- [x] Plan documentation

### Phase 2 — UI Shell
- [ ] App.tsx with view switching (target / calibrate / results)
- [ ] Basic routing via Solid.js signals (no router library needed)
- [ ] CSS layout for all three views

### Phase 3 — Camera + Grayscale
- [x] Camera access and live video display
- [x] TypeGPU initialization
- [x] Grayscale conversion compute shader
- [x] Display grayscale output to verify pipeline
- [x] Sobel edge detection with same-padding
- [x] Histogram-based adaptive threshold
- [x] Edge filtering with dynamic threshold

### Phase 4 — AprilTag Detection
- [x] Edge detection (Sobel + threshold)
- [x] Contour extraction (JFA) — see `docs/jfa.md` for TypeGPU implementation guide
  - [x] Create labelInitPipeline (edge pixels get unique label)
  - [x] Create jfaPropagatePipeline (propagate labels over N passes)
  - [x] Create ping-pong label buffers
  - [x] Integrate JFA into camera pipeline
  - [x] CPU-side region extraction and quad fitting
- [ ] Quad fitting refinement (RANSAC for better corners)
- [ ] tag36h11 decode
- [ ] Subpixel corner refinement

### Phase 5 — Pose + Solver
- [ ] EPnP implementation
- [ ] RANSAC loop
- [ ] Bundle adjustment with LM
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
- [ ] Split-screen mode

### Phase 8 — Polish
- [ ] Motion blur feedback
- [ ] FFT-based deblurring (if needed)
- [ ] Quality indicators
- [ ] Error states

---

## Open Questions (Deferred)

| Question | Decision |
|----------|----------|
| Deblurring | FFT on TypeGPU branch — use when we get there |
| Minimum good views | 10–15 before running BA |
| Partial tags | Discarded (off-screen = fine, decode errors = discard) |
| Session persistence | None — each session is clean |
