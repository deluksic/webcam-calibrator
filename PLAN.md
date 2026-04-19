# AprilTag Grid Visualization

## Goal
Render 7×7 grid lines over detected quads with perspective-correct warping via homography.

## Approach

### Shader Strategy
For each detected quad, apply a homography warp to map the unit square onto the quad's 4 corner points. Draw grid lines in the fragment shader using the warped UV coordinates.

### Data Flow
```
DetectedQuads (CPU) → computeHomography() → quadCornersBuffer (GPU)
                                                   ↓
                                           gridVizPipeline vertex shader
                                                   ↓
                                           Fragment shader draws grid lines
```

### Buffer Format
`quadCornersBuffer` follows `GridDataSchema` in `gridVizPipeline.ts`: **`MAX_INSTANCES`** entries, each a **`mat3x3f` homography** (column-major, 8 free coeffs + bottom-right 1) plus **`QuadDebug`** (`failureCode`, `edgePixelCount`, `minR2`, `intersectionCount`) and **`decodedTagId`** (u32; **`0xFFFFFFFF`** = unknown / black fill). Homography maps unit square → image quad; failure styling is driven from CPU `cornerDebug`. When a dictionary id is known, the fragment shader hashes **`decodedTagId`** for a stable fill tint (`stableHashToRgb01`).

## Status
- [x] Create gridVizPipeline.ts
- [x] Add buffer and pipeline to camera.ts
- [x] Add 'grid' display mode
- [x] Add Grid button in CalibrationView UI
- [x] Homography-based perspective warp
- [x] Fallback quads with visual distinction (red outline, 50% opacity)
- [x] Fallbk checkbox: show bbox quads and corner quads **without** dictionary decode; off = decoded-only grid
- [x] CPU AprilTag grid + dictionary decode wired from NMS-filtered Sobel readback (unweighted **τ**-voting; see **`ARCHITECTURE.md` → AprilTag grid + decode**); overlay **`?`** when no match

Full roadmap and phase checkboxes: **`docs/plan.md`**. Product-wide architecture (GPU stages, corner order, buffers): **`ARCHITECTURE.md`**.

### Note — corner order
Homography uses quad corners **TL, TR, BL, BR**. `buildTagGrid` expects **TL, TR, BR, BL**; the CPU remaps before building the 6×6 cell mesh (see `contour.ts`).
