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
`quadCornersBuffer` follows `GridDataSchema` in `gridVizPipeline.ts`: **`MAX_INSTANCES`** entries, each a **`mat3x3f` homography** (column-major, 8 free coeffs + bottom-right 1) plus **`QuadDebug`** (`failureCode`, `edgePixelCount`, `minR2`, `intersectionCount`) for overlay tinting. Homography maps unit square → image quad; failure styling is driven from CPU `cornerDebug`, not a legacy `hasCorners` vec4.

## Status
- [x] Create gridVizPipeline.ts
- [x] Add buffer and pipeline to camera.ts
- [x] Add 'grid' display mode
- [x] Add Grid button in CalibrationView UI
- [x] Homography-based perspective warp
- [x] Fallback quads with visual distinction (red outline, 50% opacity)
- [x] Fallbk checkbox to toggle fallback quad visibility
