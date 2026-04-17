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
`quadCornersBuffer` stores 3 vec4f per quad (12 f32):
- `vec4f(h1, h2, h3, h4)` — homography row 1 + start of row 2
- `vec4f(h5, h6, h7, h8)` — homography row 2 + row 3
- `vec4f(hasCorners, 0, 0, 0)` — 1.0 = real corners (blue grid), 0.0 = fallback (red outline)

## Status
- [x] Create gridVizPipeline.ts
- [x] Add buffer and pipeline to camera.ts
- [x] Add 'grid' display mode
- [x] Add Grid button in CalibrationView UI
- [x] Homography-based perspective warp
- [x] Fallback quads with visual distinction (red outline, 50% opacity)
- [x] Fallbk checkbox to toggle fallback quad visibility
