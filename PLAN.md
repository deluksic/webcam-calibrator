# AprilTag Grid Visualization Pipeline

## Goal
Render 7x7 grid lines over detected quads using TypeGPU. Grid is drawn in the fragment shader using the quad's 4 corner points.

## Approach

### Shader Strategy
For each pixel, check if it's inside any detected quad's bounding box. If so:
1. Compute barycentric-like weights based on distance to each of the 4 edges
2. Use these weights to determine position within the quad
3. Map to 7x7 grid and compute distance to nearest grid line
4. Draw grid lines (white/bright) with configurable thickness

### Data Flow
```
DetectedQuads (CPU) → extentBuffer metadata + new quadCornersBuffer (GPU)
                         ↓
                   GridVizPipeline reads corners
                         ↓
                   Fragment shader draws grid lines
```

### Files to Create/Modify

1. **New file: `src/gpu/pipelines/gridVizPipeline.ts`**
   - `createGridVizLayouts()` — bind group layout for corners buffer
   - `createGridVizPipeline()` — render pipeline with fragment shader
   - Shader: draws 7x7 grid lines inside each detected quad

2. **Modify `src/gpu/camera.ts`**
   - Add `quadCornersBuffer` — stores detected quad corners (4 points × 2 floats × MAX_TAGS)
   - Add gridVizPipeline and bind groups
   - Add `'grid'` to `DisplayMode` type
   - In `processFrame()`, render grid overlay for 'grid' mode

3. **Modify `src/components/CalibrationView.tsx`**
   - Add 'Grid' button to display mode toggle
   - Pass detected quads to pipeline for visualization

### Technical Details

**Buffer Layout:**
```
quadCornersBuffer: f32[]
  [tag0_tl.x, tag0_tl.y, tag0_tr.x, tag0_tr.y, tag0_br.x, tag0_br.y, tag0_bl.x, tag0_bl.y,
   tag1_tl.x, ...]
```

**Fragment Shader Logic:**
1. For each detected quad (up to MAX_TAGS=128), compute inclusion test
2. Use bilinear interpolation weights from corners to get (u,v) in [0,1]
3. Multiply by 6 (grid divisions) and check distance to nearest integer
4. Grid line = where `fract(pos * 6)` is near 0 or near 1
5. Output bright line color (e.g., cyan #00ffff with alpha)

**Constants:**
- `GRID_DIVISIONS = 6` (6 cells = 7 lines per axis)
- `GRID_LINE_WIDTH = 0.05` (in grid cell units)
- `MAX_DETECTED_QUADS = 128`

## Status
- [x] Create gridVizPipeline.ts
- [x] Add buffer and pipeline to camera.ts
- [x] Add 'grid' display mode
- [x] Add Grid button in CalibrationView UI
