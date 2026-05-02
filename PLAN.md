# AprilTag grid overlay

Wires each detected quad to a procedural UV grid (**`GRID_DIVISIONS` = 8** subdivisions per axis on the unit square, see [`gridVizPipeline.ts`](src/gpu/pipelines/gridVizPipeline.ts)), warped into image space with the per-quad homography in the fragment shader.

**Data path:** `DetectedQuad` (CPU) → `computeHomography()` → `quadCornersBuffer` (GPU) → `gridVizPipeline` (vertex + fragment).

**`quadCornersBuffer`:** `GridDataSchema` — up to 1024 instances (`MAX_INSTANCES`). Per instance: `mat3x3f` homography (column-major, w normalized to 1 in the last element), `QuadDebug` (`failureCode`, `edgePixelCount`, `minR2`, `intersectionCount`), and `decodedTagId` (use `0xFFFFFFFF` for unknown; shader draws black, no id hash). Homography maps **uv ∈ [0,1]²** to the image quad. Decoded tag IDs get a stable fill tint with `stableHashToRgb01`.

**UI:** In **Debug**, every display mode (Gray, Edges, NMS, Labels, **Grid**, **Debug**) and the `Fallbk` toggle live on the toolbar. In **Calibrate**, the app stays on **grid** and omits a mode switcher; collection controls only.

**Decode:** NMS-filtered readback from the GPU, CPU homography + bbox scan, tag36h11 with Hamming budget 3, UV voting as described in [`ARCHITECTURE.md`](ARCHITECTURE.md). Overlay shows the numeric id or **`?`** when decode fails.

**Corner order** everywhere: **TL, TR, BL, BR** (see `Corners` in [`geometry.ts`](src/lib/geometry.ts)).

**Broader** product and roadmap: [`docs/plan.md`](docs/plan.md). **GPU stages and failure semantics:** [`ARCHITECTURE.md`](ARCHITECTURE.md).
