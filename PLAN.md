# AprilTag grid overlay

Wires each detected quad to a procedural UV grid (**`GRID_DIVISIONS` = 8** subdivisions per axis on the unit square, see [`gridVizPipeline.ts`](src/gpu/pipelines/gridVizPipeline.ts)), warped into image space with the per-quad homography in the fragment shader.

**Data path:** `DetectedQuad` (CPU) → `computeHomography()` → `quadCornersBuffer` (GPU) → `gridVizPipeline` (vertex + fragment).

**`quadCornersBuffer`:** `GridDataSchema` — up to `MAX_INSTANCES` (= `MAX_QUADS`, 512). Per instance: `mat3x3f` homography (column-major, w normalized to 1 in the last element), `QuadDebug` (`failureCode`, `edgePixelCount`, `minR2`, `intersectionCount`), `decodedTagId`, and `decodedRotation` (zeroed after GPU canonicalize). Homography maps **uv ∈ [0,1]²** to the image quad. Decoded tag IDs get a stable fill tint with `stableHashToRgb01`; `DICT_MISS` is amber; unknown is black.

**UI:** **Debug** (`/debug`) exposes GPU display modes (gray, NMS, labels, quad grid, line debug, undistort, etc.). **Calibrate** uses the same GPU grid path with collection controls and decoded-only overlay.

**Decode:** GPU tag36h11 in [`tagDecodePipeline.ts`](src/gpu/pipelines/tagDecodePipeline.ts) (Hamming budget 3, canonicalize on GPU). Overlay shows the numeric id or **`?`** on dictionary miss.

**Corner order** everywhere: **TL, TR, BL, BR** (see `Corners` in [`geometry.ts`](src/lib/geometry.ts)).

**Broader** product and roadmap: [`docs/plan.md`](docs/plan.md). **GPU stages and failure semantics:** [`ARCHITECTURE.md`](ARCHITECTURE.md).
