# Gradient profile pipeline

GPU path used by **Debug** at `/debug` ([`GradientProfilesView.tsx`](../src/components/gradientProfiles/GradientProfilesView.tsx); `/gradient-profiles` redirects here): live Sobel edges, connected components, per-label orientation clustering, TLS line fits, **quad registration**, GPU **tag36h11 decode**, perspective **grid overlay**, gradient profiles along edge normals, undistort preview, and line-fit debug overlays.

Wiring lives in [`gradientProfilePipeline.ts`](../src/gpu/gradientProfilePipeline.ts). Per-frame compute is [`encodeGradientProfileCompute`](../src/gpu/gradientProfileComputeEncoding.ts); camera and plot present are [`gradientProfilePresentEncoding.ts`](../src/gpu/gradientProfilePresentEncoding.ts).

The live **calibration / grid** path ([`cameraPipeline.ts`](../src/gpu/cameraPipeline.ts), [`ARCHITECTURE.md`](../ARCHITECTURE.md)) shares ingest → gray → Sobel → histogram → NMS → labeling through compact labels, but does **not** run orientation clustering, line fit, profiles, or quad registration.

## Goal

**Move AprilTag detection onto the GPU** so the calibration **grid** path no longer depends on expensive full-frame CPU readbacks.

Today, **grid** mode calls [`readDetection`](../src/gpu/cameraDetection.ts) each frame (when a frame slot is free): it copies compact labels and the NMS `filteredBuffer` to the CPU (~11 MB at 1280×720), then runs per-region corner finding and tag36h11 decode on the host ([`ARCHITECTURE.md`](../ARCHITECTURE.md) — corner pipeline). That bandwidth and sync cost limits frame rate and adds latency.

This pipeline is the staging ground for a GPU-native tag path:

- **Connected components** and **per-label geometry** stay on the GPU through quad registration (`labelToQuadId`, fitted sides, `packedEdgeLabels`).
- **Gradient profiles** validate edge quality per side without readback.
- **Corner intersections + homography** (GPU) write per-quad `mat3x3f` maps into `grid.quadCornersBuffer`.
- **Tag36h11 decode** (GPU) samples luma per data module, matches the dictionary, and updates `decodedTagId` / `decodedRotation` in the same buffer—no CPU contour readback in this view.
- **Grid overlay** composites an 8×8 perspective-correct warp with stable hash tints per decoded id.

**Still CPU on the calibration path:** live **grid** mode in [`cameraPipeline.ts`](../src/gpu/cameraPipeline.ts) still uses [`readDetection`](../src/gpu/cameraDetection.ts) + [`updateQuadCornersBuffer`](../src/gpu/cameraFrame.ts). Porting calibration to this GPU decode chain is the remaining integration step.

Success for production **grid** looks like reusing this compute chain (or a shared subset), with **no** dense label or gradient buffer readback per frame—only small structured readbacks when the UI needs tag ids on the host.

---

## End-to-end flow

```
Video frame
  → ingest (external texture → luma texture)
  → grayscale buffer
  → Sobel (vec2 gradient per pixel)
  → histogram accumulate (CPU reads bins → threshold each frame)
  → NMS + edge filter (filteredBuffer: vec2, zero = no edge)
  → pointer-jump labeling (raw union-find labels)
  → canonical compact labels (0 … N−1 per pixel)
  → edge histogram cluster stage (per labelId)
        hist reset → hist accum → find peaks
        → label line fit (per labelId × 4 peak slots)
        → compact quads → quad label map → assign packed edge ids
  → line-fit debug classify (optional viz)
  → scatter quad lines → flat lineOut[MAX_FLAT_EDGES]
  → quad corner homography (intersections + DLT → grid viz buffer)
  → edge profile (64-bin normal profiles per flat edge)
  → tag decode vote pass (raster quads → per-module luma atomics)
  → tag decode compute (histogram threshold → codeword → dictionary)
  → present: camera mode + tag histogram canvas + profile plot canvas
```

---

## Compute order (one frame)

From [`encodeGradientProfileCompute`](../src/gpu/gradientProfileComputeEncoding.ts):

| Step | Stage | Notes |
|------|--------|--------|
| 1 | `ingest` | Copy camera frame to GPU luma |
| 2 | `gray` | Full-frame luma buffer |
| 3 | `sobel` | Per-pixel gradient |
| 4 | `histogram.accumulate` | Drives adaptive threshold (CPU write before pass) |
| 5 | `nms` | Thresholded, non-max-suppressed edges |
| 6 | `pointerJump` | Connected components |
| 7 | `compact` | Remap to `compactLabelBuffer` |
| 8 | `edgeHistogram.encodeCompute` | See [Histogram cluster stage](#histogram-cluster-stage) |
| 9 | `lineFitDebug` | Per-pixel failure codes (no readback) |
| 10 | `orientHistViz` (optional) | Pack histograms for side canvas |
| 11 | `lineFit` | Reset + scatter label fits → flat `lineOut` |
| 12 | `quadHomography` | Per `quadId`: adjacent line intersections → CCW corners → DLT `H` → `grid.quadCornersBuffer` |
| 13 | `profile` | Reset buckets → accum → normalize |
| 14 | `tagDecode.encodeVotes` | Render pass: warp each quad with `H`, atomic-add luma into 6×6 module sums (+ debug 16-bin hist) |
| 15 | `tagDecode.encodeDecode` | Compute pass: per-quad threshold, 36-bit codeword, Hamming match vs tag36h11 (≤3 errors) |

Label line fit runs **inside** step 8 (between find peaks and compact quads), because quad registration reads `labelLineOut`.

Steps 14–15 run **after** the main compute pass (separate render + compute passes on the same command encoder). They read/write `grid.quadCornersBuffer` in place.

---

## Histogram cluster stage

Module: [`edgeHistogramClusterPipeline.ts`](../src/gpu/pipelines/edgeHistogramClusterPipeline.ts).

Per compact `labelId` (up to `MAX_EXTENT_COMPONENTS` = 8192), per frame:

### 1. Hist reset

- Clear 64-bin `orientationHistogram`, peak slots, `peakCount`.
- Thread 0 clears global `quadCount`.

### 2. Hist accum

- For each edge pixel (`length(g) > 0`) with valid compact label:
- `gradientOrientationBin(g)` → atomic increment in that label’s histogram ([`orientPeakAssign.ts`](../src/gpu/shaders/orientPeakAssign.ts)).

No histogram smoothing pass (removed).

### 3. Find peaks

- Up to **4** local maxima in the 64-bin circular histogram.
- A bin is a candidate if count ≥ `ORIENT_PEAK_MIN_COUNT` (6) and ≥ neighbors.
- Peaks must be ≥ `MIN_PEAK_BIN_SEPARATION` (8 bins ≈ 45°) apart.
- `peakDirs[k]` = 3-bin weighted centroid around each accepted peak.

### 4. Label line fit

Module: [`labelLineFitPipeline.ts`](../src/gpu/pipelines/labelLineFitPipeline.ts).

Per slot `labelId × 4 + peakIndex`:

1. **Reset** reduce / inlier / `labelLineOut`.
2. **Accum** — edge pixels assigned to that peak (`assignPeakEdgeId`) contribute fixed-point moments.
3. **Fit** — TLS normal + coarse segment; inlier gate (`LINE_INLIER_DIST_PX`, ratio).
4. **Refine** — second TLS pass on inliers.
5. **Extent** — along-edge `tMin`/`tMax`, trim (`LINE_EXTENT_TRIM_FRAC`), endpoints `p0`/`p1`.
6. **Scatter** — write final `EdgeLineEntry` (`valid`, `inlierCount`, geometry).

Thresholds: [`lineFitThresholds.ts`](../src/gpu/lineFitThresholds.ts).

### 5. Compact quads

A label becomes a registered quad only if **all** of the following hold:

| Check | Constant / rule |
|--------|------------------|
| Exactly four histogram peaks | `peakCount === MAX_EDGES_PER_LABEL` (4) |
| Four valid TLS sides | Each peak slot: `valid` and `inlierCount ≥ MIN_QUAD_EDGE_INLIERS` (12) |
| Orientation spread | Each peak bin has ≥ 3 other peaks at circular distance ≥ `MIN_PEAK_BIN_SEPARATION` (rejects two parallel pairs) |
| Room in quad table | `quadId < MAX_QUADS` (512) |

On success:

- `labelToQuadId[labelId] = quadId`
- `quadSourceLabelId[quadId] = labelId`
- `quadPeakEdge[quadBase + q] = q` for each valid side slot (else `INVALID`)

Otherwise `labelToQuadId[labelId]` stays `INVALID`.

### 6. Write quad label map

- `quadLabelBuffer[pixel] = labelToQuadId[compactLabel]` (invalid where no quad).

### 7. Assign edges

- For edge pixels on labels with a registered quad:
- `packedEdgeLabels[pixel] = quadId × 4 + edgeId` where `edgeId` comes from `assignPeakEdgeId` (gradient aligned to nearest peak).
- Non-quad labels → `INVALID` in `packedEdgeLabels`.

---

## Flat line scatter and profiles

### Scatter (`edgeLineFitPipeline.ts`)

After quads are known:

- `lineOut[flatSlot]` copies from `labelLineOut[labelId × 4 + peakEdge]` only when `quadPeakEdge[quadId × 4 + edgeId]` is valid.
- `flatSlot = quadId × 4 + edgeId`, `flatSlot < quadCount × 4`.
- `validEdgeCount` = number of scattered valid edges (shown in UI).

`MAX_FLAT_EDGES = MAX_QUADS × 4` (2048).

### Profile (`edgeProfilePipeline.ts`)

- **Buckets:** `MAX_FLAT_EDGES × 64` — one 64-bin profile per flat edge slot.
- **Pixel assignment:** 5×5 neighborhood over edge pixels; pick the flat edge whose fitted normal is closest in signed distance (`packedEdgeLabels` indexes `lineOut`).
- **Sampling:** Signed distance `s` along normal (black → white), `|s| ≤ PROFILE_NEIGHBORHOOD_HALF` (2.5 px); along-edge `t` within `tSampleMin`/`tSampleMax`.
- **Output:** `profileAvg[bucketIdx]` normalized mean luma per bin.

Plot: [`edgeProfilePlotPipeline.ts`](../src/gpu/pipelines/edgeProfilePlotPipeline.ts) on a separate canvas (MSAA resolve).

---

## Quad corners and homography

Module: [`quadCornerHomographyPipeline.ts`](../src/gpu/pipelines/quadCornerHomographyPipeline.ts). Shaders: [`lineIntersect.ts`](../src/gpu/shaders/lineIntersect.ts), [`quadCornerOrder.ts`](../src/gpu/shaders/quadCornerOrder.ts), [`homographyDlt.ts`](../src/gpu/shaders/homographyDlt.ts).

Per `quadId < quadCount` (one thread per slot, up to `MAX_QUADS` = 512):

1. Load four scattered `lineOut` sides and `peakDirs` from `labelClusters[labelId]`.
2. **Sort sides CCW** by `atan2(peakDir)`; **intersect adjacent** infinite lines (`n·p = nDotMean`, same normal form as TLS).
3. **Sort intersection points CCW** around centroid; **degeneracy only** (signed area floor, min edge 2 px — no opposite-edge ratio checks).
4. **Try four CCW rotations** (CPU `rotateRing`); first nonsingular [`tryHomographyFromCorners`](../src/gpu/shaders/homographyDlt.ts) (8×8 DLT, same as CPU [`tryComputeHomography`](../src/lib/geometry.ts)) wins.
5. Write [`GridDataSchema`](../src/gpu/pipelines/gridVizPipeline.ts) entry: `homography`, `screenCorners`, `debug.failureCode` / `intersectionCount`, `decodedTagId = UNKNOWN`, `decodedRotation = 0`.

Slots `quadId ≥ quadCount` are cleared so stale instances do not draw.

### Strip order vs cyclic order

Corners are stored in **triangle-strip order** for rendering and DLT: `TL, TR, BL, BR` (indices 0–3). Shoelace area and edge-length degeneracy checks remap to cyclic perimeter `TL, TR, BR, BL` via `cyclicIdx` in [`quadCornerOrder.ts`](../src/gpu/shaders/quadCornerOrder.ts)—walking strip order directly would treat the quad as a bow-tie and falsely fail plausibility.

---

## GPU tag decode (tag36h11)

Module: [`tagDecodePipeline.ts`](../src/gpu/pipelines/tagDecodePipeline.ts). Dictionary: [`tag36h11.ts`](../src/lib/tag36h11.ts) / `tag36h11.json` (587 codes).

Shared buffer: `grid.quadCornersBuffer` (`GridDataSchema`, up to `MAX_INSTANCES` = 1024). Homography stage writes geometry; decode stage **only** updates `decodedTagId` and `decodedRotation` (and leaves `UNKNOWN` when `H` is degenerate).

### Stage 1 — Vote accumulation (render pass)

- **Input:** `grayTex` (camera ingest, `rgba8unorm` sampled as `texture2d(f32)`), `quads` from `quadCornersBuffer`.
- **Vertex:** Same perspective warp as grid viz: `mul(H, vec3(uv, 1))` → clip with `w = imgPos.z` (degenerate `H` → off-screen discard).
- **Fragment:** For each covered pixel:
  - `floor(uv × 8)` → module index; **data cells** are the inner 6×6 (`mx, my ∈ 1…6`).
  - `atomicAdd` fixed-point luma (`round(gray × 65536)`) into `moduleSum[quadId × 36 + cell]`, `moduleCount` likewise.
  - **Debug:** 16-bin per-quad histogram of raw `gray` (all pixels in the quad, not only data cells)—shown on the **Tag grayscale histogram** canvas.

Vote pass currently dispatches `MAX_INSTANCES` instances; slots without a valid homography contribute no samples.

### Stage 2 — Decode (compute pass)

One thread per `quadId < MAX_INSTANCES` (workgroup size 64):

1. **Module averages** — `sum / (65536 × count)` per data cell; default 0.5 if no samples.
2. **16-bin histogram** over the 36 averages; **3-wide circular smooth**; find black peak (max bin) and white peak (local max, ≥4 bins from black, fallback = farthest bin).
3. **Threshold** — midpoint of peak bin indices: `(blackPeak + whitePeak) / 2 / 16`.
4. **Codeword** — classify each cell white/black; pack 36 bits via `BIT_POS` (same spatial→bit mapping as CPU `BIT_X` / `BIT_Y`).
5. **Bit layout** — AprilTag bit index `b` is the **MSB** of the 36-bit word. Dictionary stores `low` = bits 0–31, `high` = bits 32–35. Packing uses `pos = 35 - bitIdx` (not `1 << bitIdx` on the index alone).
6. **Four rotations** — unrolled `ROT_LUTS_0…3`; Hamming distance `popcount(low ^ cw.low) + popcount(high ^ cw.high)` against all 587 entries.
7. **Write result** — if `bestDist ≤ 3`: `decodedTagId = bestId` (0…586), `decodedRotation = 0…3`; else `decodedTagId = DICT_MISS`, rotation 0.

Constants: `MAX_DICT_ERROR = 3`, `DATA_MODULES = 6`, `TAG_MODULES = 8`.

### Sentinels (`decodedTagId`)

| Value | Symbol | Grid viz (corners OK, `failureCode = 0`) |
|-------|--------|------------------------------------------|
| `0xFFFFFFFF` | `DECODED_TAG_ID_UNKNOWN` | Black 8×8 grid (no hash) |
| `0xFFFFFFFE` | `DECODED_TAG_ID_DICT_MISS` | Amber-tinted grid (pattern OK, dictionary miss) |
| `0…586` | tag36h11 id | Stable hash fill + grid lines |

Non-zero `failureCode` uses failure tints from [`gridVizFailureTintRgb`](../src/gpu/pipelines/gridVizPipeline.ts) (same bitmask order as CPU [`corners.ts`](../src/lib/corners.ts)).

---

## Grid overlay (`gridVizPipeline.ts`)

Present in **`quadGrid`** mode: grayscale camera, then instanced triangle-strip quads composited with alpha blending ([`encodeGradientProfileCameraPresent`](../src/gpu/gradientProfilePresentEncoding.ts)).

### Vertex shader

- **Perspective path:** `mul(H, vec3(uv, 1))` with `w` in clip space (same as tag vote pass).
- **Degenerate fallback:** `screenCorners` in pixel space, `w = 1` (affine UV; may show a diagonal kink).
- **UVs:** unit square `TL, TR, BL, BR` matching strip order.

### Fragment shader

- **Grid lines:** `gridTextureGradBox` — 8×8 anti-aliased lines via `dpdx`/`dpdy` on interpolated UV (`GRID_DIVISIONS = 8`, `GRID_LINE_WIDTH = 0.06`).
- **Color rules** (after computing `grid`):
  1. `failureCode = 0` and `DICT_MISS` → amber fill × grid
  2. `failureCode = 0` and id ≠ `UNKNOWN` → `stableHashToRgb01(decodedTagId)` × 0.55 fill × grid
  3. `failureCode = 0` → black grid only
  4. Else → failure tint × grid

Instance count for draw comes from **previous frame** `quadCount` readback (`lastQuadCount` in [`GradientProfilesPipeline.tsx`](../src/components/gradientProfiles/GradientProfilesPipeline.tsx))—one frame of latency vs registration.

---

## Display modes

From [`encodeGradientProfileCameraPresent`](../src/gpu/gradientProfilePresentEncoding.ts):

| Mode | Camera view | Overlay / notes |
|------|-------------|-----------------|
| `grayscale` | Luma | — |
| `edges` | Sobel magnitude | — |
| `nms` | Filtered edges | Default |
| `labels` | False-color `compactLabelBuffer` | — |
| `quads` | False-color `quadLabelBuffer` | Only registered quads |
| `edgeLabels` | False-color `packedEdgeLabels` | Flat edge id per pixel |
| `fittedLines` | Dim grayscale (0.38) | Green TLS segments — **registered quads only** |
| `quadGrid` | Full grayscale | 8×8 grid warp per quad; hash tint when decoded, amber on dict miss, failure colors when corners fail |
| `lineFitDebug` | Dim grayscale + failure colors | Same fitted-line overlay |

Side canvases (same page):

| Canvas | Source |
|--------|--------|
| Orientation histograms | `orientHistViz` — per-label 64-bin orientation |
| Tag grayscale histogram | `tagHistogramDisplay` — per-quad 16-bin luma from vote-pass debug buffer |
| Edge gradient profiles | `profilePlot` — 64-bin profiles per flat edge |

### Fitted-line overlay gating

[`edgeFittedLineOverlayPipeline.ts`](../src/gpu/pipelines/edgeFittedLineOverlayPipeline.ts) draws instance `labelId × 4 + edgeId` only when:

- `labelToQuadId[labelId] ≠ INVALID`
- `quadPeakEdge[quadId × 4 + edgeId] ≠ INVALID`
- Line has `count > 0` and span ≥ 0.5 px

This matches quad scatter: partial labels (e.g. two strong parallel edges inside a tag) do not draw unless they pass full quad registration.

### Line-fit debug

[`lineFitDebugPipeline.ts`](../src/gpu/pipelines/lineFitDebugPipeline.ts) colors each edge pixel by failure stage (`assign`, `lowSample`, `outlier`, `lineInvalid`, `quadAssign`, etc.). Legend: `LINE_FIT_DEBUG_LEGEND`.

---

## Key buffers

| Buffer | Size / indexing | Role |
|--------|------------------|------|
| `compactLabelBuffer` | W×H | Compact component id per pixel |
| `labelClusters[labelId]` | 8192 × `LabelOrientCluster` | 64-bin hist + 4 peaks |
| `labelLineOut[labelId×4+peak]` | 8192×4 | Per-label TLS fits |
| `labelToQuadId[labelId]` | 8192 | Quad id or `INVALID` |
| `quadPeakEdge[quadId×4+side]` | 512×4 | Active side index or `INVALID` |
| `quadSourceLabelId[quadId]` | 512 | Source label for scatter |
| `quadCount` | 1 (atomic) | Registered quads this frame |
| `packedEdgeLabels` | W×H | `quadId×4+edgeId` or `INVALID` |
| `lineOut[flatSlot]` | 2048 | Scattered fits for profile/plot |
| `profileAvg` | 2048×64 | Normalized profile curves |
| `grid.quadCornersBuffer[quadId]` | 1024 × `QuadData` | `homography`, `screenCorners`, `debug`, `decodedTagId`, `decodedRotation` |
| `tagDecode.moduleSum` / `moduleCount` | 1024×36 each | Per-quad module luma votes (fixed-point / count) |
| `tagDecode.histBuf` | 1024×16 | Per-quad debug grayscale histogram (vote pass) |
| `tagDecode.codewords` | 587 × `{low, high}` | tag36h11 dictionary on GPU |

`QuadData` layout: [`QuadDataGpu`](../src/gpu/pipelines/gridVizPipeline.ts).

---

## Debugging tips

| Symptom | Likely cause |
|---------|----------------|
| Solid flat fill, no grid lines | Fragment shader left in debug fill-only mode—should call `gridTextureGradBox` |
| All quads same gray, no hash colors | `DICT_MISS` for every quad—check codeword bit packing (`pos = 35 - bitIdx`), homography, or threshold |
| Purple quads but geometry looks fine | `FAIL_PLAUSIBILITY` from strip-order shoelace bug—fixed via `cyclicIdx` in `quadCornerOrder` |
| Grid “kink” on diagonal | Degenerate `H`; overlay fell back to affine `screenCorners` |
| Tag histogram flat / single peak | Weak contrast or homography missing; check **Tag grayscale histogram** panel |

---

## Threshold reference

Shared constants in [`lineFitThresholds.ts`](../src/gpu/lineFitThresholds.ts):

| Symbol | Value | Meaning |
|--------|-------|---------|
| `ORIENT_HIST_BINS` | 64 | Histogram width |
| `MAX_EDGES_PER_LABEL` | 4 | Peaks / sides per label |
| `ORIENT_PEAK_MIN_COUNT` | 6 | Min bin count for a peak |
| `MIN_PEAK_BIN_SEPARATION` | 8 | ~45° between peaks |
| `ORIENT_ASSIGN_MIN_ALIGN` | 0.5 | Min dot(ĝ, peakDir) for pixel assign |
| `LINE_INLIER_DIST_PX` | 4.0 | Perpendicular inlier gate |
| `MIN_QUAD_VALID_EDGES` | 4 | Sides required to register |
| `MIN_QUAD_EDGE_INLIERS` | 12 | Min inliers per side for quad |
| `LINE_INTERSECT_DET_EPS` | 1e-10 | Parallel-line rejection (intersection only) |
| `HOMOGRAPHY_PIVOT_EPS` | 1e-10 | DLT singularity |
| `QUAD_MIN_EDGE_PX` | 2 | Min corner edge length |
| `QUAD_MIN_SIGNED_AREA_REL` | 1e-4 | × scale² min convex area |

---

## File map

| Concern | File |
|---------|------|
| Pipeline assembly | [`gradientProfilePipeline.ts`](../src/gpu/gradientProfilePipeline.ts) |
| Compute encode | [`gradientProfileComputeEncoding.ts`](../src/gpu/gradientProfileComputeEncoding.ts) |
| Present encode | [`gradientProfilePresentEncoding.ts`](../src/gpu/gradientProfilePresentEncoding.ts) |
| Hist + peaks + quads | [`edgeHistogramClusterPipeline.ts`](../src/gpu/pipelines/edgeHistogramClusterPipeline.ts) |
| TLS per label×peak | [`labelLineFitPipeline.ts`](../src/gpu/pipelines/labelLineFitPipeline.ts) |
| Flat scatter | [`edgeLineFitPipeline.ts`](../src/gpu/pipelines/edgeLineFitPipeline.ts) |
| Corners + homography | [`quadCornerHomographyPipeline.ts`](../src/gpu/pipelines/quadCornerHomographyPipeline.ts) |
| Tag decode | [`tagDecodePipeline.ts`](../src/gpu/pipelines/tagDecodePipeline.ts) |
| Grid overlay | [`gridVizPipeline.ts`](../src/gpu/pipelines/gridVizPipeline.ts) |
| Stable hash colors | [`hashStableColor.ts`](../src/lib/hashStableColor.ts) |
| Profiles | [`edgeProfilePipeline.ts`](../src/gpu/pipelines/edgeProfilePipeline.ts) |
| Profile plot | [`edgeProfilePlotPipeline.ts`](../src/gpu/pipelines/edgeProfilePlotPipeline.ts) |
| Fitted lines overlay | [`edgeFittedLineOverlayPipeline.ts`](../src/gpu/pipelines/edgeFittedLineOverlayPipeline.ts) |
| Debug colors | [`lineFitDebugPipeline.ts`](../src/gpu/pipelines/lineFitDebugPipeline.ts) |
| Orientation helpers | [`orientPeakAssign.ts`](../src/gpu/shaders/orientPeakAssign.ts) |
| Line intersect / corners / DLT | [`lineIntersect.ts`](../src/gpu/shaders/lineIntersect.ts), [`quadCornerOrder.ts`](../src/gpu/shaders/quadCornerOrder.ts), [`homographyDlt.ts`](../src/gpu/shaders/homographyDlt.ts) |
| UI | [`GradientProfilesView.tsx`](../src/components/gradientProfiles/GradientProfilesView.tsx), [`GradientProfilesPipeline.tsx`](../src/components/gradientProfiles/GradientProfilesPipeline.tsx) |
