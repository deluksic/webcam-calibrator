# Gradient profile pipeline

GPU path used by **Gradient profiles** ([`GradientProfilesView.tsx`](../src/components/gradientProfiles/GradientProfilesView.tsx)): live Sobel edges, connected components, per-label orientation clustering, TLS line fits, **quad registration**, gradient profiles along edge normals, and debug overlays.

Wiring lives in [`gradientProfilePipeline.ts`](../src/gpu/gradientProfilePipeline.ts). Per-frame compute is [`encodeGradientProfileCompute`](../src/gpu/gradientProfileComputeEncoding.ts); camera and plot present are [`gradientProfilePresentEncoding.ts`](../src/gpu/gradientProfilePresentEncoding.ts).

The live **calibration / grid** path ([`cameraPipeline.ts`](../src/gpu/cameraPipeline.ts), [`ARCHITECTURE.md`](../ARCHITECTURE.md)) shares ingest → gray → Sobel → histogram → NMS → labeling through compact labels, but does **not** run orientation clustering, line fit, profiles, or quad registration.

## Goal

**Move AprilTag detection onto the GPU** so the calibration **grid** path no longer depends on expensive full-frame CPU readbacks.

Today, **grid** mode calls [`readDetection`](../src/gpu/cameraDetection.ts) each frame (when a frame slot is free): it copies compact labels and the NMS `filteredBuffer` to the CPU (~11 MB at 1280×720), then runs per-region corner finding and tag36h11 decode on the host ([`ARCHITECTURE.md`](../ARCHITECTURE.md) — corner pipeline). That bandwidth and sync cost limits frame rate and adds latency.

This pipeline is the staging ground for a GPU-native tag path:

- **Connected components** and **per-label geometry** already stay on the GPU through quad registration (`labelToQuadId`, fitted sides, `packedEdgeLabels`).
- **Gradient profiles** validate edge quality per side without readback.
- **Corner intersections + homography** (GPU) produce per-quad `mat3x3f` maps for the same 8×8 grid overlay as calibration **grid** mode (no tag decode yet).
- **Next steps**: on-GPU tag bit sampling and tag36h11 decode from compact buffers—replacing `readDetection` with small structured readbacks (decode results) only when needed.

Success looks like **grid** reusing the same compute chain (or a shared subset), with **no** dense label or gradient buffer readback per frame.

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
  → present: camera mode + profile plot canvas
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

Label line fit runs **inside** step 8 (between find peaks and compact quads), because quad registration reads `labelLineOut`.

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
5. Write [`GridDataSchema`](../src/gpu/pipelines/gridVizPipeline.ts) entry: `homography`, `debug.failureCode` / `intersectionCount`, `decodedTagId = UNKNOWN` (black grid, no ID tint).

Slots `quadId ≥ quadCount` are cleared so stale instances do not draw.

Present uses [`createGridVizStage`](../src/gpu/pipelines/gridVizPipeline.ts) — same 8×8 perspective warp as [`encodeAndSubmitGridPresent`](../src/gpu/cameraPresentEncoding.ts).

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
| `quadGrid` | Full grayscale | Black 8×8 grid warp per quad (`gridViz`, no tag ID coloring) |
| `lineFitDebug` | Dim grayscale + failure colors | Same fitted-line overlay |

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
| `grid.quadCornersBuffer[quadId]` | 1024 × `QuadData` | Homography + debug for grid overlay (GPU-written) |

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
| Grid overlay | [`gridVizPipeline.ts`](../src/gpu/pipelines/gridVizPipeline.ts) |
| Profiles | [`edgeProfilePipeline.ts`](../src/gpu/pipelines/edgeProfilePipeline.ts) |
| Profile plot | [`edgeProfilePlotPipeline.ts`](../src/gpu/pipelines/edgeProfilePlotPipeline.ts) |
| Fitted lines overlay | [`edgeFittedLineOverlayPipeline.ts`](../src/gpu/pipelines/edgeFittedLineOverlayPipeline.ts) |
| Debug colors | [`lineFitDebugPipeline.ts`](../src/gpu/pipelines/lineFitDebugPipeline.ts) |
| Orientation helpers | [`orientPeakAssign.ts`](../src/gpu/shaders/orientPeakAssign.ts) |
| Line intersect / corners / DLT | [`lineIntersect.ts`](../src/gpu/shaders/lineIntersect.ts), [`quadCornerOrder.ts`](../src/gpu/shaders/quadCornerOrder.ts), [`homographyDlt.ts`](../src/gpu/shaders/homographyDlt.ts) |
| UI | [`GradientProfilesView.tsx`](../src/components/gradientProfiles/GradientProfilesView.tsx), [`GradientProfilesPipeline.tsx`](../src/components/gradientProfiles/GradientProfilesPipeline.tsx) |
