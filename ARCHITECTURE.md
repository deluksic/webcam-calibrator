# Architecture

## Overview

The webcam calibrator uses WebGPU compute shaders to perform camera calibration via AprilTag detection.

## App shell (Solid)

- **Views** (`src/components/App.tsx`): **Target** (SVG grid for printing), **Calibrate** (`CalibrationView.tsx` — capture workflow, quality stats, top‑K samples; histogram still drives threshold but may be off-screen), **Results** (stub), **Debug** (`DebugView.tsx` — all pipeline display modes, visible histogram, bbox overlay, dev-oriented controls).
- **Camera:** `CameraStreamProvider` in `src/components/camera/CameraStreamContext.tsx` (app root); constraint ladder and capability upgrades in `src/lib/cameraStreamAcquire.ts`.
- **Live pipeline:** `src/components/camera/LiveCameraPipeline.tsx` mounts the WebGPU path for both Calibrate and Debug.

Product phases and page table: `docs/plan.md`.

## Coordinate Spaces

- **Frame size**: up to 1280×720 (HD)
- **Raw label values**: pixel indices (0 to area-1) — output of pointer-jump labeling
- **Compact label values**: 0 to N-1 — remapped by canonical labeling, used everywhere downstream
- **Extent buffer keys**: compact label values (must be < MAX_EXTENT_COMPONENTS = 16384)

## Pipeline Stages (per frame)

```
Frame Input
  ↓
Sobel Gradient → Histogram (adaptive threshold display)
  ↓
NMS + Edge Filter
  ↓
Pointer-Jump Labeling (iterative connected components → raw pixel-index labels)
  ↓
Canonical Labeling (3-pass remap → compact IDs 0..N-1)
  ↓
Extent Tracking (atomic min/max per compact label)
  ↓
Render (mode-dependent visualization)
```

### Pointer-Jump Labeling

GPU-only. ~10 iterations per frame via pointer doubling + atomic parent-tightening.

- **Bufs**: `pointerJumpBuffer0`, `pointerJumpBuffer1` (ping-pong), `pointerJumpAtomicBuffer`
- **Output**: raw pixel-index labels — used only as input to canonical labeling

### Canonical Labeling

GPU-only. 3-pass remap: reset → atomic root claiming → canonical read.

- **Pass 1**: Reset `canonicalRootBuffer[i] = INVALID`
- **Pass 2**: Each root atomically claims `compactCounter++` as its canonical ID
- **Pass 3**: Every pixel reads `L[i] = canonicalRootBuffer[label[i]]`
- **Output**: `compactLabelBuffer` — used by all downstream stages

### Extent Tracking

GPU-only. One reset + one track dispatch. Atomically tracks (minX, minY, maxX, maxY) per component.

- **Buf**: `extentBuffer` — sized for MAX_EXTENT_COMPONENTS entries

## Display Modes

| Mode        | GPU Compute                    | Render                                  | CPU Readback                                                                                                  |
| ----------- | ------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `grayscale` | Gray                           | grayscale                               | histogram                                                                                                     |
| `edges`     | Sobel                          | edges                                   | histogram                                                                                                     |
| `nms`       | Sobel + NMS                    | edges                                   | histogram                                                                                                     |
| `labels`    | Sobel + pointer-jump + compact | labels (hash-based colors)              | none                                                                                                          |
| `debug`     | Sobel + pointer-jump + extent  | labels + bbox overlay                   | extent buffer                                                                                                 |
| `grid`      | Sobel + pointer-jump + extent  | grayscale + homography-warped quad grid | throttled `detectContours` (compact labels + **filtered** Sobel readback → corners + **CPU AprilTag decode**) |

## CPU Readbacks

| Function             | When                          | What                                   |
| -------------------- | ----------------------------- | -------------------------------------- |
| `readExtentBuffer()` | Every frame in debug mode     | Extent entries (320 KB)                |
| `detectContours()`   | Every ~30 frames in grid mode | Full label + gradient buffers (~11 MB) |

## Corner Detection (grid mode, CPU)

`detectContours()` → `validateAndFilterQuads()` in `contour.ts` calls `findCornersFromEdgesWithDebug()` **per region** (each connected component from compact labels, after area / aspect / edge-density filters). Only runs in grid mode on a throttled cadence.

### End-to-end order (GPU → CPU)

**On the GPU (same frame submit as grid mode):** grayscale → Sobel → histogram/threshold → edge filter (NMS) → pointer-jump labeling → compact remap → extent tracking. CPU readback then gets **per-pixel compact labels** and **filtered Sobel buffer** `(gx, gy)` per pixel.

**Per region on the CPU** (`src/lib/corners.ts`), stages run **strictly in this order**. Anything that fails **through step 5** (before four deduped intersection points exist) means intersection geometry never stabilizes — so a `failureCode` that _looks_ like “intersections” can still be rooted in clustering or line fit.

| Step | What happens                                                                                                                                                                                                                                                                                                                                                                                                                        | Typical failure / debug                                                                                                                                    |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Labeled edge pixels** — pixels inside the region’s bbox whose compact label matches the region; each stores raw **Sobel** `(gx, gy)` and magnitude.                                                                                                                                                                                                                                                                               | `FAIL_INSUFFICIENT_EDGES` (bit 0) if count &lt; `minEdgePixels` (default 12).                                                                              |
| 2    | **K-means (k=4)** on gradient vectors using **cosine dissimilarity** \(1 - \cos\theta\); cluster **reference direction** = normalized sum of member gradients (not a spatial centroid).                                                                                                                                                                                                                                             | Poor splits → weak lines later; no dedicated bit here.                                                                                                     |
| 3    | **Line per cluster** — `fitLine`: RANSAC on `(x,y)` inliers, then **PCA** on inliers for the line normal; returns `null` if PCA rejects scatter.                                                                                                                                                                                                                                                                                    | `FAIL_LINE_FIT_FAILED` (bit 2) for that cluster; **fewer than four lines** reduces how many intersection pairs exist.                                      |
| 4    | **Line–line intersections** — all pairs of **non-null** lines; `lineIntersection` (parallel pairs yield no point); clip to region extent bbox ± **`extentBBoxSlack`** = `max(6px, 0.5 × max(bboxW, bboxH))`.                                                                                                                                                                                                                        | `FAIL_NO_INTERSECTIONS` (bit 4) if **&lt;4** raw hits pass the clip (often because step 3 left missing lines, parallel lines, or hits fall outside slack). |
| 5    | **Deduplicate** intersections closer than **5 px**.                                                                                                                                                                                                                                                                                                                                                                                 | Same bit 4 if **&lt;4** distinct points remain (many hits collapsed to one corner).                                                                        |
| 6    | **Order + plausibility** — among permutations of the four points, require a **strictly convex CCW** cycle (consistent turn signs; largest valid signed area); **rotate** that cycle and run **plausibility** until one passes: `R²`, corners inside extent bbox ± **the same `extentBBoxSlack`** as step 4, opposite-edge length ratios, cluster inlier counts. Emit **`[TL, TR, BL, BR]`** (image / triangle-strip order) for homography and downstream decode. | `FAIL_PLAUSIBILITY` (bit 3) if no convex cycle, no rotation passes plausibility, or a check fails (e.g. `R²`, edge ratio, sparse cluster).                 |

If `contour.ts` does not get four corners, it still builds a quad from the **region bounding box** for grid/homography, but `cornerDebug.failureCode` reflects the CPU attempt above.

### AprilTag grid + decode (same CPU pass, `contour.ts`)

After corners exist (detected **or** bbox fallback), `validateAndFilterQuads()` builds a tag grid and runs dictionary decode:

1. **Perspective 6×6 grid** — `buildTagGrid()` in `src/lib/grid.ts` takes outer corners in **one** canonical order: **TL → TR → BL → BR** (same as `computeHomography` / `DetectedQuad.corners`).
2. **Pattern read** — `decodeTagPattern(corners, sobelData, width, …)` in `grid.ts` walks **integer image pixels** inside the quad’s axis-aligned bounding box. For each pixel it uses **`imagePixelToUnitSquareUv`** (inverse of the tag homography) at **pixel centers** `(ix + 0.5, iy + 0.5)` and reads **raw** `(gx, gy)` from **`sobelData`**. That buffer is the **NMS edge-filter output** read back from GPU `filteredBuffer` (same readback as `findCornersFromEdgesWithDebug`). It is **not** the pre-NMS `sobelBuffer`. Pixels outside the unit square are skipped. Samples with **numerically zero** magnitude (`≤ 1e⁻¹²`) are skipped; there is **no** adaptive magnitude quantile or **`magCut`** inside decode—**NMS on the GPU** is the main place edges are suppressed. An optional **`edgeMask`** (`buildDecodeEdgeMask`) can gate which pixels contribute; **`contour.ts`** passes **`undefined`** on the live path. Decode models AprilTag as an **8×8** module lattice in tag UV (black border ring + inner **6×6** data): gradients are pushed to tag UV with **`imageSobelToTagGradient`**. Each sample gets a **primary** `8×8` bin from `(u,v)`, then **center+diagonal** half-spaces in local module `[0,1]²` (`du = fu−½`, `dv = fv−½`; tie-break **top → bottom → left → right**) pick which **two** adjacent bins may receive a vote; **at most two** adjacent bins each get an **unweighted** ±1 vote from **`decodeVoteBinRadialDot`** per bin (**`gᵤ(u−cu)+gᵥ(v−cv)`** toward that bin’s center). Distance from `(u,v)` to the primary cell boundary (same wedge gate) must be ≤ **`max(τ, 2/L_min, 0.5/8)`** tag UV (`τ = 0.1/8`, **`L_min`** = shortest outer-quad side in px). Inner modules **`mx, my ∈ {1…6}`** map to the **36** dictionary bits. Per-module outcome is **`0`/`1`** from a majority, **`-1`** if total directional votes fall below **`DECODE_MIN_VOTE_TOTAL`** (3), or **`-2`** if positive and negative counts tie. **`fillUnknownNeighbors6`** then fills **`-1`** cells from neighbors where possible; **`-2`** (tie) is left unchanged so it stays an explicit unknown for dictionary matching. (`decodeCell` + per-cell UV sampling remain in `grid.ts` for tests and tooling; the live pipeline uses this homography + bbox path only.)
3. **Dictionary** — `decodeTag36h11AnyRotation(pattern, maxError)` (currently `maxError = 7` at the call site) tries four 90° rotations against the tag36h11 set (`src/lib/tag36h11.ts`, 587 codewords).
4. **`DetectedQuad` fields** — `pattern` (36 values `0 | 1 | -1 | -2`: data bits, insufficient votes, or ambiguous tie), and when a match is found, `decodedTagId` and `decodedRotation`. The **Calibrate** view shows the id when decoded, **`?`** when corners succeeded but dictionary decode did not. **`updateQuadCornersBuffer`** writes **`vizTagId`** (when set) into the instanced **`decodedTagId`** field; **`0xFFFFFFFF`** means unknown — grid viz draws **black** fill with no hash. When a real id is present, **`gridVizPipeline`** tints with **`stableHashToRgb01(decodedTagId)`** (deterministic pseudo-RGB), not camera texture.

**Reliability:** Dictionary decode can still miss when corners, perspective, or the filtered readback are weak; optional **`edgeMask`** experiments aside, tuning is mostly **GPU NMS / edge strength** and corner quality, not a second magnitude gate on the CPU. See **`docs/plan.md` → Phase 4.2**.

### Failure bitmask (see `corners.ts`)

Bits 0–4 are defined there; bit 1 is reserved for future use. Intersection-related failure is **bit 4 only** today (both “too few raw intersections” and “dedupe left &lt;4” share that code path). **Bit 3** groups ordering/plausibility failures after four deduped points exist (convex cycle, rotation choice, `R²`, slack-aligned bbox, edge ratios, cluster counts).

## Homography

Per-quad 8-parameter homography solved via Gaussian elimination with partial pivoting.

```
w = h7*u + h8*v + 1
x = (h1*u + h2*v + h3) / w
y = (h4*u + h5*v + h6) / w
```

Vertex shader passes `w` in `outPos.w` for automatic perspective-correct interpolation.

## GPU Buffers

| Buffer                    | Size                                                                                                                  | Type                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `sobelBuffer`             | W×H×vec2f                                                                                                             | gradient (gx, gy)                                            |
| `filteredBuffer`          | W×H×vec2f                                                                                                             | NMS-suppressed gradient                                      |
| `pointerJumpBuffer0/1`    | W×H×u32                                                                                                               | labels (ping-pong)                                           |
| `pointerJumpAtomicBuffer` | W×H×atomic u32                                                                                                        | atomic labels                                                |
| `compactLabelBuffer`      | W×H×u32                                                                                                               | compact labels                                               |
| `canonicalRootBuffer`     | area×atomic u32                                                                                                       | canonical root IDs                                           |
| `histogramBuffer`         | 256×atomic u32                                                                                                        | edge histogram                                               |
| `extentBuffer`            | MAX_EXTENT_COMPONENTS×ExtentEntry                                                                                     | bounding boxes                                               |
| `quadCornersBuffer`       | `MAX_INSTANCES` × (`GridDataSchema` in `gridVizPipeline.ts`: `mat3x3f` homography + `QuadDebug` + `decodedTagId` u32) | instanced grid viz (`MAX_INSTANCES` in that file, e.g. 1024) |
