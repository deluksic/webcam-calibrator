# GPU refined-peak detector and neighbour-filter worker

The browser ROI detector now mirrors the simplified Python model. The GPU stops
after producing refined peaks and their shape features; the 4σ same-polarity
neighbour filter runs in a worker, then the existing identity pipeline consumes
the surviving points.

## Split

```text
ONNX ROI mask + mask-aware background
  → GPU: Gaussian σ1.0 blur of the flattened luma
  → GPU: 256-bin histogram of the blurred image, masked
  → CPU: black/white cuts from the blurred histogram
  → GPU: raw high-pass black/white classification (the "Blobs" view)
  → GPU: 3×3 extrema of the blurred image inside that class
  → GPU: 5×5 10-term cubic fit, Newton peak, physical Hessian
  → GPU readback: { x, y, polarity, peak, λ1, λ2, hxy }
  → worker: same-polarity kNN, trim 2σ, recompute, accept 4σ
  → identity worker / matcher
```

No CCL, 2×2 opening, centroid NMS, patch clustering, or 4σ filtering executes on
the GPU.

## Candidate stage

- Blur: 9-tap OpenCV `GaussianBlur(0, 0, sigmaX=1.0)` kernel, `BORDER_REFLECT`.
- Histogram: blurred value, masked with bilinear mask sampling. The cut walk
  stops at 1.9% of the gray peak (down from 2%), so both cuts sit slightly
  farther from gray.
- Blob class: raw high-pass value (the same signal as the ROI Lab "Blobs"
  view), tested against the cuts from the blurred histogram. The blurred value
  is not re-tested here.
- Extrema: 3×3 minima for black, maxima for white on the blurred image; one
  representative per equal-valued plateau. All pixels are eligible, including
  the border. The 3×3 test only considers in-frame neighbours; the 5×5 fit
  clamps its reads at the frame edge.
- Fit: fixed 25×10 design `[1,u,v,u²,uv,v²,u³,u²v,uv²,v³]`, `u=dx/2`,
  `v=dy/2`; least squares is a precomputed 10×25 pseudo-inverse.
- Newton: 25 iterations, step limit 0.5, f32-safe convergence `1e-6`,
  determinant guard `1e-9`, vertex bound `1 ± 1e-5`.
- Hessian: physical units (`/4`); white requires both eigenvalues below
  `-1e-7`, black requires both above `+1e-7`.
- Subpixel position: `x = pixelX + 2u + 0.5`, same for `y`.

## Readback

`readPeaks()` copies a 16-byte counter header plus `4096 × 32` records:

| field      | type |
| ---------- | ---- |
| `x`        | f32  |
| `y`        | f32  |
| `peak`     | f32  |
| `lambda1`  | f32  |
| `lambda2`  | f32  |
| `hxy`      | f32  |
| `polarity` | u32  |
| padding    | u32  |

The header stores candidate, refined, and rejected counts. Only definite-Hessian
peaks are read back; saddles and fit failures stay in the GPU reject buffer for
the optional overlay.

## Worker filter

- Defaults: **40 same-polarity neighbours, trim 2σ, accept 4σ**.
- Features: `peak` (hard gate), `λ1/c`, `λ2/c`, `hxy/c`, where
  `c = |peak - 0.5|` is the peak contrast. Only brightness is removed; target
  size is retained.
- Hessian vote: at least **2 of the 3** Hessian features must pass 4σ
  (`ROI_MIN_PASSING_HESSIAN_FEATURES = 2`).
- For each point and feature:
  - initial neighbour mean/std (`ddof=1`);
  - trim neighbours outside `mean ± 2σ`;
  - recompute once if at least 3 inliers remain;
  - if fewer than 3 inliers remain, leave that feature untrimmed;
  - `z = (value - mean) / max(std, floor)`.
- The point is kept when `peak` satisfies `|z| <= 4` **and** at least two of
  `λ1/c`, `λ2/c`, `hxy/c` satisfy `|z| <= 4`.
- Fewer than four same-polarity points or fewer than three neighbours: keep the
  point unchanged.
- kNN selection uses a max-heap of the current worst neighbours: `O(N log K)`
  instead of `O(N K)`.
- Per-point feature values are precomputed once per frame; neighbour values are
  read from that matrix.
- The existing identity worker is warmed when the pipeline starts. If it fails,
  filtering falls back to the main thread.

## Files

- `src/gpu/roiLab/roiRefinedPeakPipeline.ts` — GPU blur, candidates, fit, readback.
- `src/gpu/roiLab/roiPeakTypes.ts` — shared schemas and readback types.
- `src/lib/roiPeakFit.ts` — fixed cubic design, pseudo-inverse, CPU reference,
  blur kernels, tolerances.
- `src/lib/roiNeighbourFilter.ts` — trim/recompute/accept filter.
- `src/workers/identity.worker.ts` — exposes the filter alongside identity.
- `src/workers/identityClient.ts` — shared worker lifecycle and filter client.
- `src/lib/roiPeakFit.test.ts`, `src/lib/roiNeighbourFilter.test.ts` — unit tests.

## Validation

- `pnpm type`, `pnpm test`, `pnpm build`.
- ROI Lab “Peaks” shows the pre-filter candidates; “Identity” shows the filtered
  graph.
- Python reference, current raw-class/no-border candidate stage:
  - synthetic `oblique_1280x960` sample 0: 481 accepted, 455 kept
    (455 TP / 0 FP / 21 FN at 1 px);
  - `real_positive_01`: 477 accepted, 473 kept;
  - `real_positive_02`: 426 accepted, 419 kept.

## ROI Lab inspector

- Switch to the **Peaks** view and click a refined peak (including a point that
  the worker filter dropped).
- The sidebar plots the point's same-polarity neighbour values for `peak`,
  `λ1/c`, `λ2/c`, and `hxy/c`, with:
  - the neighbour histogram;
  - the neighbour mean after trimming;
  - the `mean ± 4σ` band;
  - the point's own value (green accepted, red rejected);
  - red ticks for trimmed neighbour values.
- The panel reports the point coordinates, polarity, neighbour count, z-scores,
  and the failed Hessian features. A kept point with one failing Hessian feature
  is shown as `kept (2/3 Hessian): … outside 4σ`; a failed peak is shown as
  `dropped: peak outside 4σ`.
- The selected peak is drawn with a larger yellow ring in the Peaks view; the
  ring follows zoom/pan and clears when clicking empty space. The analysis
  follows the presented frame, so it stays aligned while the live view advances.

Implementation: [`PeakInspector.tsx`](../src/components/roiLab/PeakInspector.tsx),
`describeRefinedPeak` in
[`roiNeighbourFilter.ts`](../src/lib/roiNeighbourFilter.ts), and
`getPresentedPeakAnalysis` in
[`roiLabPipeline.ts`](../src/gpu/roiLab/roiLabPipeline.ts).
