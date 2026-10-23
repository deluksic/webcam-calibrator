# Hex Blob Detection — Deferred Plan

This document records the detection plan for the hex square target. Only the target renderer is implemented at present.

## Goal

Detect well-isolated black/white blobs on a gray target, classify their color, and reject environmental clutter before it reaches identity or calibration.

The marks are rendered as squares because that is easy to display cleanly. The detector should treat them as blobs, not as squares with corners or quads.

## Target assumptions

- One square per sampled honeycomb cell-edge point, matching the Point Lab pattern.
- Four points per cell side by default (configurable).
- Flat-top hexagonal cells.
- Marker side is roughly 45% of the nearest marker spacing.
- Every marker has a known gray padding ring.
- Background is neutral gray.
- Markers are pure black or pure white.
- Colors are deterministic binary values from a seed.
- Each marker carries a stable unique `designId` taken from its source point index.
- The target runs at the maximum 4:3 pixel resolution that fits the available display stage.
- The TypeGPU backing store is exactly that resolution; CSS only centers it and sets its 1:1 display size.
- The measured resolution changes on resize/fullscreen and is included in the JSON export.
- The target has at least 10% blank padding on each axis, stored in the layout so other renderers can reproduce it.
- A JSON export contains the resolution, padding, parameters, and exact integer quads.

The target layout should remain available in device-independent form for later detection work.


## Implemented first milestone: Hex Debug

`/hex-debug` now contains the first GPU component lab:

- gray ingest only, with no tag stages
- 1×1 pixel initialization
- union-find component merging
- merge predicate based on component mean and variance
- compact component readback and camera overlay

The merge is intentionally stricter when either component is uniform:

```text
varianceFloor = min(varianceA, varianceB)
tolerance = min(maxTolerance, baseTolerance + z × sqrt(varianceFloor))
merge if abs(meanA - meanB) <= tolerance
```

The lab exposes merge iterations, tolerance, minimum/maximum area, expected marker count, pause, a Gray/Components view toggle, and live metrics. Component labels are resolved with the existing pointer-jump (jump-flooding) step pipeline for final root compaction.

## Proposed pipeline

```text
gray camera frame
→ dual-threshold mask
→ 8-connected components
→ component moments and core statistics
→ content gate
→ surrounding-context gate
→ compact blob list
→ hex lattice fit
→ temporal tracking
```

Only the generic video ingest and gray conversion should be reused. Do not reuse the AprilTag Sobel, edge, quad, or dictionary stages.

## Threshold mask

Three values:

- `0` background
- `1` dark blob
- `2` light blob

Start with fixed thresholds around the known gray background. Add automatic threshold selection only after measuring fixed-threshold behavior.

Small morphological opening may remove single-pixel noise. Avoid aggressive blur because it shifts centroids.

## Connected components

- 8-connected components.
- Labels only for mask pixels.
- Compact roots to dense IDs.
- Cap the blob count.
- Keep dark and light components separate.

A pointer-jumping implementation is acceptable initially. If it becomes the bottleneck, use block-local union-find or a half-resolution mask with full-resolution centroid refinement.

## Component statistics

Accumulate per label:

- area
- `sumX`, `sumY`
- `sumXX`, `sumXY`, `sumYY`
- bounding box
- core area
- core gray sum and sum of squares

Define core pixels as mask pixels whose eight neighbours share the same label.

Use full-component moments for:

```text
cx = sumX / area
cy = sumY / area
```

Use core pixels for color mean, color variance, and rejecting edge-only components.

## Content gate

Reject a component when:

- core area is too small
- core standard deviation is too high
- area is far outside the expected range
- bounding-box fill ratio is implausibly low
- covariance is extremely elongated
- centroid lies outside the image

Do not require a near-square aspect ratio or four corners.

## Surrounding-context gate

This is the primary clutter filter.

Every target marker is surrounded by gray padding. Analyze a ring outside the component:

- expand the component bounding box by a padding derived from its size
- sample the expanded rectangle perimeter with a fixed number of samples
- require all samples to be background, not another component
- require the ring mean and standard deviation to represent one uniform background color
- use the ring mean as the local gray reference for color classification

Suggested sampling:

- padding: `clamp(round(0.15 × min(bboxWidth, bboxHeight)), 2, 8)`
- eight samples per side of the expanded bbox
- reject samples outside the image
- reject when too few valid samples remain

A component passes only if both its core and its surrounding ring are effectively single-color.

If the ring intersects a neighboring marker, the target padding is too small or the perspective is too severe. Reject the blob; do not guess.

## Color

Use the local ring mean as background:

```text
delta = coreMean - ringMean
white if delta > margin
black if delta < -margin
unknown otherwise
```

Initial margin: `0.08`. Confidence is `clamp(abs(delta) / margin, 0, 1)`.

Reject low-contrast, high-variance, or unknown cores.

## Compact blob output

Read back only accepted blobs:

```ts
type BlobDetection = {
  index: number
  center: Vec2
  area: number
  coreArea: number
  bbox: { minX; maxX; minY; maxY }
  covariance: { xx; xy; yy }
  coreMean: number
  coreStdDev: number
  ringMean: number
  ringStdDev: number
  color: 0 | 1
  contrast: number
  confidence: number
}
```

Never read back the full gray frame during normal operation.

## Hex lattice cleanup

The context gate removes most clutter. Remaining blobs must still form the target lattice.

### Neighbour filter

- estimate median nearest-neighbour spacing `s`
- reject blobs with fewer than two neighbours in a reasonable spacing band
- reject blobs without two distinct lattice directions
- reject isolated area outliers

### Lattice fit

1. Build a neighbour graph.
2. Seed from a blob with two non-collinear neighbours.
3. Use those neighbours as a local lattice basis.
4. Assign integer lattice coordinates to all candidates.
5. Fit an affine or projective transform from lattice coordinates to image centers.
6. Count inliers with a residual threshold around `0.25s`.
7. Re-seed and keep the best hypothesis.
8. Refit a homography with all inliers.
9. Reject candidates outside the final gate.

Reject the frame as “no lattice” when inlier count/fraction is too low or residual is too high.

## Temporal tracking

After a lattice lock:

- predict lattice points from the previous frame’s homography
- match new blobs within a small gate
- update the homography
- require a new track to survive several frames
- drop tracks missing for several frames

Static environmental clutter can persist, so tracking supplements lattice fitting rather than replacing it.

## Testing

Synthetic scenes should include:

- uniform gray background
- isolated black and white blobs
- perspective projection
- blur, noise, and exposure shifts
- textured surroundings
- neighboring blobs inside the expected padding ring
- random high-contrast clutter
- partial targets

Metrics:

- detection recall
- color accuracy
- false-positive rate after context and lattice gates
- centroid error
- lattice inlier count and residual
- frame-to-frame track stability

Initial acceptance targets:

- ≥99% recall at nominal framing
- ≥99.5% color accuracy among accepted blobs
- <0.1% false blobs after lattice cleanup in cluttered scenes
- centroid error <0.2 px
- 60 fps at 1280×720

## Identity recovery

The blob detector feeds a separate point-identity stage. For each ambiguous
point, high-proof hash anchors in its observed-graph support fit a local
homography; the local observed points are then matched partially against the
global design by color and predicted position, and the homography is refit from
all matched pairs. Point Lab exposes the full experiment; ROI Lab runs the same
recovery behind its “Local homography recovery” toggle. This is documented in
[`point-identity-icp-recovery.md`](point-identity-icp-recovery.md). It does not
change the blob detection pipeline above.

## Implementation phases

1. CPU reference detector and synthetic tests.
2. GPU threshold, connected components, moments, and context ring.
3. Compact readback and GPU/CPU parity tests.
4. Lattice fit and outlier tests.
5. Temporal tracking.
6. Live `/blob-lab` view and tuning.
7. Optional identity/calibration integration using lattice IDs and colors.

## Out of scope

- Square/quad detection.
- AprilTag dictionary or tag decode reuse.
- Calibration worker integration.
- Single-bit square identity.
- Printed targets.
