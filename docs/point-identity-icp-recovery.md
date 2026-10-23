# Local Hop-Graph Point Identity Recovery

Status: active in Point Lab, ROI Lab, and the production Calibrate path. ROI Lab
runs the same matcher behind the “Local homography recovery” toggle; Calibrate
always enables recovery and filters accepted points before they reach the
calibration frame pool.

## Problem

The hex-target hash is graph connectivity plus angular color order. It resolves
interior points well, but partial views and dropout break the local graph at the
edges. A single global homography is the wrong model here: it can only hold over
small areas when the view includes lens distortion or non-planar effects.

The active matcher therefore uses **local transforms only**, initialized from
anchors inside a graph-hop support region and refined with every local point it
intends to match.

## Algorithm

Run per point whose hash result is **ambiguous or unresolved** and which is not
an anchor:

1. **Collect local support anchors.**
   - BFS the **observed** graph from the ambiguous point.
   - Default `hopRadius = 8`; Point Lab slider 1–10.
   - Collect the anchors inside that observed support.
   - Fewer than 4 support anchors means the point stays ambiguous.
   - Hash is used only to admit anchors; it does not propose a design root.

2. **Fit the local homography.**
   - Fit a robust normalized homography observed → design from the support
     anchors.
   - Require at least 4 inliers, a valid orientation, and a finite inverse.
   - Identity depth is fixed at the maximum constant
     (`MAX_FINGERPRINT_DEPTH = 12`). **Anchor proof power** is an explicit slider
     (default 7, range 2–12) and is a hard floor.

3. **Match partially against the global design.**
   - Transform the local observed support into design space with the homography.
   - For each observed point, find the nearest same-color design point inside
     the gate (0.35 × minimum lattice spacing).
   - One-to-one greedy assignment.
   - The design side is global; only the observed side is local. Many points on
     both sides remain unmatched — this is a partial match, not a global
     registration.

4. **ICP-refine from all matched pairs.**
   - Refit the homography from all matched pairs, not just anchors.
   - Anchor guard: support anchors must stay within 0.5 design spacings.
   - Repeat up to 3 iterations.

5. **Process farthest-first and confirm the subgraph.**
   - Order ambiguous points by observed-graph hop distance to the nearest
     anchor, farthest first.
   - If the current point matches, confirm **every ambiguous point matched in
     that local subgraph**, not just the root point.
   - Confirmed points are removed from later work, so one good subgraph can
     replace many individual refinements.
   - Skip points whose nearest anchor is farther than the support radius.

6. **Accept the point.**
   - The design point matched to the ambiguous observed point is its identity.
   - There is no pre-selected design root and no root-match step.
   - Points with no match, or fewer than 4 support anchors, stay ambiguous.
   - Points already resolved by the hash are reserved and not overwritten.

Each still-unconfirmed ambiguous point gets its own local homography. No global
transform is fitted or iterated.

## Color and distance

- Colors are compared exactly. A wrong-color candidate is not used.
- Distances are measured in **design space**, so perspective squish does not
  change the gate.
- The distance gate is 0.35 × the minimum design-neighbor spacing.
- The support anchors are the guard: if a refit drifts away from them, that
  update is rejected.
- Every local homography must be orientation-preserving, finite, and invertible.
  Mirroring and singular fits are rejected after the initial fit and after every
  refit.
- The predicted-lattice overlay is drawn only from usable homography inverses
  and assigned design ids. Unmatched graph nodes are no longer drawn as tiny
  dots, which removes the apparent “splatter.”

## Baseline color-indexing bug (fixed)

`matchObservedIdentity` indexes colors by the graph point's `.index` while
computing fingerprints. In Point Lab that index is the design id, so the
baseline must be passed a **graph-indexed** color array. Passing the
position-indexed observed colors made almost every point look ambiguous.
Point Lab already passed design colors; the stress harness and several ad-hoc
tests did not. Those paths now pass a graph-indexed array.

This bug is why early local results looked like a failed transform: the matcher
was asked to resolve nearly the whole field instead of only the genuinely
ambiguous edge points.

## Point Lab

The **Geometric recovery** panel exposes:

- enable/disable local ICP;
- **support radius** in graph hops, default 8, slider 1–10;
- **anchor proof power**, default 7, slider 2–12;
- overlay toggles for anchors, warped lattice, and predicted missing.

Identity depth itself is fixed at the maximum 12 hops; the anchor slider controls
how many of those hops an anchor must prove.

Metrics show anchors, registrations, iterations, ICP correspondences, coverage, design
RMS, recovered, recovered false, ambiguous input/left, unresolved input/left,
predicted missing, conflicts, proof power, and color rejects.

Rendering:

- green diamond = anchor;
- cyan ring = recovered correctly;
- **red ring = recovered but wrong** (visible failure, not suppressed);
- magenta ring = conflict;
- hollow dashed circle = predicted missing.

The inspector shows proof power, ICP residual, votes, source, and conflict.

## Measured results

Anchor-first local homography, 20 stress runs, seed range 1–20, identity depth
fixed at 12, anchor proof power 7, 8 columns, 4 points/edge, observed hop radius
8:

| Scenario | Final coverage | Accepted | Recovered | Recovered false |
|---|---:|---:|---:|---:|
| Crop left 20% | 97.35% mean | 20/20 | 51.0 | 0 |
| Crop left 30% | 100% | 20/20 | 56.0 | 0 |
| Dropout 2% | 93.62% mean | 20/20 | 216.9 | 0 |
| Dropout 5% | 64.96% mean | 20/20 | 215.8 | 0 |
| Dropout 10% | 16.50% mean | 16/20 | 56.6 | 0 |
| Crop 20% + distortion 40% | 99.73% mean | 20/20 | 54.0 | 0 |
| Yaw 40° + crop 10% | 97.70% mean | 20/20 | 51.0 | 0 |
| Pitch 30° + crop top 10% | 100% | 20/20 | 91.0 | 0 |

The design side is global; only the observed local support is transformed. The
support anchors identify the local homography, and the ambiguous point's own
match against the global design identifies it. There is no hash-prefix root
search and no image-space nearest-anchor fallback. Farthest-first ordering plus
whole-subgraph confirmation cuts registrations sharply: crop 20% now needs ~12
registrations per frame instead of hundreds.

False matches are intentionally left visible on the Point Lab canvas and in the
metrics. They are diagnostic, not silently converted to safe failures.

## Limitations

- **Dropout with sparse anchors:** many support graphs have fewer than four
  anchors and correctly give up. Raise the support radius, or defer temporal
  accumulation. The failures are visible.
- **False local matches:** a local homography can still map a wrong same-color
  point into the gate. Point Lab draws these red; the next step is a local
  color-patch or residual validation if needed.
- **ROI Lab cost:** local recovery runs in the per-frame identity callback; the
  mask/input/labels views can use the recovery toggle if the extra cost matters.
- **Calibration gates:** production calibration accepts hash-resolved points and
  recovered points that are not geometry conflicts and stay inside the design
  distance gate. Predicted-missing points are lab diagnostics only, and a frame
  that assigns the same design id twice is rejected.

## Commands

```bash
pnpm test
pnpm type
pnpm stress:identity runs=20 depth=12 pointsPerEdge=4 columns=8 cropLeft=20 recover=1
pnpm stress:identity runs=20 depth=12 pointsPerEdge=4 columns=8 cropLeft=20 hopRadius=10 recover=1
pnpm stress:identity runs=20 depth=12 pointsPerEdge=4 columns=8 dropout=10 recover=1
pnpm stress:identity runs=20 depth=12 pointsPerEdge=4 columns=8 cropLeft=20 distortion=0.4 recover=1
```

`hopRadius` defaults to 8; the accepted range is 1–10.

## Implementation map

- `src/lib/pointIdentity/localIcpRefiner.ts` — anchor-first local homography,
  partial global-design match, ambiguous-only.
- `src/lib/pointIdentity/homographyFit.ts` — robust/normalized homography fit,
  orientation check.
- `src/lib/pointIdentity/localAffine.ts` — legacy affine fit, no longer used by
  the active matcher.
- `src/lib/pointIdentity/anchors.ts` — proof-power anchors.
- `src/lib/pointIdentity/identity.ts` — fingerprint prefix keys and indexes.
- `src/components/pointLab/PointLabView.tsx` — controls, metrics, inspector.
- `src/components/pointLab/PointLabCanvas.tsx` — overlays and red false rings.
- `src/components/roiLab/RoiLabView.tsx` — ROI identity integration and toggle.
- `src/gpu/roiLab/roiGraphPipeline.ts` — ROI identity status rendering, including
  recovered points.
- `src/lib/pointIdentity/stress.ts` — recovery stress reporting.
- `scripts/pointIdentityStress.ts` — CLI.
- `src/lib/pointIdentity/icpRefiner.ts` / `designSpaceIcp.ts` — legacy global
  model kept for comparison and shared result types; not used by Point Lab or
  the stress harness.
