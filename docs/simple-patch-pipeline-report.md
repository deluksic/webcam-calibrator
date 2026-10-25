# Simplified local-peak pipeline — current result

One synthetic oblique 1280×960 frame. The synthetic target mask is used directly;
no ONNX run.

## Exact pipeline

1. Gray input, `GaussianBlur(σ=0.8)`.
2. Production histogram cuts inside the synthetic target mask:
   - `black_cut=120`, `white_cut=134`, `background=127`.
   - `black_mask = target_mask & (bin < black_cut)`.
   - `white_mask = target_mask & (bin >= white_cut)`.
3. 3×3 extrema inside the corresponding mask:
   - black: 3×3 local minima in `black_mask`;
   - white: 3×3 local maxima in `white_mask`;
   - one representative per plateau; 2 px border excluded.
4. 5×5 square around each candidate; 10-term bivariate cubic least-squares.
5. Newton solve of `∇z=0`; vertex must be inside the 5×5.
6. Physical Hessian at the vertex: keep only negative-definite for white maxima,
   positive-definite for black minima. Indefinite/wrong sign = saddle, rejected.
7. Patch radius = synthetic target-mask bbox diagonal / 4.
   - mask bbox diagonal: `604.3 px`;
   - patch radius: `151.1 px`;
   - patch centers spaced by `0.75 × radius` (`113.3 px`, 25% overlap).
8. Per patch and **per polarity**, cluster `peak + λ1 + λ2`:
   - robust standardization per patch/polarity;
   - KMeans `K=2`;
   - keep the larger cluster;
   - fuse overlapping patches by vote (kept in ≥50% of containing patches).

## Candidate result

| quantity | value |
|---|---:|
| visible truth | 476 |
| raw candidates | 456 |
| accepted (definite Hessian) | 456 |
| accepted TP / FP / FN | **456 / 0 / 20** |
| saddle/indefinite | 0 |
| cubic-fit failures | 0 |

The histogram support mask removed all garbage. The 20 misses are at the target
mask/threshold boundary; there are no false positives.

## Cluster result

| quantity | value |
|---|---:|
| accepted candidates | 456 |
| cluster-kept | 338 |
| cluster-kept TP / FP / FN | **338 / 0 / 138** |
| patches with clusters | 41 |
| patches with accepted points | 23 |

The current per-patch `K=2` majority rule removes **118 true markers and zero
false positives**. It is splitting valid marker shape distributions, not
separating markers from clutter. Do not use it as a hard filter yet. The
cluster plots make this visible: green/orange are mixed in many patches.

## Plots

- Refined candidates and saddle rejects:
  `python/notebooks/blob_lab_data/simple_patch_pipeline/refined_candidates.png`
- All patch cluster scatters:
  `python/notebooks/blob_lab_data/simple_patch_pipeline/cluster_patches.png`
- Per-patch feature histograms:
  `python/notebooks/blob_lab_data/simple_patch_pipeline/feature_histograms_per_patch.png`
- Counts and patch metadata:
  `python/notebooks/blob_lab_data/simple_patch_pipeline/summary.json`

## Exact clustering features

- `peak`: cubic surface value at the refined stationary point.
- `lambda_1`: smaller physical Hessian eigenvalue at the refined point.
- `lambda_2`: larger physical Hessian eigenvalue at the refined point.

No cross-polarity clusters. No spatial coordinates are features; locality comes
from the patch.

## Next decision

- Keep the candidate stage; it is clean.
- Either stop at the candidate stage, or replace the `K=2` majority rule with a
  rule that does not split a single valid shape mode (e.g. one-cluster null test,
  silhouette threshold, or per-patch distribution fitting).
