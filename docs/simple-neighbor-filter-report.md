# Neighbour-consistency filter — historical result

> **Historical.** This report predates the raw-high-pass candidate class, the
> removal of the 2 px border crop, and the 40-neighbour normalized-Hessian
> filter. The current pipeline is documented in
> [`gpu-refined-peak-pipeline.md`](gpu-refined-peak-pipeline.md). The tables
> below describe the older 20-neighbour, raw-λ variant and should not be used to
> interpret current ROI Lab counts.

## Pipeline

1. `GaussianBlur(σ=0.8)`.
2. Production histogram cuts inside the CNN/synthetic target mask.
3. Separate black/white support masks.
4. 3×3 minima inside black, maxima inside white; one per plateau.
5. 5×5 cubic fit; Newton stationary point; keep only negative-definite for
   white maxima / positive-definite for black minima.
6. **Neighbour test:**
   - 20 nearest neighbours in image space, **same polarity only**;
   - features: `peak`, `λ1`, `λ2`;
   - for each feature:
     `value ∈ mean_neighbours ± sigma × std_neighbours`;
   - reject if any feature fails;
   - `sigma` is configurable; default now **4.0**.

## Per-point plots vs aggregate plots

- `neighbor_feature_histograms.png` is **aggregate**: top row pools raw feature
  values over all kept/removed points; bottom row pools neighbour z-scores.
  It is not a per-point view.
- `per_point/removed.png` and `per_point/kept_examples.png` are **per-point**.
  Each point has three panels (peak, λ1, λ2). Each panel shows:
  - histogram of that point's 20 neighbour values;
  - blue line: neighbour mean;
  - green band: mean ± sigma × std;
  - point's own value: green if inside the band, red if outside;
  - title: value, neighbour mean, std, z-score;
  - row label: point index, coordinates, polarity, failed feature(s).

## Synthetic oblique 1280×960

Candidate stage unchanged: **456 / 476**, **0 FP**, **0 saddles/failures**.

| sigma | removed | kept |
|---:|---:|---:|
| 3.0 | 9 | 447 |
| 3.5 | 2 | 454 |
| 4.0 | **0** | **456** |
| 4.5 | 0 | 456 |
| 5.0 | 0 | 456 |

At 3σ the filter removed 9 true markers. At 4σ it removes nothing. On this
clean candidate set there is no garbage for it to remove.

## Real photo `real_positive_01.jpg`

ONNX ROI model input `256×144`; production-style high-pass:

- model-resolution mask-weighted **25×25 box blur** (radius 12), divided by the
  blurred mask weight, upsampled;
- `flattened = clip(0.5 + (gray − background) × residual_gain, 0, 1)`,
  `residual_gain=1.0`;
- histogram cuts on the flattened image: black **114**, white **139**,
  background **126**.

Candidate stage: **477 accepted**, 0 saddles, 0 fit failures. There is one more
point than the 476-marker target; the extra is the non-marker blot near the
bottom of the screen.

| sigma | removed | kept |
|---:|---:|---:|
| 3.0 | 7 | 470 |
| 3.5 | 2 | 475 |
| 4.0 | **1** | **476** |
| 4.5 | 1 | 476 |
| 5.0 | 1 | 476 |

At 4σ the only removed point is the non-marker blot. Its per-point plot:
`peak = 0.8654`, neighbour mean `0.7411`, std `0.0099`, **z = +12.52**; λ1/λ2 are
inside the band. The result is exactly 476 kept points, matching the target
marker count.

There is no point ground truth for this photo, so the 476 count is an expected
target count, not a per-point verified match.

## Files

- Real photo preprocessing (input, background, flattened, masks):
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_01/preprocessing.png`
- Real photo per-point removed point:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_01/per_point/removed.png`
- Real photo per-point kept examples:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_01/per_point/kept_examples.png`
- Real photo neighbour filter at 4σ:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_01/neighbor_filter.png`
- Aggregate feature/z histograms:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_01/neighbor_feature_histograms.png`
- Synthetic per-point examples:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/per_point/kept_examples.png`
- Summary:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_01/summary.json`

## Second real photo: `real_positive_02.png`

Closer view of the same phone target, with specular reflection on the right side
of the screen.

- ONNX ROI model input `256×144`; high-pass as above.
- Histogram cuts: black **118**, white **138**, background **126**.
- Candidate stage: raw **435**, accepted **433**, saddle/indefinite **2**,
  fit failures **0**.
- Accepted count is below the 476-marker target; the missing markers are mostly
  in the reflective/compressed right side of the screen.

| sigma | removed | kept |
|---:|---:|---:|
| 3.0 | 11 | 422 |
| 3.5 | 5 | 428 |
| 4.0 | **2** | **431** |
| 4.5 | 1 | 432 |
| 5.0 | 1 | 432 |

Per-point removal at 4σ:

- point 124: black, peak `0.456` vs black-neighbour mean `0.143`, z `+4.05`;
- point 216: black, peak `0.454` vs black-neighbour mean `0.094`, z `+25.9`.

Both are black markers whose local brightness is far above the black-neighbour
distribution, consistent with the screen reflection. At 4.5σ only the extreme
point 216 is removed. The 3σ setting removes 11 points, again too aggressive.

- Per-point removed panels:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_02/per_point/removed.png`
- Candidate stage:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_02/refined_candidates.png`
- Neighbor filter:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_02/neighbor_filter.png`
- Summary:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/real_positive_02/summary.json`

## Other real data

- `python/notebooks/test_images/real/roi_real_01.jpg`: real hex target,
  640×480, no annotated mask; not yet run.
- `python/notebooks/test_images/run-1/` and `run-2/`: 13 webcam frames, but
  they show the old **AprilTag** target, not the hex target.
- `python/notebooks/natural_images/` (34 office + 97 wall/art/negative images):
  backgrounds/negatives, not target positives.
