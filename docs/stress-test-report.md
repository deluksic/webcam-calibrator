# Simplified pipeline — historical stress test

> **Historical.** These counts predate the raw-high-pass candidate class, the
> removal of the 2 px border crop, and the current 40-neighbour filter. The
> current pipeline is documented in
> [`gpu-refined-peak-pipeline.md`](gpu-refined-peak-pipeline.md).

40 examples:

- 28 synthetic: the 14-case angle suite × clean / camera-sharpened, with truth.
- 3 real hex-target photos: the two annotated positives and the unannotated
  `roi_real_01.jpg`.
- 9 negatives: 6 office + 3 wall images with no target.

Candidate stage: ONNX ROI + production-style high-pass for real images; synthetic
target mask for synthetic images; histogram cuts; 3×3 extrema; 5×5 cubic;
definite-Hessian filter. Neighbour filter at 20 and 40 same-polarity neighbours,
trim 2σ, accept 4σ.

## Synthetic aggregate (visible truth, match at 1 px)

| angle | truth | accepted | accepted F1 | kept20 F1 | kept20 recall | kept40 F1 | kept40 recall | mean error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frontal | 2734 | 2612 | 0.927 | 0.932 | 0.898 | 0.933 | 0.900 | 0.25 |
| moderate | 2856 | 1827 | 0.746 | 0.748 | 0.609 | 0.747 | 0.609 | 0.25 |
| oblique | 3784 | 3119 | 0.873 | 0.870 | 0.788 | 0.871 | 0.792 | 0.32 |
| steep | 3798 | 2122 | 0.674 | 0.665 | 0.512 | 0.670 | 0.520 | 0.30 |

- Precision stays high (0.94–0.97); the failure is **recall**.
- 20 vs 40 neighbours is marginal: 40 slightly higher recall, slightly lower
  precision.
- Low recall is concentrated in a few 640×480 scenes:

| case | truth | accepted | kept40 | F1@40 |
|---|---:|---:|---:|---:|
| moderate_640 clean 00 | 476 | 131 | 127 | 0.362 |
| moderate_640 clean 01 | 476 | 221 | 218 | 0.591 |
| oblique_640 clean 01 | 471 | 228 | 224 | 0.596 |
| steep_640 clean 01 | 476 | 104 | 99 | 0.282 |
| steep_640 clean 02 | 476 | 115 | 114 | 0.295 |
| steep_640 sharpened 02 | 476 | 140 | 139 | 0.358 |

The 1280×960 cases are around 0.96–0.98 F1. The failure mode is the small,
blurred 640 target: the histogram support masks miss low-contrast markers before
the cubic/neighbour stages are reached.

## Real photos (no point truth)

| image | accepted | kept20 | kept40 | notes |
|---|---:|---:|---:|---|
| `real_positive_01` | 477 | 472 | 473 | 99.8% inside manual mask; target 476 |
| `real_positive_02` | 433 | 423 | 419 | 100% inside manual mask; 2 saddles; target 476; misses in reflection |
| `roi_real_01` | 420 | 410 | 417 | no annotated mask; target 476 |

## Negatives

8/9 negatives produce zero candidates. `office_03.jpg` produces one candidate,
which survives both 20 and 40 neighbours. One false point across nine negatives.

## Conclusion

- The model is precise on real frames and negatives.
- The bottleneck is candidate recall on small/blurred 640×480 synthetic markers,
  not the cubic fit or neighbour filter.
- 40 neighbours does not materially change the synthetic result.
- On real `real_positive_02`, the trim-2σ/accept-4σ risk is still present: 40
  neighbours keeps 419 vs 431 for the no-trim 4σ rule.

## Files

- Summary plot:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/stress/stress_summary.png`
- Center-error CDF:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/stress/stress_center_error_cdf.png`
- Per-case JSON:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/stress/synthetic.json`,
  `real.json`, `negatives.json`
- Combined:
  `python/notebooks/blob_lab_data/simple_neighbor_pipeline/stress/summary.json`
