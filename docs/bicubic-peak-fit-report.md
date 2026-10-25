# Bicubic local-peak fit — current report (corrected support + local clustering)

Status: **experiment complete; local clustered filtering is the next online
stage.** No production TypeScript/WebGPU changes, and no CNN checkpoint was
retrained.

## What was wrong in the first pass

Two mistakes are now fixed:

1. **Support scale was global.** `R_max` and `R_compact` were derived from one
   median nearest-candidate distance for the whole image. Under perspective the
   compressed side needs a smaller support than the stretched side, so this was
   not content-adaptive.
2. **Clustering was global.** KMeans ran over every fitted candidate on the
   screen. Marker shape changes systematically across the target (compressed vs
   stretched), so true markers split into several clusters and the PCA spread
   was dominated by position/angle, not by target-vs-clutter.

The corrected version uses per-candidate local support radii and overlapping
local-patch clustering. The connected curvature region you asked about is now
implemented as `curvature_region`.

## Support mechanism (exactly what is used now)

### Candidate detection

- Detection image: `GaussianBlur(luma, σ=1.2)`.
- White candidates = local maxima of `+detection`; black = local maxima of
  `−detection`; plateaus collapse to one representative.
- Gate: peak minus the minimum in a **5×5** window ≥ `min_prominence`
  (0.015 recall-oriented run; 0.05 cleaner run). The window is local, but this
  threshold is still one global value. That is the remaining non-local
  parameter and a candidate for a noise-relative local gate next.
- Fit image: a separate `GaussianBlur(luma, σ=0.8)`. Candidate peak/valley/
  prominence values are refreshed on this image before support selection.

### Local support scale

- For each candidate, take the 4 nearest other candidates (`cKDTree`) and use
  their median distance as the **local pitch**.
- `R_max = clamp(0.45 × local pitch, 2.5, 7.0)`;
  `R_compact = max(2.0, 0.60 × R_max)`.
- Both are per candidate, so the compressed side of an oblique target gets a
  smaller support than the stretched side.

### Variants

| variant | support |
|---|---|
| `compact` | disk at local `R_compact` |
| `full` | disk at local `R_max` (the large-window control) |
| `adaptive_circle` | starts at local compact, grows 0.5 px at a time while the 24-point ring is still falling, is above `valley + 0.10 × prominence`, has ≥80% falling slopes, ≥60% non-positive radial second derivative, and no ring sample above the peak |
| `adaptive_circle_concave` | the same disk intersected with the Hessian-sign mask (negative-definite for white, positive-definite for black); falls back to the disk if <10 pixels |
| `adaptive_ellipse_concave` | Hessian eigenvectors; curvature ratio sets the aspect; ray stops to 20% of the local prominence; broad axis = farther side, narrow = nearer side; clamped to local `R_max`; ellipse ∩ curvature mask |
| `curvature_region` | **connected** BFS region of correct-sign-curvature pixels above the local gray floor, bounded by the local radius; fits the connected region; falls back to `adaptive_circle_concave` if <10 pixels |

The `curvature_region` variant is the actual connected curvature support. The
earlier `concave` variants were only a disk/mask intersection, not a grown
region.

### Fit

- 10-term bivariate cubic least-squares in normalized coordinates, small ridge.
- Newton solve of `∇z = 0`; accept only a negative-definite Hessian at the
  vertex and a vertex inside the support.
- Falls back to the candidate pixel center if the support is too small, the
  design is singular, Newton fails, or the vertex is outside the support.
- `support_kind`, axes, pixel count, curvature-mask use, eigenvalues,
  anisotropy, residual, and offset are written to `descriptors.csv`.

## Fit results — clean, recall-oriented gate

Visible truth only; points within 8 px of the frame border are excluded because
they are partially off-screen. 6,586 visible truth points, 7,730 predictions.

| variant | F1@1.0 | mean err | median | p90 | fallback |
|---|---:|---:|---:|---:|---:|
| `compact` | 0.813 | 0.227 | 0.190 | 0.403 | 257 |
| `full` | 0.797 | 0.323 | 0.271 | 0.555 | 880 |
| `adaptive_circle` | 0.813 | 0.227 | 0.190 | 0.404 | 250 |
| `adaptive_circle_concave` | 0.813 | **0.212** | **0.178** | **0.393** | 303 |
| `adaptive_ellipse_concave` | 0.808 | 0.218 | 0.184 | 0.418 | 272 |
| `curvature_region` | 0.809 | 0.215 | 0.181 | 0.402 | 366 |

Per foreshortening bin (recall@1.0 / mean error px):

| bin | truth | full | circle concave | ellipse concave | curvature region |
|---|---:|---|---|---|---|
| frontal | 1367 | 0.980 / 0.260 | 0.980 / 0.168 | 0.979 / 0.176 | 0.982 / 0.173 |
| moderate | 1116 | 0.850 / 0.330 | 0.880 / **0.204** | 0.874 / 0.215 | 0.875 / 0.216 |
| oblique | 1331 | 0.878 / 0.330 | 0.890 / 0.214 | 0.885 / 0.222 | 0.885 / **0.214** |
| steep | 2772 | 0.811 / 0.355 | 0.833 / 0.241 | 0.826 / 0.241 | 0.828 / **0.239** |

The large fixed support is now clearly the worst: it pulls in neighbouring
markers and background, with a 0.323 px mean error and 880 fallbacks. The local
adaptive supports are stable; `adaptive_circle_concave` is the best overall,
and `curvature_region` is best/tied at oblique/steep and has the best connected
shape semantics.

## Camera sharpening

Mixed run (9/14 scenes sharpened), same local supports:

| variant | F1@1.0 | mean err | oblique mean | steep mean |
|---|---:|---:|---:|---:|
| `full` | 0.743 | 0.331 | 0.346 | 0.381 |
| `adaptive_circle_concave` | 0.762 | 0.198 | 0.215 | 0.213 |
| `adaptive_ellipse_concave` | 0.758 | 0.201 | 0.221 | 0.212 |
| `curvature_region` | 0.761 | **0.195** | **0.211** | **0.209** |

Sharpening adds halos/extra local structure and lowers F1 slightly, but the
local supports remain far better than the fixed large window. The sharpening
family is applied in the dataset generators to the final target-containing
frame; the next CNN training run will see it automatically.

## Local-patch clustering (the corrected filter diagnostic)

Why local: a target marker's shape is only comparable to nearby markers. The
compressed and stretched sides of the same target have different aspect ratios;
a global KMeans splits true markers along that gradient. The first-pass PCA made
this obvious.

Current diagnostic:

- Patch radius: `clamp(8 × median candidate spacing, 24, 96)` px.
- 50% overlap (`step = 0.5 × radius`).
- Per patch: standardize the shape descriptors, KMeans `K=2..4`, keep the
  majority cluster; all-true patches keep everything, all-false patches keep
  nothing.
- Fusion: a candidate is kept if it was kept in ≥50% of the patches containing
  it.
- `shape` feature set: peak, prominence, anisotropy, support radius,
  eigenvalues, residual, support-axis ratio. `full` adds support pixel count,
  curvature-mask use, and subpixel offset.

Local vs global clustering, clean recall-oriented run (`unclustered F1` is the
same candidate set with no filter):

| variant | feature set | scope | purity | recall | F1 | unclustered F1 |
|---|---|---|---:|---:|---:|---:|
| `curvature_region` | shape | **local** | 0.944 | 0.848 | **0.887** | 0.833 |
| `curvature_region` | shape | global | 0.932 | 0.812 | 0.854 | 0.833 |
| `adaptive_ellipse_concave` | shape | local | 0.949 | 0.814 | 0.873 | 0.828 |
| `adaptive_circle_concave` | shape | local | 0.948 | 0.814 | 0.872 | 0.835 |
| `adaptive_circle` | shape | local | 0.957 | 0.798 | 0.865 | 0.833 |
| `full` | shape | local | 0.943 | 0.786 | 0.852 | 0.843 |

Local patches improve every variant over global clustering. The best current
configuration is **`curvature_region` + `shape` descriptors + overlapping local
patches**: F1 0.887 vs 0.833 unfiltered, at 85% true-marker recall. The next
lever is a multi-cluster/multi-patch rule (coverage or size-ratio) to recover
the remaining true-marker loss without readmitting false positives.

## Representative examples

- Local-patch PCA: true markers vs false positives, and the local clusters
  (`cx=429, cy=593, r=49`, K=3, purity 1.000):
  `python/notebooks/blob_lab_data/roi_bicubic/clean_lowgate/cluster_pca.png`
- Per-bin recall / mean error, showing the large fixed support (orange) losing
  badly at oblique/steep:
  `python/notebooks/blob_lab_data/roi_bicubic/clean_lowgate/error_vs_angle.png`
- Oblique 1280×960, clean, with local-support fits:
  `python/notebooks/blob_lab_data/roi_bicubic/clean_lowgate/overlays/oblique_1280x960_00_zoom.png`
- Steep 640×480 stress case; yellow = visible truth, dim gray = excluded
  border/off-screen truth, colored = fits:
  `python/notebooks/blob_lab_data/roi_bicubic/clean_lowgate/overlays/steep_640x480_00_zoom.png`
- Descriptor distributions:
  `python/notebooks/blob_lab_data/roi_bicubic/clean_lowgate/feature_histograms.png`
- Support-size distribution:
  `python/notebooks/blob_lab_data/roi_bicubic/clean_lowgate/support_hist.png`

## Next step

1. Promote the local-patch majority filter to an online stage: patch grid over
   the candidate set, per-patch KMeans on the `shape` features, vote fusion.
2. Add a multi-cluster rule inside each patch (keep clusters covering a
   fraction of the patch or within a size ratio of the largest) to recover the
   remaining ~15% true-marker loss.
3. Keep `adaptive_circle_concave` as the default fit; `curvature_region` is the
   best connected-curvature alternative and the best clusterable shape so far.
4. Make the prominence gate noise-relative locally; it is the last global
   threshold in the pipeline.
5. Evaluate the filtered detector on the same fixed angle groups, visible truth
   only, against the unfiltered and fixed-large-window baselines.

## Reproduce

```bash
cd python
uv run python scripts/roi_bicubic_fits.py \
  --sharpen-probability 0 --min-prominence 0.015 \
  --out notebooks/blob_lab_data/roi_bicubic/clean_lowgate
uv run python scripts/roi_bicubic_fits.py \
  --out notebooks/blob_lab_data/roi_bicubic/mixed
uv run python scripts/analyze_peak_shapes.py \
  --descriptors notebooks/blob_lab_data/roi_bicubic/clean_lowgate/descriptors.csv
```

Camera sharpening and oblique pose sampling are configured by
`CameraSharpenConfig` and `PoseAugmentConfig`; no training script changes are
needed for the next run.
