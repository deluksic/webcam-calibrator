#!/usr/bin/env python3
"""Focused visual: visible ground truth vs the best local-peak pipeline.

This is deliberately not the six-variant comparison.  It runs the recommended
end-to-end configuration (connected curvature-region fit + local-patch
majority filter) on one representative scene and draws only:

  * visible ground-truth markers as large hollow green circles,
  * fit-valid detections as small red dots,
  * a zoomed local patch before and after the cluster filter.

It also writes the exact descriptor list used by the local clustering/PCA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.vq import kmeans2

from blob_lab.augment import CameraSharpenConfig
from blob_lab.camera import PoseAugmentConfig
from blob_lab.heatmap_dataset import HeatmapDatasetConfig, generate_heatmap_dataset
from blob_lab.local_peak_fit import (
    BORDER_MARGIN,
    curvature_maps,
    detect_local_peaks,
    estimate_spacing,
    fit_peaks,
    refresh_candidate,
    smooth_luma,
    support_radii,
)
from blob_lab.target_layout import build_hex_target_layout

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from roi_bicubic_fits import (  # noqa: E402
    build_case_specs,
    make_config,
    match_from_distances,
    point_distances,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_bicubic" / "best_pipeline"

# Exactly the feature list used by the local clustering and PCA.
SHAPE_FEATURES: tuple[tuple[str, str], ...] = (
    ("log_peak", "log1p(10 * |fitted peak height|)"),
    ("log_prominence", "log1p(10 * local peak-to-valley prominence)"),
    ("anisotropy", "sqrt(min |Hessian eigenvalue| / max |Hessian eigenvalue|) at the fitted vertex"),
    ("log_support_radius", "log1p(broad support semi-axis / support radius)"),
    ("log_abs_eigenvalue_1", "log1p(10 * |smallest Hessian eigenvalue|)"),
    ("log_abs_eigenvalue_2", "log1p(10 * |largest Hessian eigenvalue|)"),
    ("log_residual_rms", "log1p(100 * RMS residual of the cubic fit)"),
    ("support_axis_ratio", "min(support semi-axis) / max(support semi-axis)"),
)

FULL_FEATURES_EXTRA: tuple[tuple[str, str], ...] = (
    ("log_support_pixels", "log1p(number of support pixels)"),
    ("used_curvature_mask", "1 if the fit used the curvature-sign support, else 0"),
    ("log_offset", "log1p(10 * |fitted vertex - candidate pixel| in px)"),
)


def _features(peaks) -> np.ndarray:
    rows = []
    for peak in peaks:
        eigen_1 = abs(float(peak.eigenvalues[0])) if np.isfinite(peak.eigenvalues[0]) else 0.0
        eigen_2 = abs(float(peak.eigenvalues[1])) if np.isfinite(peak.eigenvalues[1]) else 0.0
        axis_1 = max(1e-6, float(peak.support_axes[0]))
        axis_2 = max(1e-6, float(peak.support_axes[1]))
        residual = float(peak.residual_rms) if np.isfinite(peak.residual_rms) else 0.0
        rows.append(
            [
                np.log1p(10.0 * abs(float(peak.peak))),
                np.log1p(10.0 * max(0.0, float(peak.prominence))),
                float(peak.anisotropy) if np.isfinite(peak.anisotropy) else 0.0,
                np.log1p(float(peak.support_radius)),
                np.log1p(10.0 * eigen_1),
                np.log1p(10.0 * eigen_2),
                np.log1p(100.0 * max(0.0, residual)),
                min(axis_1, axis_2) / max(axis_1, axis_2),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    scale = q75 - q25
    fallback = matrix.std(axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return np.clip((matrix - median) / scale, -8.0, 8.0)


def _local_patch_keep(
    peaks,
    *,
    patch_radius: float,
    overlap: float = 0.5,
    clusters: int = 2,
    vote_threshold: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Per-patch KMeans + majority cluster + overlap vote.

    No ground truth is used here.  This is a first online-style version of the
    diagnostic filter; the offline analysis selected K using truth, so its
    reported F1 is optimistic.  This plot uses a fixed K=2 and reports the
    resulting counts separately.
    """
    features = _features(peaks)
    coords = np.asarray([[peak.x, peak.y] for peak in peaks], dtype=np.float64)
    n = coords.shape[0]
    keep_votes = np.zeros(n, dtype=np.float64)
    total_votes = np.zeros(n, dtype=np.float64)
    radius = max(8.0, float(patch_radius))
    step = max(6.0, radius * (1.0 - float(np.clip(overlap, 0.0, 0.9))))
    x_centers = np.arange(coords[:, 0].min(), coords[:, 0].max() + step, step)
    y_centers = np.arange(coords[:, 1].min(), coords[:, 1].max() + step, step)
    patches = 0
    k = max(2, int(clusters))
    for cx in x_centers:
        for cy in y_centers:
            inside = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2 <= radius * radius
            indices = np.nonzero(inside)[0]
            if indices.size < max(8, 2 * k):
                continue
            matrix = _standardize(features[indices])
            try:
                _centroids, labels = kmeans2(
                    matrix,
                    k,
                    iter=40,
                    minit="++",
                    seed=0,
                )
            except Exception:
                continue
            if labels.shape[0] != indices.size or (labels < 0).any():
                continue
            counts = np.bincount(labels, minlength=k)
            if int((counts > 0).sum()) < 2:
                keep_sub = np.ones(indices.size, dtype=bool)
            else:
                keep_sub = labels == int(np.argmax(counts))
            total_votes[indices] += 1.0
            keep_votes[indices] += keep_sub.astype(np.float64)
            patches += 1
    ratio = np.divide(
        keep_votes,
        np.maximum(total_votes, 1.0),
        out=np.ones(n, dtype=np.float64),
        where=total_votes > 0.0,
    )
    keep = ratio >= float(vote_threshold)
    keep[total_votes <= 0.0] = True
    return keep, patches


def _draw_points(axis, image, truth, detections, title, *, marker_scale=1.0):
    axis.imshow(
        np.clip(image, 0.0, 1.0),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=(-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5),
    )
    if truth.size:
        axis.scatter(
            truth[:, 0],
            truth[:, 1],
            s=45.0 * marker_scale,
            facecolors="none",
            edgecolors="#39ff14",
            linewidths=1.1 * max(0.6, marker_scale),
            label="ground truth (visible)",
            zorder=4,
        )
    if detections.size:
        axis.scatter(
            detections[:, 0],
            detections[:, 1],
            s=9.0 * marker_scale,
            c="#ff2d55",
            edgecolors="white",
            linewidths=0.3,
            label="detected",
            zorder=5,
        )
    axis.set_title(title, fontsize=8.5)
    axis.set_xticks([])
    axis.set_yticks([])


def _counts(detections: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    if detections.size == 0 or truth.size == 0:
        return 0, int(detections.shape[0]), int(truth.shape[0])
    distances = point_distances(detections, truth)
    matches = match_from_distances(distances, 1.0)
    tp = len(matches)
    return tp, int(detections.shape[0]) - tp, int(truth.shape[0]) - tp


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_case_specs(
        seed=args.seed,
        samples_scale=1.0,
        include_large=True,
        max_cases=None,
    )
    by_name = {spec.name: spec for spec in specs}
    if args.case not in by_name:
        raise SystemExit(f"unknown case {args.case!r}; choices: {sorted(by_name)}")
    spec = by_name[args.case]
    config = make_config(spec, sharpen_probability=args.sharpen_probability)
    layout = build_hex_target_layout(width_px=spec.width, height_px=spec.height)
    dataset = generate_heatmap_dataset(config, layout, count=spec.samples)
    sample = min(max(0, int(args.sample)), len(dataset) - 1)
    gray = np.asarray(dataset.images[sample, :, :, 0], dtype=np.float32)
    meta = dataset.metadata[sample]
    truth_all = np.asarray(meta.get("points") or [], dtype=np.float64).reshape(-1, 2)
    margin = float(args.visibility_margin)
    visible = (
        (truth_all[:, 0] >= margin)
        & (truth_all[:, 0] <= spec.width - 1 - margin)
        & (truth_all[:, 1] >= margin)
        & (truth_all[:, 1] <= spec.height - 1 - margin)
    )
    truth = truth_all[visible]

    detection_response = smooth_luma(gray, args.detect_sigma)
    fit_response = smooth_luma(gray, args.fit_sigma)
    curvature = curvature_maps(fit_response)
    candidates = detect_local_peaks(
        detection_response,
        min_prominence=args.min_prominence,
        prominence_window=args.prominence_window,
        border=BORDER_MARGIN,
    )
    candidates = [
        refresh_candidate(fit_response, candidate, window=args.support_window)
        for candidate in candidates
    ]
    spacing = estimate_spacing(candidates, fallback=float(layout.spacing_px))
    compact, maximum = support_radii(spacing)
    peaks = fit_peaks(
        fit_response,
        candidates,
        spacing=spacing,
        mode=args.fit_mode,
        compact_radius=compact,
        max_radius=maximum,
        curvature=curvature,
        local_radii=True,
    )
    fit_peaks_ok = [peak for peak in peaks if peak.fit_ok]
    raw_points = np.asarray([[peak.x, peak.y] for peak in fit_peaks_ok], dtype=np.float64)
    if fit_peaks_ok:
        keep, patches = _local_patch_keep(
            fit_peaks_ok,
            patch_radius=max(24.0, min(96.0, 8.0 * spacing)),
            clusters=args.filter_clusters,
            vote_threshold=args.vote_threshold,
        )
        filtered_points = raw_points[keep]
    else:
        patches = 0
        filtered_points = raw_points

    raw_tp, raw_fp, raw_fn = _counts(raw_points, truth)
    filt_tp, filt_fp, filt_fn = _counts(filtered_points, truth)

    # Pick a local patch with many visible truth points for the zoom.
    patch_radius = max(24.0, min(96.0, 8.0 * spacing))
    best_center = (float(truth[:, 0].mean()), float(truth[:, 1].mean())) if truth.size else (spec.width / 2, spec.height / 2)
    best_count = -1
    for cx in np.arange(0, spec.width, max(8.0, patch_radius * 0.5)):
        for cy in np.arange(0, spec.height, max(8.0, patch_radius * 0.5)):
            inside = (truth[:, 0] - cx) ** 2 + (truth[:, 1] - cy) ** 2 <= patch_radius * patch_radius
            count = int(inside.sum())
            if count > best_count:
                best_count = count
                best_center = (float(cx), float(cy))
    cx, cy = best_center
    x0 = int(max(0, np.floor(cx - patch_radius * 1.15)))
    x1 = int(min(spec.width, np.ceil(cx + patch_radius * 1.15)))
    y0 = int(max(0, np.floor(cy - patch_radius * 1.15)))
    y1 = int(min(spec.height, np.ceil(cy + patch_radius * 1.15)))
    crop = gray[y0:y1, x0:x1]
    crop_truth = truth[
        (truth[:, 0] >= x0) & (truth[:, 0] < x1) & (truth[:, 1] >= y0) & (truth[:, 1] < y1)
    ].copy()
    crop_truth[:, 0] -= x0
    crop_truth[:, 1] -= y0
    crop_raw = raw_points[
        (raw_points[:, 0] >= x0) & (raw_points[:, 0] < x1) & (raw_points[:, 1] >= y0) & (raw_points[:, 1] < y1)
    ].copy()
    crop_raw[:, 0] -= x0
    crop_raw[:, 1] -= y0
    crop_filtered = filtered_points[
        (filtered_points[:, 0] >= x0)
        & (filtered_points[:, 0] < x1)
        & (filtered_points[:, 1] >= y0)
        & (filtered_points[:, 1] < y1)
    ].copy()
    crop_filtered[:, 0] -= x0
    crop_filtered[:, 1] -= y0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.0),
        gridspec_kw={"width_ratios": (1.0, 1.25, 1.25)},
    )
    mask = np.asarray(dataset.masks[sample, :, :, 0], dtype=np.float32)
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        xs = truth[:, 0] if truth.size else np.asarray([0])
        ys = truth[:, 1] if truth.size else np.asarray([0])
    pad = 12
    fx0, fx1 = max(0, int(xs.min()) - pad), min(spec.width, int(xs.max()) + pad)
    fy0, fy1 = max(0, int(ys.min()) - pad), min(spec.height, int(ys.max()) + pad)
    full = gray[fy0:fy1, fx0:fx1]
    full_truth = truth[
        (truth[:, 0] >= fx0) & (truth[:, 0] < fx1) & (truth[:, 1] >= fy0) & (truth[:, 1] < fy1)
    ].copy()
    full_truth[:, 0] -= fx0
    full_truth[:, 1] -= fy0
    full_raw = raw_points[
        (raw_points[:, 0] >= fx0) & (raw_points[:, 0] < fx1) & (raw_points[:, 1] >= fy0) & (raw_points[:, 1] < fy1)
    ].copy()
    full_raw[:, 0] -= fx0
    full_raw[:, 1] -= fy0
    _draw_points(
        axes[0],
        full,
        full_truth,
        full_raw,
        f"context: visible truth {len(truth)} · raw fits {len(raw_points)}\n"
        f"whole scene TP={raw_tp} FP={raw_fp} FN={raw_fn}",
        marker_scale=0.42,
    )
    axes[0].add_patch(
        plt.Rectangle(
            (cx - patch_radius - fx0, cy - patch_radius - fy0),
            2 * patch_radius,
            2 * patch_radius,
            fill=False,
            edgecolor="#00d5ff",
            linewidth=1.2,
            linestyle="--",
        )
    )
    crop_raw_tp, crop_raw_fp, crop_raw_fn = _counts(crop_raw, crop_truth)
    crop_filt_tp, crop_filt_fp, crop_filt_fn = _counts(crop_filtered, crop_truth)
    _draw_points(
        axes[1],
        crop,
        crop_truth,
        crop_raw,
        f"local patch · raw fits\nGT={len(crop_truth)} points={len(crop_raw)} "
        f"TP={crop_raw_tp} FP={crop_raw_fp} FN={crop_raw_fn}",
        marker_scale=1.15,
    )
    _draw_points(
        axes[2],
        crop,
        crop_truth,
        crop_filtered,
        f"local patch · K={args.filter_clusters} majority vote\n"
        f"GT={len(crop_truth)} points={len(crop_filtered)} "
        f"TP={crop_filt_tp} FP={crop_filt_fp} FN={crop_filt_fn}",
        marker_scale=1.15,
    )
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor="none",
               markeredgecolor="#39ff14", markeredgewidth=1.4, label="ground truth (visible)"),
        Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="#ff2d55",
               markeredgecolor="white", markeredgewidth=0.3, label="detected"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.85)
    fig.suptitle(
        f"Best local-peak pipeline · {args.fit_mode} + local-patch majority filter · "
        f"spacing≈{spacing:.2f}px · patch r≈{patch_radius:.0f}px",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_path = out_dir / f"{args.case}_best_pipeline.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    feature_payload = {
        "case": spec.name,
        "sample": int(sample),
        "fit_mode": args.fit_mode,
        "detect_sigma": args.detect_sigma,
        "fit_sigma": args.fit_sigma,
        "min_prominence": args.min_prominence,
        "prominence_window": args.prominence_window,
        "local_patch_filter": {
            "clusters": int(args.filter_clusters),
            "vote_threshold": float(args.vote_threshold),
            "patch_radius_px": float(patch_radius),
            "overlap": 0.5,
            "standardization": "per-patch median/IQR robust z-score, clipped to ±8",
        },
        "shape_features": [
            {"name": name, "definition": definition} for name, definition in SHAPE_FEATURES
        ],
        "full_features_extra": [
            {"name": name, "definition": definition} for name, definition in FULL_FEATURES_EXTRA
        ],
        "pca": "SVD of the standardized shape-feature matrix inside one local patch; axes are PC1/PC2",
        "counts": {
            "visible_truth": int(len(truth)),
            "raw_fit_points": int(len(raw_points)),
            "filtered_points": int(len(filtered_points)),
            "raw_tp_fp_fn": [int(raw_tp), int(raw_fp), int(raw_fn)],
            "filtered_tp_fp_fn": [int(filt_tp), int(filt_fp), int(filt_fn)],
            "patches": int(patches),
        },
        "plot": str(plot_path),
    }
    (out_dir / f"{args.case}_best_pipeline.json").write_text(json.dumps(feature_payload, indent=2))
    print(json.dumps(feature_payload, indent=2))
    print(f"wrote {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="oblique_1280x960",
        help="case name from roi_bicubic_fits (e.g. oblique_1280x960, steep_640x480)",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sharpen-probability", type=float, default=0.0)
    parser.add_argument("--detect-sigma", type=float, default=1.2)
    parser.add_argument("--fit-sigma", type=float, default=0.8)
    parser.add_argument("--min-prominence", type=float, default=0.015)
    parser.add_argument("--prominence-window", type=int, default=5)
    parser.add_argument("--support-window", type=int, default=11)
    parser.add_argument(
        "--fit-mode",
        default="curvature_region",
        choices=("curvature_region", "adaptive_circle_concave"),
    )
    parser.add_argument("--filter-clusters", type=int, default=2)
    parser.add_argument("--vote-threshold", type=float, default=0.5)
    parser.add_argument("--visibility-margin", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
