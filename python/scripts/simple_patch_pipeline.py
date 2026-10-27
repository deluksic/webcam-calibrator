#!/usr/bin/env python3
"""Simplified local-peak pipeline.

Exact pipeline requested:
  raw high-pass input, blurred with sigma 1.0
  -> production histogram cuts on the blurred image, restricted to the mask
  -> separate raw high-pass black/white support masks
  -> 3x3 minima/maxima of the blurred image inside each support mask
  -> 10-term cubic fit on the 5x5 square around each candidate (edge-clamped)
  -> Newton stationary point; keep only a definite Hessian of the correct sign
  -> local-patch clustering per patch and per polarity, using peak + Hessian
     eigenvalues only
  -> plots: refined candidates + saddle rejects, all patch cluster scatters,
     per-patch feature histograms.

The synthetic target mask from the dataset is used directly; no model run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.vq import kmeans2

from blob_lab.heatmap_dataset import generate_heatmap_dataset
from blob_lab.target_layout import build_hex_target_layout

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from roi_bicubic_fits import (  # noqa: E402
    build_case_specs,
    make_config,
    match_from_distances,
    point_distances,
)
from roi_histogram_nms import histogram_cuts, segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "python" / "notebooks" / "blob_lab_data" / "simple_patch_pipeline"

FEATURE_KEYS = ("peak", "eigenvalue_1", "eigenvalue_2")
FEATURE_LABELS = ("peak", "lambda_1", "lambda_2")


def _fit_cubic_5x5(image: np.ndarray, x: int, y: int) -> dict[str, Any] | None:
    """Fit the 10-term cubic on the 5x5 square and Newton-refine its extremum.

    Edge candidates are allowed. Their 5x5 window repeats the nearest edge
    pixel, matching the GPU's clamped reads.
    """
    height, width = image.shape
    dx = np.asarray([side for side in (-2, -1, 0, 1, 2)], dtype=np.float64)
    grid_x, grid_y = np.meshgrid(dx, dx, indexing="xy")
    u = grid_x.ravel() / 2.0
    v = grid_y.ravel() / 2.0
    design = np.stack(
        [
            np.ones_like(u),
            u,
            v,
            u * u,
            u * v,
            v * v,
            u * u * u,
            u * u * v,
            u * v * v,
            v * v * v,
        ],
        axis=1,
    )
    xs = np.clip(np.arange(x - 2, x + 3), 0, width - 1)
    ys = np.clip(np.arange(y - 2, y + 3), 0, height - 1)
    values = image[np.ix_(ys, xs)].reshape(-1).astype(np.float64)
    try:
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coefficients).all():
        return None
    c = coefficients
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = c[1:].tolist()

    uu, vv = 0.0, 0.0
    converged = False
    for _iteration in range(25):
        gx = c1 + 2 * c3 * uu + c4 * vv + 3 * c6 * uu * uu + 2 * c7 * uu * vv + c8 * vv * vv
        gy = c2 + c4 * uu + 2 * c5 * vv + c7 * uu * uu + 2 * c8 * uu * vv + 3 * c9 * vv * vv
        hxx = 2 * c3 + 6 * c6 * uu + 2 * c7 * vv
        hxy = c4 + 2 * c7 * uu + 2 * c8 * vv
        hyy = 2 * c5 + 2 * c8 * uu + 6 * c9 * vv
        hessian = np.asarray([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
        det = float(np.linalg.det(hessian))
        if not np.isfinite(det) or abs(det) < 1e-12:
            break
        step = np.linalg.solve(hessian, -np.asarray([gx, gy], dtype=np.float64))
        if not np.isfinite(step).all():
            break
        scale = 1.0
        if float(np.linalg.norm(step)) > 0.5:
            scale = 0.5 / float(np.linalg.norm(step))
        uu += step[0] * scale
        vv += step[1] * scale
        if float(np.linalg.norm(step * scale)) < 1e-8:
            converged = True
            break
    if not converged:
        return None
    if abs(uu) > 1.0 + 1e-6 or abs(vv) > 1.0 + 1e-6:
        return None

    # Re-evaluate the Hessian at the converged vertex, then convert from
    # normalized (u,v) to physical pixel units: d/dx = (1/2) d/du.
    hxx = 2 * c3 + 6 * c6 * uu + 2 * c7 * vv
    hxy = c4 + 2 * c7 * uu + 2 * c8 * vv
    hyy = 2 * c5 + 2 * c8 * uu + 6 * c9 * vv
    physical_hessian = np.asarray([[hxx, hxy], [hxy, hyy]], dtype=np.float64) / 4.0
    eigenvalues = np.linalg.eigvalsh(physical_hessian)
    peak = float(
        c[0]
        + c1 * uu
        + c2 * vv
        + c3 * uu * uu
        + c4 * uu * vv
        + c5 * vv * vv
        + c6 * uu**3
        + c7 * uu * uu * vv
        + c8 * uu * vv * vv
        + c9 * vv**3
    )
    return {
        "x": float(x) + 2.0 * uu + 0.5,
        "y": float(y) + 2.0 * vv + 0.5,
        "offset_x": 2.0 * uu,
        "offset_y": 2.0 * vv,
        "peak": peak,
        "eigenvalue_1": float(eigenvalues[0]),
        "eigenvalue_2": float(eigenvalues[1]),
        "hessian_xy": float(physical_hessian[0, 1]),
    }


def _plateau_representatives(mask: np.ndarray, values: np.ndarray) -> list[tuple[int, int]]:
    """One candidate per equal-valued plateau (raster tie-break)."""
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    keep: list[tuple[int, int]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        value = values[y, x]
        earlier = False
        for dy, dxx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            ny, nx = y + dy, x + dxx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and mask[ny, nx]
                and abs(float(values[ny, nx]) - float(value)) <= 1e-7
            ):
                earlier = True
                break
        if not earlier:
            keep.append((x, y))
    return keep


def _candidates(
    image: np.ndarray,
    support: np.ndarray,
    polarity: int,
) -> list[tuple[int, int]]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    if polarity == 1:
        local = cv2.dilate(image, kernel, borderType=cv2.BORDER_REPLICATE)
        extrema = image >= local - 1e-7
    else:
        local = cv2.erode(image, kernel, borderType=cv2.BORDER_REPLICATE)
        extrema = image <= local + 1e-7
    extrema &= support
    # No border crop: every pixel in the blob class is eligible. The 3x3
    # support is replicate-padded and the 5x5 fit clamps at the edge.
    return _plateau_representatives(extrema, image)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    scale = q75 - q25
    fallback = matrix.std(axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return np.clip((matrix - median) / scale, -8.0, 8.0)


def _cluster_patch(
    points: list[dict[str, Any]],
    radius: float,
    center: tuple[float, float],
    polarity: int,
) -> dict[str, Any] | None:
    selected = [
        point
        for point in points
        if point["polarity"] == polarity
        and (point["x"] - center[0]) ** 2 + (point["y"] - center[1]) ** 2 <= radius * radius
    ]
    if len(selected) < 10:
        return None
    matrix = np.asarray(
        [[point["peak"], point["eigenvalue_1"], point["eigenvalue_2"]] for point in selected],
        dtype=np.float64,
    )
    matrix = _standardize(matrix)
    try:
        _centroids, labels = kmeans2(matrix, 2, iter=40, minit="++", seed=0)
    except Exception:
        return None
    if labels.shape[0] != len(selected) or (labels < 0).any():
        return None
    counts = np.bincount(labels, minlength=2)
    if int((counts > 0).sum()) < 2:
        return None
    majority = int(np.argmax(counts))
    keep = labels == majority
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T
    return {
        "center": (float(center[0]), float(center[1])),
        "radius": float(radius),
        "polarity": int(polarity),
        "points": selected,
        "labels": labels,
        "keep": keep,
        "projection": projection,
        "counts": [int(value) for value in counts],
        "majority": majority,
    }


def _crop_bounds(mask: np.ndarray, margin: int = 24) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        height, width = mask.shape
        return 0, width, 0, height
    return (
        max(0, int(xs.min()) - margin),
        min(mask.shape[1], int(xs.max()) + margin),
        max(0, int(ys.min()) - margin),
        min(mask.shape[0], int(ys.max()) + margin),
    )


def _plot_candidates(
    path: Path,
    image: np.ndarray,
    cnn_mask: np.ndarray,
    support_black: np.ndarray,
    support_white: np.ndarray,
    accepted: list[dict[str, Any]],
    saddles: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    truth: np.ndarray,
    title: str,
) -> None:
    x0, x1, y0, y1 = _crop_bounds(cnn_mask)
    crop = image[y0:y1, x0:x1]
    fig, axis = plt.subplots(figsize=(9.0, 7.0))
    axis.imshow(crop, cmap="gray", vmin=0.0, vmax=1.0)
    axis.contour(
        cnn_mask[y0:y1, x0:x1].astype(np.float32),
        levels=[0.5],
        colors=["#00d5ff"],
        linewidths=1.2,
    )

    def xy(points, key_x="x", key_y="y"):
        return np.asarray(
            [[point[key_x] - 0.5 - x0, point[key_y] - 0.5 - y0] for point in points],
            dtype=np.float64,
        )

    if truth.size:
        visible = truth[
            (truth[:, 0] >= x0)
            & (truth[:, 0] < x1)
            & (truth[:, 1] >= y0)
            & (truth[:, 1] < y1)
        ]
        if visible.size:
            axis.scatter(
                visible[:, 0] - 0.5 - x0,
                visible[:, 1] - 0.5 - y0,
                s=26,
                facecolors="none",
                edgecolors="#39ff14",
                linewidths=0.8,
                label="ground truth",
                zorder=6,
            )
    white = [point for point in accepted if point["polarity"] == 1]
    black = [point for point in accepted if point["polarity"] == 0]
    if white:
        pts = xy(white)
        axis.scatter(pts[:, 0], pts[:, 1], s=13, c="#ff2d55", edgecolors="white",
                     linewidths=0.25, label=f"accepted white ({len(white)})", zorder=5)
    if black:
        pts = xy(black)
        axis.scatter(pts[:, 0], pts[:, 1], s=13, c="#2d7dff", edgecolors="white",
                     linewidths=0.25, label=f"accepted black ({len(black)})", zorder=5)
    if saddles:
        pts = xy(saddles)
        axis.scatter(pts[:, 0], pts[:, 1], s=34, marker="x", c="#ffb000",
                     linewidths=1.1, label=f"saddle/indefinite ({len(saddles)})", zorder=5)
    if failures:
        pts = xy(failures)
        axis.scatter(pts[:, 0], pts[:, 1], s=18, marker="+", c="#888888",
                     linewidths=0.8, label=f"fit failure ({len(failures)})", zorder=4)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {spec.name: spec for spec in build_case_specs(
        seed=args.seed, samples_scale=1.0, include_large=True, max_cases=None
    )}
    spec = specs[args.case]
    config = make_config(spec, sharpen_probability=0.0)
    layout = build_hex_target_layout(width_px=spec.width, height_px=spec.height)
    dataset = generate_heatmap_dataset(config, layout, count=spec.samples)
    sample = min(max(0, args.sample), len(dataset) - 1)
    gray = np.asarray(dataset.images[sample, :, :, 0], dtype=np.float32)
    cnn_mask = np.asarray(dataset.masks[sample, :, :, 0], dtype=np.float32) > 0.5
    truth = np.asarray(dataset.metadata[sample].get("points") or [], dtype=np.float64).reshape(-1, 2)
    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=args.blur_sigma,
        sigmaY=args.blur_sigma,
        borderType=cv2.BORDER_REFLECT,
    )

    cuts = histogram_cuts(blurred, cnn_mask)
    # The blob class comes from the raw high-pass image, matching the ROI Lab
    # "Blobs" view. Only the 3x3 extrema and cubic fit use the blurred image.
    black_mask, white_mask = segment(gray, cnn_mask, cuts)
    if not black_mask.any() or not white_mask.any():
        raise SystemExit(
            f"histogram masks are empty: cuts={cuts['black_cut']}/{cuts['white_cut']} "
            f"background={cuts['background']}"
        )

    raw_candidates: list[dict[str, Any]] = []
    for polarity, support, name in ((0, black_mask, "black"), (1, white_mask, "white")):
        for x, y in _candidates(blurred, support, polarity):
            raw_candidates.append({"x": float(x), "y": float(y), "polarity": polarity})

    accepted: list[dict[str, Any]] = []
    saddles: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        fit = _fit_cubic_5x5(blurred, int(candidate["x"]), int(candidate["y"]))
        if fit is None:
            failures.append(candidate)
            continue
        eigen_1 = float(fit["eigenvalue_1"])
        eigen_2 = float(fit["eigenvalue_2"])
        if candidate["polarity"] == 1:
            definite = eigen_1 < -args.hessian_epsilon and eigen_2 < -args.hessian_epsilon
        else:
            definite = eigen_1 > args.hessian_epsilon and eigen_2 > args.hessian_epsilon
        refined = {
            **candidate,
            **fit,
        }
        if definite:
            accepted.append(refined)
        else:
            saddles.append(refined)

    ys, xs = np.where(cnn_mask)
    if len(xs) == 0:
        raise SystemExit("synthetic CNN mask is empty")
    mask_width = float(xs.max() - xs.min() + 1)
    mask_height = float(ys.max() - ys.min() + 1)
    diagonal = float(np.hypot(mask_width, mask_height))
    patch_radius = max(16.0, diagonal / 4.0)
    step = max(8.0, patch_radius * 0.75)
    centers = [
        (float(cx), float(cy))
        for cy in np.arange(float(ys.min()), float(ys.max()) + step, step)
        for cx in np.arange(float(xs.min()), float(xs.max()) + step, step)
    ]

    patches: list[dict[str, Any]] = []
    for index, center in enumerate(centers):
        for polarity in (0, 1):
            cluster = _cluster_patch(accepted, patch_radius, center, polarity)
            if cluster is not None:
                cluster["index"] = index
                patches.append(cluster)

    # Vote fusion across overlapping patches; no votes => keep.
    votes = defaultdict(float)
    total = defaultdict(float)
    for cluster in patches:
        for point, keep in zip(cluster["points"], cluster["keep"]):
            key = (point["x"], point["y"], point["polarity"])
            total[key] += 1.0
            votes[key] += float(keep)
    for point in accepted:
        key = (point["x"], point["y"], point["polarity"])
        point["cluster_votes"] = total.get(key, 0.0)
        point["cluster_keep"] = (
            True if total.get(key, 0.0) <= 0.0 else votes[key] / total[key] >= 0.5
        )

    margin = 8.0
    visible_truth = truth[
        (truth[:, 0] >= margin)
        & (truth[:, 0] <= spec.width - 1 - margin)
        & (truth[:, 1] >= margin)
        & (truth[:, 1] <= spec.height - 1 - margin)
    ] if truth.size else truth

    def _match_counts(points: list[dict[str, Any]]) -> tuple[int, int, int]:
        if not points or visible_truth.size == 0:
            return 0, len(points), int(visible_truth.shape[0])
        coordinates = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float64)
        distances = point_distances(coordinates, visible_truth)
        matches = match_from_distances(distances, 1.0)
        tp = len(matches)
        return tp, len(points) - tp, int(visible_truth.shape[0]) - tp

    accepted_tp, accepted_fp, accepted_fn = _match_counts(accepted)
    kept_points = [point for point in accepted if point["cluster_keep"]]
    kept_tp, kept_fp, kept_fn = _match_counts(kept_points)

    counts = {
        "visible_truth": int(visible_truth.shape[0]),
        "accepted_tp_fp_fn": [int(accepted_tp), int(accepted_fp), int(accepted_fn)],
        "cluster_kept_tp_fp_fn": [int(kept_tp), int(kept_fp), int(kept_fn)],
        "raw_candidates": len(raw_candidates),
        "accepted": len(accepted),
        "saddle_or_indefinite": len(saddles),
        "fit_failure": len(failures),
        "accepted_white": sum(1 for point in accepted if point["polarity"] == 1),
        "accepted_black": sum(1 for point in accepted if point["polarity"] == 0),
        "cluster_kept": sum(1 for point in accepted if point["cluster_keep"]),
        "cluster_removed": sum(1 for point in accepted if not point["cluster_keep"]),
        "patches_with_clusters": len(patches),
    }
    summary = {
        "case": spec.name,
        "sample": int(sample),
        "blur_sigma": float(args.blur_sigma),
        "histogram": {
            "background": int(cuts["background"]),
            "black_cut": int(cuts["black_cut"]),
            "white_cut": int(cuts["white_cut"]),
            "masked_pixels": int(cuts["masked_pixels"]),
        },
        "cnn_mask_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "cnn_mask_diagonal_px": diagonal,
        "patch_radius_px": patch_radius,
        "patch_step_px": step,
        "features": list(FEATURE_KEYS),
        "feature_definition": {
            "peak": "cubic surface value at the refined stationary point",
            "eigenvalue_1": "smaller physical Hessian eigenvalue at the refined point",
            "eigenvalue_2": "larger physical Hessian eigenvalue at the refined point",
        },
        "counts": counts,
        "patches": [
            {
                "index": cluster["index"],
                "center": cluster["center"],
                "polarity": cluster["polarity"],
                "count": len(cluster["points"]),
                "cluster_sizes": cluster["counts"],
            }
            for cluster in patches
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    title = (
        f"{spec.name} sample {sample}: raw {len(raw_candidates)} · accepted {len(accepted)} "
        f"(TP={accepted_tp} FP={accepted_fp} FN={accepted_fn}) "
        f"· saddle {len(saddles)} · fit fail {len(failures)}\n"
        f"hist cuts {cuts['black_cut']}/{cuts['white_cut']} · mask diag {diagonal:.1f}px "
        f"· patch r {patch_radius:.1f}px"
    )
    _plot_candidates(
        out_dir / "refined_candidates.png",
        gray,
        cnn_mask,
        black_mask,
        white_mask,
        accepted,
        saddles,
        failures,
        truth,
        title,
    )

    # All patch cluster scatters in one figure.
    if patches:
        ncols = min(6, len(patches))
        nrows = int(np.ceil(len(patches) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, cluster in zip(axes.ravel(), patches):
            axis.axis("on")
            projection = cluster["projection"]
            keep = cluster["keep"]
            axis.scatter(
                projection[keep, 0],
                projection[keep, 1],
                s=14,
                c="#2ca02c",
                label=f"majority ({int(keep.sum())})",
                alpha=0.8,
            )
            axis.scatter(
                projection[~keep, 0],
                projection[~keep, 1],
                s=14,
                c="#ff7f0e",
                label=f"other ({int((~keep).sum())})",
                alpha=0.8,
            )
            polarity_name = "white" if cluster["polarity"] == 1 else "black"
            axis.set_title(
                f"patch {cluster['index']} · {polarity_name} · "
                f"center=({cluster['center'][0]:.0f},{cluster['center'][1]:.0f})",
                fontsize=7,
            )
            axis.set_xlabel("PC 1", fontsize=6)
            axis.set_ylabel("PC 2", fontsize=6)
            axis.tick_params(labelsize=5)
            axis.grid(alpha=0.2)
        axes.ravel()[0].legend(fontsize=5)
        fig.suptitle("All local-patch cluster scatters from one image", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / "cluster_patches.png", dpi=150)
        plt.close(fig)
    else:
        (out_dir / "cluster_patches.png").write_bytes(b"")
        print("warning: no patch had >=6 candidates of one polarity; no cluster plots")

    # Per-patch feature histograms.
    patches_with_points = []
    for index, center in enumerate(centers):
        selected = [
            point
            for point in accepted
            if (point["x"] - center[0]) ** 2 + (point["y"] - center[1]) ** 2 <= patch_radius**2
        ]
        if selected:
            patches_with_points.append((index, center, selected))
    if patches_with_points:
        nrows = len(patches_with_points)
        fig, axes = plt.subplots(
            nrows,
            3,
            figsize=(11.0, max(2.0, 1.5 * nrows)),
            squeeze=False,
        )
        for row, (index, center, selected) in enumerate(patches_with_points):
            for col, feature in enumerate(FEATURE_KEYS):
                axis = axes[row, col]
                white = np.asarray(
                    [point[feature] for point in selected if point["polarity"] == 1],
                    dtype=np.float64,
                )
                black = np.asarray(
                    [point[feature] for point in selected if point["polarity"] == 0],
                    dtype=np.float64,
                )
                values = np.concatenate([white, black]) if white.size or black.size else np.asarray([])
                if values.size:
                    low = float(np.min(values))
                    high = float(np.max(values))
                    if high - low < 1e-9:
                        low -= 0.5
                        high += 0.5
                    bins = np.linspace(low, high, 16)
                    if white.size:
                        axis.hist(white, bins=bins, color="#ff7f0e", alpha=0.65, density=True)
                    if black.size:
                        axis.hist(black, bins=bins, color="#1f77b4", alpha=0.55, density=True)
                axis.set_title(
                    f"patch {index} · {FEATURE_LABELS[col]}" if col == 0 else FEATURE_LABELS[col],
                    fontsize=6,
                )
                axis.tick_params(labelsize=5)
                axis.grid(alpha=0.2)
        fig.legend(
            handles=[
                Line2D([], [], color="#ff7f0e", linewidth=5, label="white candidates"),
                Line2D([], [], color="#1f77b4", linewidth=5, label="black candidates"),
            ],
            loc="upper right",
            fontsize=8,
        )
        fig.suptitle(
            "Per-patch feature histograms (all accepted candidates in each patch)",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 0.93, 0.96))
        fig.savefig(out_dir / "feature_histograms_per_patch.png", dpi=150)
        plt.close(fig)
    else:
        print("warning: no accepted candidates for feature histograms")

    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="oblique_1280x960")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--hessian-epsilon", type=float, default=1e-7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
