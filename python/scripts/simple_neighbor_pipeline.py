#!/usr/bin/env python3
"""Simplified pipeline with a per-point neighbour-consistency filter.

Candidate stage:
  raw high-pass blob class -> 3x3 extrema of the blurred image -> 5x5 cubic
  -> definite-Hessian filter. No border crop; edge windows are clamped.

Instead of patches/clustering, each accepted point is compared with its nearest
same-polarity neighbours in image space. Features are peak, lambda_1/c,
lambda_2/c, and hxy/c, where c = abs(peak - 0.5). The peak must pass the
accept_sigma test and at least two of the three Hessian features must pass.
This matches the browser's 40-neighbour filter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

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
from simple_patch_pipeline import (  # noqa: E402
    _candidates,
    _crop_bounds,
    _fit_cubic_5x5,
    _plot_candidates,
)

ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = ROOT / "public" / "models" / "roi_tiny_no_concat.onnx"
DEFAULT_OUT = ROOT / "python" / "notebooks" / "blob_lab_data" / "simple_neighbor_pipeline"

FEATURE_KEYS = ("peak", "eigenvalue_1", "eigenvalue_2", "hessian_xy")
FEATURE_LABELS = ("peak", "lambda_1/c", "lambda_2/c", "hxy/c")
PEAK_FEATURE = 0
MIN_PASSING_HESSIAN_FEATURES = 2
CONTRAST_FLOOR = 1e-6


def _feature_values(point: dict[str, Any]) -> list[float]:
    scale = max(abs(float(point["peak"]) - 0.5), CONTRAST_FLOOR)
    return [
        float(point["peak"]),
        float(point["eigenvalue_1"]) / scale,
        float(point["eigenvalue_2"]) / scale,
        float(point["hessian_xy"]) / scale,
    ]


def _features(accepted: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_feature_values(point) for point in accepted], dtype=np.float64)


def _neighbor_filter(
    accepted: list[dict[str, Any]],
    *,
    neighbours: int,
    sigma: float,
    trim_sigma: float,
) -> dict[str, Any]:
    """Per-polarity spatial kNN mean/std test. Returns keep mask + z-scores."""
    features = _features(accepted)
    keep = np.ones(len(accepted), dtype=bool)
    z_scores = np.full_like(features, np.nan)
    neighbour_counts = np.zeros(len(accepted), dtype=np.int32)
    neighbour_values: list[np.ndarray | None] = [None] * len(accepted)
    trim_masks: list[np.ndarray | None] = [None] * len(accepted)
    neighbour_mean = np.full_like(features, np.nan)
    neighbour_std = np.full_like(features, np.nan)
    neighbour_mean_initial = np.full_like(features, np.nan)
    neighbour_std_initial = np.full_like(features, np.nan)
    trimmed_counts = np.zeros_like(features, dtype=np.int32)
    rejection_by_feature: Counter[str] = Counter()
    per_polarity: dict[str, Any] = {}
    for polarity, name in ((0, "black"), (1, "white")):
        indices = [index for index, point in enumerate(accepted) if point["polarity"] == polarity]
        if len(indices) < 4:
            per_polarity[name] = {"points": len(indices), "kept": len(indices), "note": "too few"}
            continue
        coords = np.asarray(
            [[accepted[index]["x"], accepted[index]["y"]] for index in indices],
            dtype=np.float64,
        )
        local_features = features[indices]
        tree = cKDTree(coords)
        k = min(int(neighbours) + 1, len(indices))
        _distances, neighbours_idx = tree.query(coords, k=k)
        if neighbours_idx.ndim == 1:
            neighbours_idx = neighbours_idx[:, None]
        polarity_keep = np.ones(len(indices), dtype=bool)
        for local_index, global_index in enumerate(indices):
            neighbour_rows = neighbours_idx[local_index, 1:]
            if neighbour_rows.size < 3:
                continue
            neighbour_features = local_features[neighbour_rows]
            mean_initial = neighbour_features.mean(axis=0)
            std_initial = neighbour_features.std(axis=0, ddof=1)
            floor = 1e-8 * np.maximum(1.0, np.abs(mean_initial))
            mean_final = mean_initial.copy()
            std_final = std_initial.copy()
            point_trim_masks = np.ones_like(neighbour_features, dtype=bool)
            for feature_index in range(neighbour_features.shape[1]):
                column = neighbour_features[:, feature_index]
                safe_initial = max(float(std_initial[feature_index]), float(floor[feature_index]))
                inliers = np.abs(column - mean_initial[feature_index]) <= float(trim_sigma) * safe_initial
                if int(inliers.sum()) >= 3:
                    point_trim_masks[:, feature_index] = inliers
                    mean_final[feature_index] = float(column[inliers].mean())
                    std_final[feature_index] = float(column[inliers].std(ddof=1))
                else:
                    point_trim_masks[:, feature_index] = True
                trimmed_counts[global_index, feature_index] = int((~point_trim_masks[:, feature_index]).sum())
            safe_final = np.where(std_final > floor, std_final, floor)
            z = (local_features[local_index] - mean_final) / safe_final
            z_scores[global_index] = z
            neighbour_counts[global_index] = int(neighbour_rows.size)
            neighbour_values[global_index] = neighbour_features.copy()
            trim_masks[global_index] = point_trim_masks
            neighbour_mean[global_index] = mean_final
            neighbour_std[global_index] = std_final
            neighbour_mean_initial[global_index] = mean_initial
            neighbour_std_initial[global_index] = std_initial
            inside = np.abs(z) <= float(sigma)
            peak_pass = bool(inside[PEAK_FEATURE])
            hessian_passing = int(inside[PEAK_FEATURE + 1 :].sum())
            if not (peak_pass and hessian_passing >= MIN_PASSING_HESSIAN_FEATURES):
                polarity_keep[local_index] = False
                for feature_index, key in enumerate(FEATURE_LABELS):
                    if not inside[feature_index]:
                        rejection_by_feature[key] += 1
        keep[indices] = polarity_keep
        per_polarity[name] = {
            "points": int(len(indices)),
            "kept": int(polarity_keep.sum()),
            "removed": int((~polarity_keep).sum()),
        }
    return {
        "keep": keep,
        "z_scores": z_scores,
        "neighbour_values": neighbour_values,
        "trim_masks": trim_masks,
        "neighbour_mean": neighbour_mean,
        "neighbour_std": neighbour_std,
        "neighbour_mean_initial": neighbour_mean_initial,
        "neighbour_std_initial": neighbour_std_initial,
        "trimmed_counts": trimmed_counts,
        "neighbour_counts": neighbour_counts,
        "trim_sigma": float(trim_sigma),
        "rejection_by_feature": dict(rejection_by_feature),
        "per_polarity": per_polarity,
    }



def _load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.0


def _onnx_roi_and_background(image_path: Path, *, residual_gain: float) -> dict[str, Any]:
    """Production-style CNN ROI + mask-aware long-wavelength high-pass.

    Mirrors roiBackgroundPipeline: model-resolution mask-weighted luma is box
    blurred with radius 12, divided by the blurred mask weight, upsampled, and
    subtracted from the full-resolution luma:
        flattened = clamp(0.5 + (luma - background) * residual_gain, 0, 1)
    """
    import onnxruntime as ort

    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(image_path)
    height, width = color.shape[:2]
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    stats_small = cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA)
    stats_luma = stats_small @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    mean = float(stats_luma.mean())
    std = max(0.05, float(stats_luma.std()))

    model_width = 256
    raw_height = round(model_width * height / width)
    model_height = min(224, max(96, int(np.ceil(raw_height / 16.0)) * 16))
    resized = cv2.resize(rgb, (model_width, model_height), interpolation=cv2.INTER_LINEAR)
    luma_model = resized @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    normalized = ((luma_model - mean) / std)[None, None].astype(np.float32)

    session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    output = session.run(["mask"], {"input": normalized})[0]
    probability = np.asarray(output[0, 0], dtype=np.float32)
    probability_model = cv2.resize(
        probability, (model_width, model_height), interpolation=cv2.INTER_LINEAR
    )
    probability_model = np.clip(probability_model, 0.0, 1.0)

    weighted = luma_model * probability_model
    weight = probability_model
    box = 25  # radius 12, as in roiBackgroundPipeline
    weighted_blur = cv2.blur(
        weighted, (box, box), borderType=cv2.BORDER_REPLICATE
    )
    weight_blur = cv2.blur(
        weight, (box, box), borderType=cv2.BORDER_REPLICATE
    )
    background_model = np.where(
        weight_blur > 1e-3,
        weighted_blur / np.maximum(weight_blur, 1e-3),
        mean,
    ).astype(np.float32)
    background = cv2.resize(
        background_model, (width, height), interpolation=cv2.INTER_LINEAR
    )
    flattened = np.clip(
        0.5 + (gray - background) * float(residual_gain), 0.0, 1.0
    ).astype(np.float32)
    roi_mask = cv2.resize(
        probability, (width, height), interpolation=cv2.INTER_LINEAR
    ) >= 0.5
    return {
        "gray": gray,
        "flattened": flattened,
        "background": background,
        "roi_mask": roi_mask,
        "probability": probability,
        "model_width": model_width,
        "model_height": model_height,
        "stats_mean": mean,
        "stats_std": std,
        "residual_gain": float(residual_gain),
    }


def _load_manual_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask >= 127


def _plot_preprocessing(
    path: Path,
    gray: np.ndarray,
    background: np.ndarray,
    flattened: np.ndarray,
    cnn_mask: np.ndarray,
    manual_mask: np.ndarray | None,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    for axis, image, name in (
        (axes[0], gray, "input gray"),
        (axes[1], background, "estimated background (25x25 @ 256w)"),
        (axes[2], flattened, "flattened: 0.5 + (gray - background)"),
    ):
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        axis.contour(cnn_mask.astype(np.float32), levels=[0.5], colors=["#00d5ff"], linewidths=1.2)
        if manual_mask is not None:
            axis.contour(
                manual_mask.astype(np.float32),
                levels=[0.5],
                colors=["#ff2dff"],
                linewidths=0.8,
                alpha=0.7,
            )
        axis.set_title(name, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _match_counts(points: list[dict[str, Any]], truth: np.ndarray) -> tuple[int, int, int]:
    if not points or truth.size == 0:
        return 0, len(points), int(truth.shape[0])
    coords = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float64)
    distances = point_distances(coords, truth)
    matches = match_from_distances(distances, 1.0)
    tp = len(matches)
    return tp, len(points) - tp, int(truth.shape[0]) - tp


def _plot_neighbor_filter(
    path: Path,
    image: np.ndarray,
    cnn_mask: np.ndarray,
    truth: np.ndarray,
    kept: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    title: str,
    sigma: float,
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
    visible_truth = truth[
        (truth[:, 0] >= x0)
        & (truth[:, 0] < x1)
        & (truth[:, 1] >= y0)
        & (truth[:, 1] < y1)
    ] if truth.size else truth
    if visible_truth.size:
        axis.scatter(
            visible_truth[:, 0] - 0.5 - x0,
            visible_truth[:, 1] - 0.5 - y0,
            s=26,
            facecolors="none",
            edgecolors="#39ff14",
            linewidths=0.8,
            label="ground truth",
            zorder=4,
        )
    if kept:
        pts = np.asarray([[p["x"] - 0.5 - x0, p["y"] - 0.5 - y0] for p in kept])
        axis.scatter(pts[:, 0], pts[:, 1], s=13, c="#2ca02c", edgecolors="white",
                     linewidths=0.25, label=f"kept ({len(kept)})", zorder=5)
    if removed:
        pts = np.asarray([[p["x"] - 0.5 - x0, p["y"] - 0.5 - y0] for p in removed])
        axis.scatter(pts[:, 0], pts[:, 1], s=38, marker="x", c="#ff7f0e",
                     linewidths=1.1, label=f"removed by {sigma:g}σ ({len(removed)})", zorder=6)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_features(
    path: Path,
    accepted: list[dict[str, Any]],
    keep: np.ndarray,
    z_scores: np.ndarray,
    sigma: float,
) -> None:
    features = _features(accepted)
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.0))
    for column, (key, label) in enumerate(zip(FEATURE_KEYS, FEATURE_LABELS)):
        values = features[:, column]
        kept_values = values[keep]
        removed_values = values[~keep]
        all_values = np.concatenate([kept_values, removed_values])
        if all_values.size == 0:
            continue
        low = float(np.min(all_values))
        high = float(np.max(all_values))
        if high - low < 1e-9:
            low -= 0.5
            high += 0.5
        bins = np.linspace(low, high, 30)
        axis = axes[0, column]
        if kept_values.size:
            axis.hist(kept_values, bins=bins, color="#2ca02c", alpha=0.6, density=True, label="kept")
        if removed_values.size:
            axis.hist(removed_values, bins=bins, color="#ff7f0e", alpha=0.7, density=True, label="removed")
        axis.set_title(f"{label} · raw values", fontsize=8)
        axis.grid(alpha=0.2)
        if column == 0:
            axis.legend(fontsize=6)

        axis = axes[1, column]
        z = z_scores[:, column]
        finite = np.isfinite(z)
        kept_z = z[finite & keep]
        removed_z = z[finite & ~keep]
        z_values = np.concatenate([kept_z, removed_z])
        if z_values.size == 0:
            continue
        z_low = float(min(-sigma - 1.0, np.min(z_values)))
        z_high = float(max(sigma + 1.0, np.max(z_values)))
        zbins = np.linspace(z_low, z_high, 40)
        if kept_z.size:
            axis.hist(kept_z, bins=zbins, color="#2ca02c", alpha=0.6, density=True)
        if removed_z.size:
            axis.hist(removed_z, bins=zbins, color="#ff7f0e", alpha=0.7, density=True)
        axis.axvline(-sigma, color="#d62728", linestyle="--", linewidth=1.0)
        axis.axvline(sigma, color="#d62728", linestyle="--", linewidth=1.0)
        axis.set_title(f"{label} · neighbour z-score", fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(f"Per-point neighbour test (same-polarity neighbours, ±{sigma:g}σ)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)



def _plot_point_feature(axis, index, feature_index, accepted, result, sigma):
    point = accepted[index]
    values = result["neighbour_values"][index]
    z = result["z_scores"][index]
    feature = FEATURE_LABELS[feature_index]
    if values is None:
        axis.axis("off")
        axis.text(0.5, 0.5, "fewer than 3 neighbours", ha="center", va="center", fontsize=7)
        return
    sample = values[:, feature_index]
    mean = float(result["neighbour_mean"][index, feature_index])
    std = float(result["neighbour_std"][index, feature_index])
    value = float(_feature_values(point)[feature_index])
    trim_mask = result["trim_masks"][index]
    trimmed = sample[~trim_mask[:, feature_index]] if trim_mask is not None else np.asarray([])
    low = float(np.min(np.concatenate([sample, [value]])))
    high = float(np.max(np.concatenate([sample, [value]])))
    if high - low < 1e-9:
        low -= 0.5
        high += 0.5
    bins = np.linspace(low, high, 10)
    axis.hist(sample, bins=bins, color="#9aa0a6", edgecolor="#5f6368", alpha=0.8)
    if trimmed.size:
        axis.plot(trimmed, np.full(trimmed.size, 0.06), "|", color="#d62728",
                  markersize=13, markeredgewidth=1.4, label=f"trimmed >{result['trim_sigma']:g}σ")
    axis.axvspan(mean - sigma * std, mean + sigma * std, color="#2ca02c", alpha=0.14,
                 label=f"mean ± {sigma:g}σ")
    axis.axvline(mean, color="#1f77b4", linewidth=1.2, label="neighbour mean")
    inside = abs(float(z[feature_index])) <= sigma
    axis.axvline(value, color="#2ca02c" if inside else "#d62728", linewidth=1.6,
                 label="point value")
    axis.set_title(
        f"{feature} · value={value:.4g} · mean={mean:.4g} · σ={std:.4g} · "
        f"z={float(z[feature_index]):+.2f} · trimmed={trimmed.size}",
        fontsize=7,
    )
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.2)
    if not inside:
        axis.patch.set_facecolor("#fff2f0")


def _plot_per_point(
    out_dir: Path,
    accepted: list[dict[str, Any]],
    result: dict[str, Any],
    sigma: float,
    *,
    kept_examples: int,
) -> list[str]:
    keep = result["keep"]
    removed = [index for index, flag in enumerate(keep) if not flag]
    kept = [index for index, flag in enumerate(keep) if flag]
    written: list[str] = []

    def make_figure(indices: list[int], name: str) -> None:
        if not indices:
            return
        fig, axes = plt.subplots(
            len(indices), len(FEATURE_LABELS), figsize=(16.0, 2.6 * len(indices)), squeeze=False
        )
        for row, index in enumerate(indices):
            for column in range(len(FEATURE_LABELS)):
                _plot_point_feature(axes[row, column], index, column, accepted, result, sigma)
            point = accepted[index]
            polarity = "white" if point["polarity"] == 1 else "black"
            failed = [
                FEATURE_LABELS[column]
                for column in range(len(FEATURE_LABELS))
                if abs(float(result["z_scores"][index, column])) > sigma
            ]
            axes[row, 0].set_ylabel(
                f"point {index}\n({point['x']:.0f},{point['y']:.0f}) {polarity}"
                + ("\nfail: " + ", ".join(failed) if failed else ""),
                fontsize=7,
            )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(handles, labels, fontsize=5, loc="upper right")
        fig.suptitle(
            f"Per-point neighbour test · {sigma:g}σ · {name}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out_dir / "per_point" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
        if len(indices) > 1:
            # A compact contact sheet is the same figure; keep the name explicit.
            contact = out_dir / f"per_point_{name}.png"
            contact.write_bytes(path.read_bytes())
            written.append(str(contact))

    make_figure(removed, "removed")
    if kept_examples > 0 and kept:
        step = max(1, len(kept) // kept_examples)
        make_figure(kept[::step][:kept_examples], "kept_examples")
    for index in removed:
        make_figure([index], f"point_{index:04d}")
    return written

def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manual_mask = None
    preprocessing = None
    if args.image is not None:
        image_path = Path(args.image)
        prep = _onnx_roi_and_background(image_path, residual_gain=args.residual_gain)
        gray = prep["gray"]
        processed = prep["flattened"]
        cnn_mask = prep["roi_mask"]
        truth = np.empty((0, 2), dtype=np.float64)
        spec_name = image_path.name
        sample = 0
        manual_mask = _load_manual_mask(args.manual_mask, gray.shape)
        preprocessing = prep
    else:
        specs = {
            spec.name: spec
            for spec in build_case_specs(
                seed=args.seed,
                samples_scale=1.0,
                include_large=True,
                max_cases=None,
            )
        }
        spec = specs[args.case]
        config = make_config(spec, sharpen_probability=0.0)
        layout = build_hex_target_layout(width_px=spec.width, height_px=spec.height)
        dataset = generate_heatmap_dataset(config, layout, count=spec.samples)
        sample = min(max(0, args.sample), len(dataset) - 1)
        gray = np.asarray(dataset.images[sample, :, :, 0], dtype=np.float32)
        cnn_mask = np.asarray(dataset.masks[sample, :, :, 0], dtype=np.float32) > 0.5
        truth = np.asarray(dataset.metadata[sample].get("points") or [], dtype=np.float64).reshape(-1, 2)
        spec_name = spec.name
        processed = gray
    blurred = cv2.GaussianBlur(
        processed,
        (0, 0),
        sigmaX=args.blur_sigma,
        sigmaY=args.blur_sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    cuts = histogram_cuts(blurred, cnn_mask)
    # Candidate membership is the raw high-pass class, not a second test of the
    # blurred value. The histogram cuts themselves still come from the blurred
    # image. This matches the browser candidate stage.
    black_mask, white_mask = segment(processed, cnn_mask, cuts)
    if not black_mask.any() or not white_mask.any():
        raise SystemExit(
            f"histogram masks are empty: cuts={cuts['black_cut']}/{cuts['white_cut']}"
        )

    raw_candidates: list[dict[str, Any]] = []
    for polarity, support in ((0, black_mask), (1, white_mask)):
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
        definite = (
            eigen_1 < -args.hessian_epsilon and eigen_2 < -args.hessian_epsilon
            if candidate["polarity"] == 1
            else eigen_1 > args.hessian_epsilon and eigen_2 > args.hessian_epsilon
        )
        refined = {**candidate, **fit}
        (accepted if definite else saddles).append(refined)

    margin = 8.0
    visible_truth = truth[
        (truth[:, 0] >= margin)
        & (truth[:, 0] <= gray.shape[1] - 1 - margin)
        & (truth[:, 1] >= margin)
        & (truth[:, 1] <= gray.shape[0] - 1 - margin)
    ] if truth.size else truth

    has_point_truth = bool(visible_truth.size > 0)
    accepted_tp, accepted_fp, accepted_fn = _match_counts(accepted, visible_truth)
    filter_result = _neighbor_filter(
        accepted,
        neighbours=args.neighbours,
        sigma=args.sigma,
        trim_sigma=args.trim_sigma,
    )
    keep = filter_result["keep"]
    kept = [point for point, flag in zip(accepted, keep) if flag]
    removed = [point for point, flag in zip(accepted, keep) if not flag]
    kept_tp, kept_fp, kept_fn = _match_counts(kept, visible_truth)

    sigma_scan = {}
    for candidate_sigma in (3.0, 3.5, 4.0, 4.5, 5.0):
        scan = _neighbor_filter(
            accepted,
            neighbours=args.neighbours,
            sigma=candidate_sigma,
            trim_sigma=args.trim_sigma,
        )
        sigma_scan[f"{candidate_sigma:g}"] = int((~scan["keep"]).sum())

    summary = {
        "case": spec_name,
        "sample": int(sample),
        "blur_sigma": float(args.blur_sigma),
        "image": str(args.image) if args.image is not None else None,
        "onnx_model": str(ONNX_MODEL) if args.image is not None else None,
        "background": {
            "method": "mask-weighted 25x25 box blur at 256px-wide model resolution, upsample, subtract",
            "residual_gain": float(args.residual_gain),
        }
        if args.image is not None
        else None,
        "manual_mask": str(args.manual_mask) if args.manual_mask is not None else None,
        "histogram": {
            "background": int(cuts["background"]),
            "black_cut": int(cuts["black_cut"]),
            "white_cut": int(cuts["white_cut"]),
            "source": "blurred high-pass",
            "candidate_class": "raw high-pass",
            "border_margin": 0,
        },
        "neighbour_filter": {
            "neighbours": int(args.neighbours),
            "trim_sigma": float(args.trim_sigma),
            "accept_sigma": float(args.sigma),
            "sigma": float(args.sigma),
            "sigma_scan_removed": sigma_scan,
            "same_polarity": True,
            "features": list(FEATURE_LABELS),
            "peak_gate": True,
            "min_passing_hessian_features": MIN_PASSING_HESSIAN_FEATURES,
            "rejection_by_feature": filter_result["rejection_by_feature"],
            "per_polarity": filter_result["per_polarity"],
        },
        "counts": {
            "has_point_truth": has_point_truth,
            "visible_truth": int(visible_truth.shape[0]),
            "raw_candidates": len(raw_candidates),
            "accepted": len(accepted),
            "accepted_tp_fp_fn": [int(accepted_tp), int(accepted_fp), int(accepted_fn)]
            if has_point_truth
            else None,
            "neighbor_kept": len(kept),
            "neighbor_removed": len(removed),
            "neighbor_kept_tp_fp_fn": [int(kept_tp), int(kept_fp), int(kept_fn)]
            if has_point_truth
            else None,
            "saddle_or_indefinite": len(saddles),
            "fit_failure": len(failures),
        },
        "removed_points": [
            {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "polarity": int(point["polarity"]),
                "z": [float(value) for value in filter_result["z_scores"][index]],
                "value": _feature_values(point),
                "neighbour_mean": [
                    float(value) for value in filter_result["neighbour_mean"][index]
                ],
                "neighbour_std": [
                    float(value) for value in filter_result["neighbour_std"][index]
                ],
                "trimmed_counts": [
                    int(value) for value in filter_result["trimmed_counts"][index]
                ],
            }
            for index, point in enumerate(accepted)
            if not keep[index]
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if visible_truth.size > 0:
        title = (
            f"{spec_name} sample {sample}: accepted {len(accepted)} "
            f"(TP={accepted_tp} FP={accepted_fp} FN={accepted_fn}) · "
            f"removed by {args.sigma:g}σ (trim {args.trim_sigma:g}σ) {len(removed)}\n"
            f"kept {len(kept)} (TP={kept_tp} FP={kept_fp} FN={kept_fn}) · "
            f"hist cuts {cuts['black_cut']}/{cuts['white_cut']}"
        )
    else:
        title = (
            f"{spec_name}: accepted {len(accepted)} · removed by {args.sigma:g}σ "
            f"(trim {args.trim_sigma:g}σ) {len(removed)} · kept {len(kept)}\n"
            f"hist cuts {cuts['black_cut']}/{cuts['white_cut']}"
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
        (
            f"{spec_name}: raw {len(raw_candidates)} · "
            f"accepted {len(accepted)} · saddle {len(saddles)} · fit fail {len(failures)}\n"
            f"hist cuts {cuts['black_cut']}/{cuts['white_cut']}"
        ),
    )
    _plot_neighbor_filter(
        out_dir / "neighbor_filter.png",
        gray,
        cnn_mask,
        truth,
        kept,
        removed,
        title,
        args.sigma,
    )
    _plot_features(
        out_dir / "neighbor_feature_histograms.png",
        accepted,
        keep,
        filter_result["z_scores"],
        args.sigma,
    )
    per_point_paths = _plot_per_point(
        out_dir,
        accepted,
        filter_result,
        args.sigma,
        kept_examples=args.per_point_kept,
    )
    summary["per_point_plots"] = per_point_paths
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if preprocessing is not None:
        _plot_preprocessing(
            out_dir / "preprocessing.png",
            gray,
            preprocessing["background"],
            preprocessing["flattened"],
            cnn_mask,
            manual_mask,
            f"{spec_name}: CNN ROI + mask-weighted long-wavelength high-pass "
            f"(model {preprocessing['model_width']}x{preprocessing['model_height']})",
        )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="oblique_1280x960")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Real image path. If omitted, the synthetic case is used.",
    )
    parser.add_argument(
        "--manual-mask",
        type=Path,
        default=None,
        help="Optional manual target mask, drawn as a reference contour.",
    )
    parser.add_argument("--residual-gain", type=float, default=1.0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--hessian-epsilon", type=float, default=1e-7)
    parser.add_argument("--neighbours", type=int, default=40)
    parser.add_argument("--trim-sigma", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--per-point-kept", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
