#!/usr/bin/env python3
"""Oblique, sharpening-aware comparison of local cubic peak fits.

The current production chain is histogram mask -> CCL -> centroid -> NMS.  This
experiment replaces the localization stage with local extrema of the smoothed
luma and a locally fitted bivariate cubic.  It deliberately does not compare
against the quadratic/CCL baselines: the goal is to choose a cubic support rule
under the conditions that matter (oblique views and camera sharpening), and to
produce the descriptor table used to design the shape-cluster filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import numpy as np

from blob_lab.augment import CameraSharpenConfig
from blob_lab.camera import (
    CameraRanges,
    PoseAugmentConfig,
    TILT_BINS,
    local_foreshortening,
    tilt_bin,
)
from blob_lab.heatmap_dataset import HeatmapDatasetConfig, generate_heatmap_dataset
from blob_lab.local_peak_fit import (
    BORDER_MARGIN,
    CurvatureData,
    LocalPeak,
    LocalPeakCandidate,
    curvature_maps,
    detect_local_peaks,
    estimate_spacing,
    fit_peaks,
    refresh_candidate,
    smooth_luma,
    support_radii,
)
from blob_lab.target_layout import build_hex_target_layout

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_bicubic"

VARIANTS: tuple[str, ...] = (
    "compact",
    "full",
    "adaptive_circle",
    "adaptive_circle_concave",
    "adaptive_ellipse_concave",
    "curvature_region",
)

VARIANT_COLORS: dict[str, tuple[float, float, float]] = {
    "compact": (0.10, 0.85, 0.20),
    "full": (1.00, 0.55, 0.10),
    "adaptive_circle": (0.85, 0.85, 0.20),
    "adaptive_circle_concave": (0.10, 0.80, 1.00),
    "adaptive_ellipse_concave": (1.00, 0.20, 0.90),
    "curvature_region": (0.95, 0.95, 0.95),
}

# name, yaw, pitch, samples at 640x480, samples at 1280x960
ANGLE_GROUPS: tuple[tuple[str, float, float, int, int], ...] = (
    ("frontal", 0.0, 0.0, 2, 1),
    ("moderate", 30.0, 20.0, 2, 1),
    ("oblique", 50.0, 30.0, 3, 1),
    ("steep", 60.0, 40.0, 3, 1),
)

TOLERANCES: tuple[float, ...] = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    group: str
    yaw_deg: float
    pitch_deg: float
    width: int
    height: int
    samples: int
    seed: int


def build_case_specs(
    *,
    seed: int,
    samples_scale: float,
    include_large: bool,
    max_cases: int | None,
) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for group, yaw, pitch, count_640, count_1280 in ANGLE_GROUPS:
        for width, height, count in (
            (640, 480, count_640),
            (1280, 960, count_1280),
        ):
            if width == 1280 and not include_large:
                continue
            samples = max(1, int(round(count * max(0.1, samples_scale)))) if count else 0
            if samples <= 0:
                continue
            specs.append(
                CaseSpec(
                    name=f"{group}_{width}x{height}",
                    group=group,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    width=width,
                    height=height,
                    samples=samples,
                    seed=seed + len(specs),
                )
            )
    if max_cases is not None:
        specs = specs[: max(0, int(max_cases))]
    return specs


def make_config(spec: CaseSpec, *, sharpen_probability: float | None) -> HeatmapDatasetConfig:
    camera = CameraRanges(
        yaw_deg=(spec.yaw_deg, spec.yaw_deg),
        pitch_deg=(spec.pitch_deg, spec.pitch_deg),
        roll_deg=(-10.0, 10.0),
        distance=(2.8, 3.8),
        focal=(2.0, 2.7),
        tx=(-0.08, 0.08),
        ty=(-0.08, 0.08),
    )
    sharpen = (
        CameraSharpenConfig(probability=float(sharpen_probability))
        if sharpen_probability is not None
        else CameraSharpenConfig()
    )
    return HeatmapDatasetConfig(
        output_width=spec.width,
        output_height=spec.height,
        samples=spec.samples,
        natural_fraction=0.0,
        randomize_colors=False,
        blur_sigma=(0.8, 1.6),
        noise_sigma=(0.0, 0.01),
        seed=spec.seed,
        camera=camera,
        pose_augment=PoseAugmentConfig(oblique_fraction=0.0),
        sharpen=sharpen,
        min_visible_marker_fraction=0.30,
    )


def point_distances(points: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if points.size == 0 or truth.size == 0:
        return np.zeros((points.shape[0], truth.shape[0]), dtype=np.float64)
    delta = points[:, None, :] - truth[None, :, :]
    return np.sqrt((delta * delta).sum(axis=-1))


def match_from_distances(
    distances: np.ndarray,
    tolerance: float,
) -> list[tuple[float, int, int]]:
    if distances.size == 0:
        return []
    indices = np.argwhere(distances <= tolerance)
    if indices.size == 0:
        return []
    order = np.argsort(distances[indices[:, 0], indices[:, 1]], kind="stable")
    indices = indices[order]
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for predicted, truth in indices.tolist():
        if predicted in used_predicted or truth in used_truth:
            continue
        used_predicted.add(predicted)
        used_truth.add(truth)
        matches.append((float(distances[predicted, truth]), int(predicted), int(truth)))
    return matches


class MetricAccumulator:
    def __init__(self) -> None:
        self.truth_count = 0
        self.predicted_count = 0
        self.fit_ok = 0
        self.fallback = 0
        self.matched = {tolerance: 0 for tolerance in TOLERANCES}
        self.errors = {tolerance: [] for tolerance in TOLERANCES}
        self.fit_errors = {tolerance: [] for tolerance in TOLERANCES}
        self.signed_x = {tolerance: [] for tolerance in TOLERANCES}
        self.signed_y = {tolerance: [] for tolerance in TOLERANCES}
        self.bin_truth: Counter[str] = Counter()
        self.bin_matched: Counter[str] = Counter()
        self.bin_errors: dict[str, list[float]] = defaultdict(list)
        self.polarity_truth: Counter[str] = Counter()
        self.polarity_matched: Counter[str] = Counter()
        self.polarity_mismatch = 0
        self.support_radii: list[float] = []
        self.support_kinds: Counter[str] = Counter()
        self.mask_used = 0

    def add(
        self,
        peaks: list[LocalPeak],
        truth: np.ndarray,
        truth_colors: np.ndarray,
        truth_bins: list[str],
    ) -> None:
        self.truth_count += int(truth.shape[0])
        self.predicted_count += len(peaks)
        self.fit_ok += sum(1 for peak in peaks if peak.fit_ok)
        self.fallback += sum(1 for peak in peaks if peak.fallback)
        for peak in peaks:
            self.support_radii.append(float(peak.support_radius))
            self.support_kinds[peak.support_kind] += 1
            if peak.used_curvature_mask:
                self.mask_used += 1
        self.bin_truth.update(truth_bins)
        for color in truth_colors.tolist():
            self.polarity_truth["white" if int(color) == 1 else "black"] += 1
        if not peaks or truth.size == 0:
            return
        points = np.asarray([[peak.x, peak.y] for peak in peaks], dtype=np.float64)
        distances = point_distances(points, truth)
        for tolerance in TOLERANCES:
            matches = match_from_distances(distances, tolerance)
            self.matched[tolerance] += len(matches)
            for distance, predicted, truth_index in matches:
                self.errors[tolerance].append(float(distance))
                self.signed_x[tolerance].append(float(points[predicted, 0] - truth[truth_index, 0]))
                self.signed_y[tolerance].append(float(points[predicted, 1] - truth[truth_index, 1]))
                if peaks[predicted].fit_ok:
                    self.fit_errors[tolerance].append(float(distance))
                if tolerance == 1.0:
                    bin_name = truth_bins[truth_index]
                    self.bin_matched[bin_name] += 1
                    self.bin_errors[bin_name].append(float(distance))
                    color_name = "white" if int(truth_colors[truth_index]) == 1 else "black"
                    self.polarity_matched[color_name] += 1
                    if int(peaks[predicted].polarity) != int(truth_colors[truth_index]):
                        self.polarity_mismatch += 1

    def _error_summary(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {
                "mean_error_px": 0.0,
                "median_error_px": 0.0,
                "p90_error_px": 0.0,
                "rmse_px": 0.0,
                "bias_x_px": 0.0,
                "bias_y_px": 0.0,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_error_px": float(array.mean()),
            "median_error_px": float(np.median(array)),
            "p90_error_px": float(np.percentile(array, 90)),
            "rmse_px": float(np.sqrt(np.mean(array * array))),
            "bias_x_px": 0.0,
            "bias_y_px": 0.0,
        }

    def summary(self) -> dict[str, Any]:
        overall: dict[str, Any] = {}
        for tolerance in TOLERANCES:
            matched = self.matched[tolerance]
            precision = matched / self.predicted_count if self.predicted_count else 0.0
            recall = matched / self.truth_count if self.truth_count else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            error = self._error_summary(self.errors[tolerance])
            if self.signed_x[tolerance]:
                error["bias_x_px"] = float(np.mean(self.signed_x[tolerance]))
                error["bias_y_px"] = float(np.mean(self.signed_y[tolerance]))
            error.update(
                {
                    "tolerance_px": tolerance,
                    "matched": matched,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "mean_fit_error_px": float(np.mean(self.fit_errors[tolerance]))
                    if self.fit_errors[tolerance]
                    else 0.0,
                }
            )
            overall[f"{tolerance:.1f}"] = error
        per_bin: dict[str, Any] = {}
        for name, _lower, _upper in TILT_BINS:
            truth = int(self.bin_truth.get(name, 0))
            matched = int(self.bin_matched.get(name, 0))
            values = self.bin_errors.get(name, [])
            entry = self._error_summary(values)
            entry.update(
                {
                    "truth": truth,
                    "matched": matched,
                    "recall": float(matched / truth) if truth else 0.0,
                }
            )
            per_bin[name] = entry
        by_polarity: dict[str, Any] = {}
        for name in ("black", "white"):
            truth = int(self.polarity_truth.get(name, 0))
            matched = int(self.polarity_matched.get(name, 0))
            by_polarity[name] = {
                "truth": truth,
                "matched": matched,
                "recall": float(matched / truth) if truth else 0.0,
            }
        support = {
            "mean_radius": float(np.mean(self.support_radii)) if self.support_radii else 0.0,
            "median_radius": float(np.median(self.support_radii)) if self.support_radii else 0.0,
            "kinds": dict(self.support_kinds),
            "curvature_mask_used": int(self.mask_used),
        }
        return {
            "truth": int(self.truth_count),
            "predicted": int(self.predicted_count),
            "fit_ok": int(self.fit_ok),
            "fallback": int(self.fallback),
            "polarity_mismatch": int(self.polarity_mismatch),
            "overall": overall,
            "per_bin": per_bin,
            "by_polarity": by_polarity,
            "support": support,
        }


def _truth_bins(meta: dict[str, Any], truth: np.ndarray) -> list[str]:
    targets = meta.get("point_targets") or []
    homography = meta.get("H")
    if homography is not None and len(targets) == truth.shape[0]:
        result: list[str] = []
        for target in targets:
            _ratio, tilt = local_foreshortening(homography, target)
            result.append(tilt_bin(tilt))
        return result
    case_tilt = meta.get("tilt_deg")
    return [tilt_bin(case_tilt) if case_tilt is not None else "frontal"] * truth.shape[0]


def _truth_colors(meta: dict[str, Any], gray: np.ndarray, truth: np.ndarray) -> np.ndarray:
    colors = meta.get("point_colors")
    if colors is not None and len(colors) == truth.shape[0]:
        return np.asarray(colors, dtype=np.int64)
    # Fallback for older metadata: sample the rendered image at the truth point.
    result = []
    height, width = gray.shape
    for x, y in truth:
        ix = int(np.clip(round(float(x)), 0, width - 1))
        iy = int(np.clip(round(float(y)), 0, height - 1))
        result.append(1 if float(gray[iy, ix]) > 0.5 else 0)
    return np.asarray(result, dtype=np.int64)


def _descriptor_rows(
    *,
    case: CaseSpec,
    sample_index: int,
    variant: str,
    peaks: list[LocalPeak],
    candidates: list[LocalPeakCandidate],
    truth: np.ndarray,
    truth_colors: np.ndarray,
    truth_bins: list[str],
    meta: dict[str, Any],
    spacing: float,
) -> tuple[list[dict[str, Any]], list[tuple[float, int, int]]]:
    if not peaks:
        return [], []
    points = np.asarray([[peak.x, peak.y] for peak in peaks], dtype=np.float64)
    distances = point_distances(points, truth)
    matches = match_from_distances(distances, 1.0)
    matched_truth = {predicted: truth_index for _distance, predicted, truth_index in matches}
    nearest_distance = distances.min(axis=1) if truth.size else np.full(len(peaks), np.nan)
    nearest_index = distances.argmin(axis=1) if truth.size else np.full(len(peaks), -1)
    sharpen = meta.get("sharpen") or {}
    bin_counts = Counter(truth_bins)
    polarity_counts = Counter(
        "white" if int(color) == 1 else "black" for color in truth_colors.tolist()
    )
    rows: list[dict[str, Any]] = []
    for index, peak in enumerate(peaks):
        truth_index = int(nearest_index[index]) if truth.size else -1
        bin_name = truth_bins[truth_index] if 0 <= truth_index < len(truth_bins) else "unmatched"
        row: dict[str, Any] = {
            "case": case.name,
            "group": case.group,
            "sample": sample_index,
            "variant": variant,
            "truth_count": int(truth.shape[0]),
            "candidate_x": float(peak.candidate_x),
            "candidate_y": float(peak.candidate_y),
            "fit_x": float(peak.x),
            "fit_y": float(peak.y),
            "pred_polarity": int(peak.polarity),
            "truth_polarity": int(truth_colors[truth_index])
            if 0 <= truth_index < len(truth_colors)
            else -1,
            "nearest_truth_x": float(truth[truth_index, 0]) if 0 <= truth_index else float("nan"),
            "nearest_truth_y": float(truth[truth_index, 1]) if 0 <= truth_index else float("nan"),
            "nearest_truth_bin": bin_name,
            "match_distance": float(nearest_distance[index]),
            "matched_truth_index": int(matched_truth.get(index, -1)),
            "is_true_marker": 1 if index in matched_truth else 0,
            "truth_bin_count": int(bin_counts.get(bin_name, 0)),
            "truth_polarity_count": int(
                polarity_counts.get(
                    "white"
                    if 0 <= truth_index < len(truth_colors)
                    and int(truth_colors[truth_index]) == 1
                    else "black",
                    0,
                )
            ),
            "peak": float(peak.peak),
            "prominence": float(peak.prominence),
            "support_kind": peak.support_kind,
            "support_radius": float(peak.support_radius),
            "support_axis_1": float(peak.support_axes[0]),
            "support_axis_2": float(peak.support_axes[1]),
            "support_pixels": int(peak.support_pixels),
            "used_curvature_mask": int(bool(peak.used_curvature_mask)),
            "fit_ok": int(bool(peak.fit_ok)),
            "fallback": int(bool(peak.fallback)),
            "eigenvalue_1": float(peak.eigenvalues[0]),
            "eigenvalue_2": float(peak.eigenvalues[1]),
            "anisotropy": float(peak.anisotropy),
            "orientation_deg": float(peak.orientation_deg),
            "residual_rms": float(peak.residual_rms),
            "iterations": int(peak.iterations),
            "offset_x": float(peak.offset_x),
            "offset_y": float(peak.offset_y),
            "case_tilt_deg": float(meta.get("tilt_deg")) if meta.get("tilt_deg") is not None else float("nan"),
            "spacing_est": float(spacing),
            "sharpened": int(meta.get("sharpen") is not None),
            "sharpen_mode": sharpen.get("mode", ""),
            "sharpen_space": sharpen.get("space", ""),
            "sharpen_amount": float(sharpen.get("amount", 0.0)),
        }
        rows.append(row)
    return rows, matches


def _draw_overlay(
    path: Path,
    gray: np.ndarray,
    mask: np.ndarray,
    truth: np.ndarray,
    truth_excluded: np.ndarray,
    peaks_by_variant: dict[str, list[LocalPeak]],
    title: str,
) -> None:
    display = np.repeat(np.clip(gray, 0.0, 1.0)[:, :, None], 3, axis=2)
    def to_index(value: float) -> int:
        return int(round(float(value) - 0.5))

    for x, y in truth_excluded:
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            cv2.drawMarker(
                display,
                (to_index(x), to_index(y)),
                (0.55, 0.55, 0.55),
                cv2.MARKER_CROSS,
                3,
                1,
            )
    for x, y in truth:
        cv2.drawMarker(
            display,
            (to_index(x), to_index(y)),
            (1.0, 1.0, 0.15),
            cv2.MARKER_CROSS,
            4,
            1,
        )
    for variant, peaks in peaks_by_variant.items():
        color = VARIANT_COLORS[variant]
        for peak in peaks:
            cv2.circle(
                display,
                (to_index(peak.x), to_index(peak.y)),
                1,
                color,
                1,
            )
    # Draw the support of the recommended oblique variant on top, but keep it
    # thin so hundreds of markers remain readable.
    for peak in peaks_by_variant.get("adaptive_ellipse_concave", []):
        if not peak.fit_ok:
            continue
        center = (int(round(peak.candidate_x)), int(round(peak.candidate_y)))
        if peak.support_kind == "ellipse":
            axes = (max(1, int(round(peak.support_axes[0]))), max(1, int(round(peak.support_axes[1]))))
            cv2.ellipse(
                display,
                center,
                axes,
                float(peak.orientation_deg),
                0.0,
                360.0,
                (0.4, 0.9, 1.0),
                1,
            )
        else:
            cv2.circle(display, center, max(1, int(round(peak.support_radius))), (0.4, 0.9, 1.0), 1)
    cv2.putText(
        display,
        title,
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (1.0, 1.0, 1.0),
        1,
        cv2.LINE_AA,
    )
    bgr = cv2.cvtColor(np.clip(display * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 and truth.size:
        xs = truth[:, 0]
        ys = truth[:, 1]
    if len(xs) == 0:
        return
    pad = 12
    x0, x1 = max(0, int(xs.min()) - pad), min(gray.shape[1], int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(gray.shape[0], int(ys.max()) + pad)
    crop = display[y0:y1, x0:x1]
    if crop.size == 0:
        return
    scale = max(1, int(round(720 / max(1, crop.shape[1]))))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(
        str(path.with_name(path.stem + "_zoom.png")),
        cv2.cvtColor(np.clip(crop * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )


def _make_plots(
    out_dir: Path,
    variant_errors: dict[str, list[float]],
    variant_support: dict[str, list[float]],
    aggregate: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - plotting is optional
        print(f"[warn] matplotlib unavailable, skipping plots: {error}", file=sys.stderr)
        return

    plt.figure(figsize=(6.4, 4.0))
    for variant in VARIANTS:
        values = np.sort(np.asarray(variant_errors.get(variant, []), dtype=np.float64))
        if values.size == 0:
            continue
        cumulative = np.arange(1, values.size + 1) / values.size
        plt.plot(values, cumulative, label=variant)
    plt.xlabel("center error (px)")
    plt.ylabel("fraction of matched peaks")
    plt.title("Matched-peak center error CDF (all angle bins)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "error_cdf.png", dpi=140)
    plt.close()

    plt.figure(figsize=(6.4, 4.0))
    for variant in ("adaptive_circle_concave", "adaptive_ellipse_concave"):
        values = np.asarray(variant_support.get(variant, []), dtype=np.float64)
        if values.size == 0:
            continue
        plt.hist(values, bins=20, alpha=0.5, label=variant)
    plt.xlabel("support radius / broad semi-axis (px)")
    plt.ylabel("peaks")
    plt.title("Chosen support size")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "support_hist.png", dpi=140)
    plt.close()

    bin_names = [name for name, _lower, _upper in TILT_BINS]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    width = 0.8 / max(1, len(VARIANTS))
    for index, variant in enumerate(VARIANTS):
        entry = aggregate.get("by_variant", {}).get(variant, {})
        per_bin = entry.get("per_bin", {})
        recalls = [per_bin.get(name, {}).get("recall", 0.0) for name in bin_names]
        errors = [per_bin.get(name, {}).get("mean_error_px", 0.0) for name in bin_names]
        offsets = np.arange(len(bin_names)) + (index - len(VARIANTS) / 2.0) * width + width / 2.0
        axes[0].bar(offsets, recalls, width=width, label=variant)
        axes[1].bar(offsets, errors, width=width, label=variant)
    for axis, title in ((axes[0], "recall@1.0 px"), (axes[1], "mean center error (px)")):
        axis.set_xticks(np.arange(len(bin_names)))
        axis.set_xticklabels(bin_names)
        axis.set_title(title)
        axis.grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_angle.png", dpi=140)
    plt.close(fig)

    if aggregate.get("by_sharpen"):
        fig, axis = plt.subplots(figsize=(6.4, 3.8))
        for variant in ("adaptive_circle_concave", "adaptive_ellipse_concave"):
            clean = aggregate["by_sharpen"].get(variant, {}).get("clean", {})
            sharpened = aggregate["by_sharpen"].get(variant, {}).get("sharpened", {})
            x = [0, 1]
            y = [
                clean.get("overall", {}).get("1.0", {}).get("mean_error_px", 0.0),
                sharpened.get("overall", {}).get("1.0", {}).get("mean_error_px", 0.0),
            ]
            axis.plot(x, y, marker="o", label=variant)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["clean", "sharpened"])
        axis.set_ylabel("mean center error (px)")
        axis.set_title("Camera sharpening effect")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out_dir / "sharpening_delta.png", dpi=140)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    if not args.no_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    specs = build_case_specs(
        seed=args.seed,
        samples_scale=args.samples_scale,
        include_large=not args.no_large,
        max_cases=args.max_cases,
    )
    if not specs:
        raise SystemExit("no cases selected")

    cases: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    overall: dict[str, MetricAccumulator] = {variant: MetricAccumulator() for variant in VARIANTS}
    by_sharpen: dict[str, dict[str, MetricAccumulator]] = {
        variant: {"clean": MetricAccumulator(), "sharpened": MetricAccumulator()}
        for variant in VARIANTS
    }
    variant_errors: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
    variant_support: dict[str, list[float]] = {variant: [] for variant in VARIANTS}

    for spec_index, spec in enumerate(specs):
        print(f"[{spec_index + 1}/{len(specs)}] {spec.name} yaw={spec.yaw_deg:g} pitch={spec.pitch_deg:g}")
        config = make_config(spec, sharpen_probability=args.sharpen_probability)
        layout = build_hex_target_layout(width_px=spec.width, height_px=spec.height)
        try:
            dataset = generate_heatmap_dataset(config, layout, count=spec.samples)
        except RuntimeError as error:
            print(f"  [warn] {error}")
            continue
        for sample_index in range(len(dataset)):
            gray = np.asarray(dataset.images[sample_index, :, :, 0], dtype=np.float32)
            meta = dataset.metadata[sample_index]
            truth_all = np.asarray(meta.get("points") or [], dtype=np.float64).reshape(-1, 2)
            if truth_all.shape[0] == 0:
                continue
            truth_bins_all = _truth_bins(meta, truth_all)
            truth_colors_all = _truth_colors(meta, gray, truth_all)
            margin = max(0.0, float(args.visibility_margin))
            visible = (
                (truth_all[:, 0] >= margin)
                & (truth_all[:, 0] <= spec.width - 1 - margin)
                & (truth_all[:, 1] >= margin)
                & (truth_all[:, 1] <= spec.height - 1 - margin)
            )
            if not visible.any():
                continue
            truth = truth_all[visible]
            truth_bins = [name for name, keep in zip(truth_bins_all, visible) if keep]
            truth_colors = truth_colors_all[visible]
            detection_response = smooth_luma(gray, args.smooth_sigma)
            fit_response = smooth_luma(gray, args.fit_sigma)
            curvature: CurvatureData = curvature_maps(fit_response)
            candidates = detect_local_peaks(
                detection_response,
                min_prominence=args.min_prominence,
                prominence_window=args.prominence_window,
                border=BORDER_MARGIN,
            )
            candidates = [
                refresh_candidate(
                    fit_response,
                    candidate,
                    window=args.support_window,
                )
                for candidate in candidates
            ]
            spacing = estimate_spacing(candidates, fallback=float(layout.spacing_px))
            compact, maximum = support_radii(spacing)
            peaks_by_variant: dict[str, list[LocalPeak]] = {}
            case_variants: dict[str, Any] = {}
            for variant in VARIANTS:
                peaks = fit_peaks(
                    fit_response,
                    candidates,
                    spacing=spacing,
                    mode=variant,
                    compact_radius=compact,
                    max_radius=maximum,
                    curvature=curvature,
                    local_radii=True,
                )
                peaks_by_variant[variant] = peaks
                accumulator = MetricAccumulator()
                accumulator.add(peaks, truth, truth_colors, truth_bins)
                case_variants[variant] = accumulator.summary()
                overall[variant].add(peaks, truth, truth_colors, truth_bins)
                sharpened = meta.get("sharpen") is not None
                by_sharpen[variant]["sharpened" if sharpened else "clean"].add(
                    peaks, truth, truth_colors, truth_bins
                )
                variant_errors[variant].extend(accumulator.errors[2.0])
                for peak in peaks:
                    if peak.fit_ok:
                        variant_support[variant].append(float(peak.support_radius))
                rows, _matches = _descriptor_rows(
                    case=spec,
                    sample_index=sample_index,
                    variant=variant,
                    peaks=peaks,
                    candidates=candidates,
                    truth=truth,
                    truth_colors=truth_colors,
                    truth_bins=truth_bins,
                    meta=meta,
                    spacing=spacing,
                )
                descriptors.extend(rows)

            if not args.no_overlays:
                title = (
                    f"{spec.name} sample={sample_index} group={spec.group} "
                    f"truth={truth.shape[0]}/{truth_all.shape[0]} "
                    f"cand={len(candidates)} sharpened={meta.get('sharpen') is not None}"
                )
                _draw_overlay(
                    overlay_dir / f"{spec.name}_{sample_index:02d}.png",
                    gray,
                    np.asarray(dataset.masks[sample_index, :, :, 0], dtype=np.float32),
                    truth,
                    truth_all[~visible],
                    peaks_by_variant,
                    title,
                )

            cases.append(
                {
                    "name": spec.name,
                    "group": spec.group,
                    "sample": sample_index,
                    "width": spec.width,
                    "height": spec.height,
                    "yaw_deg": spec.yaw_deg,
                    "pitch_deg": spec.pitch_deg,
                    "truth": int(truth.shape[0]),
                    "truth_all": int(truth_all.shape[0]),
                    "border_excluded": int(truth_all.shape[0] - truth.shape[0]),
                    "visibility_margin_px": float(margin),
                    "candidates": len(candidates),
                    "spacing_est": float(spacing),
                    "compact_radius": float(compact),
                    "max_radius": float(maximum),
                    "case_tilt_deg": meta.get("tilt_deg"),
                    "sharpen": meta.get("sharpen"),
                    "variants": case_variants,
                }
            )

    if not cases:
        raise SystemExit("no usable cases were generated")

    aggregate = {
        "by_variant": {variant: overall[variant].summary() for variant in VARIANTS},
        "by_sharpen": {
            variant: {
                name: accumulator.summary()
                for name, accumulator in by_sharpen[variant].items()
            }
            for variant in VARIANTS
        },
        "descriptor_count": len(descriptors),
    }

    descriptor_path = out_dir / "descriptors.csv"
    if descriptors:
        with descriptor_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(descriptors[0].keys()))
            writer.writeheader()
            writer.writerows(descriptors)

    result = {
        "config": {
            "smooth_sigma": args.smooth_sigma,
            "fit_sigma": args.fit_sigma,
            "min_prominence": args.min_prominence,
            "prominence_window": args.prominence_window,
            "support_window": args.support_window,
            "visibility_margin": args.visibility_margin,
            "sharpen_probability": args.sharpen_probability,
            "variants": list(VARIANTS),
            "angle_groups": [
                {"name": name, "yaw_deg": yaw, "pitch_deg": pitch}
                for name, yaw, pitch, _a, _b in ANGLE_GROUPS
            ],
            "tilt_bins": [name for name, _lower, _upper in TILT_BINS],
        },
        "cases": cases,
        "aggregate": aggregate,
        "descriptors_csv": str(descriptor_path) if descriptors else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    if not args.skip_plots:
        _make_plots(out_dir, variant_errors, variant_support, aggregate)

    print(f"\nwrote {out_dir / 'summary.json'}")
    if descriptors:
        print(f"wrote {descriptor_path} ({len(descriptors)} rows)")
    if not args.no_overlays:
        print(f"wrote overlays to {overlay_dir}")

    print("\nOVERALL")
    for variant in VARIANTS:
        entry = aggregate["by_variant"][variant]
        overall_entry = entry["overall"]
        print(
            f"{variant:26s} truth={entry['truth']:5d} pred={entry['predicted']:5d} "
            f"fit={entry['fit_ok']:5d} fb={entry['fallback']:4d} "
            f"F1@0.5={overall_entry['0.5']['f1']:.3f} "
            f"F1@1.0={overall_entry['1.0']['f1']:.3f} "
            f"F1@2.0={overall_entry['2.0']['f1']:.3f} "
            f"mean={overall_entry['1.0']['mean_error_px']:.3f} "
            f"p90={overall_entry['1.0']['p90_error_px']:.3f} "
            f"bias=({overall_entry['1.0']['bias_x_px']:+.3f},{overall_entry['1.0']['bias_y_px']:+.3f})"
        )

    print("\nPER FORESHORTENING BIN (recall@1.0 / mean error)")
    header = f"{'bin':10s} {'truth':>6s} " + " ".join(f"{variant:>26s}" for variant in VARIANTS)
    print(header)
    for name, _lower, _upper in TILT_BINS:
        cells = []
        truth_count = 0
        for variant in VARIANTS:
            entry = aggregate["by_variant"][variant]["per_bin"][name]
            truth_count = max(truth_count, int(entry["truth"]))
            cells.append(f"{entry['recall']:.3f}/{entry['mean_error_px']:.3f}")
        print(f"{name:10s} {truth_count:6d} " + " ".join(f"{cell:>26s}" for cell in cells))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--fit-sigma", type=float, default=0.8)
    parser.add_argument("--min-prominence", type=float, default=0.05)
    parser.add_argument("--prominence-window", type=int, default=5)
    parser.add_argument("--support-window", type=int, default=11)
    parser.add_argument(
        "--visibility-margin",
        type=float,
        default=8.0,
        help="Exclude truth points within this many px of the frame border from the metrics.",
    )
    parser.add_argument(
        "--sharpen-probability",
        type=float,
        default=None,
        help="Override the camera sharpening probability; 0 gives the clean control.",
    )
    parser.add_argument("--samples-scale", type=float, default=1.0)
    parser.add_argument("--no-large", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
