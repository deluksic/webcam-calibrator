"""Separate peak detector and keypoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Peak:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class PeakMetrics:
    true_count: int
    predicted_count: int
    matched: int
    precision: float
    recall: float
    f1: float
    mean_error_px: float


def detect_peaks(
    heatmap: np.ndarray,
    *,
    threshold: float = 0.25,
    nms_radius: int = 2,
    max_peaks: int = 5000,
) -> list[Peak]:
    """Threshold + local-maximum + greedy non-max suppression."""
    image = heatmap[..., 0] if heatmap.ndim == 3 else heatmap
    image = np.asarray(image, dtype=np.float32)
    radius = max(1, int(nms_radius))
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    local_max = cv2.dilate(image, kernel)
    maxima = ((image >= threshold) & (image >= local_max - 1e-7)).astype(np.uint8)
    if not maxima.any():
        return []

    # Collapse flat local-maxima plateaus to one representative per connected
    # component before doing distance suppression.
    count, labels = cv2.connectedComponents(maxima, connectivity=8)
    flat_labels = labels.ravel()
    flat_scores = image.ravel()
    component_max = np.zeros(count, dtype=np.float32)
    np.maximum.at(component_max, flat_labels, flat_scores)
    seen = np.zeros(count, dtype=bool)
    candidates: list[tuple[float, int, int]] = []
    for flat_index in np.flatnonzero(flat_scores >= component_max[flat_labels] - 1e-7):
        label = int(flat_labels[flat_index])
        if label == 0 or seen[label]:
            continue
        seen[label] = True
        y, x = np.unravel_index(int(flat_index), image.shape)
        candidates.append((float(flat_scores[flat_index]), int(x), int(y)))
    candidates.sort(reverse=True)

    # Grid-accelerated NMS; avoids comparing every candidate against every
    # already-selected peak when a background model produces many maxima.
    cell = max(1, radius)
    grid: dict[tuple[int, int], list[Peak]] = {}
    selected: list[Peak] = []
    min_distance_sq = float(radius * radius)
    for score, x, y in candidates:
        gx, gy = x // cell, y // cell
        suppressed = False
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for peak in grid.get((gx + ox, gy + oy), ()):
                    if (x - peak.x) ** 2 + (y - peak.y) ** 2 < min_distance_sq:
                        suppressed = True
                        break
                if suppressed:
                    break
            if suppressed:
                break
        if suppressed:
            continue
        peak = Peak(float(x), float(y), score)
        selected.append(peak)
        grid.setdefault((gx, gy), []).append(peak)
        if len(selected) >= max_peaks:
            break
    return selected


def match_peaks(
    predicted: Iterable[Peak],
    truth: Iterable[tuple[float, float]],
    *,
    tolerance: float = 2.0,
) -> PeakMetrics:
    predicted_list = list(predicted)
    truth_list = [(float(x), float(y)) for x, y in truth]
    used: set[int] = set()
    errors: list[float] = []
    cell = max(1.0, float(tolerance))
    truth_grid: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for index, (x, y) in enumerate(truth_list):
        truth_grid.setdefault((int(x // cell), int(y // cell)), []).append((index, x, y))
    for point in predicted_list:
        gx, gy = int(point.x // cell), int(point.y // cell)
        best_index = -1
        best_distance = float("inf")
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for index, x, y in truth_grid.get((gx + ox, gy + oy), ()):
                    if index in used:
                        continue
                    distance = float(np.hypot(point.x - x, point.y - y))
                    if distance < best_distance:
                        best_distance = distance
                        best_index = index
        if best_index >= 0 and best_distance <= tolerance:
            used.add(best_index)
            errors.append(best_distance)
    matched = len(errors)
    precision = matched / len(predicted_list) if predicted_list else 0.0
    recall = matched / len(truth_list) if truth_list else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PeakMetrics(
        true_count=len(truth_list),
        predicted_count=len(predicted_list),
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_error_px=float(np.mean(errors)) if errors else 0.0,
    )


def evaluate_heatmaps(
    heatmaps: np.ndarray,
    metadata: list[dict],
    *,
    threshold: float = 0.25,
    nms_radius: int = 2,
    tolerance: float = 2.0,
) -> PeakMetrics:
    predicted: list[Peak] = []
    truth: list[tuple[float, float]] = []
    for heatmap, meta in zip(heatmaps, metadata):
        predicted.extend(detect_peaks(heatmap, threshold=threshold, nms_radius=nms_radius))
        truth.extend(meta.get("points", []))
    return match_peaks(predicted, truth, tolerance=tolerance)


def overlay_peaks(image: np.ndarray, peaks: Iterable[Peak], *, radius: int = 2) -> np.ndarray:
    base = image[..., 0] if image.ndim == 3 else image
    rgb = np.repeat(np.asarray(base, dtype=np.float32)[:, :, None], 3, axis=2)
    for peak in peaks:
        x, y = int(round(peak.x)), int(round(peak.y))
        cv2.circle(rgb, (x, y), radius, (1.0, 0.0, 0.0), 1)
    return np.clip(rgb, 0.0, 1.0)
