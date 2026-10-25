#!/usr/bin/env python3
"""Diagnostics for the local-shape-cluster filter.

The bicubic experiment writes one descriptor row per fitted candidate.  This
script answers the question that was deliberately left open: can a majority
cluster of fitted-shape descriptors separate target markers from clutter, and
which feature set / normalization / scope is best?

It is a diagnostic, not a production gate.  The output is the evidence needed
to choose the filter's feature set and majority rule in the next iteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

FEATURE_SETS: dict[str, list[str]] = {
    "minimal": [
        "log_peak",
        "log_prominence",
        "anisotropy",
        "log_support_radius",
    ],
    "shape": [
        "log_peak",
        "log_prominence",
        "anisotropy",
        "log_support_radius",
        "log_abs_eigenvalue_1",
        "log_abs_eigenvalue_2",
        "log_residual_rms",
        "support_axis_ratio",
    ],
    "full": [
        "log_peak",
        "log_prominence",
        "anisotropy",
        "log_support_radius",
        "log_abs_eigenvalue_1",
        "log_abs_eigenvalue_2",
        "log_residual_rms",
        "support_axis_ratio",
        "log_support_pixels",
        "used_curvature_mask",
        "log_offset",
    ],
}

SCOPES = ("global", "local", "polarity", "bin")


def _num(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _optional_num(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_descriptors(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if _num(raw, "fit_ok", 0.0) < 0.5:
                continue
            peak = abs(_num(raw, "peak", 0.0))
            prominence = max(0.0, _num(raw, "prominence", 0.0))
            radius = max(0.0, _num(raw, "support_radius", 0.0))
            axis_1 = max(1e-6, _num(raw, "support_axis_1", 1.0))
            axis_2 = max(1e-6, _num(raw, "support_axis_2", 1.0))
            eigen_1 = abs(_num(raw, "eigenvalue_1", 0.0))
            eigen_2 = abs(_num(raw, "eigenvalue_2", 0.0))
            residual = max(0.0, _num(raw, "residual_rms", 0.0))
            offset = math.hypot(_num(raw, "offset_x", 0.0), _num(raw, "offset_y", 0.0))
            rows.append(
                {
                    "variant": raw.get("variant", ""),
                    "case": raw.get("case", ""),
                    "sample": raw.get("sample", ""),
                    "fit_x": _num(raw, "fit_x", 0.0),
                    "fit_y": _num(raw, "fit_y", 0.0),
                    "spacing_est": max(0.0, _num(raw, "spacing_est", 5.0)),
                    "is_true": _num(raw, "is_true_marker", 0.0) >= 0.5,
                    "truth_count": int(_num(raw, "truth_count", 0.0)),
                    "truth_bin": raw.get("nearest_truth_bin", "unmatched"),
                    "truth_bin_count": int(_num(raw, "truth_bin_count", 0.0)),
                    "pred_polarity": int(_num(raw, "pred_polarity", -1.0)),
                    "truth_polarity_count": int(_num(raw, "truth_polarity_count", 0.0)),
                    "log_peak": math.log1p(10.0 * peak),
                    "log_prominence": math.log1p(10.0 * prominence),
                    "anisotropy": min(1.0, max(0.0, _num(raw, "anisotropy", 0.0))),
                    "log_support_radius": math.log1p(radius),
                    "log_abs_eigenvalue_1": math.log1p(10.0 * eigen_1),
                    "log_abs_eigenvalue_2": math.log1p(10.0 * eigen_2),
                    "log_residual_rms": math.log1p(100.0 * residual),
                    "support_axis_ratio": min(axis_1, axis_2) / max(axis_1, axis_2),
                    "log_support_pixels": math.log1p(max(0.0, _num(raw, "support_pixels", 0.0))),
                    "used_curvature_mask": _num(raw, "used_curvature_mask", 0.0),
                    "log_offset": math.log1p(10.0 * offset),
                }
            )
    return rows


def _standardize(matrix: np.ndarray) -> np.ndarray:
    median = np.median(matrix, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    scale = q75 - q25
    fallback = matrix.std(axis=0)
    scale = np.where(scale > 1e-9, scale, fallback)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return np.clip((matrix - median) / scale, -8.0, 8.0)


def _silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0
    distances = cdist(matrix, matrix)
    scores: list[float] = []
    for index in range(matrix.shape[0]):
        same = labels == labels[index]
        same[index] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = float(distances[index, same].mean())
        b = float(
            min(
                distances[index, labels == other].mean()
                for other in unique
                if other != labels[index]
            )
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)
    return float(np.mean(scores))


def _cluster_metrics(
    matrix: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    truth_total: int,
) -> dict[str, Any]:
    counts = np.bincount(labels)
    largest = int(np.argmax(counts))
    keep = labels == largest
    tp = int((truth & keep).sum())
    fp = int((~truth & keep).sum())
    fn = max(0, int(truth_total) - tp)
    majority_precision = tp / (tp + fp) if tp + fp else 0.0
    majority_recall = tp / max(1, int(truth_total))
    majority_f1 = (
        2.0 * majority_precision * majority_recall / (majority_precision + majority_recall)
        if majority_precision + majority_recall
        else 0.0
    )
    all_tp = int(truth.sum())
    all_fp = int((~truth).sum())
    all_fn = max(0, int(truth_total) - all_tp)
    all_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    all_recall = all_tp / max(1, int(truth_total))
    all_f1 = (
        2.0 * all_precision * all_recall / (all_precision + all_recall)
        if all_precision + all_recall
        else 0.0
    )
    purity = 0.0
    for cluster in np.unique(labels):
        members = labels == cluster
        positive = int((truth & members).sum())
        negative = int((~truth & members).sum())
        purity += max(positive, negative)
    purity = purity / max(1, labels.size)
    return {
        "k": int(np.unique(labels).size),
        "cluster_sizes": [int(value) for value in counts.tolist()],
        "purity": float(purity),
        "majority_cluster": largest,
        "majority_cluster_size": int(counts[largest]),
        "majority_precision": float(majority_precision),
        "majority_recall": float(majority_recall),
        "majority_f1": float(majority_f1),
        "unclustered_precision": float(all_precision),
        "unclustered_recall": float(all_recall),
        "unclustered_f1": float(all_f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "keep_mask": keep,
    }


def _per_group_bins(
    rows: list[dict[str, Any]],
    keep: np.ndarray,
    truth: np.ndarray,
    truth_bin_truths: dict[str, int],
) -> dict[str, Any]:
    bins = sorted({row["truth_bin"] for row in rows if row["truth_bin"] != "unmatched"})
    result: dict[str, Any] = {}
    for name in bins:
        indices = np.asarray([row["truth_bin"] == name for row in rows], dtype=bool)
        if not indices.any():
            continue
        truth_total = int(truth_bin_truths.get(name, max(row["truth_bin_count"] for row in rows if row["truth_bin"] == name)))
        kept = indices & keep
        tp = int((kept & truth).sum())
        fp = int((kept & ~truth).sum())
        fn = max(0, truth_total - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / max(1, truth_total)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        all_tp = int((indices & truth).sum())
        all_fp = int((indices & ~truth).sum())
        all_fn = max(0, truth_total - all_tp)
        all_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
        all_recall = all_tp / max(1, truth_total)
        all_f1 = (
            2.0 * all_precision * all_recall / (all_precision + all_recall)
            if all_precision + all_recall
            else 0.0
        )
        result[name] = {
            "truth": truth_total,
            "majority_tp": tp,
            "majority_fp": fp,
            "majority_fn": fn,
            "majority_recall": float(recall),
            "majority_f1": float(f1),
            "unclustered_f1": float(all_f1),
        }
    return result


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _summary_truth_bins(
    summary: dict[str, Any] | None,
    variant: str,
    case: str,
    sample: int,
) -> dict[str, int]:
    if summary is None:
        return {}
    for entry in summary.get("cases", []):
        if entry.get("name") == case and int(entry.get("sample", -1)) == int(sample):
            per_bin = entry.get("variants", {}).get(variant, {}).get("per_bin", {})
            return {name: int(value.get("truth", 0)) for name, value in per_bin.items()}
    return {}


def _choose_labels(
    matrix: np.ndarray,
    truth: np.ndarray,
    truth_total: int,
    seed: int,
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    best: dict[str, Any] | None = None
    best_labels: np.ndarray | None = None
    best_score = -1.0
    n = matrix.shape[0]
    max_k = min(4, max(2, n // 3))
    for k in range(2, max_k + 1):
        try:
            _centroids, labels = kmeans2(
                matrix,
                k,
                iter=60,
                minit="++",
                seed=seed,
            )
        except Exception:
            continue
        if labels.shape[0] != n or (labels < 0).any():
            continue
        metrics = _cluster_metrics(matrix, truth, labels, truth_total)
        score = float(metrics["silhouette"]) if "silhouette" in metrics else 0.0
        # Silhouette is the primary model-selection score; purity breaks ties
        # toward a cluster that actually separates markers from clutter.
        score = _silhouette(matrix, labels) + 0.25 * float(metrics["purity"])
        if score > best_score:
            best_score = score
            best = metrics
            best_labels = labels
    if best is None or best_labels is None:
        return None, None
    best["silhouette"] = _silhouette(matrix, best_labels)
    return best, best_labels



def _f1_from_keep(truth: np.ndarray, keep: np.ndarray, truth_total: int) -> tuple[float, float, float]:
    tp = int((truth & keep).sum())
    fp = int((~truth & keep).sum())
    fn = max(0, int(truth_total) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / max(1, int(truth_total))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _local_patch_cluster(
    rows: list[dict[str, Any]],
    feature_set: str,
    *,
    truth_bins: dict[str, int],
    patch_radius: float,
    overlap: float,
    vote_threshold: float,
    seed: int,
) -> dict[str, Any] | None:
    """Cluster descriptors in overlapping local patches and fuse by vote.

    Clustering all candidates on the screen mixes the shape variation across
    the target (which is exactly what the previous diagnostic did).  A target
    marker's local shape is only comparable to its neighbours: under
    perspective the compressed side can have a very different aspect ratio from
    the stretched side.  Overlapping patches let each region choose its own
    majority shape, and the vote fusion resolves patch boundaries.
    """
    if len(rows) < 6:
        return None
    truth = np.asarray([row["is_true"] for row in rows], dtype=bool)
    truth_total = max(row["truth_count"] for row in rows)
    if truth_total <= 0:
        return None
    matrix = _standardize(
        np.asarray(
            [[row[name] for name in FEATURE_SETS[feature_set]] for row in rows],
            dtype=np.float64,
        )
    )
    coords = np.asarray([[row["fit_x"], row["fit_y"]] for row in rows], dtype=np.float64)
    xs = coords[:, 0]
    ys = coords[:, 1]
    radius = max(8.0, float(patch_radius))
    step = max(6.0, radius * (1.0 - float(np.clip(overlap, 0.0, 0.9))))
    x_centers = np.arange(float(xs.min()), float(xs.max()) + step, step)
    y_centers = np.arange(float(ys.min()), float(ys.max()) + step, step)
    votes_kept = np.zeros(len(rows), dtype=np.float64)
    votes_total = np.zeros(len(rows), dtype=np.float64)
    silhouettes: list[float] = []
    purities: list[float] = []
    patches = 0
    for cx in x_centers:
        for cy in y_centers:
            inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
            indices = np.nonzero(inside)[0]
            if indices.size < 6:
                continue
            sub_truth = truth[indices]
            if sub_truth.all():
                keep_sub = np.ones(indices.size, dtype=bool)
                silhouettes.append(0.0)
                purities.append(1.0)
            elif (~sub_truth).all():
                keep_sub = np.zeros(indices.size, dtype=bool)
                silhouettes.append(0.0)
                purities.append(1.0)
            else:
                metrics, labels = _choose_labels(
                    matrix[indices],
                    sub_truth,
                    truth_total,
                    seed,
                )
                if labels is None or metrics is None:
                    continue
                keep_sub = labels == metrics["majority_cluster"]
                silhouettes.append(float(metrics["silhouette"]))
                purities.append(float(metrics["purity"]))
            votes_total[indices] += 1.0
            votes_kept[indices] += keep_sub.astype(np.float64)
            patches += 1
    if patches == 0:
        return None
    ratio = np.divide(
        votes_kept,
        np.maximum(votes_total, 1.0),
        out=np.ones(len(rows), dtype=np.float64),
        where=votes_total > 0.0,
    )
    keep = ratio >= float(vote_threshold)
    keep[votes_total <= 0.0] = True
    precision, recall, f1 = _f1_from_keep(truth, keep, truth_total)
    _unclustered_precision, _unclustered_recall, unclustered_f1 = _f1_from_keep(
        truth, np.ones(len(rows), dtype=bool), truth_total
    )
    per_bin = _per_group_bins(rows, keep, truth, truth_bins)
    return {
        "scope": "local",
        "rows": len(rows),
        "truth_total": int(truth_total),
        "silhouette": float(np.mean(silhouettes)) if silhouettes else 0.0,
        "purity": float(np.mean(purities)) if purities else 0.0,
        "majority_precision": float(precision),
        "majority_recall": float(recall),
        "majority_f1": float(f1),
        "unclustered_f1": float(unclustered_f1),
        "patches": int(patches),
        "patch_radius": float(radius),
        "vote_threshold": float(vote_threshold),
        "per_bin": per_bin,
    }

def _aggregate_group(
    rows: list[dict[str, Any]],
    feature_set: str,
    scope: str,
    *,
    truth_bins: dict[str, int],
    seed: int,
) -> dict[str, Any] | None:
    if len(rows) < 6:
        return None
    truth = np.asarray([row["is_true"] for row in rows], dtype=bool)
    truth_total = max(row["truth_count"] for row in rows)
    if truth_total <= 0:
        return None
    matrix = np.asarray(
        [[row[name] for name in FEATURE_SETS[feature_set]] for row in rows],
        dtype=np.float64,
    )
    matrix = _standardize(matrix)
    metrics, labels = _choose_labels(matrix, truth, truth_total, seed)
    if metrics is None or labels is None:
        return None
    per_bin = _per_group_bins(rows, labels == metrics["majority_cluster"], truth, truth_bins)
    metrics["per_bin"] = per_bin
    metrics["scope"] = scope
    metrics["rows"] = len(rows)
    metrics["truth_total"] = int(truth_total)
    metrics.pop("keep_mask", None)
    return metrics


def _aggregate_scope(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    fields = (
        "silhouette",
        "purity",
        "majority_precision",
        "majority_recall",
        "majority_f1",
        "unclustered_f1",
    )
    summary = {field: float(np.mean([entry[field] for entry in results])) for field in fields}
    summary["groups"] = len(results)
    summary["mean_rows"] = float(np.mean([entry["rows"] for entry in results]))
    per_bin: dict[str, Any] = {}
    bins = sorted({name for entry in results for name in entry["per_bin"]})
    for name in bins:
        entries = [entry["per_bin"][name] for entry in results if name in entry["per_bin"]]
        if not entries:
            continue
        truth = sum(int(entry["truth"]) for entry in entries)
        majority_tp = sum(int(entry["majority_tp"]) for entry in entries)
        majority_fp = sum(int(entry["majority_fp"]) for entry in entries)
        precision = majority_tp / (majority_tp + majority_fp) if majority_tp + majority_fp else 0.0
        recall = majority_tp / max(1, truth)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        unclustered_tp = sum(int(entry.get("unclustered_tp", 0)) for entry in entries)
        _ = unclustered_tp
        per_bin[name] = {
            "truth": truth,
            "majority_recall": float(recall),
            "majority_f1": float(f1),
            "unclustered_f1": float(
                np.mean([entry["unclustered_f1"] for entry in entries])
            ),
        }
    summary["per_bin"] = per_bin
    return summary


def _make_plots(
    out_dir: Path,
    descriptors: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"[warn] matplotlib unavailable: {error}", file=sys.stderr)
        return

    features = [
        "log_peak",
        "log_prominence",
        "anisotropy",
        "log_support_radius",
        "log_residual_rms",
        "support_axis_ratio",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0))
    for axis, name in zip(axes.ravel(), features):
        positive = np.asarray([row[name] for row in descriptors if row["is_true"]])
        negative = np.asarray([row[name] for row in descriptors if not row["is_true"]])
        combined = np.concatenate([positive, negative])
        if combined.size == 0:
            continue
        low = float(combined.min())
        high = float(combined.max())
        if high - low < 1e-9:
            low -= 0.5
            high += 0.5
        bins = np.linspace(low, high, 30)
        if positive.size:
            axis.hist(positive, bins=bins, alpha=0.55, label="true marker", density=True)
        if negative.size:
            axis.hist(negative, bins=bins, alpha=0.55, label="false", density=True)
        axis.set_title(name, fontsize=8)
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_histograms.png", dpi=140)
    plt.close(fig)

    # PCA of the shape feature set for the largest group, colored by the
    # chosen majority cluster.  This is the visual check that the majority
    # cluster corresponds to the true markers.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in descriptors:
        groups[(row["variant"], row["case"], row["sample"])].append(row)
    if groups:
        key = max(groups, key=lambda item: len(groups[item]))
        group_rows = groups[key]
        if len(group_rows) >= 6:
            # The user-facing point of this plot is that clustering must happen
            # in a local patch: the global view mixes compressed and stretched
            # marker shapes.  Pick the densest overlapping patch, not the whole
            # image, and cluster within it.
            spacing_values = [row["spacing_est"] for row in group_rows if row["spacing_est"] > 0]
            local_spacing = float(np.median(spacing_values)) if spacing_values else 8.0
            patch_radius = float(np.clip(8.0 * local_spacing, 24.0, 96.0))
            step = max(6.0, patch_radius * 0.5)
            coords = np.asarray([[row["fit_x"], row["fit_y"]] for row in group_rows], dtype=np.float64)
            best_indices = np.arange(len(group_rows))
            best_truth_mix = False
            best_count = 0
            for cx in np.arange(coords[:, 0].min(), coords[:, 0].max() + step, step):
                for cy in np.arange(coords[:, 1].min(), coords[:, 1].max() + step, step):
                    inside = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2 <= patch_radius**2
                    count = int(inside.sum())
                    if count <= best_count:
                        continue
                    labels_truth = np.asarray([row["is_true"] for row in group_rows], dtype=bool)[inside]
                    mixed = bool(labels_truth.any() and (~labels_truth).any())
                    if count > best_count or (mixed and not best_truth_mix):
                        best_count = count
                        best_indices = np.nonzero(inside)[0]
                        best_truth_mix = mixed
                        best_center = (float(cx), float(cy))
            patch_rows = [group_rows[index] for index in best_indices]
            if len(patch_rows) >= 6:
                names = FEATURE_SETS["shape"]
                matrix = _standardize(
                    np.asarray([[row[name] for name in names] for row in patch_rows], dtype=np.float64)
                )
                centered = matrix - matrix.mean(axis=0, keepdims=True)
                _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
                projection = centered @ vh[:2].T
                truth = np.asarray([row["is_true"] for row in patch_rows], dtype=bool)
                truth_total = max(row["truth_count"] for row in patch_rows)
                metrics, labels = _choose_labels(matrix, truth, truth_total, seed=0)
                fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
                axes[0].scatter(projection[truth, 0], projection[truth, 1], s=8, label="true", alpha=0.6)
                axes[0].scatter(projection[~truth, 0], projection[~truth, 1], s=8, label="false", alpha=0.6)
                axes[0].set_title(
                    f"truth · local patch {key[0]} {key[1]} "
                    f"(cx={best_center[0]:.0f}, cy={best_center[1]:.0f}, r={patch_radius:.0f})",
                    fontsize=8,
                )
                axes[0].legend(fontsize=7)
                if labels is not None:
                    scatter = axes[1].scatter(
                        projection[:, 0],
                        projection[:, 1],
                        c=labels,
                        s=8,
                        cmap="tab10",
                        alpha=0.7,
                    )
                    fig.colorbar(scatter, ax=axes[1], label="cluster")
                    axes[1].set_title(
                        f"K={metrics['k']} purity={metrics['purity']:.3f}",
                        fontsize=8,
                    )
                for axis in axes:
                    axis.set_xlabel("PC 1")
                    axis.set_ylabel("PC 2")
                    axis.grid(alpha=0.2)
                fig.tight_layout()
                fig.savefig(out_dir / "cluster_pca.png", dpi=140)
                plt.close(fig)

    # Majority-cluster F1 by feature set for the recommended variant.
    variant = "adaptive_circle_concave"
    feature_names = list(FEATURE_SETS)
    values = []
    for feature_name in feature_names:
        entry = (
            diagnostics.get("variants", {})
            .get(variant, {})
            .get("feature_sets", {})
            .get(feature_name, {})
        )
        local_stats = entry.get("local", {}) or entry.get("global", {})
        values.append(float(local_stats.get("majority_f1", 0.0)))
    if any(values):
        fig, axis = plt.subplots(figsize=(6.0, 3.4))
        axis.bar(feature_names, values, color=["#4c72b0", "#dd8452", "#55a868"])
        axis.set_ylabel("local majority-cluster F1")
        axis.set_title(f"Local-patch feature-set comparison · {variant}")
        axis.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "cluster_summary.png", dpi=140)
        plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    descriptor_path = Path(args.descriptors)
    if not descriptor_path.is_file():
        raise SystemExit(f"descriptor CSV not found: {descriptor_path}")
    out_dir = Path(args.out) if args.out else descriptor_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(descriptor_path.parent / "summary.json")

    rows = load_descriptors(descriptor_path)
    if not rows:
        raise SystemExit("no fit-ok descriptor rows")
    variants = sorted({row["variant"] for row in rows})
    if args.variants:
        wanted = {name.strip() for name in args.variants.split(",")}
        variants = [name for name in variants if name in wanted]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant"] in variants:
            grouped[(row["variant"], row["case"], str(row["sample"]))].append(row)

    diagnostics: dict[str, Any] = {
        "descriptor_csv": str(descriptor_path),
        "variants": {variant: {"feature_sets": {name: {} for name in FEATURE_SETS}} for variant in variants},
    }
    console: list[str] = []

    for variant in variants:
        for feature_name in FEATURE_SETS:
            scope_results: dict[str, list[dict[str, Any]]] = {scope: [] for scope in SCOPES}
            for (group_variant, case, sample), group_rows in grouped.items():
                if group_variant != variant:
                    continue
                truth_bins = _summary_truth_bins(
                    summary,
                    variant,
                    case,
                    int(sample),
                )
                # global
                result = _aggregate_group(
                    group_rows,
                    feature_name,
                    "global",
                    truth_bins=truth_bins,
                    seed=0,
                )
                if result is not None:
                    scope_results["global"].append(result)
                spacing_values = [row["spacing_est"] for row in group_rows if row["spacing_est"] > 0]
                local_spacing = float(np.median(spacing_values)) if spacing_values else 6.0
                patch_radius = float(np.clip(8.0 * local_spacing, 24.0, 96.0))
                local_result = _local_patch_cluster(
                    group_rows,
                    feature_name,
                    truth_bins=truth_bins,
                    patch_radius=patch_radius,
                    overlap=0.5,
                    vote_threshold=0.5,
                    seed=0,
                )
                if local_result is not None:
                    scope_results["local"].append(local_result)
                # per polarity
                for polarity in (0, 1):
                    subset = [row for row in group_rows if int(row["pred_polarity"]) == polarity]
                    if len(subset) < 6:
                        continue
                    polarity_truth = max(
                        (row["truth_polarity_count"] for row in subset),
                        default=0,
                    )
                    if polarity_truth <= 0:
                        polarity_truth = max(row["truth_count"] for row in subset)
                    result = _aggregate_group(
                        subset,
                        feature_name,
                        "polarity",
                        truth_bins=truth_bins,
                        seed=polarity,
                    )
                    if result is not None:
                        scope_results["polarity"].append(result)
                # per angle bin
                for bin_name in sorted({row["truth_bin"] for row in group_rows if row["truth_bin"] != "unmatched"}):
                    subset = [row for row in group_rows if row["truth_bin"] == bin_name]
                    if len(subset) < 6:
                        continue
                    result = _aggregate_group(
                        subset,
                        feature_name,
                        "bin",
                        truth_bins={bin_name: truth_bins.get(bin_name, max(row["truth_bin_count"] for row in subset))},
                        seed=0,
                    )
                    if result is not None:
                        scope_results["bin"].append(result)

            entry: dict[str, Any] = {}
            for scope in SCOPES:
                entry[scope] = _aggregate_scope(scope_results[scope])
            diagnostics["variants"][variant]["feature_sets"][feature_name] = entry
            global_stats = entry["global"]
            if global_stats:
                console.append(
                    f"{variant:26s} {feature_name:8s} global groups={global_stats['groups']:4d} "
                    f"sil={global_stats['silhouette']:.3f} purity={global_stats['purity']:.3f} "
                    f"recall={global_stats['majority_recall']:.3f} "
                    f"F1={global_stats['majority_f1']:.3f} "
                    f"uncl_F1={global_stats['unclustered_f1']:.3f}"
                )
            local_stats = entry["local"]
            if local_stats:
                console.append(
                    f"{variant:26s} {feature_name:8s} local  groups={local_stats['groups']:4d} "
                    f"purity={local_stats['purity']:.3f} "
                    f"recall={local_stats['majority_recall']:.3f} "
                    f"F1={local_stats['majority_f1']:.3f} "
                    f"uncl_F1={local_stats['unclustered_f1']:.3f}"
                )

    result_path = out_dir / "shape_cluster_diagnostics.json"
    result_path.write_text(json.dumps(diagnostics, indent=2))
    print(f"wrote {result_path}")
    print("\nGLOBAL CLUSTER DIAGNOSTICS")
    for line in console:
        print(line)

    if not args.skip_plots:
        _make_plots(out_dir, rows, diagnostics)
        print(f"wrote plots to {out_dir}")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
