"""Small NumPy metric helpers for the blob-detector harness."""

from __future__ import annotations

import numpy as np


def confusion_matrix(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    predictions = scores >= threshold
    actual = labels >= 0.5
    return {
        "tp": int(np.sum(predictions & actual)),
        "tn": int(np.sum(~predictions & ~actual)),
        "fp": int(np.sum(predictions & ~actual)),
        "fn": int(np.sum(~predictions & actual)),
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    cm = confusion_matrix(labels, scores, threshold)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate([scores, [0.0, 0.5, 1.0]]))
    best = (-1.0, 0.5, binary_metrics(labels, scores, 0.5))
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, float(threshold))
        if metrics["f1"] > best[0] or (metrics["f1"] == best[0] and metrics["accuracy"] > best[2]["accuracy"]):
            best = (metrics["f1"], float(threshold), metrics)
    return best[1], best[2]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    actual = labels >= 0.5
    positives = int(np.sum(actual))
    negatives = int(np.sum(~actual))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum = float(np.sum(ranks[actual]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(np.concatenate([scores, [0.0, 1.0]]))[::-1]
    tprs = []
    fprs = []
    actual = labels >= 0.5
    positives = max(1, int(np.sum(actual)))
    negatives = max(1, int(np.sum(~actual)))
    for threshold in thresholds:
        predicted = scores >= threshold
        tprs.append(float(np.sum(predicted & actual)) / positives)
        fprs.append(float(np.sum(predicted & ~actual)) / negatives)
    return np.asarray(fprs), np.asarray(tprs)
