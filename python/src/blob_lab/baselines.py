"""Reference patch scores for blob/no-blob experiments."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class PatchMethod(Protocol):
    name: str

    def fit(self, images: np.ndarray, labels: np.ndarray) -> None: ...

    def score(self, images: np.ndarray) -> np.ndarray: ...


def _as_2d(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4:
        return images[..., 0]
    return images


class DohMethod:
    name = "DoH center"

    def __init__(self, sigmas: tuple[float, ...] = (1.0, 1.5, 2.0)):
        self.sigmas = sigmas

    def fit(self, images: np.ndarray, labels: np.ndarray) -> None:
        return

    def score(self, images: np.ndarray) -> np.ndarray:
        patches = _as_2d(images)
        scores = np.full((patches.shape[0],), -np.inf, dtype=np.float32)
        for sigma in self.sigmas:
            for index, patch in enumerate(patches):
                blurred = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), sigma)
                dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
                dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
                dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
                response = (sigma ** 4) * (dxx * dyy - dxy * dxy)
                center = float(response[patch.shape[0] // 2, patch.shape[1] // 2])
                scores[index] = max(scores[index], center)
        return scores


class RingMethod:
    name = "center vs ring"

    def __init__(self, inner_radius: int = 3, outer_radius: int = 8):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def fit(self, images: np.ndarray, labels: np.ndarray) -> None:
        return

    def score(self, images: np.ndarray) -> np.ndarray:
        patches = _as_2d(images)
        scores = np.zeros((patches.shape[0],), dtype=np.float32)
        for index, patch in enumerate(patches):
            h, w = patch.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.mgrid[:h, :w]
            radius = np.hypot(yy - cy, xx - cx)
            inner = patch[radius <= self.inner_radius]
            ring = patch[(radius >= self.inner_radius + 1) & (radius <= self.outer_radius)]
            if inner.size == 0 or ring.size == 0:
                continue
            contrast = abs(float(inner.mean()) - float(ring.mean()))
            scores[index] = contrast / (float(ring.std()) + 1e-4)
        return scores


class CnnMethod:
    name = "tiny CNN"

    def __init__(self, model):
        self.model = model

    def fit(self, images: np.ndarray, labels: np.ndarray) -> None:
        return

    def score(self, images: np.ndarray) -> np.ndarray:
        return self.model.score(images)


def compare_methods(methods: list[PatchMethod], images: np.ndarray, labels: np.ndarray) -> list[dict]:
    from blob_lab.metrics import best_threshold, binary_metrics, roc_auc

    results = []
    for method in methods:
        scores = method.score(images)
        threshold, metrics = best_threshold(labels, scores)
        results.append(
            {
                "method": method.name,
                "threshold": threshold,
                "auc": roc_auc(labels, scores),
                **metrics,
            }
        )
    return results
