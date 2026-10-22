#!/usr/bin/env python3
"""Measure all-color fitted-peak spacing before NMS.

The app currently uses a resolution-scaled NMS radius. This script instead
looks at the actual pairwise/nearest-neighbor distances between fitted peaks
after the local-max and gray-gap gates, so we can see whether the spacing has
a usable mode independent of frame width.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_histogram_nms import (  # noqa: E402
    _contrast_response,
    histogram_cuts,
    quadratic_maximum,
    roi_mask_from_onnx,
    segment,
)

GAP_FRACTION = 0.07
MIN_RESPONSE = 0.004
HIST_MAX_PX = 60.0
HIST_BIN_PX = 1.0


def fitted_local_maxima(
    response: np.ndarray,
    threshold: float,
) -> list[tuple[float, float, float, int]]:
    """5x5 local maxima, plateau collapse, quadratic fit; no greedy NMS."""
    local_max = cv2.dilate(response, np.ones((5, 5), dtype=np.uint8))
    maxima = (response >= threshold) & (response >= local_max - 1e-6)
    if not maxima.any():
        return []
    count, labels = cv2.connectedComponents(maxima.astype(np.uint8), connectivity=8)
    out: list[tuple[float, float, float, int]] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        values = response[ys, xs]
        best = int(np.argmax(values))
        x = int(xs[best])
        y = int(ys[best])
        fit = quadratic_maximum(
            response,
            x,
            y,
            fit_radius=1,
            min_curvature=0.0,
            min_peak=0.0,
        )
        if fit is None:
            continue
        fx, fy, peak = fit
        support = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                yy = int(np.clip(round(fy) + dy, 0, response.shape[0] - 1))
                xx = int(np.clip(round(fx) + dx, 0, response.shape[1] - 1))
                if response[yy, xx] >= peak * 0.5:
                    support += 1
        out.append((fx, fy, peak, support))
    return out


def peaks_for_image(path: Path) -> tuple[list[dict], dict]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    roi = roi_mask_from_onnx(path)
    cuts = histogram_cuts(gray, roi)
    black, white = segment(gray, roi, cuts)
    peaks: list[dict] = []
    for polarity, mask, name in [(-1, black, "black"), (1, white, "white")]:
        response = _contrast_response(gray, mask, polarity, background_sigma=2.5)
        if response is None:
            continue
        gap = (
            cuts["background"] - cuts["black_cut"]
            if name == "black"
            else cuts["white_cut"] - cuts["background"]
        )
        threshold = max(MIN_RESPONSE * 255.0, gap * GAP_FRACTION)
        for x, y, peak, support in fitted_local_maxima(response, threshold):
            peaks.append(
                {
                    "x": x,
                    "y": y,
                    "peak": peak,
                    "support": support,
                    "color": name,
                }
            )
    return peaks, cuts


def pairwise_distances(peaks: list[dict], colors: tuple[str, ...]) -> np.ndarray:
    distances: list[np.ndarray] = []
    for name in colors:
        pts = np.array([(p["x"], p["y"]) for p in peaks if p["color"] == name], dtype=np.float64)
        if len(pts) < 2:
            continue
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        iu = np.triu_indices(len(pts), k=1)
        distances.append(dist[iu])
    return np.concatenate(distances) if distances else np.empty(0)


def all_pairwise_distances(peaks: list[dict]) -> np.ndarray:
    pts = np.array([(p["x"], p["y"]) for p in peaks], dtype=np.float64)
    if len(pts) < 2:
        return np.empty(0)
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    iu = np.triu_indices(len(pts), k=1)
    return dist[iu]


def all_nearest_neighbor_distances(peaks: list[dict]) -> np.ndarray:
    pts = np.array([(p["x"], p["y"]) for p in peaks], dtype=np.float64)
    if len(pts) < 2:
        return np.empty(0)
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist, axis=1)


def nearest_neighbor_distances(peaks: list[dict], colors: tuple[str, ...]) -> np.ndarray:
    out: list[np.ndarray] = []
    for name in colors:
        pts = np.array([(p["x"], p["y"]) for p in peaks if p["color"] == name], dtype=np.float64)
        if len(pts) < 2:
            continue
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1))
        np.fill_diagonal(dist, np.inf)
        out.append(np.min(dist, axis=1))
    return np.concatenate(out) if out else np.empty(0)


def summarize(name: str, distances: np.ndarray) -> None:
    if distances.size == 0:
        print(f"{name}: no pairs")
        return
    nn_median = float(np.median(distances))
    cutoff = nn_median * 0.4
    print(f"{name}: median={nn_median:.2f}  below0.4*median={int(np.sum(distances < cutoff))} "
          f"({np.mean(distances < cutoff) * 100:.1f}%)")
    bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40, 60]
    hist, _ = np.histogram(np.clip(distances, 0, 59.999), bins=bins)
    print(f"{name}: n={distances.size} min={distances.min():.2f} "
          f"p10={np.percentile(distances, 10):.2f} median={np.median(distances):.2f} "
          f"p90={np.percentile(distances, 90):.2f}")
    print("  bins:", " ".join(f"{bins[i]}-{bins[i+1]}:{int(hist[i])}" for i in range(len(hist))))
    # Coarse mode in the 0-40 px range, on a 1 px grid.
    hist1, _ = np.histogram(distances[(distances > 0) & (distances < 40)], bins=np.arange(0, 41, 1))
    if hist1.size:
        top = np.argsort(hist1)[-6:][::-1]
        print("  top bins:", ", ".join(f"{int(i)}px:{int(hist1[i])}" for i in top if hist1[i] > 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=Path("/tmp/roi_photo_2560.jpg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks/blob_lab_data/roi_nms/distance_histogram_all.png",
    )
    args = parser.parse_args()

    source = args.image
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read {source}")
    height, width = image.shape[:2]
    widths = [640, 1280, 2560]
    if width < 1280:
        widths = [width]

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for target_width in widths:
            if width == target_width:
                path = source
            else:
                ratio = target_width / width
                resized = cv2.resize(
                    image,
                    (target_width, max(1, round(height * ratio))),
                    interpolation=cv2.INTER_AREA,
                )
                path = Path(tmp) / f"photo_{target_width}.jpg"
                cv2.imwrite(str(path), resized)
            peaks, cuts = peaks_for_image(path)
            pair = all_pairwise_distances(peaks)
            nn = all_nearest_neighbor_distances(peaks)
            pair_same = pairwise_distances(peaks, ("black", "white"))
            nn_same = nearest_neighbor_distances(peaks, ("black", "white"))
            print(f"\n{target_width}px: fitted peaks={len(peaks)} "
                  f"bg={cuts['background']} cuts={cuts['black_cut']}/{cuts['white_cut']}")
            summarize("  all pairwise", pair)
            summarize("  all nearest ", nn)
            summarize("  same nearest", nn_same)
            rows.append((target_width, pair, nn, len(peaks)))

    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.2 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = np.array([axes])
    edges = np.arange(0, HIST_MAX_PX + HIST_BIN_PX, HIST_BIN_PX)
    for row, (target_width, pair, nn, count) in enumerate(rows):
        for col, (values, label) in enumerate([(pair, "all pairs (colors mixed)"), (nn, "nearest (colors mixed)")]):
            ax = axes[row, col]
            values = values[values < HIST_MAX_PX]
            ax.hist(values, bins=edges,
                    color="#4f8cff" if col == 0 else "#ff9f1a", alpha=0.85)
            radius = max(2.0, min(8.0, target_width / 400.0))
            ax.axvline(radius, color="#e03434", linestyle="--", linewidth=1.2,
                       label=f"current NMS r={radius:.1f}px")
            if values.size:
                nn_median = float(np.median(values))
                ax.axvline(nn_median * 0.4, color="#2fbf71", linestyle=":",
                           linewidth=1.2, label="0.4 × median nearest")
            ax.set_title(f"{target_width}px · {label} · n={values.size} (peaks={count})",
                         fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("pairs")
            if row == len(rows) - 1:
                ax.set_xlabel("distance (px)")
    fig.suptitle("Fitted-peak distance histogram before NMS — all polarities", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
