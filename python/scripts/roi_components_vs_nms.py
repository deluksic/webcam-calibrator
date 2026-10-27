#!/usr/bin/env python3
"""Compare histogram-mask components with contrast/NMS peaks.

For each black/white component, count how many fitted NMS peaks land inside it.
That gives the hybrid rule:
  0 peaks -> centroid fallback
  1 peak  -> fitted peak
  2+ peaks -> merged component; keep the peaks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_histogram_nms import (  # noqa: E402
    _contrast_response,
    histogram_cuts,
    quadratic_maximum,
    roi_mask_from_checkpoint,
    segment,
)

GAP_FRACTION = 0.07
MIN_RESPONSE = 0.004
MIN_AREA = 3


def fitted_local_maxima(response: np.ndarray, threshold: float) -> list[dict]:
    local_max = cv2.dilate(response, np.ones((5, 5), dtype=np.uint8))
    maxima = (response >= threshold) & (response >= local_max - 1e-6)
    count, labels = cv2.connectedComponents(maxima.astype(np.uint8), connectivity=8)
    out: list[dict] = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        values = response[ys, xs]
        best = int(np.argmax(values))
        x, y = int(xs[best]), int(ys[best])
        fit = quadratic_maximum(response, x, y, fit_radius=1, min_curvature=0.0, min_peak=0.0)
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
        out.append({"x": fx, "y": fy, "peak": peak, "support": support})
    return out


def suppress_all_color(peaks: list[dict]) -> list[dict]:
    """Mirror the GPU: mean nearest distance, 0.6x radius, peak/support quality."""
    if len(peaks) < 2:
        return list(peaks)
    pts = np.array([(p["x"], p["y"]) for p in peaks])
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((delta * delta).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    radius = float(np.clip(0.6 * nearest.mean(), 1.5, 16.0))
    order = sorted(
        range(len(peaks)),
        key=lambda i: (-peaks[i]["peak"], -peaks[i]["support"], i),
    )
    keep: list[int] = []
    for index in order:
        x, y = peaks[index]["x"], peaks[index]["y"]
        if any((x - peaks[j]["x"]) ** 2 + (y - peaks[j]["y"]) ** 2 < radius * radius for j in keep):
            continue
        keep.append(index)
    return [peaks[i] for i in sorted(keep)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?", default=Path("/tmp/roi_photo_2560.jpg"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/roi_components_vs_nms.png"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    roi = roi_mask_from_checkpoint(args.image)
    cuts = histogram_cuts(gray, roi)
    black, white = segment(gray, roi, cuts)

    component_data: dict[str, dict] = {}
    peaks_by_color: dict[str, list[dict]] = {}
    for polarity, mask, name in [(-1, black, "black"), (1, white, "white")]:
        response = _contrast_response(gray, mask, polarity, background_sigma=2.5)
        if response is None:
            continue
        gap = cuts["background"] - cuts["black_cut"] if name == "black" else cuts["white_cut"] - cuts["background"]
        threshold = max(MIN_RESPONSE * 255.0, gap * GAP_FRACTION)
        peaks = suppress_all_color(fitted_local_maxima(response, threshold))
        peaks_by_color[name] = peaks

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        comps = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < MIN_AREA:
                continue
            cx, cy = centroids[label]
            comps.append({
                "label": label,
                "area": area,
                "x": float(cx),
                "y": float(cy),
                "bbox": tuple(int(v) for v in stats[label, :4]),
            })
        component_data[name] = {"labels": labels, "components": comps}

    total_components = 0
    total_peaks = 0
    counts = {0: 0, 1: 0, 2: 0}
    rows: list[str] = []
    for name in ("black", "white"):
        comps = component_data[name]["components"]
        labels = component_data[name]["labels"]
        peaks = peaks_by_color[name]
        total_components += len(comps)
        total_peaks += len(peaks)
        per_component: dict[int, int] = {c["label"]: 0 for c in comps}
        outside = 0
        for peak in peaks:
            xi = int(np.clip(round(peak["x"]), 0, labels.shape[1] - 1))
            yi = int(np.clip(round(peak["y"]), 0, labels.shape[0] - 1))
            label = int(labels[yi, xi])
            if label in per_component:
                per_component[label] += 1
            else:
                outside += 1
        local = {0: 0, 1: 0, 2: 0}
        area_by_count: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for c in comps:
            n = per_component[c["label"]]
            local[min(n, 2)] += 1
            counts[min(n, 2)] += 1
            area_by_count[min(n, 2)].append(c["area"])
        areas = np.array([c["area"] for c in comps])
        rows.append(
            f"{name}: components={len(comps)} peaks={len(peaks)} "
            f"0-peak={local[0]} 1-peak={local[1]} 2+-peak={local[2]} "
            f"peaks-outside-components={outside}"
        )
        if areas.size:
            rows.append(
                f"  area min/median/p90/max = {areas.min()}/{np.median(areas):.1f}/"
                f"{np.percentile(areas, 90):.1f}/{areas.max()}"
            )
        for n in (0, 1, 2):
            values = area_by_count[n]
            if values:
                rows.append(
                    f"  {n}-peak component area median/mean/max = "
                    f"{np.median(values):.1f}/{np.mean(values):.1f}/{max(values)} (n={len(values)})"
                )
    print("\n".join(rows))
    print(f"total components={total_components} total NMS peaks={total_peaks} "
          f"0-peak={counts[0]} 1-peak={counts[1]} 2+-peak={counts[2]}")

    # Overlay: component centroids green; NMS peaks blue/white; 0-peak centroids cyan;
    # merged components yellow boxes.
    display = np.repeat(gray[:, :, None], 3, axis=2)
    for name, color in [("black", (0.1, 0.5, 1.0)), ("white", (1.0, 0.75, 0.1))]:
        for peak in peaks_by_color[name]:
            cv2.circle(display, (int(round(peak["x"])), int(round(peak["y"]))), 2, color, 1)
        labels = component_data[name]["labels"]
        per_component: dict[int, int] = {}
        for c in component_data[name]["components"]:
            per_component[c["label"]] = 0
        for peak in peaks_by_color[name]:
            xi = int(np.clip(round(peak["x"]), 0, labels.shape[1] - 1))
            yi = int(np.clip(round(peak["y"]), 0, labels.shape[0] - 1))
            label = int(labels[yi, xi])
            if label in per_component:
                per_component[label] += 1
        for c in component_data[name]["components"]:
            n = per_component[c["label"]]
            if n == 0:
                cv2.drawMarker(display, (int(round(c["x"])), int(round(c["y"]))),
                               (0.0, 1.0, 1.0), cv2.MARKER_CROSS, 9, 1)
            elif n >= 2:
                x, y, w, h = c["bbox"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (1.0, 1.0, 0.0), 1)
    cv2.imwrite(str(args.output), cv2.cvtColor(np.clip(display * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
