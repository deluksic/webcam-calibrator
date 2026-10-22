"""ROI histogram mask + NMS blob-center experiment.

Reproduces the browser ROI-lab segmentation in Python:

  full-res gray -> ROI mask -> masked luma histogram -> peak/noise-floor cuts
  -> black/white masks -> connected components or NMS centers.

The synthetic part uses the existing hex-target renderer and reports center
metrics against known marker positions. The real part runs on the existing
hand-masked hex-target photos and on the demo images (ROI net for the mask).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blob_lab.heatmap_dataset import HeatmapDatasetConfig, generate_heatmap_dataset
from blob_lab.peaks import Peak, detect_peaks, match_peaks
from blob_lab.target_layout import build_hex_target_layout

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_nms"
ONNX_MODEL = ROOT / "public" / "models" / "roi_tiny_no_concat.onnx"


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.0


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask >= 127


def roi_mask_from_onnx(image_path: Path, model_path: Path = ONNX_MODEL) -> np.ndarray:
    import onnxruntime as ort

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    stats_small = cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA)
    stats_luma = stats_small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mean = float(stats_luma.mean())
    std = max(0.05, float(stats_luma.std()))

    model_width = 256
    raw_height = round(model_width * height / width)
    model_height = min(224, max(96, math.ceil(raw_height / 16) * 16))
    resized = cv2.resize(rgb, (model_width, model_height), interpolation=cv2.INTER_LINEAR)
    luma = resized @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    normalized = ((luma - mean) / std)[None, None].astype(np.float32)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output = session.run(["mask"], {"input": normalized})[0]
    mask = np.asarray(output[0, 0], dtype=np.float32)
    full = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return full >= 0.5


def histogram_cuts(gray: np.ndarray, roi: np.ndarray, noise_floor_fraction: float = 0.02) -> dict:
    values = np.clip(np.floor(gray[roi] * 255.0 + 0.5), 0, 255).astype(np.int32)
    hist = np.bincount(values, minlength=256).astype(np.float64)
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64)
    kernel /= kernel.sum()
    smoothed = np.convolve(hist, kernel, mode="same")
    background = int(np.argmax(smoothed))
    floor = float(smoothed[background] * noise_floor_fraction)

    black_cut = 0
    for bin_index in range(background - 1, -1, -1):
        if smoothed[bin_index] <= floor:
            black_cut = bin_index + 1
            break
    white_cut = 256
    for bin_index in range(background + 1, 256):
        if smoothed[bin_index] <= floor:
            white_cut = bin_index
            break
    return {
        "hist": hist,
        "smoothed": smoothed,
        "background": background,
        "black_cut": black_cut,
        "white_cut": white_cut,
        "masked_pixels": int(values.size),
    }


def segment(gray: np.ndarray, roi: np.ndarray, cuts: dict) -> tuple[np.ndarray, np.ndarray]:
    bins = np.clip(np.floor(gray * 255.0 + 0.5), 0, 255).astype(np.int32)
    black = roi & (bins < cuts["black_cut"])
    white = roi & (bins >= cuts["white_cut"])
    return black, white


def connected_component_centers(mask: np.ndarray) -> list[tuple[float, float, int]]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    centers: list[tuple[float, float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 1:
            continue
        x, y = centroids[index]
        centers.append((float(x), float(y), area))
    return centers


def refine_peak(response: np.ndarray, x: int, y: int) -> tuple[float, float]:
    height, width = response.shape
    if x <= 0 or x >= width - 1 or y <= 0 or y >= height - 1:
        return float(x), float(y)
    center = float(response[y, x])
    denom_x = float(response[y, x - 1]) - 2.0 * center + float(response[y, x + 1])
    denom_y = float(response[y - 1, x]) - 2.0 * center + float(response[y + 1, x])
    dx = 0.0
    dy = 0.0
    if abs(denom_x) > 1e-6:
        dx = 0.5 * (float(response[y, x - 1]) - float(response[y, x + 1])) / denom_x
    if abs(denom_y) > 1e-6:
        dy = 0.5 * (float(response[y - 1, x]) - float(response[y + 1, x])) / denom_y
    if not (-1.0 <= dx <= 1.0):
        dx = 0.0
    if not (-1.0 <= dy <= 1.0):
        dy = 0.0
    return float(x) + dx, float(y) + dy


def nms_distance_centers(mask: np.ndarray, nms_radius: int) -> list[tuple[float, float, float]]:
    if not mask.any():
        return []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = detect_peaks(distance, threshold=0.5, nms_radius=nms_radius)
    centers = []
    for peak in peaks:
        x, y = refine_peak(distance, int(round(peak.x)), int(round(peak.y)))
        centers.append((x, y, float(distance[int(round(y)), int(round(x))])))
    return centers


def nms_intensity_centers(
    gray: np.ndarray,
    mask: np.ndarray,
    cut: int,
    polarity: int,
    nms_radius: int,
) -> list[tuple[float, float, float]]:
    if not mask.any():
        return []
    value = gray * 255.0
    response = np.where(mask, (cut - value) if polarity < 0 else (value - cut), 0.0)
    response = np.maximum(response, 0.0)
    response = cv2.GaussianBlur(response, (0, 0), 0.8)
    peaks = detect_peaks(response, threshold=0.5, nms_radius=nms_radius)
    centers = []
    for peak in peaks:
        x, y = refine_peak(response, int(round(peak.x)), int(round(peak.y)))
        centers.append((x, y, float(peak.score)))
    return centers


def quadratic_maximum(
    response: np.ndarray,
    x: int,
    y: int,
    fit_radius: int = 1,
    min_curvature: float = 0.0,
    min_peak: float = 0.0,
) -> tuple[float, float, float] | None:
    """Fit z = ax^2 + bxy + cy^2 + dx + ey + f around a candidate.

    Keep only negative-definite Hessians (a true local maximum) whose fitted
    vertex is inside the window. The vertex is also the subpixel center.
    """
    height, width = response.shape
    x0 = max(0, x - fit_radius)
    x1 = min(width, x + fit_radius + 1)
    y0 = max(0, y - fit_radius)
    y1 = min(height, y + fit_radius + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float64) - x,
        np.arange(y0, y1, dtype=np.float64) - y,
    )
    values = response[y0:y1, x0:x1].astype(np.float64)
    design = np.stack(
        [
            (grid_x * grid_x).ravel(),
            (grid_x * grid_y).ravel(),
            (grid_y * grid_y).ravel(),
            grid_x.ravel(),
            grid_y.ravel(),
            np.ones(grid_x.size),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(design, values.ravel(), rcond=None)
    a, b, c, d, e, _f = coefficients
    hessian = np.array([[2.0 * a, b], [b, 2.0 * c]])
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[1] >= 0.0 or -eigenvalues[1] < min_curvature:
        return None
    try:
        offset = np.linalg.solve(hessian, -np.array([d, e]))
    except np.linalg.LinAlgError:
        return None
    if abs(offset[0]) > fit_radius + 0.5 or abs(offset[1]) > fit_radius + 0.5:
        return None
    vx, vy = float(offset[0]), float(offset[1])
    peak = float(
        a * vx * vx + b * vx * vy + c * vy * vy + d * vx + e * vy + _f
    )
    if peak < min_peak:
        return None
    return float(x) + vx, float(y) + vy, peak


def _contrast_response(
    gray: np.ndarray,
    mask: np.ndarray,
    polarity: int,
    background_sigma: float = 3.0,
) -> np.ndarray | None:
    if not mask.any():
        return None
    background = cv2.GaussianBlur(gray, (0, 0), background_sigma)
    contrast = (gray - background) * 255.0
    response = np.where(mask, contrast if polarity > 0 else -contrast, 0.0)
    response = np.maximum(response, 0.0)
    return cv2.GaussianBlur(response, (0, 0), 0.6)


def nms_contrast_raw_centers(
    gray: np.ndarray,
    mask: np.ndarray,
    polarity: int,
    nms_radius: int,
) -> list[tuple[float, float, float]]:
    response = _contrast_response(gray, mask, polarity)
    if response is None:
        return []
    peaks = detect_peaks(response, threshold=0.2, nms_radius=nms_radius)
    return [
        (*refine_peak(response, int(round(peak.x)), int(round(peak.y))), float(peak.score))
        for peak in peaks
    ]


def nms_contrast_centers(
    gray: np.ndarray,
    mask: np.ndarray,
    polarity: int,
    nms_radius: int,
    fit_radius: int = 1,
    min_curvature: float = 0.0,
    min_peak: float = 1.0,
    peak_fraction: float = 0.2,
) -> list[tuple[float, float, float]]:
    """Local center-surround contrast -> NMS -> quadratic fit -> dynamic height gate.

    The height gate is a fraction of this color's median fitted peak, so it
    adapts to exposure/target contrast. `min_peak` is an absolute floor.
    """
    response = _contrast_response(gray, mask, polarity)
    if response is None:
        return []
    peaks = detect_peaks(response, threshold=0.2, nms_radius=nms_radius)
    centers = []
    for peak in peaks:
        fitted = quadratic_maximum(
            response,
            int(round(peak.x)),
            int(round(peak.y)),
            fit_radius=fit_radius,
            min_curvature=min_curvature,
            min_peak=0.0,
        )
        if fitted is not None:
            centers.append(fitted)
    if not centers:
        return []
    heights = np.asarray([center[2] for center in centers], dtype=np.float64)
    threshold = max(min_peak, float(np.median(heights)) * peak_fraction)
    return [center for center in centers if center[2] >= threshold]


def median_nearest(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1.0
    differences = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((differences * differences).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def evaluate(
    name: str,
    gray: np.ndarray,
    roi: np.ndarray,
    truth: list[tuple[float, float]],
    cuts: dict,
    black: np.ndarray,
    white: np.ndarray,
    nms_radius: int,
    fit_radius: int = 1,
    min_curvature: float = 0.0,
    min_peak: float = 1.0,
    peak_fraction: float = 0.2,
) -> dict:
    black_cc = connected_component_centers(black)
    white_cc = connected_component_centers(white)
    cc_points = [(x, y) for x, y, _area in black_cc] + [(x, y) for x, y, _area in white_cc]

    black_nms = nms_distance_centers(black, nms_radius)
    white_nms = nms_distance_centers(white, nms_radius)
    nms_points = [(x, y) for x, y, _score in black_nms] + [
        (x, y) for x, y, _score in white_nms
    ]

    black_int = nms_intensity_centers(gray, black, cuts["black_cut"], -1, nms_radius)
    white_int = nms_intensity_centers(gray, white, cuts["white_cut"], +1, nms_radius)
    intensity_points = [(x, y) for x, y, _score in black_int] + [
        (x, y) for x, y, _score in white_int
    ]

    black_contrast = nms_contrast_centers(
        gray,
        black,
        -1,
        nms_radius,
        fit_radius=fit_radius,
        min_curvature=min_curvature,
        min_peak=min_peak,
        peak_fraction=peak_fraction,
    )
    white_contrast = nms_contrast_centers(
        gray,
        white,
        +1,
        nms_radius,
        fit_radius=fit_radius,
        min_curvature=min_curvature,
        min_peak=min_peak,
        peak_fraction=peak_fraction,
    )
    contrast_points = [(x, y) for x, y, _score in black_contrast] + [
        (x, y) for x, y, _score in white_contrast
    ]

    black_raw = nms_contrast_raw_centers(gray, black, -1, nms_radius)
    white_raw = nms_contrast_raw_centers(gray, white, +1, nms_radius)
    contrast_raw_points = [(x, y) for x, y, _score in black_raw] + [
        (x, y) for x, y, _score in white_raw
    ]

    truth_array = np.asarray(truth, dtype=np.float32).reshape(-1, 2)
    tolerance = min(2.0, max(1.0, 0.3 * median_nearest(truth_array))) if len(truth) else 1.5
    truth_list = [(float(x), float(y)) for x, y in truth]

    def metrics(points: list[tuple[float, float]]):
        return match_peaks(
            [Peak(float(x), float(y), 1.0) for x, y in points],
            truth_list,
            tolerance=tolerance,
        )

    result = {
        "name": name,
        "size": [int(gray.shape[1]), int(gray.shape[0])],
        "roi_pixels": int(roi.sum()),
        "masked_pixels": cuts["masked_pixels"],
        "background": cuts["background"],
        "black_cut": cuts["black_cut"],
        "white_cut": cuts["white_cut"],
        "truth": len(truth),
        "cc_count": len(cc_points),
        "nms_distance_count": len(nms_points),
        "nms_intensity_count": len(intensity_points),
        "nms_contrast_count": len(contrast_points),
        "nms_contrast_raw_count": len(contrast_raw_points),
        "tolerance": tolerance,
        "cc": metrics(cc_points).__dict__,
        "nms_distance": metrics(nms_points).__dict__,
        "nms_intensity": metrics(intensity_points).__dict__,
        "nms_contrast": metrics(contrast_points).__dict__,
        "nms_contrast_raw": metrics(contrast_raw_points).__dict__,
    }
    return result


def draw_overlay(
    path: Path,
    gray: np.ndarray,
    roi: np.ndarray,
    black: np.ndarray,
    white: np.ndarray,
    truth: list[tuple[float, float]],
    cc_points: list[tuple[float, float]],
    nms_points: list[tuple[float, float]],
    title: str,
) -> None:
    display = np.repeat(np.clip(gray, 0, 1)[:, :, None], 3, axis=2)
    display[black] = 0.45 * display[black] + 0.55 * np.array([0.10, 0.40, 1.0])
    display[white] = 0.45 * display[white] + 0.55 * np.array([1.0, 0.70, 0.10])
    contours, _ = cv2.findContours(roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (0.1, 1.0, 0.2), 1)
    for x, y in truth:
        cv2.drawMarker(display, (int(round(x)), int(round(y))), (1.0, 1.0, 0.1), cv2.MARKER_CROSS, 3, 1)
    for x, y in cc_points:
        cv2.circle(display, (int(round(x)), int(round(y))), 2, (1.0, 0.15, 0.15), 1)
    for x, y in nms_points:
        cv2.circle(display, (int(round(x)), int(round(y))), 2, (0.1, 1.0, 0.3), 1)
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

    ys, xs = np.where(roi)
    if len(xs) == 0:
        return
    pad = 12
    x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad)
    crop = display[y0:y1, x0:x1]
    if crop.size:
        scale = max(1, int(round(720 / max(1, crop.shape[1]))))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        crop_bgr = cv2.cvtColor(np.clip(crop * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path.with_name(path.stem + "_zoom.png")), crop_bgr)


def run_synthetic(
    width: int,
    height: int,
    samples: int,
    seed: int,
    out_dir: Path,
    fit_radius: int = 1,
    min_curvature: float = 0.0,
    min_peak: float = 1.0,
    peak_fraction: float = 0.2,
) -> list[dict]:
    layout = build_hex_target_layout(width_px=width, height_px=height)
    config = HeatmapDatasetConfig(
        output_width=width,
        output_height=height,
        samples=samples,
        natural_fraction=0.0,
        randomize_colors=False,
        blur_sigma=(0.8, 1.6),
        noise_sigma=(0.0, 0.01),
        seed=seed,
    )
    dataset = generate_heatmap_dataset(config, layout, count=samples)
    results = []
    for index in range(len(dataset)):
        gray = np.asarray(dataset.images[index, :, :, 0], dtype=np.float32)
        roi = np.asarray(dataset.masks[index, :, :, 0]) >= 0.5
        truth = [(float(x), float(y)) for x, y in dataset.metadata[index]["points"]]
        cuts = histogram_cuts(gray, roi)
        black, white = segment(gray, roi, cuts)
        result = evaluate(
            f"synthetic_{width}x{height}_{index:02d}",
            gray,
            roi,
            truth,
            cuts,
            black,
            white,
            2,
            fit_radius=fit_radius,
            min_curvature=min_curvature,
            min_peak=min_peak,
            peak_fraction=peak_fraction,
        )
        result["marker_side_px"] = layout.side_px
        result["spacing_px"] = layout.spacing_px
        result["blur_sigma"] = float(dataset.metadata[index].get("blur_sigma", 0.0))
        results.append(result)

        cc_points = [(x, y) for x, y, _ in connected_component_centers(black)] + [
            (x, y) for x, y, _ in connected_component_centers(white)
        ]
        contrast_points = [
            (x, y)
            for x, y, _ in nms_contrast_centers(
                gray, black, -1, 2, fit_radius=fit_radius, min_curvature=min_curvature, min_peak=min_peak, peak_fraction=peak_fraction
            )
        ] + [
            (x, y)
            for x, y, _ in nms_contrast_centers(
                gray, white, 1, 2, fit_radius=fit_radius, min_curvature=min_curvature, min_peak=min_peak, peak_fraction=peak_fraction
            )
        ]
        draw_overlay(
            out_dir / f"{result['name']}.png",
            gray,
            roi,
            black,
            white,
            truth,
            cc_points,
            contrast_points,
            f"{width}x{height} cc={result['cc_count']} nmsC={result['nms_contrast_count']} gt={len(truth)}",
        )
    return results


def run_real_case(
    name: str,
    image_path: Path,
    mask_path: Path | None,
    out_dir: Path,
    fit_radius: int = 1,
    min_curvature: float = 0.0,
    min_peak: float = 1.0,
    peak_fraction: float = 0.2,
) -> dict:
    gray = load_gray(image_path)
    if mask_path is not None:
        roi = load_mask(mask_path, gray.shape)
    else:
        roi = roi_mask_from_onnx(image_path)
    cuts = histogram_cuts(gray, roi)
    black, white = segment(gray, roi, cuts)
    result = evaluate(
        name,
        gray,
        roi,
        [],
        cuts,
        black,
        white,
        2,
        fit_radius=fit_radius,
        min_curvature=min_curvature,
        min_peak=min_peak,
        peak_fraction=peak_fraction,
    )
    cc_points = [(x, y) for x, y, _ in connected_component_centers(black)] + [
        (x, y) for x, y, _ in connected_component_centers(white)
    ]
    contrast_points = [
        (x, y)
        for x, y, _ in nms_contrast_centers(
            gray, black, -1, 2, fit_radius=fit_radius, min_curvature=min_curvature, min_peak=min_peak, peak_fraction=peak_fraction
        )
    ] + [
        (x, y)
        for x, y, _ in nms_contrast_centers(
            gray, white, 1, 2, fit_radius=fit_radius, min_curvature=min_curvature, min_peak=min_peak, peak_fraction=peak_fraction
        )
    ]
    draw_overlay(
        out_dir / f"{name}.png",
        gray,
        roi,
        black,
        white,
        [],
        cc_points,
        contrast_points,
        f"{name} cc={result['cc_count']} nmsC={result['nms_contrast_count']} cuts={cuts['black_cut']}/{cuts['white_cut']}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--fit-radius", type=int, default=1)
    parser.add_argument("--min-curvature", type=float, default=0.0)
    parser.add_argument("--min-peak", type=float, default=1.0)
    parser.add_argument("--peak-fraction", type=float, default=0.2)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {"synthetic": [], "real": []}
    if not args.skip_synthetic:
        results["synthetic"] += run_synthetic(
            640,
            480,
            args.samples,
            7,
            args.out,
            args.fit_radius,
            args.min_curvature,
            args.min_peak,
            args.peak_fraction,
        )
        results["synthetic"] += run_synthetic(
            1280,
            960,
            max(2, args.samples // 2),
            11,
            args.out,
            args.fit_radius,
            args.min_curvature,
            args.min_peak,
            args.peak_fraction,
        )

    if not args.skip_real:
        real_cases = [
            (
                "real_positive_01",
                ROOT / "python" / "notebooks" / "real_positives" / "real_positive_01.jpg",
                ROOT / "python" / "notebooks" / "real_positives" / "real_positive_01_mask.png",
            ),
            (
                "real_positive_02",
                ROOT / "python" / "notebooks" / "real_positives" / "real_positive_02.png",
                ROOT / "python" / "notebooks" / "real_positives" / "real_positive_02_mask.png",
            ),
            (
                "demo_target",
                ROOT / "public" / "demo" / "roi_target.png",
                None,
            ),
            (
                "demo_real_01",
                ROOT / "public" / "demo" / "roi_real_01.jpg",
                None,
            ),
        ]
        for name, image_path, mask_path in real_cases:
            if not image_path.is_file():
                continue
            results["real"].append(
                run_real_case(
                    name,
                    image_path,
                    mask_path,
                    args.out,
                    args.fit_radius,
                    args.min_curvature,
                    args.min_peak,
                    args.peak_fraction,
                )
            )

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {summary_path}")

    print("\nSYNTHETIC")
    for result in results["synthetic"]:
        print(
            f"{result['name']:28s} gt={result['truth']:4d} "
            f"cc={result['cc_count']:4d} "
            f"ccF1={result['cc']['f1']:.3f} "
            f"nmsD={result['nms_distance_count']:4d} "
            f"nmsDF1={result['nms_distance']['f1']:.3f} "
            f"nmsI={result['nms_intensity_count']:4d} "
            f"nmsIF1={result['nms_intensity']['f1']:.3f} "
            f"nmsC={result['nms_contrast_count']:4d} "
            f"nmsCF1={result['nms_contrast']['f1']:.3f} "
            f"rawC={result['nms_contrast_raw_count']:4d} "
            f"rawCF1={result['nms_contrast_raw']['f1']:.3f} "
            f"cuts={result['black_cut']}/{result['white_cut']}"
        )
    print("\nREAL")
    for result in results["real"]:
        print(
            f"{result['name']:28s} roi={result['roi_pixels']:7d} "
            f"cc={result['cc_count']:5d} nmsD={result['nms_distance_count']:5d} "
            f"nmsI={result['nms_intensity_count']:5d} nmsC={result['nms_contrast_count']:5d} "
            f"rawC={result['nms_contrast_raw_count']:5d} "
            f"cuts={result['black_cut']}/{result['white_cut']}"
        )


if __name__ == "__main__":
    main()
