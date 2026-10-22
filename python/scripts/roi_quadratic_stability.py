"""Compare raw NMS peak values with quadratic-fit peak values.

For baseline candidates, measures:
  * correlation / bias of fitted vs raw peak height
  * per-candidate stability under synthetic sensor noise
  * quadratic-fit failure rate

The point is to decide whether the fitted maximum is a better detection score
than the raw local maximum before using it as a height gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blob_lab.heatmap_dataset import HeatmapDatasetConfig, generate_heatmap_dataset
from blob_lab.peaks import detect_peaks
from blob_lab.target_layout import build_hex_target_layout

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_histogram_nms import (  # noqa: E402
    _contrast_response,
    histogram_cuts,
    load_gray,
    load_mask,
    quadratic_maximum,
    segment,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_nms" / "quadratic_stability.json"


def _candidates(response: np.ndarray):
    if response is None:
        return []
    peaks = detect_peaks(response, threshold=0.2, nms_radius=2)
    return [(int(round(peak.x)), int(round(peak.y))) for peak in peaks]


def _values(response: np.ndarray, candidates):
    raw = []
    fitted = []
    for x, y in candidates:
        y0, y1 = max(0, y - 1), min(response.shape[0], y + 2)
        x0, x1 = max(0, x - 1), min(response.shape[1], x + 2)
        raw.append(float(response[y0:y1, x0:x1].max()))
        fit = quadratic_maximum(
            response, x, y, fit_radius=1, min_curvature=0.0, min_peak=-1e9
        )
        fitted.append(fit[2] if fit is not None else np.nan)
    return np.asarray(raw), np.asarray(fitted)


def analyze_case(
    name: str,
    gray: np.ndarray,
    black: np.ndarray,
    white: np.ndarray,
    sigmas=(0.005, 0.01, 0.02),
    trials: int = 6,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    results = {"name": name, "polarities": []}
    for polarization, mask in (("black", black), ("white", white)):
        response = _contrast_response(gray, mask, -1 if polarization == "black" else 1)
        candidates = _candidates(response)
        if not candidates:
            continue
        raw0, fit0 = _values(response, candidates)
        valid = ~np.isnan(fit0)

        raw_trials = []
        fit_trials = []
        for sigma in sigmas:
            for _ in range(trials):
                noisy = np.clip(
                    gray + rng.normal(0.0, sigma, gray.shape).astype(np.float32), 0.0, 1.0
                )
                noisy_response = _contrast_response(
                    noisy, mask, -1 if polarization == "black" else 1
                )
                raw, fitted = _values(noisy_response, candidates)
                raw_trials.append(raw)
                fit_trials.append(fitted)
        raw_trials = np.asarray(raw_trials)
        fit_trials = np.asarray(fit_trials)

        raw_std = np.nanstd(raw_trials, axis=0)
        fit_std = np.nanstd(fit_trials, axis=0)
        raw_mean = np.nanmean(raw_trials, axis=0)
        fit_mean = np.nanmean(fit_trials, axis=0)
        valid_all = valid & ~np.isnan(fit_trials).any(axis=0)

        entry = {
            "polarity": polarization,
            "candidates": len(candidates),
            "fit_valid_baseline": int(valid.sum()),
            "fit_failure_baseline": int((~valid).sum()),
            "fit_failure_under_noise": int((~valid_all).sum()),
            "correlation": float(np.corrcoef(raw0[valid], fit0[valid])[0, 1])
            if valid.sum() > 2
            else None,
            "raw_mean": float(np.mean(raw0[valid])) if valid.any() else None,
            "fit_mean": float(np.mean(fit0[valid])) if valid.any() else None,
            "bias_percent": float(
                100.0 * (np.mean(fit0[valid]) - np.mean(raw0[valid])) / max(1e-6, np.mean(raw0[valid]))
            )
            if valid.any()
            else None,
            "median_std_raw": float(np.nanmedian(raw_std[valid_all])) if valid_all.any() else None,
            "median_std_fit": float(np.nanmedian(fit_std[valid_all])) if valid_all.any() else None,
            "median_cv_raw": float(
                np.nanmedian(raw_std[valid_all] / np.maximum(np.abs(raw_mean[valid_all]), 1e-6))
            )
            if valid_all.any()
            else None,
            "median_cv_fit": float(
                np.nanmedian(fit_std[valid_all] / np.maximum(np.abs(fit_mean[valid_all]), 1e-6))
            )
            if valid_all.any()
            else None,
            "stability_ratio_std": float(
                np.nanmedian(fit_std[valid_all]) / max(1e-6, np.nanmedian(raw_std[valid_all]))
            )
            if valid_all.any()
            else None,
        }
        results["polarities"].append(entry)
    return results


def main() -> None:
    cases = []
    for width, height, samples, seed in ((640, 480, 2, 7), (1280, 960, 1, 11)):
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
        for index in range(len(dataset)):
            gray = np.asarray(dataset.images[index, :, :, 0], dtype=np.float32)
            roi = np.asarray(dataset.masks[index, :, :, 0]) >= 0.5
            cuts = histogram_cuts(gray, roi)
            black, white = segment(gray, roi, cuts)
            cases.append((f"synthetic_{width}x{height}_{index:02d}", gray, black, white))

    real_root = ROOT / "python" / "notebooks" / "real_positives"
    for name, image, mask in (
        ("real_positive_01", "real_positive_01.jpg", "real_positive_01_mask.png"),
        ("real_positive_02", "real_positive_02.png", "real_positive_02_mask.png"),
    ):
        image_path = real_root / image
        mask_path = real_root / mask
        if not image_path.is_file():
            continue
        gray = load_gray(image_path)
        roi = load_mask(mask_path, gray.shape)
        cuts = histogram_cuts(gray, roi)
        black, white = segment(gray, roi, cuts)
        cases.append((name, gray, black, white))

    results = [analyze_case(name, gray, black, white) for name, gray, black, white in cases]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")
    for result in results:
        print(result["name"])
        for entry in result["polarities"]:
            print(
                f"  {entry['polarity']:5s} n={entry['candidates']:4d} "
                f"fail={entry['fit_failure_baseline']:3d} "
                f"corr={entry['correlation']:.3f} "
                f"bias={entry['bias_percent']:+.1f}% "
                f"std raw={entry['median_std_raw']:.3f} fit={entry['median_std_fit']:.3f} "
                f"ratio={entry['stability_ratio_std']:.3f} "
                f"cv raw={entry['median_cv_raw']:.4f} fit={entry['median_cv_fit']:.4f}"
            )


if __name__ == "__main__":
    main()
