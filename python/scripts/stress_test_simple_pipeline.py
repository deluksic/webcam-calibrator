#!/usr/bin/env python3
"""Stress-test the simplified local-peak pipeline over many examples.

Synthetic: the 14-case angle suite, clean and camera-sharpened, with exact truth.
Real: the two annotated real positives and the unannotated real hex photo.
Negatives: office/wall images with no target.

The candidate stage is run once per example (raw high-pass blob class, sigma 1.0
blur, 3x3 extrema, 5x5 edge-clamped cubic, no border crop). The neighbour filter
is evaluated at 20 and 40 same-polarity neighbours, trim 2 sigma, accept 4 sigma.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/webcam-calibrator-mpl")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
from simple_neighbor_pipeline import (  # noqa: E402
    _load_manual_mask,
    _neighbor_filter,
    _roi_and_background,
)
from simple_patch_pipeline import _candidates, _fit_cubic_5x5  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "python" / "notebooks" / "blob_lab_data" / "simple_neighbor_pipeline" / "stress"
NATURAL = ROOT / "python" / "notebooks" / "natural_images"
REAL_POSITIVES = ROOT / "python" / "notebooks" / "real_positives"
REAL_UNANNOTATED = ROOT / "python" / "notebooks" / "test_images" / "real" / "roi_real_01.jpg"
NEIGHBOUR_COUNTS = (20, 40)
TRIM_SIGMA = 2.0
ACCEPT_SIGMA = 4.0


@dataclass
class CandidateResult:
    raw: int
    accepted: list[dict[str, Any]]
    saddles: int
    failures: int


def _candidate_stage(processed: np.ndarray, mask: np.ndarray) -> CandidateResult:
    blurred = cv2.GaussianBlur(
        processed,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
        borderType=cv2.BORDER_REFLECT,
    )
    if not mask.any():
        return CandidateResult(raw=0, accepted=[], saddles=0, failures=0)
    cuts = histogram_cuts(blurred, mask)
    # Raw high-pass class; the blurred image only supplies the 3x3 extrema and
    # the 5x5 fit. No border crop.
    black_mask, white_mask = segment(processed, mask, cuts)
    raw_candidates: list[dict[str, Any]] = []
    for polarity, support in ((0, black_mask), (1, white_mask)):
        for x, y in _candidates(blurred, support, polarity):
            raw_candidates.append({"x": float(x), "y": float(y), "polarity": polarity})
    accepted: list[dict[str, Any]] = []
    saddles = 0
    failures = 0
    for candidate in raw_candidates:
        fit = _fit_cubic_5x5(blurred, int(candidate["x"]), int(candidate["y"]))
        if fit is None:
            failures += 1
            continue
        eigen_1 = float(fit["eigenvalue_1"])
        eigen_2 = float(fit["eigenvalue_2"])
        definite = (
            eigen_1 < -1e-7 and eigen_2 < -1e-7
            if candidate["polarity"] == 1
            else eigen_1 > 1e-7 and eigen_2 > 1e-7
        )
        if definite:
            accepted.append({**candidate, **fit})
        else:
            saddles += 1
    return CandidateResult(raw=len(raw_candidates), accepted=accepted, saddles=saddles, failures=failures)


def _match_metrics(
    points: list[dict[str, Any]],
    truth: np.ndarray,
    *,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    if truth.size == 0:
        return {
            "predicted": len(points),
            "truth": 0,
            "tp": 0,
            "fp": len(points),
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_error_px": 0.0,
            "median_error_px": 0.0,
            "p90_error_px": 0.0,
            "errors": [],
        }
    if not points:
        return {
            "predicted": 0,
            "truth": int(truth.shape[0]),
            "tp": 0,
            "fp": 0,
            "fn": int(truth.shape[0]),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_error_px": 0.0,
            "median_error_px": 0.0,
            "p90_error_px": 0.0,
            "errors": [],
        }
    coordinates = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float64)
    distances = point_distances(coordinates, truth)
    matches = match_from_distances(distances, tolerance)
    tp = len(matches)
    fp = len(points) - tp
    fn = int(truth.shape[0]) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.asarray([match[0] for match in matches], dtype=np.float64)
    return {
        "predicted": len(points),
        "truth": int(truth.shape[0]),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_error_px": float(errors.mean()) if errors.size else 0.0,
        "median_error_px": float(np.median(errors)) if errors.size else 0.0,
        "p90_error_px": float(np.percentile(errors, 90)) if errors.size else 0.0,
        "errors": [float(value) for value in errors],
    }


def _visible_truth(truth: np.ndarray, shape: tuple[int, int], margin: float = 8.0) -> np.ndarray:
    if truth.size == 0:
        return truth
    height, width = shape
    return truth[
        (truth[:, 0] >= margin)
        & (truth[:, 0] <= width - 1 - margin)
        & (truth[:, 1] >= margin)
        & (truth[:, 1] <= height - 1 - margin)
    ]


def _evaluate_candidates(
    result: CandidateResult,
    truth: np.ndarray,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "raw": result.raw,
        "accepted": len(result.accepted),
        "saddles": result.saddles,
        "fit_failures": result.failures,
        "accepted_metrics": _match_metrics(result.accepted, truth),
        "kept": {},
    }
    for neighbours in NEIGHBOUR_COUNTS:
        filter_result = _neighbor_filter(
            result.accepted,
            neighbours=neighbours,
            sigma=ACCEPT_SIGMA,
            trim_sigma=TRIM_SIGMA,
        )
        kept = [
            point for point, keep in zip(result.accepted, filter_result["keep"]) if keep
        ]
        entry["kept"][str(neighbours)] = {
            "kept": len(kept),
            "removed": len(result.accepted) - len(kept),
            "metrics": _match_metrics(kept, truth),
            "per_polarity": filter_result["per_polarity"],
            "rejection_by_feature": filter_result["rejection_by_feature"],
        }
    return entry


def run_synthetic(out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    specs = build_case_specs(
        seed=7,
        samples_scale=1.0,
        include_large=True,
        max_cases=None,
    )
    for sharpen_label, sharpen_probability in (("clean", 0.0), ("sharpened", 0.7)):
        for spec in specs:
            config = make_config(spec, sharpen_probability=sharpen_probability)
            layout = build_hex_target_layout(width_px=spec.width, height_px=spec.height)
            try:
                dataset = generate_heatmap_dataset(config, layout, count=spec.samples)
            except RuntimeError as error:
                print(f"[warn] {spec.name} {sharpen_label}: {error}")
                continue
            for sample in range(len(dataset)):
                gray = np.asarray(dataset.images[sample, :, :, 0], dtype=np.float32)
                mask = np.asarray(dataset.masks[sample, :, :, 0], dtype=np.float32) > 0.5
                truth = np.asarray(
                    dataset.metadata[sample].get("points") or [], dtype=np.float64
                ).reshape(-1, 2)
                truth = _visible_truth(truth, gray.shape)
                result = _candidate_stage(gray, mask)
                entry = _evaluate_candidates(result, truth)
                entry.update(
                    {
                        "kind": "synthetic",
                        "name": f"{spec.name}_{sharpen_label}_{sample:02d}",
                        "group": spec.group,
                        "width": spec.width,
                        "height": spec.height,
                        "sharpen": sharpen_label,
                        "sharpen_meta": dataset.metadata[sample].get("sharpen"),
                    }
                )
                records.append(entry)
                print(
                    f"[synthetic] {entry['name']:38s} truth={truth.shape[0]:4d} "
                    f"accepted={entry['accepted']:4d} "
                    f"kept20={entry['kept']['20']['kept']:4d}/{entry['kept']['20']['metrics']['f1']:.3f} "
                    f"kept40={entry['kept']['40']['kept']:4d}/{entry['kept']['40']['metrics']['f1']:.3f}"
                )
    (out_dir / "synthetic.json").write_text(json.dumps(records, indent=2))
    return records


def run_real(out_dir: Path) -> list[dict[str, Any]]:
    cases = [
        (REAL_POSITIVES / "real_positive_01.jpg", REAL_POSITIVES / "real_positive_01_mask.png"),
        (REAL_POSITIVES / "real_positive_02.png", REAL_POSITIVES / "real_positive_02_mask.png"),
        (REAL_UNANNOTATED, None),
    ]
    records: list[dict[str, Any]] = []
    for image_path, manual_path in cases:
        prep = _roi_and_background(image_path, residual_gain=1.0)
        result = _candidate_stage(prep["flattened"], prep["roi_mask"])
        empty_truth = np.zeros((0, 2), dtype=np.float64)
        entry = _evaluate_candidates(result, empty_truth)
        manual_mask = _load_manual_mask(manual_path, prep["gray"].shape)
        coverage = None
        if manual_mask is not None and result.accepted:
            inside = 0
            for point in result.accepted:
                x = int(np.clip(round(point["x"]), 0, manual_mask.shape[1] - 1))
                y = int(np.clip(round(point["y"]), 0, manual_mask.shape[0] - 1))
                inside += int(manual_mask[y, x])
            coverage = inside / len(result.accepted)
        entry.update(
            {
                "kind": "real",
                "name": image_path.name,
                "path": str(image_path),
                "manual_mask": str(manual_path) if manual_path else None,
                "mask_pixels": int(prep["roi_mask"].sum()),
                "manual_mask_coverage": coverage,
                "expected_target_markers": 476,
            }
        )
        records.append(entry)
        print(
            f"[real] {entry['name']:32s} accepted={entry['accepted']:4d} "
            f"kept20={entry['kept']['20']['kept']:4d} "
            f"kept40={entry['kept']['40']['kept']:4d} "
            f"manual_coverage={coverage}"
        )
    (out_dir / "real.json").write_text(json.dumps(records, indent=2))
    return records


def run_negatives(out_dir: Path, limit_per_group: int) -> list[dict[str, Any]]:
    office = sorted(NATURAL.glob("office_*.jpg"))[:limit_per_group]
    wall = sorted(NATURAL.glob("wall_user_*.jpg"))[:limit_per_group]
    records: list[dict[str, Any]] = []
    for path in [*office, *wall]:
        prep = _roi_and_background(path, residual_gain=1.0)
        result = _candidate_stage(prep["flattened"], prep["roi_mask"])
        empty_truth = np.zeros((0, 2), dtype=np.float64)
        entry = _evaluate_candidates(result, empty_truth)
        entry.update(
            {
                "kind": "negative",
                "name": path.name,
                "path": str(path),
                "mask_pixels": int(prep["roi_mask"].sum()),
            }
        )
        records.append(entry)
        print(
            f"[negative] {entry['name']:28s} mask={entry['mask_pixels']:6d} "
            f"accepted={entry['accepted']:3d} kept20={entry['kept']['20']['kept']:3d} "
            f"kept40={entry['kept']['40']['kept']:3d}"
        )
    (out_dir / "negatives.json").write_text(json.dumps(records, indent=2))
    return records


def _aggregate(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "cases": 0,
            "truth": 0,
            "accepted": 0,
            "saddles": 0,
            "fit_failures": 0,
            "accepted_tp": 0,
            "accepted_fp": 0,
            "accepted_fn": 0,
            "kept": {
                str(count): {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "kept": 0,
                    "predicted": 0,
                    "errors": [],
                }
                for count in NEIGHBOUR_COUNTS
            },
        }
    )
    for record in records:
        group = record[key]
        entry = groups[group]
        entry["cases"] += 1
        entry["truth"] += int(record["accepted_metrics"]["truth"])
        entry["accepted"] += int(record["accepted"])
        entry["saddles"] += int(record["saddles"])
        entry["fit_failures"] += int(record["fit_failures"])
        entry["accepted_tp"] += int(record["accepted_metrics"]["tp"])
        entry["accepted_fp"] += int(record["accepted_metrics"]["fp"])
        entry["accepted_fn"] += int(record["accepted_metrics"]["fn"])
        for count in NEIGHBOUR_COUNTS:
            kept = record["kept"][str(count)]
            metrics = kept["metrics"]
            target = entry["kept"][str(count)]
            target["kept"] += int(kept["kept"])
            target["predicted"] += int(metrics["predicted"])
            target["tp"] += int(metrics["tp"])
            target["fp"] += int(metrics["fp"])
            target["fn"] += int(metrics["fn"])
            target["errors"].extend(metrics["errors"])
    for entry in groups.values():
        truth = entry["truth"]
        accepted_tp = entry["accepted_tp"]
        accepted_precision = accepted_tp / max(1, accepted_tp + entry["accepted_fp"])
        accepted_recall = accepted_tp / max(1, truth)
        entry["accepted_precision"] = accepted_precision
        entry["accepted_recall"] = accepted_recall
        entry["accepted_f1"] = (
            2 * accepted_precision * accepted_recall / (accepted_precision + accepted_recall)
            if accepted_precision + accepted_recall
            else 0.0
        )
        for count, target in entry["kept"].items():
            precision = target["tp"] / max(1, target["tp"] + target["fp"])
            recall = target["tp"] / max(1, truth)
            target["precision"] = precision
            target["recall"] = recall
            target["f1"] = (
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
            errors = np.asarray(target["errors"], dtype=np.float64)
            target["mean_error_px"] = float(errors.mean()) if errors.size else 0.0
            target["median_error_px"] = float(np.median(errors)) if errors.size else 0.0
            target["p90_error_px"] = float(np.percentile(errors, 90)) if errors.size else 0.0
            target["removed"] = int(entry["accepted"] - target["kept"])
    return groups


def _plot_results(
    out_dir: Path,
    synthetic: list[dict[str, Any]],
    real: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    groups: dict[str, Any],
) -> None:
    group_names = ["frontal", "moderate", "oblique", "steep"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    width = 0.2
    x = np.arange(len(group_names))
    for index, count in enumerate(NEIGHBOUR_COUNTS):
        recalls = [
            groups[name]["kept"][str(count)]["recall"] if name in groups else 0.0
            for name in group_names
        ]
        precisions = [
            groups[name]["kept"][str(count)]["precision"] if name in groups else 0.0
            for name in group_names
        ]
        axes[0].bar(x + (index - 0.5) * width, recalls, width=width, label=f"recall {count}nn")
        axes[0].bar(x + (index + 0.5) * width, precisions, width=width, label=f"precision {count}nn")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(group_names)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Synthetic: recall / precision")
    axes[0].grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=6)

    for index, count in enumerate(NEIGHBOUR_COUNTS):
        errors = [
            groups[name]["kept"][str(count)]["mean_error_px"] if name in groups else 0.0
            for name in group_names
        ]
        axes[1].bar(x + (index - 0.5) * width, errors, width=width, label=f"kept {count}nn")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(group_names)
    axes[1].set_title("Synthetic: mean center error (px)")
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].legend(fontsize=6)

    real_names = [record["name"] for record in real]
    real_x = np.arange(len(real_names))
    for index, count in enumerate(NEIGHBOUR_COUNTS):
        values = [record["kept"][str(count)]["kept"] for record in real]
        axes[2].bar(real_x + (index - 0.5) * width, values, width=width, label=f"kept {count}nn")
    negative_total = sum(record["kept"]["40"]["kept"] for record in negatives)
    axes[2].set_xticks(real_x)
    axes[2].set_xticklabels(real_names, fontsize=7)
    axes[2].set_title(f"Real kept counts · negatives kept(40nn)={negative_total}")
    axes[2].grid(alpha=0.2, axis="y")
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "stress_summary.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    for count in NEIGHBOUR_COUNTS:
        errors = [
            value
            for record in synthetic
            for value in record["kept"][str(count)]["metrics"]["errors"]
        ]
        if not errors:
            continue
        values = np.sort(np.asarray(errors, dtype=np.float64))
        axis.plot(values, np.arange(1, values.size + 1) / values.size, label=f"kept {count}nn")
    accepted_errors = [
        value for record in synthetic for value in record["accepted_metrics"]["errors"]
    ]
    if accepted_errors:
        values = np.sort(np.asarray(accepted_errors, dtype=np.float64))
        axis.plot(values, np.arange(1, values.size + 1) / values.size, label="accepted")
    axis.set_xlabel("center error (px)")
    axis.set_ylabel("fraction matched")
    axis.set_title("Synthetic center error CDF")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "stress_center_error_cdf.png", dpi=150)
    plt.close(fig)


def _print_table(groups: dict[str, Any]) -> None:
    print("\nSYNTHETIC")
    header = (
        f"{'group':10s} {'truth':>6s} {'acc':>5s} {'aF1':>5s} "
        + " ".join(f"{'k'+str(c)+'_F1':>8s} {'k'+str(c)+'_rec':>9s} {'k'+str(c)+'_err':>9s}" for c in NEIGHBOUR_COUNTS)
    )
    print(header)
    for name in ("frontal", "moderate", "oblique", "steep"):
        if name not in groups:
            continue
        entry = groups[name]
        cells = []
        for count in NEIGHBOUR_COUNTS:
            target = entry["kept"][str(count)]
            cells.append(
                f"{target['f1']:.3f} {target['recall']:.3f} {target['mean_error_px']:.3f}"
            )
        print(
            f"{name:10s} {entry['truth']:6d} {entry['accepted']:5d} "
            f"{entry['accepted_f1']:.3f} " + " ".join(cells)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--negative-per-group", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    synthetic = run_synthetic(args.out)
    real = run_real(args.out)
    negatives = run_negatives(args.out, args.negative_per_group)
    groups = _aggregate(synthetic, "group")
    summary = {
        "config": {
            "neighbour_counts": [int(count) for count in NEIGHBOUR_COUNTS],
            "trim_sigma": TRIM_SIGMA,
            "accept_sigma": ACCEPT_SIGMA,
            "synthetic_cases": len(synthetic),
            "real_cases": len(real),
            "negative_cases": len(negatives),
        },
        "aggregate": groups,
        "real": real,
        "negatives": negatives,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    _print_table(groups)
    print("\nREAL")
    for record in real:
        print(
            f"{record['name']:32s} accepted={record['accepted']:4d} "
            f"kept20={record['kept']['20']['kept']:4d} "
            f"kept40={record['kept']['40']['kept']:4d} "
            f"coverage={record['manual_mask_coverage']}"
        )
    print("\nNEGATIVES")
    for record in negatives:
        print(
            f"{record['name']:32s} mask={record['mask_pixels']:6d} "
            f"accepted={record['accepted']:3d} "
            f"kept20={record['kept']['20']['kept']:3d} "
            f"kept40={record['kept']['40']['kept']:3d}"
        )
    _plot_results(args.out, synthetic, real, negatives, groups)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
