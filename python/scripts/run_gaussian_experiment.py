"""Focused test of small-Gaussian keypoint targets."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from blob_lab import (
    HeatmapDatasetConfig,
    UNetTorchConfig,
    build_hex_target_layout,
    detect_peaks,
    evaluate_heatmaps,
    generate_heatmap_dataset,
    train_unet_torch,
)

OUT = Path("notebooks/blob_lab_data/experiments")
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def log(message: str) -> None:
    print(f"[{time.monotonic() - START:7.1f}s] {message}", flush=True)


def best(heatmaps, metadata):
    rows = []
    for threshold in THRESHOLDS:
        metrics = evaluate_heatmaps(heatmaps, metadata, threshold=threshold, nms_radius=2, tolerance=2.0)
        rows.append({
            "threshold": threshold,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mean_error_px": metrics.mean_error_px,
            "predicted": metrics.predicted_count,
            "truth": metrics.true_count,
        })
    return max(rows, key=lambda row: row["f1"])


def main() -> None:
    layout = build_hex_target_layout(width_px=1280, height_px=960, columns=7, points_per_edge=4, marker_scale=0.45, color_seed=1)
    dataset = generate_heatmap_dataset(
        HeatmapDatasetConfig(
            output_width=640,
            output_height=480,
            samples=80,
            natural_fraction=0.35,
            natural_dir="notebooks/natural_images",
            randomize_colors=True,
            keypoint_sigma=0.8,
            seed=23,
        ),
        layout,
    )
    train, validation, test = dataset.split((0.7, 0.15, 0.15))
    log(f"dataset train={len(train)} val={len(validation)} test={len(test)} center_heat={float(train.heatmaps.sum()):.1f} junction_heat={float(train.junctions.sum()):.1f}")

    variants = [
        ("gaussian_center", False, False),
        ("gaussian_center_area_junction", True, True),
    ]
    results = []
    for name, use_area, use_junction in variants:
        log(f"START {name}")
        model = train_unet_torch(
            train.images,
            train.heatmaps,
            validation.images,
            validation.heatmaps,
            UNetTorchConfig(filters=8, input_channels=4, epochs=15, batch_size=4, seed=31),
            train_features=train.features,
            validation_features=validation.features,
            train_masks=train.masks if use_area else None,
            validation_masks=validation.masks if use_area else None,
            train_junctions=train.junctions if use_junction else None,
            validation_junctions=validation.junctions if use_junction else None,
            on_epoch=lambda e, n, tr, va: log(f"{name} {e}/{n} train={tr:.4f} val={va:.4f}") if e in (1, n) or e % 5 == 0 else None,
        )
        heat = model.score(test.features)
        row = {
            "name": name,
            "sigma": dataset.config.keypoint_sigma,
            "center": best(heat, test.metadata),
            "train_loss": model.history["train_loss"][-1],
            "val_loss": model.history["val_loss"][-1],
        }
        if use_junction:
            _, _, junction_heat = model.score_heads(test.features)
            junction_meta = [{"points": meta.get("junctions", [])} for meta in test.metadata]
            row["junction"] = best(junction_heat, junction_meta)
        results.append(row)
        (OUT / "gaussian_results.json").write_text(json.dumps(results, indent=2))
        log(f"DONE {name}: {json.dumps(row)}")
    log("finished")


if __name__ == "__main__":
    main()
