"""Background ablation for center/area/junction heads and input features.

Keeps the dataset fixed and trains a small number of variants. Results are
written incrementally under notebooks/blob_lab_data/experiments.
"""

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
    match_peaks,
    torch_unet_parameter_count,
    train_unet_torch,
)

OUT = Path("notebooks/blob_lab_data/experiments")
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
LIMIT_SECONDS = 3300.0
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
TOLERANCE = 2.0
NMS_RADIUS = 2


def log(message: str) -> None:
    elapsed = time.monotonic() - START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def best_metrics(heatmaps, metadata, key: str):
    best = None
    for threshold in THRESHOLDS:
        metrics = evaluate_heatmaps(
            heatmaps,
            metadata,
            threshold=threshold,
            nms_radius=NMS_RADIUS,
            tolerance=TOLERANCE,
        )
        row = {
            "threshold": threshold,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "mean_error_px": metrics.mean_error_px,
            "predicted": metrics.predicted_count,
            "truth": metrics.true_count,
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best


def junction_center_similarity(heatmaps, metadata) -> dict:
    center_distances = []
    junction_distances = []
    for heatmap, meta in zip(heatmaps, metadata):
        peaks = detect_peaks(heatmap, threshold=0.2, nms_radius=1)
        centers = meta.get("points", [])
        junctions = meta.get("junctions", [])
        for peak in peaks:
            if centers:
                center_distances.append(min(float(np.hypot(peak.x - x, peak.y - y)) for x, y in centers))
            if junctions:
                junction_distances.append(min(float(np.hypot(peak.x - x, peak.y - y)) for x, y in junctions))
    return {
        "mean_pred_to_center_px": float(np.mean(center_distances)) if center_distances else None,
        "mean_pred_to_junction_px": float(np.mean(junction_distances)) if junction_distances else None,
        "predicted_junction_peaks": len(center_distances),
    }


def main() -> None:
    layout = build_hex_target_layout(
        width_px=1280,
        height_px=960,
        columns=7,
        points_per_edge=4,
        marker_scale=0.45,
        color_seed=1,
    )
    log(f"layout quads={len(layout.quads)} hex_cells={len(layout.cell_polygons)} junctions={len(layout.three_way_junction_points)}")

    dataset = generate_heatmap_dataset(
        HeatmapDatasetConfig(
            output_width=640,
            output_height=480,
            samples=100,
            natural_fraction=0.35,
            natural_dir="notebooks/natural_images",
            randomize_colors=True,
            seed=11,
        ),
        layout,
    )
    train, validation, test = dataset.split((0.7, 0.15, 0.15))
    log(
        f"dataset train={len(train)} val={len(validation)} test={len(test)} "
        f"train_positive={int(train.heatmaps.sum())} test_junctions={int(test.junctions.sum())}"
    )

    variants = [
        ("center_gray", "gray", False, False),
        ("center_features", "features", False, False),
        ("center_area_features", "features", True, False),
        ("center_area_junction_gray", "gray", True, True),
        ("center_area_junction_features", "features", True, True),
    ]

    results = []
    for name, input_mode, use_area, use_junction in variants:
        if time.monotonic() - START > LIMIT_SECONDS:
            log(f"stopping before {name}: time limit")
            break
        log(f"START {name} input={input_mode} area={use_area} junction={use_junction}")
        config = UNetTorchConfig(
            filters=8,
            input_channels=1 if input_mode == "gray" else 4,
            epochs=20,
            batch_size=4,
            learning_rate=2e-3,
            seed=17,
        )
        features = dataset.features if input_mode == "features" else None
        train_features = train.features if input_mode == "features" else None
        val_features = validation.features if input_mode == "features" else None
        test_features = test.features if input_mode == "features" else None

        def on_epoch(epoch, total, train_loss, val_loss):
            if epoch == 1 or epoch == total or epoch % 5 == 0:
                log(f"{name} epoch {epoch}/{total} train={train_loss:.4f} val={val_loss:.4f}")

        model = train_unet_torch(
            train.images,
            train.heatmaps,
            validation.images,
            validation.heatmaps,
            config,
            on_epoch=on_epoch,
            train_features=train_features,
            validation_features=val_features,
            train_masks=train.masks if use_area else None,
            validation_masks=validation.masks if use_area else None,
            train_junctions=train.junctions if use_junction else None,
            validation_junctions=validation.junctions if use_junction else None,
        )
        log(f"trained {name} params={torch_unet_parameter_count(model.model)}")

        center_heatmaps = model.score(test_features if test_features is not None else test.images)
        center_metrics = best_metrics(center_heatmaps, test.metadata, "points")
        row = {
            "name": name,
            "input": input_mode,
            "area": use_area,
            "junction": use_junction,
            "params": torch_unet_parameter_count(model.model),
            "device": str(model.device),
            "final_train_loss": model.history["train_loss"][-1] if model.history["train_loss"] else None,
            "final_val_loss": model.history["val_loss"][-1] if model.history["val_loss"] else None,
            "center": center_metrics,
        }

        if use_junction:
            _, _, junction_heatmaps = model.score_heads(test_features if test_features is not None else test.images)
            row["junction"] = best_metrics(
                junction_heatmaps,
                [{"points": meta.get("junctions", [])} for meta in test.metadata],
                "junctions",
            )
            row["junction_center_similarity"] = junction_center_similarity(junction_heatmaps, test.metadata)
        results.append(row)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        log(f"DONE {name}: center={center_metrics}")
        log(f"RESULT {json.dumps(row)}")

    log(f"finished {len(results)} variants; results={OUT / 'results.json'}")


if __name__ == "__main__":
    main()
