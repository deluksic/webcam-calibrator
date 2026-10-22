"""Can the junction head learn on its own, with the center loss disabled?"""

from __future__ import annotations

import json
import time
from pathlib import Path

from blob_lab import (
    HeatmapDatasetConfig,
    UNetTorchConfig,
    build_hex_target_layout,
    evaluate_heatmaps,
    generate_heatmap_dataset,
    train_unet_torch,
)

OUT = Path("notebooks/blob_lab_data/experiments")
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def log(message: str) -> None:
    print(f"[{time.monotonic() - START:7.1f}s] {message}", flush=True)


def best(heatmaps, metadata):
    rows = []
    for threshold in THRESHOLDS:
        m = evaluate_heatmaps(heatmaps, metadata, threshold=threshold, nms_radius=2, tolerance=2.0)
        rows.append({
            "threshold": threshold,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "mean_error_px": m.mean_error_px,
            "predicted": m.predicted_count,
            "truth": m.true_count,
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
    log(f"train={len(train)} val={len(validation)} test={len(test)} junctions_train={float(train.junctions.sum()):.1f}")

    model = train_unet_torch(
        train.images,
        train.heatmaps,
        validation.images,
        validation.heatmaps,
        UNetTorchConfig(
            filters=8,
            input_channels=4,
            epochs=20,
            batch_size=4,
            seed=41,
            center_weight=0.0,
            area_weight=0.0,
            junction_weight=1.0,
        ),
        train_features=train.features,
        validation_features=validation.features,
        train_junctions=train.junctions,
        validation_junctions=validation.junctions,
        on_epoch=lambda e, n, tr, va: log(f"epoch {e}/{n} train={tr:.4f} val={va:.4f}") if e in (1, n) or e % 5 == 0 else None,
    )
    _, _, junction_heatmaps = model.score_heads(test.features)
    junction_meta = [{"points": meta.get("junctions", [])} for meta in test.metadata]
    row = {
        "name": "junction_only",
        "sigma": dataset.config.keypoint_sigma,
        "junction": best(junction_heatmaps, junction_meta),
        "train_loss": model.history["train_loss"][-1],
        "val_loss": model.history["val_loss"][-1],
    }
    (OUT / "junction_only_results.json").write_text(json.dumps([row], indent=2))
    log(json.dumps(row))


if __name__ == "__main__":
    main()
