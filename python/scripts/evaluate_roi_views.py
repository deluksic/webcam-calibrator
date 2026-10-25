"""Compare ROI checkpoints on far, close-up, full-view, cropped, and negative frames.

The mixed ROI metric hides scale-specific regressions. This script builds one
held-out synthetic suite with every view branch and reports each checkpoint's
containment/IoU per branch, including the full-view cases where the marker
pattern covers almost the whole frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from blob_lab import (
    BROWSER_TARGET_PADDING,
    CameraRanges,
    RoiDatasetConfig,
    build_hex_target_layout,
    evaluate_roi,
    evaluate_roi_groups,
    generate_roi_dataset,
    load_roi_checkpoint,
    predict_roi,
)

ROOT = Path(__file__).resolve().parents[1]
NATURAL_DIR = ROOT / "notebooks" / "natural_images"
NEGATIVE_DIR = NATURAL_DIR / "negatives"
OUT_DIR = ROOT / "notebooks" / "blob_lab_data"
DEFAULT_CHECKPOINTS = [OUT_DIR / "roi_net_best.pt"]


def _srgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    )


def _overlay(
    image: np.ndarray,
    probability: np.ndarray,
    truth: np.ndarray,
    meta: dict,
    path: Path,
) -> None:
    height, width = image.shape[:2]
    base = np.repeat(
        (_srgb(np.clip(image, 0.0, 1.0)) * 255).astype(np.uint8)[:, :, None],
        3,
        axis=2,
    )
    predicted = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    truth_contours, _ = cv2.findContours(
        (np.asarray(truth) >= 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    predicted_contours, _ = cv2.findContours(
        (predicted >= 0.5).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(base, truth_contours, -1, (0, 220, 255), 1)
    cv2.drawContours(base, predicted_contours, -1, (0, 255, 0), 1)
    for x, y in meta.get("points", []):
        cv2.circle(base, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
    cv2.putText(
        base,
        f"marker width {meta.get('marker_width_fraction', 0.0):.2f}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoints",
        nargs="*",
        type=Path,
        default=DEFAULT_CHECKPOINTS,
        help="ROI checkpoints to compare (default: roi_net_best.pt)",
    )
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=480)
    parser.add_argument("--natural-fraction", type=float, default=0.30)
    parser.add_argument("--cropped-fraction", type=float, default=0.20)
    parser.add_argument("--closeup-fraction", type=float, default=0.20)
    parser.add_argument("--full-view-fraction", type=float, default=0.60)
    parser.add_argument("--full-view-min", type=float, default=0.95)
    parser.add_argument("--full-view-max", type=float, default=1.15)
    parser.add_argument("--full-view-min-height", type=float, default=0.80)
    parser.add_argument("--out-json", type=Path, default=OUT_DIR / "roi_view_eval.json")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="optional directory for full-view prediction overlays",
    )
    args = parser.parse_args()

    start = time.monotonic()

    def log(message: str) -> None:
        print(f"[{time.monotonic() - start:7.1f}s] {message}", flush=True)

    layout = build_hex_target_layout(
        width_px=1280,
        height_px=960,
        columns=7,
        points_per_edge=4,
        marker_scale=0.45,
        color_seed=1,
        padding_fraction=BROWSER_TARGET_PADDING,
    )
    dataset = generate_roi_dataset(
        RoiDatasetConfig(
            output_width=args.output_width,
            output_height=args.output_height,
            samples=args.samples,
            natural_fraction=args.natural_fraction,
            natural_dir=str(NEGATIVE_DIR),
            negative_dir=str(NEGATIVE_DIR),
            background_dir=str(NATURAL_DIR),
            target_width_fraction=(0.15, 0.72),
            closeup_fraction=args.closeup_fraction,
            closeup_width_fraction=(0.75, 0.95),
            full_view_fraction=args.full_view_fraction,
            full_view_marker_width_fraction=(args.full_view_min, args.full_view_max),
            full_view_min_marker_height_fraction=args.full_view_min_height,
            cropped_fraction=args.cropped_fraction,
            cropped_width_fraction=(0.45, 0.90),
            heavy_blur_fraction=0.20,
            frame_blur_fraction=0.15,
            canvas_aspects=(4 / 3, 16 / 9, 1.0),
            camera=CameraRanges(roll_deg=(-180.0, 180.0)),
            target_gray_range=(0.38, 0.62),
            mask_mode="hull",
            randomize_colors=True,
            seed=args.seed,
        ),
        layout,
    )
    branch_counts: dict[str, int] = {}
    for meta in dataset.metadata:
        branch = str(meta.get("view_branch", "negative"))
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
    log(f"evaluation set samples={len(dataset)} branches={branch_counts}")

    full_view_indices = [
        index for index, meta in enumerate(dataset.metadata) if meta.get("full_view")
    ][:4]
    report: dict = {
        "dataset": {
            "samples": len(dataset),
            "seed": args.seed,
            "natural_fraction": args.natural_fraction,
            "cropped_fraction": args.cropped_fraction,
            "closeup_fraction": args.closeup_fraction,
            "full_view_fraction": args.full_view_fraction,
            "full_view_marker_width_fraction": [
                args.full_view_min,
                args.full_view_max,
            ],
            "full_view_min_marker_height_fraction": args.full_view_min_height,
            "branch_counts": branch_counts,
        },
        "checkpoints": {},
    }

    for checkpoint in args.checkpoints:
        model = load_roi_checkpoint(checkpoint, None)
        if model is None:
            log(f"skipping unreadable checkpoint {checkpoint}")
            continue
        probabilities, presence = predict_roi(model, dataset.images)
        metrics = evaluate_roi(probabilities, presence, dataset)
        groups = evaluate_roi_groups(probabilities, presence, dataset)
        report["checkpoints"][str(checkpoint)] = {
            "config": {
                "channels": list(model.config.channels),
                "input": [model.config.input_width, model.config.input_height],
            },
            "overall": metrics,
            "groups": groups,
        }
        log(
            f"{checkpoint}: containment={metrics['center_containment']:.4f} "
            f"iou={metrics['mask_iou']:.4f} presence={metrics['presence_accuracy']:.4f}"
        )
        for group_name, group in groups.items():
            log(
                f"  {group_name:10s} containment={group['center_containment']:.4f} "
                f"iou={group['mask_iou']:.4f} area_ratio={group['area_ratio']:.3f} "
                f"n={group['samples']}"
            )
        if args.preview_dir is not None:
            checkpoint_name = checkpoint.stem
            for index in full_view_indices:
                _overlay(
                    dataset.images[index, :, :, 0],
                    probabilities[index],
                    dataset.masks[index, :, :, 0],
                    dataset.metadata[index],
                    args.preview_dir / f"{checkpoint_name}_full_view_{index:04d}.png",
                )
            log(f"  wrote {len(full_view_indices)} overlays to {args.preview_dir}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    log(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
