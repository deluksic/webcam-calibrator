"""Train the low-resolution target-area (ROI) segmenter on office composites."""

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
    load_real_positives,
    RoiNetConfig,
    build_hex_target_layout,
    evaluate_roi,
    generate_roi_dataset,
    roi_parameter_count,
    save_roi_checkpoint,
    train_roi,
)

ROOT = Path(__file__).resolve().parents[1]
NATURAL_DIR = ROOT / "notebooks" / "natural_images"
NEGATIVE_DIR = NATURAL_DIR / "negatives"
REAL_POSITIVE_DIR = ROOT / "notebooks" / "real_positives"
OUT_DIR = ROOT / "notebooks" / "blob_lab_data"
CHECKPOINT = OUT_DIR / "roi_net.pt"
PREVIEW_DIR = OUT_DIR / "roi_preview"


def _srgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    )


def _draw_preview(
    image: np.ndarray,
    mask_probability: np.ndarray,
    truth_mask: np.ndarray,
    points: list[tuple[float, float]],
    path: Path,
) -> None:
    height, width = image.shape[:2]
    base = np.repeat((_srgb(np.clip(image, 0.0, 1.0)) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    predicted = cv2.resize(
        np.asarray(mask_probability, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    predicted_contours, _ = cv2.findContours(
        (predicted >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    truth_contours, _ = cv2.findContours(
        (truth_mask >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(base, truth_contours, -1, (0, 220, 255), 1)
    cv2.drawContours(base, predicted_contours, -1, (0, 255, 0), 1)
    for x, y in points:
        cv2.circle(base, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=480)
    parser.add_argument("--input-width", type=int, default=192)
    parser.add_argument("--input-height", type=int, default=144)
    parser.add_argument("--channels", default="tiny", help="preset or comma-separated c1,c2,c3,c4")
    parser.add_argument("--mask-mode", choices=("hull", "canvas", "cells"), default="hull")
    parser.add_argument("--natural-fraction", type=float, default=0.45)
    parser.add_argument("--negative-dir", type=Path, default=NEGATIVE_DIR)
    parser.add_argument("--real-positive-dir", type=Path, default=REAL_POSITIVE_DIR)
    parser.add_argument("--real-positive-repeat", type=int, default=4)
    parser.add_argument("--target-min", type=float, default=0.15)
    parser.add_argument("--target-max", type=float, default=0.75)
    parser.add_argument("--closeup-fraction", type=float, default=0.30)
    parser.add_argument("--closeup-min", type=float, default=0.70)
    parser.add_argument("--closeup-max", type=float, default=0.90)
    parser.add_argument("--cropped-fraction", type=float, default=0.30)
    parser.add_argument("--cropped-min", type=float, default=0.45)
    parser.add_argument("--cropped-max", type=float, default=0.90)
    parser.add_argument("--heavy-blur-fraction", type=float, default=0.20)
    parser.add_argument("--heavy-blur-min", type=float, default=0.6)
    parser.add_argument("--heavy-blur-max", type=float, default=1.8)
    parser.add_argument("--frame-blur-fraction", type=float, default=0.15)
    parser.add_argument("--frame-blur-min", type=float, default=0.3)
    parser.add_argument("--frame-blur-max", type=float, default=1.2)
    parser.add_argument("--roll-min", type=float, default=-180.0)
    parser.add_argument("--roll-max", type=float, default=180.0)
    parser.add_argument("--canvas-aspects", type=str, default="1.3333333,1.7777778,1.0")
    parser.add_argument("--target-gray-min", type=float, default=0.38)
    parser.add_argument("--target-gray-max", type=float, default=0.62)
    parser.add_argument("--false-negative-weight", type=float, default=10.0)
    parser.add_argument("--tversky-weight", type=float, default=1.0)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
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
    log(
        f"layout quads={len(layout.quads)} side={layout.side_px}px "
        f"spacing={layout.spacing_px}px padding={layout.padding_x_px}x{layout.padding_y_px}"
    )

    dataset = generate_roi_dataset(
        RoiDatasetConfig(
            output_width=args.output_width,
            output_height=args.output_height,
            samples=args.samples,
            natural_fraction=args.natural_fraction,
            natural_dir=str(args.negative_dir),
            negative_dir=str(args.negative_dir),
            background_dir=str(NATURAL_DIR),
            target_width_fraction=(args.target_min, args.target_max),
            closeup_fraction=args.closeup_fraction,
            closeup_width_fraction=(args.closeup_min, args.closeup_max),
            cropped_fraction=args.cropped_fraction,
            cropped_width_fraction=(args.cropped_min, args.cropped_max),
            heavy_blur_fraction=args.heavy_blur_fraction,
            heavy_blur_sigma=(args.heavy_blur_min, args.heavy_blur_max),
            frame_blur_fraction=args.frame_blur_fraction,
            frame_blur_sigma=(args.frame_blur_min, args.frame_blur_max),
            canvas_aspects=tuple(
                float(part) for part in args.canvas_aspects.split(",") if part.strip()
            ),
            camera=CameraRanges(roll_deg=(args.roll_min, args.roll_max)),
            target_gray_range=(args.target_gray_min, args.target_gray_max),
            mask_mode=args.mask_mode,
            randomize_colors=True,
            seed=args.seed,
        ),
        layout,
    )
    train, validation, test = dataset.shuffled(args.seed).split((0.7, 0.15, 0.15))
    real = load_real_positives(args.real_positive_dir, dataset.config)
    if real is not None:
        real = real.repeated(args.real_positive_repeat)
        train = train.concatenated(real)
        log(
            f"real positives train={len(real)} "
            f"({len(real) / max(1, args.real_positive_repeat)} unique)"
        )
    if not args.no_flip:
        train = train.augmented(horizontal=True)
    log(
        f"dataset train={len(train)} val={len(validation)} test={len(test)} "
        f"negatives_train={int((train.presence < 0.5).sum())} "
        f"mean_mask_area={float(dataset.masks.mean()):.3f}"
    )

    channel_options = {
        "micro": (2, 4, 8, 16),
        "tiny": (3, 6, 12, 24),
        "small": (4, 8, 16, 32),
        "medium": (8, 16, 32, 64),
    }
    if "," in args.channels:
        channels = tuple(int(part) for part in args.channels.split(","))
        if len(channels) != 4:
            raise SystemExit("--channels must be a preset or four comma-separated widths")
    else:
        try:
            channels = channel_options[args.channels]
        except KeyError as error:
            raise SystemExit(f"unknown channel preset: {args.channels}") from error
    config = RoiNetConfig(
        channels=channels,
        input_width=args.input_width,
        input_height=args.input_height,
        output_stride=2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        false_negative_weight=args.false_negative_weight,
        tversky_weight=args.tversky_weight,
        seed=args.seed,
    )

    def on_epoch(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
        if epoch == 1 or epoch == total or epoch % 5 == 0:
            log(f"epoch {epoch}/{total} train={train_loss:.4f} val={val_loss:.4f}")

    model = train_roi(
        train.images,
        train.masks,
        train.presence,
        validation.images,
        validation.masks,
        validation.presence,
        config,
        on_epoch=on_epoch,
    )
    log(f"trained params={roi_parameter_count(model.model)} device={model.device}")

    result: dict = {
        "params": roi_parameter_count(model.model),
        "channels": channels,
        "input": [config.input_width, config.input_height],
        "mask_mode": args.mask_mode,
        "train": model.history,
    }
    split_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, split in (("validation", validation), ("test", test)):
        probabilities, presence = model.predict(split.images)
        split_predictions[split_name] = (probabilities, presence)
        metrics = evaluate_roi(probabilities, presence, split)
        result[split_name] = metrics
        log(f"{split_name}: {json.dumps(metrics)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_roi_checkpoint(CHECKPOINT, model.model, config)
    result["checkpoint"] = str(CHECKPOINT)
    (OUT_DIR / "roi_metrics.json").write_text(json.dumps(result, indent=2))
    log(f"saved {CHECKPOINT}")

    if not args.no_preview:
        test_probabilities, _test_presence = split_predictions["test"]
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        positive_indices = [
            index for index, meta in enumerate(test.metadata) if meta.get("points")
        ][:4]
        for index in positive_indices:
            _draw_preview(
                test.images[index, :, :, 0],
                test_probabilities[index],
                test.masks[index, :, :, 0],
                test.metadata[index]["points"],
                PREVIEW_DIR / f"test_{index:03d}.png",
            )
        log(f"saved {len(positive_indices)} previews to {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
