"""Generate and save office-composite previews for the ROI dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from blob_lab import (
    BROWSER_TARGET_PADDING,
    CameraRanges,
    RoiDatasetConfig,
    build_hex_target_layout,
    generate_roi_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NATURAL_DIR = ROOT / "notebooks" / "natural_images"
NEGATIVE_DIR = NATURAL_DIR / "negatives"
OUT_DIR = ROOT / "notebooks" / "blob_lab_data" / "roi_dataset_preview"


def _srgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=480)
    parser.add_argument("--natural-fraction", type=float, default=0.45)
    parser.add_argument("--negative-dir", type=Path, default=NEGATIVE_DIR)
    parser.add_argument("--mask-mode", choices=("hull", "canvas", "cells"), default="hull")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

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
            natural_dir=str(args.negative_dir),
            negative_dir=str(args.negative_dir),
            background_dir=str(NATURAL_DIR),
            target_width_fraction=(0.15, 0.72),
            closeup_fraction=0.35,
            closeup_width_fraction=(0.70, 0.90),
            cropped_fraction=0.35,
            cropped_width_fraction=(0.45, 0.90),
            heavy_blur_fraction=0.20,
            frame_blur_fraction=0.15,
            canvas_aspects=(4 / 3, 16 / 9, 1.0),
            camera=CameraRanges(roll_deg=(-180.0, 180.0)),
            mask_mode=args.mask_mode,
            randomize_colors=True,
            seed=args.seed,
        ),
        layout,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"layout quads={len(layout.quads)} side={layout.side_px}px "
        f"spacing={layout.spacing_px}px padding={layout.padding_x_px}x{layout.padding_y_px}"
    )
    for index, meta in enumerate(dataset.metadata):
        image = dataset.images[index, :, :, 0]
        mask = dataset.masks[index, :, :, 0]
        base = np.repeat(
            (_srgb(np.clip(image, 0.0, 1.0)) * 255).astype(np.uint8)[:, :, None],
            3,
            axis=2,
        )
        contours, _ = cv2.findContours(
            (mask >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(base, contours, -1, (0, 255, 0), 1)
        canvas = np.asarray(meta.get("canvas_corners", []), dtype=np.float32)
        if canvas.shape[0] >= 3:
            cv2.polylines(
                base,
                [np.round(canvas).astype(np.int32)],
                True,
                (0, 220, 255),
                1,
            )
        for x, y in meta.get("points", []):
            cv2.circle(base, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
        cv2.imwrite(str(args.out_dir / f"sample_{index:03d}.png"), base)
        cv2.imwrite(
            str(args.out_dir / f"sample_{index:03d}_mask.png"),
            (mask * 255).astype(np.uint8),
        )
        print(
            index,
            "natural" if meta.get("natural") else "target",
            f"points={len(meta.get('points', []))}",
            f"mask_area={float(mask.mean()):.3f}",
            f"target_width={meta.get('target_width_px', 0):.1f}px",
            f"board_gray={meta.get('board_gray', 0):.2f}",
            f"closeup={meta.get('closeup', False)}",
            f"cropped={meta.get('cropped', False)}",
            f"visible={meta.get('visible_fraction', 0):.2f}",
            f"blur={meta.get('blur_sigma', 0):.2f}",
            f"frame_blur={meta.get('frame_blur_sigma', 0):.2f}",
            f"aspect={meta.get('canvas_aspect', 0):.2f}",
        )
    print(f"saved {len(dataset)} previews to {args.out_dir}")


if __name__ == "__main__":
    main()
