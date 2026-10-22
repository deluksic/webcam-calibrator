"""Dump the exact ROI training split plus previews for inspection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from blob_lab import (
    BROWSER_TARGET_PADDING,
    CameraRanges,
    RoiDatasetConfig,
    build_hex_target_layout,
    generate_roi_dataset,
    load_real_positives,
)

ROOT = Path(__file__).resolve().parents[1]
NATURAL_DIR = ROOT / "notebooks" / "natural_images"
NEGATIVE_DIR = NATURAL_DIR / "negatives"
REAL_DIR = ROOT / "notebooks" / "real_positives"
DEFAULT_OUT = ROOT / "notebooks" / "blob_lab_data" / "training_dump"


def srgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.where(image <= 0.0031308, image * 12.92, 1.055 * np.power(np.maximum(image, 0), 1 / 2.4) - 0.055)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real-positive-repeat", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    layout = build_hex_target_layout(
        width_px=1280,
        height_px=960,
        columns=7,
        points_per_edge=4,
        marker_scale=0.45,
        color_seed=1,
        padding_fraction=BROWSER_TARGET_PADDING,
    )
    config = RoiDatasetConfig(
        output_width=640,
        output_height=480,
        samples=args.samples,
        natural_fraction=0.45,
        natural_dir=str(NEGATIVE_DIR),
        negative_dir=str(NEGATIVE_DIR),
        background_dir=str(NATURAL_DIR),
        target_width_fraction=(0.15, 0.75),
        closeup_fraction=0.30,
        closeup_width_fraction=(0.70, 0.90),
        cropped_fraction=0.30,
        cropped_width_fraction=(0.45, 0.90),
        heavy_blur_fraction=0.20,
        heavy_blur_sigma=(0.6, 1.8),
        frame_blur_fraction=0.15,
        frame_blur_sigma=(0.3, 1.2),
        canvas_aspects=(4 / 3, 16 / 9, 1.0),
        camera=CameraRanges(roll_deg=(-180.0, 180.0)),
        target_gray_range=(0.38, 0.62),
        mask_mode="hull",
        randomize_colors=True,
        seed=args.seed,
    )
    dataset = generate_roi_dataset(config, layout)
    train, validation, test = dataset.shuffled(args.seed).split((0.7, 0.15, 0.15))
    real = load_real_positives(REAL_DIR, dataset.config)
    if real is not None:
        train = train.concatenated(real.repeated(args.real_positive_repeat))
    train = train.augmented(horizontal=True)
    print(f"train={len(train)} val={len(validation)} test={len(test)} real={0 if real is None else len(real)}")

    rows = []
    kinds = []
    for index in range(len(train)):
        meta = train.metadata[index]
        if meta.get("real_positive"):
            kind = "real_positive"
        elif meta.get("presence", 1) < 0.5 or meta.get("natural"):
            kind = "negative"
        else:
            kind = "positive"
        kinds.append(kind)
        name = f"{index:05d}_{kind}"
        image = np.clip(np.asarray(train.images[index, :, :, 0], dtype=np.float32), 0, 1)
        gray = (srgb(image) * 255).astype(np.uint8)
        mask = (np.asarray(train.masks[index, :, :, 0]) > 0).astype(np.uint8) * 255
        cv2.imwrite(str(images_dir / f"{name}.jpg"), gray, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(str(masks_dir / f"{name}.png"), mask)
        rows.append(
            {
                "file": name,
                "kind": kind,
                "presence": float(train.presence[index]),
                "real_positive": bool(meta.get("real_positive", False)),
                "closeup": bool(meta.get("closeup", False)),
                "cropped": bool(meta.get("cropped", False)),
                "heavy_blur": bool(meta.get("heavy_blur", False)),
                "frame_blur_sigma": float(meta.get("frame_blur_sigma", 0.0) or 0.0),
                "canvas_aspect": float(meta.get("canvas_aspect", 0.0) or 0.0),
                "target_width_px": float(meta.get("target_width_px", 0.0) or 0.0),
                "source": str(meta.get("source", "")),
            }
        )

    with (out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "kind_counts": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
        "closeup": sum(row["closeup"] for row in rows),
        "cropped": sum(row["cropped"] for row in rows),
        "heavy_blur": sum(row["heavy_blur"] for row in rows),
        "frame_blur": sum(row["frame_blur_sigma"] > 0 for row in rows),
        "aspects": {
            key: sum(abs(row["canvas_aspect"] - float(key)) < 0.01 for row in rows)
            for key in ("1.3333333333333333", "1.7777777777777777", "1.0")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Contact sheet: evenly sampled positives/negatives plus every real frame.
    def pick(kind: str, count: int) -> list[int]:
        indices = [i for i, value in enumerate(kinds) if value == kind]
        if len(indices) <= count:
            return indices
        return [indices[int(round(i * (len(indices) - 1) / (count - 1)))] for i in range(count)]

    selected = pick("positive", 60) + pick("negative", 60) + pick("real_positive", 20)
    thumb_w, thumb_h, cols = 160, 120, 10
    rows_count = (len(selected) + cols - 1) // cols
    sheet = np.zeros((rows_count * (thumb_h + 20), cols * thumb_w, 3), dtype=np.uint8)
    for position, index in enumerate(selected):
        image = (srgb(np.clip(np.asarray(train.images[index, :, :, 0], dtype=np.float32), 0, 1)) * 255).astype(np.uint8)
        mask = (np.asarray(train.masks[index, :, :, 0]) > 0).astype(np.uint8)
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scaled = [np.round(contour.astype(np.float32) * [thumb_w / mask.shape[1], thumb_h / mask.shape[0]]).astype(np.int32) for contour in contours]
        cv2.drawContours(thumb, scaled, -1, (0, 255, 0), 1)
        row, col = divmod(position, cols)
        top = row * (thumb_h + 20)
        left = col * thumb_w
        sheet[top : top + thumb_h, left : left + thumb_w] = thumb
        cv2.putText(sheet, f"{kinds[index]} {index}", (left + 3, top + thumb_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "contact_sheet.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
