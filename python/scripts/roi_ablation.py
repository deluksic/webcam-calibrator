"""Ablate ROI model size, input resolution, and false-negative weighting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from blob_lab import (
    BROWSER_TARGET_PADDING,
    RoiDatasetConfig,
    RoiNetConfig,
    build_hex_target_layout,
    evaluate_roi,
    generate_roi_dataset,
    predict_roi,
    roi_parameter_count,
    save_roi_checkpoint,
    train_roi,
)

ROOT = Path(__file__).resolve().parents[1]
NATURAL_DIR = ROOT / "notebooks" / "natural_images"
OUT_DIR = ROOT / "notebooks" / "blob_lab_data"
PREVIEW_DIR = OUT_DIR / "roi_ablation_preview"


def _srgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    )


def _preview(
    image: np.ndarray,
    probability: np.ndarray,
    truth: np.ndarray,
    points: list[tuple[float, float]],
    path: Path,
) -> None:
    height, width = image.shape[:2]
    base = np.repeat(
        (_srgb(np.clip(image, 0.0, 1.0)) * 255).astype(np.uint8)[:, :, None], 3, axis=2
    )
    predicted = cv2.resize(
        np.asarray(probability, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    predicted_contours, _ = cv2.findContours(
        (predicted >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    truth_contours, _ = cv2.findContours(
        np.asarray(truth, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(base, truth_contours, -1, (0, 220, 255), 1)
    cv2.drawContours(base, predicted_contours, -1, (0, 255, 0), 1)
    for x, y in points:
        cv2.circle(base, (int(round(x)), int(round(y))), 1, (255, 0, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), base)


def _flops(model: torch.nn.Module, config: RoiNetConfig) -> float:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        example = torch.randn(1, 1, config.input_height, config.input_width)
        with FlopCounterMode(display=False) as counter:
            with torch.no_grad():
                model.cpu().eval()(example)
        return float(counter.get_total_flops()) / 1e9
    except Exception:
        return 0.0


def _latency_ms(model: torch.nn.Module, config: RoiNetConfig, device: torch.device) -> float:
    example = torch.randn(1, 1, config.input_height, config.input_width, device=device)
    model.to(device).eval()
    with torch.no_grad():
        for _ in range(5):
            model(example)
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            model(example)
        if device.type == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - start) * 1000.0 / 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=480)
    parser.add_argument("--target-gray-min", type=float, default=0.38)
    parser.add_argument("--target-gray-max", type=float, default=0.62)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--result-name", type=str, default="roi_ablation")
    parser.add_argument("--checkpoint-name", type=str, default="roi_net_best.pt")
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
            natural_fraction=0.30,
            natural_dir=str(NATURAL_DIR),
            background_dir=str(NATURAL_DIR),
            target_width_fraction=(0.30, 0.75),
            target_gray_range=(args.target_gray_min, args.target_gray_max),
            mask_mode="hull",
            randomize_colors=True,
            seed=args.seed,
        ),
        layout,
    )
    shuffled = dataset.shuffled(args.seed)
    train, validation, test = shuffled.split((0.7, 0.15, 0.15))
    train = train.augmented(horizontal=True)
    log(
        f"dataset total={len(dataset)} train={len(train)} val={len(validation)} "
        f"test={len(test)} backgrounds={len(list(NATURAL_DIR.glob('*.jpg')))} "
        f"mean_mask_area={float(dataset.masks.mean()):.3f}"
    )

    variants = [
        ("micro_96_fn5", (2, 4, 8, 16), 96, 72, 5.0, 0.5),
        ("micro_128_fn5", (2, 4, 8, 16), 128, 96, 5.0, 0.5),
        ("micro_192_fn10", (2, 4, 8, 16), 192, 144, 10.0, 1.0),
        ("tiny_128_fn10", (3, 6, 12, 24), 128, 96, 10.0, 1.0),
        ("tiny_192_fn10", (3, 6, 12, 24), 192, 144, 10.0, 1.0),
        ("small_128_fn1", (4, 8, 16, 32), 128, 96, 1.0, 0.0),
        ("small_128_fn5", (4, 8, 16, 32), 128, 96, 5.0, 0.5),
        ("small_192_fn1", (4, 8, 16, 32), 192, 144, 1.0, 0.0),
        ("small_192_fn5", (4, 8, 16, 32), 192, 144, 5.0, 0.5),
        ("small_192_fn10", (4, 8, 16, 32), 192, 144, 10.0, 1.0),
        ("medium_192_fn1", (8, 16, 32, 64), 192, 144, 1.0, 0.0),
        ("medium_192_fn5", (8, 16, 32, 64), 192, 144, 5.0, 0.5),
        ("medium_192_fn10", (8, 16, 32, 64), 192, 144, 10.0, 1.0),
    ]
    if args.only:
        wanted = {part.strip() for part in args.only.split(",") if part.strip()}
        variants = [variant for variant in variants if variant[0] in wanted]

    results: list[dict] = []
    best_valid: dict | None = None
    best_valid_score: tuple[float, ...] | None = None
    best_any: dict | None = None
    best_any_score: tuple[float, ...] | None = None
    best_any_model = None
    best_any_config: RoiNetConfig | None = None
    for name, channels, input_width, input_height, fn_weight, tversky_weight in variants:
        log(f"START {name}: channels={channels} input={input_width}x{input_height} fn={fn_weight}")
        config = RoiNetConfig(
            channels=channels,
            input_width=input_width,
            input_height=input_height,
            output_stride=2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=1e-3,
            false_negative_weight=fn_weight,
            tversky_weight=tversky_weight,
            seed=args.seed,
        )
        model = train_roi(
            train.images,
            train.masks,
            train.presence,
            validation.images,
            validation.masks,
            validation.presence,
            config,
            on_epoch=lambda epoch, total, train_loss, val_loss: log(
                f"{name} {epoch}/{total} train={train_loss:.4f} val={val_loss:.4f}"
            )
            if epoch == 1 or epoch == total
            else None,
        )
        probabilities, presence = model.predict(test.images)
        metrics = evaluate_roi(probabilities, presence, test)
        result = {
            "name": name,
            "channels": channels,
            "input": [input_width, input_height],
            "params": roi_parameter_count(model.model),
            "gflops": _flops(model.model, config),
            "latency_ms": _latency_ms(model.model, config, model.device),
            "false_negative_weight": fn_weight,
            "tversky_weight": tversky_weight,
            "metrics": metrics,
            "train_loss": model.history["train_loss"][-1],
            "val_loss": model.history["val_loss"][-1],
        }
        results.append(result)
        log(f"DONE {name}: {json.dumps(result)}")
        (OUT_DIR / f"{args.result_name}.json").write_text(json.dumps(results, indent=2))
        score = (
            metrics["center_containment"],
            metrics["positive_mask_iou"],
            -metrics["false_negative_rate"],
            -metrics["area_ratio"],
        )
        if best_any is None or score > best_any_score:
            best_any = result
            best_any_score = score
            best_any_model = model.model
            best_any_config = config
        usable = (
            metrics["center_containment"] >= 0.98
            and metrics["area_ratio"] <= 1.8
            and metrics["negative_mask_area"] <= 0.02
        )
        if usable and (best_valid is None or score > best_valid_score):
            best_valid = result
            best_valid_score = score
            save_roi_checkpoint(OUT_DIR / args.checkpoint_name, model.model, config)
            (OUT_DIR / f"{args.checkpoint_name}.json").write_text(
                json.dumps(
                    {
                        "name": name,
                        "channels": channels,
                        "input": [input_width, input_height],
                        "output_stride": 2,
                        "false_negative_weight": fn_weight,
                        "tversky_weight": tversky_weight,
                    },
                    indent=2,
                )
            )

    best = best_valid or best_any
    if best_valid is None and best_any is not None and best_any_model is not None:
        # No variant met the containment/area gate; keep the best fallback
        # checkpoint so previews and the report still have a model.
        log("WARNING: no variant met the usable mask gate")
        assert best_any_config is not None
        save_roi_checkpoint(OUT_DIR / args.checkpoint_name, best_any_model, best_any_config)
        (OUT_DIR / f"{args.checkpoint_name}.json").write_text(
            json.dumps(
                {
                    "name": best_any["name"],
                    "channels": best_any["channels"],
                    "input": best_any["input"],
                    "output_stride": 2,
                    "false_negative_weight": best_any["false_negative_weight"],
                    "tversky_weight": best_any["tversky_weight"],
                },
                indent=2,
            )
        )
    if best is not None:
        # Rebuild the best model for previews from its saved checkpoint.
        from blob_lab.roi_net import load_roi_checkpoint

        best_config = RoiNetConfig(
            channels=tuple(best["channels"]),
            input_width=int(best["input"][0]),
            input_height=int(best["input"][1]),
            output_stride=2,
            false_negative_weight=float(best["false_negative_weight"]),
            tversky_weight=float(best["tversky_weight"]),
        )
        best_model = load_roi_checkpoint(
            OUT_DIR / args.checkpoint_name, best_config, device=None
        )
        if best_model is not None:
            probabilities, _ = predict_roi(best_model, test.images, best_config)
            for index in [i for i, meta in enumerate(test.metadata) if meta.get("points")][:4]:
                _preview(
                    test.images[index, :, :, 0],
                    probabilities[index],
                    test.masks[index, :, :, 0],
                    test.metadata[index]["points"],
                    PREVIEW_DIR / f"{best['name']}_{index:03d}.png",
                )
        log(f"best={best['name']} previews={PREVIEW_DIR}")

    (OUT_DIR / f"{args.result_name}.json").write_text(json.dumps(results, indent=2))
    log(f"wrote {OUT_DIR / args.result_name}.json")


if __name__ == "__main__":
    main()
