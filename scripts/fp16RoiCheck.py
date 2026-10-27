"""Measure fp16 inference error for the production ROI net on real dataset frames.

Compares f32 vs f16 forward passes over the checkpoint that produced
the deployed ROI model, using the same per-image z-score normalisation as
training and as src/gpu/roiLab/roiLabPipeline.ts:191.

This is a pessimistic bound: PyTorch rounds *all* activations and accumulates
in f16, whereas runntime's conv kernels accumulate several taps in f32.
"""

import sys
from pathlib import Path

import numpy as np
import torch

PY = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PY / "src"))

from blob_lab import load_roi_checkpoint  # noqa: E402

DATA = PY / "notebooks" / "blob_lab_data"
CKPT = DATA / "roi_net_best.pt"


def luma_bilinear(rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """sRGB bytes -> luma, bilinear-resized to (h, w), like the GPU sampler."""
    rgb = rgb.astype(np.float32) / 255.0
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    src = torch.from_numpy(luma)[None, None]
    out = torch.nn.functional.interpolate(
        src, size=(h, w), mode="bilinear", align_corners=False
    )
    return out[0, 0].numpy()


def normalize(images: np.ndarray) -> np.ndarray:
    mean = images.mean(axis=(1, 2), keepdims=True)
    std = images.std(axis=(1, 2), keepdims=True)
    return (images - mean) / np.maximum(std, 1e-5)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def main() -> None:
    from PIL import Image

    model = load_roi_checkpoint(CKPT, None, device="cpu")
    if model is None:
        raise SystemExit(f"could not load {CKPT}")
    model.eval()

    # Prefer MPS for a GPU-faithful f16 path; fall back to CPU.
    half_device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"half device: {half_device}")

    model_h = load_roi_checkpoint(CKPT, None, device=half_device).eval().half()

    samples = sorted(p for p in DATA.glob("roi_dataset_preview/sample_*.png") if "_mask" not in p.name)
    print(f"samples: {len(samples)}")

    heights = [96, 112, 192, 224]
    W = 256

    rows = []
    for h in heights:
        imgs = []
        for p in samples:
            rgb = np.asarray(Image.open(p).convert("RGB"))
            imgs.append(luma_bilinear(rgb, W, h))
        batch = normalize(np.stack(imgs))
        x32 = torch.from_numpy(batch[:, None].astype(np.float32))

        with torch.no_grad():
            # forward() returns (mask_logits, presence); sigmoid is applied by the caller.
            p32 = torch.sigmoid(model(x32)[0]).numpy()
            p16 = torch.sigmoid(model_h(x32.to(half_device).half())[0]).float().cpu().numpy()

        d = np.abs(p16 - p32)
        ious = [iou(p32 > t, p16 > t) for t in (0.3, 0.5, 0.7)]
        flips = float((np.abs(p16 - p32) > 0.5 * np.maximum(p32, 1e-6)).mean())
        rows.append(
            {
                "h": h,
                "max": float(d.max()),
                "mean": float(d.mean()),
                "p999": float(np.quantile(d, 0.999)),
                "iou50": ious[1],
                "iou30": ious[0],
                "iou70": ious[2],
                "maxprob_err": float(np.abs(p16.max(axis=(1, 2)) - p32.max(axis=(1, 2))).max()),
            }
        )
        del p32, p16, x32

    print(
        f"\n{'H':>4} {'maxΔ':>9} {'meanΔ':>9} {'p99.9Δ':>9} "
        f"{'IoU@.3':>8} {'IoU@.5':>8} {'IoU@.7':>8} {'maxProbΔ':>9}"
    )
    for r in rows:
        print(
            f"{r['h']:>4} {r['max']:>9.2e} {r['mean']:>9.2e} {r['p999']:>9.2e} "
            f"{r['iou30']:>8.5f} {r['iou50']:>8.5f} {r['iou70']:>8.5f} {r['maxprob_err']:>9.2e}"
        )


if __name__ == "__main__":
    main()
