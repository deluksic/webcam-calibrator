"""Direct PyTorch inference for the ROI segmenter.

The browser runs the checkpoint through the vendored runntime core; these
helpers reproduce the research preprocessing exactly so scripts can score
masks without a graph runtime in the loop. Inputs are HxWx3 RGB frames in
[0,1], as read from a camera or image file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from blob_lab.roi_net import TinyRoiNet, load_roi_checkpoint

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "notebooks" / "blob_lab_data" / "roi_net_best.pt"
)
MODEL_WIDTH = 256
STATS_SIZE = (64, 36)
LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)

_MODELS: dict[tuple[Path, str], TinyRoiNet] = {}


def load_roi_model(
    checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> TinyRoiNet:
    """Load and cache an eval-mode model for ``checkpoint`` on ``device``."""
    path = Path(checkpoint).resolve() if checkpoint is not None else DEFAULT_CHECKPOINT
    key = (path, str(device))
    cached = _MODELS.get(key)
    if cached is not None:
        return cached
    model = load_roi_checkpoint(path, None, device)
    if model is None:
        raise FileNotFoundError(f"Could not load ROI checkpoint {path}")
    model.eval()
    _MODELS[key] = model
    return model


def model_height(width: int, height: int) -> int:
    """Aspect-preserving model height, a multiple of 16 in [96, 224]."""
    raw = round(MODEL_WIDTH * height / width)
    return min(224, max(96, int(np.ceil(raw / 16.0)) * 16))


def luma_statistics(image_rgb: np.ndarray) -> tuple[float, float]:
    """Frame-level luma mean/std from the 64x36 area-downscaled stats image."""
    small = cv2.resize(image_rgb, STATS_SIZE, interpolation=cv2.INTER_AREA)
    luma = small @ LUMA
    return float(luma.mean()), max(0.05, float(luma.std()))


def roi_mask_model_resolution(
    image_rgb: np.ndarray,
    *,
    model: TinyRoiNet | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Predict the probability map plus the model-space luma it was built from.

    Returns ``(probability, luma, height)``, all float32 at ``(height, 256)``.
    ``luma`` is the resized luma *before* standardization; the background
    estimator needs it alongside the mask.
    """
    rgb = np.asarray(image_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {rgb.shape}")
    image_height, image_width = rgb.shape[:2]
    if image_height == 0 or image_width == 0:
        raise ValueError("Empty image")

    mean, std = luma_statistics(rgb)
    height = model_height(image_width, image_height)
    resized = cv2.resize(rgb, (MODEL_WIDTH, height), interpolation=cv2.INTER_LINEAR)
    luma = (resized @ LUMA).astype(np.float32)
    normalized = ((luma - mean) / std)[None, None].astype(np.float32)

    model = model or load_roi_model()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits, _presence = model(torch.from_numpy(normalized).to(device))
        probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    probability = cv2.resize(
        probability, (MODEL_WIDTH, height), interpolation=cv2.INTER_LINEAR
    )
    return np.ascontiguousarray(probability, dtype=np.float32), luma, height


def roi_mask_probability(
    image_rgb: np.ndarray,
    *,
    model: TinyRoiNet | None = None,
) -> tuple[np.ndarray, int]:
    """Predict the target-area probability map and its model height.

    Returns ``(probability, height)`` where ``probability`` is float32 at
    ``(height, 256)``. The frame is standardized with its 64x36 luma
    statistics before inference.
    """
    probability, _luma, height = roi_mask_model_resolution(image_rgb, model=model)
    return probability, height
