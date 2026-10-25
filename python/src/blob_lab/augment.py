"""Camera-style sharpening augmentation for synthetic training data.

Real cameras do not output the clean optical image: the ISP applies one or more
edge-enhancement stages (unsharp masking, coring, two-scale detail, overshoot
limiting) before the frame is written.  The synthetic target renders only model
the lens/PSF, so the CNN and the point detector never see those halos.

This module is deliberately NumPy/OpenCV rather than the differentiable JAX
``render_model.sharpen`` path: dataset generation is a one-shot CPU operation,
and the camera family here also covers thresholded and multi-scale variants that
the differentiable camera fit does not need.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraSharpenConfig:
    """Distribution of camera sharpening applied to a dataset sample.

    ``probability`` leaves a fraction of samples completely clean.  The other
    parameters are sampled per image; ``amount`` can be zero even when the
    sample is selected, so a selected sample may still be a no-op.
    """

    probability: float = 0.7
    amount: tuple[float, float] = (0.0, 1.5)
    sigma: tuple[float, float] = (0.5, 2.0)
    threshold: tuple[float, float] = (0.0, 0.01)
    multiscale_probability: float = 0.3
    display_space_probability: float = 0.5
    overshoot_clip_probability: float = 0.5
    overshoot_limit: tuple[float, float] = (0.1, 0.3)
    # Real positive frames were captured by a real camera and already contain
    # in-camera sharpening; only double-sharpen them occasionally.
    real_positive_probability: float = 0.25


@dataclass(frozen=True)
class SharpenParams:
    mode: str  # "usm" | "multiscale"
    amount: float
    sigma: float
    threshold: float
    space: str  # "linear" | "display"
    secondary_sigma: float = 0.0
    secondary_weight: float = 0.0
    overshoot_limit: float | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "mode": self.mode,
            "amount": float(self.amount),
            "sigma": float(self.sigma),
            "threshold": float(self.threshold),
            "space": self.space,
            "secondary_sigma": float(self.secondary_sigma),
            "secondary_weight": float(self.secondary_weight),
            "overshoot_limit": None
            if self.overshoot_limit is None
            else float(self.overshoot_limit),
        }


def sample_augmentation_rng(seed: int, index: int) -> np.random.Generator:
    """Independent per-sample RNG.

    Keeping augmentation draws out of the scene RNG means toggling sharpening
    on or off does not change poses, colours, or truth points.  That is what
    makes clean/sharpened runs directly comparable.
    """
    return np.random.default_rng((int(seed), int(index), 0x5A17))


def sample_sharpen_params(
    config: CameraSharpenConfig,
    rng: np.random.Generator,
    *,
    probability: float | None = None,
) -> SharpenParams | None:
    selected = config.probability if probability is None else float(probability)
    if selected <= 0.0 or rng.random() >= min(1.0, selected):
        return None

    amount = float(rng.uniform(*config.amount))
    sigma = float(rng.uniform(*config.sigma))
    threshold = float(rng.uniform(*config.threshold))
    mode = "multiscale" if rng.random() < config.multiscale_probability else "usm"
    space = "display" if rng.random() < config.display_space_probability else "linear"
    secondary_sigma = sigma * float(rng.uniform(1.5, 2.5))
    secondary_weight = float(rng.uniform(0.3, 0.7))
    overshoot_limit = (
        float(rng.uniform(*config.overshoot_limit))
        if rng.random() < config.overshoot_clip_probability
        else None
    )
    return SharpenParams(
        mode=mode,
        amount=amount,
        sigma=sigma,
        threshold=threshold,
        space=space,
        secondary_sigma=secondary_sigma,
        secondary_weight=secondary_weight,
        overshoot_limit=overshoot_limit,
    )


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(np.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _apply_usm(image: np.ndarray, params: SharpenParams) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=max(1e-3, float(params.sigma)),
        sigmaY=max(1e-3, float(params.sigma)),
        borderType=cv2.BORDER_REFLECT,
    )
    detail = image - blurred
    if params.mode == "multiscale":
        coarse = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=max(1e-3, float(params.secondary_sigma)),
            sigmaY=max(1e-3, float(params.secondary_sigma)),
            borderType=cv2.BORDER_REFLECT,
        )
        detail = (
            float(params.secondary_weight) * detail
            + (1.0 - float(params.secondary_weight)) * (image - coarse)
        )
    threshold = max(0.0, float(params.threshold))
    if threshold > 0.0:
        detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
    sharpened = image + float(params.amount) * detail
    if params.overshoot_limit is not None:
        limit = max(0.0, float(params.overshoot_limit))
        sharpened = image + np.clip(sharpened - image, -limit, limit)
    return sharpened


def apply_camera_sharpen(
    image: np.ndarray,
    params: SharpenParams | None,
) -> np.ndarray:
    """Apply a sampled camera sharpening stage to a linear-light image.

    ``image`` may be ``(H, W)``, ``(H, W, 1)``, or ``(H, W, C)``.  The operation
    is applied independently to each channel, which matches a luma/chroma ISP
    closely enough for a single-channel synthetic dataset.  The output has the
    same shape as the input and is clipped to ``[0, 1]``.
    """
    if params is None:
        return np.asarray(image, dtype=np.float32)
    array = np.asarray(image, dtype=np.float32)
    original_shape = array.shape
    squeezed = array[:, :, 0] if array.ndim == 3 and array.shape[2] == 1 else array
    working = np.asarray(squeezed, dtype=np.float32)
    if working.ndim == 2:
        channels = [working]
    elif working.ndim == 3:
        channels = [working[:, :, index] for index in range(working.shape[2])]
    else:
        raise ValueError(f"expected an image with 2 or 3 dimensions, got {array.shape}")

    display_space = params.space == "display"
    result = []
    for channel in channels:
        source = linear_to_srgb(channel) if display_space else np.clip(channel, 0.0, 1.0)
        sharpened = _apply_usm(source, params)
        if display_space:
            sharpened = srgb_to_linear(sharpened)
        result.append(np.clip(sharpened, 0.0, 1.0))
    output = result[0] if working.ndim == 2 else np.stack(result, axis=-1)
    return output.reshape(original_shape).astype(np.float32)
