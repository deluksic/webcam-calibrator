"""Direct black/white inference from target photo + PSF-blurred AA classification."""

from __future__ import annotations

import jax.numpy as jnp

from render_model.gamma import invert_gamma
from render_model.pipeline import RenderModelParams, sanitize_levels
from render_model.psf import apply_gaussian_psf
from render_model.renderer import render_tag_antialiased, tag_cell_modules_at_pixels
# Linear thresholds on PSF-blurred AA render (exclude mid-tones / cell edges).
_LINEAR_BLACK_MAX = jnp.float32(0.2)
_LINEAR_WHITE_MIN = jnp.float32(0.8)
# Minimum mask weight (≈ pixel count) per bucket; below this, keep existing levels.
_MIN_LEVEL_SAMPLE_PIXELS = 8.0
# Fixed PSF for classification masks (not the optimizer's current psf_sigma).
_LEVEL_INFERENCE_PSF_SIGMA = jnp.float32(0.8)


def aa_black_white_sample_masks(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    black_level: jnp.ndarray,
    white_level: jnp.ndarray,
    *,
    loss_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-pixel masks from PSF-blurred AA render in linear light.

    Uses fixed σ = 0.8 for blur (see ``_LEVEL_INFERENCE_PSF_SIGMA``). Samples are
    limited to ``loss_mask`` when given (same region as the LM loss); black if
    blurred linear < 0.2, white if > 0.8.

    Classification uses scalar ``black_level`` / ``white_level`` only (no planar
    lighting); LM may still optimize ``light_grad_*`` separately.
    """
    aa = render_tag_antialiased(
        H,
        tag_pattern,
        height,
        width,
        black_level=black_level,
        white_level=white_level,
    )
    blurred = apply_gaussian_psf(aa, _LEVEL_INFERENCE_PSF_SIGMA)
    _, valid = tag_cell_modules_at_pixels(H, tag_pattern, height, width)
    weight = valid if loss_mask is None else valid * loss_mask

    is_black = weight * (blurred < _LINEAR_BLACK_MAX)
    is_white = weight * (blurred > _LINEAR_WHITE_MIN)
    return is_black, is_white


def infer_black_white_from_target(
    target: jnp.ndarray,
    is_black: jnp.ndarray,
    is_white: jnp.ndarray,
    gamma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mean linear light on masked pixels (per-pixel inv-γ, then average)."""
    linear = invert_gamma(target, gamma)
    sum_b = jnp.sum(is_black)
    sum_w = jnp.sum(is_white)
    black = jnp.sum(linear * is_black) / jnp.maximum(sum_b, 1.0)
    white = jnp.sum(linear * is_white) / jnp.maximum(sum_w, 1.0)
    return sanitize_levels(black, white)


def infer_black_white_levels(
    target: jnp.ndarray,
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    params: RenderModelParams,
    *,
    loss_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Linear black/white from target using current γ; masks use fixed PSF σ = 0.8."""
    is_black, is_white = aa_black_white_sample_masks(
        H,
        tag_pattern,
        height,
        width,
        params.black_level,
        params.white_level,
        loss_mask=loss_mask,
    )
    if (
        float(jnp.sum(is_black)) < _MIN_LEVEL_SAMPLE_PIXELS
        or float(jnp.sum(is_white)) < _MIN_LEVEL_SAMPLE_PIXELS
    ):
        return sanitize_levels(params.black_level, params.white_level)
    return infer_black_white_from_target(
        target, is_black, is_white, params.gamma
    )


def camera_with_inferred_levels(
    params: RenderModelParams,
    target: jnp.ndarray,
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    loss_mask: jnp.ndarray | None = None,
) -> RenderModelParams:
    """``params`` with black/white from :func:`infer_black_white_levels`, or unchanged if too few mask pixels."""
    black, white = infer_black_white_levels(
        target,
        H,
        tag_pattern,
        height,
        width,
        params,
        loss_mask=loss_mask,
    )
    return RenderModelParams(
        psf_sigma=params.psf_sigma,
        sharpen_amount=params.sharpen_amount,
        sharpen_sigma=params.sharpen_sigma,
        gamma=params.gamma,
        black_level=black,
        white_level=white,
        light_grad_u=params.light_grad_u,
        light_grad_v=params.light_grad_v,
    )
