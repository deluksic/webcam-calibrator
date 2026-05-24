"""Full camera render pipeline: supersample → PSF → bin → sharpen → gamma."""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

from render_model.renderer import render_tag_antialiased
from render_model.gamma import apply_gamma
from render_model.psf import apply_gaussian_psf
from render_model.sharpen import apply_sharpening

SUPERSAMPLE = 1


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RenderModelParams:
    """Differentiable camera-response parameters (all in output-pixel units)."""

    psf_sigma: jax.Array
    sharpen_amount: jax.Array
    sharpen_sigma: jax.Array
    gamma: jax.Array
    black_level: jax.Array
    white_level: jax.Array


class RenderModelStages(NamedTuple):
    hi_res: jax.Array
    hi_blurred: jax.Array
    binned: jax.Array
    sharpened: jax.Array
    final: jax.Array


def default_params() -> RenderModelParams:
    return RenderModelParams(
        psf_sigma=jnp.float32(0.8),
        sharpen_amount=jnp.float32(0.5),
        sharpen_sigma=jnp.float32(1.0),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.0),
        white_level=jnp.float32(1.0),
    )


def scale_homography(H: jax.Array, factor: jax.Array | float) -> jax.Array:
    """Scale output-side image coordinates (e.g. 4 for 4x supersampling)."""
    scale = jnp.asarray([factor, factor, 1.0], dtype=H.dtype)
    return jnp.diag(scale) @ H


def bin_down(image: jax.Array, factor: int = SUPERSAMPLE) -> jax.Array:
    """Average ``factor×factor`` blocks (sensor integration)."""
    height, width = image.shape
    return image.reshape(height // factor, factor, width // factor, factor).mean(
        axis=(1, 3)
    )


def apply_render_model(image: jax.Array, params: RenderModelParams) -> jax.Array:
    """Post-process an existing image: sharpen → gamma."""
    sharpened = apply_sharpening(image, params.sharpen_amount, params.sharpen_sigma)
    return apply_gamma(sharpened, params.gamma)


def render_with_model_stages(
    H: jax.Array,
    tag_pattern: jax.Array,
    height: int,
    width: int,
    params: RenderModelParams,
    *,
    supersample: int = SUPERSAMPLE,
) -> RenderModelStages:
    """Render tag through the full camera model, returning each pipeline stage."""
    hi_height = height * supersample
    hi_width = width * supersample
    H_hi = scale_homography(H, supersample)

    hi_res = render_tag_antialiased(
        H_hi,
        tag_pattern,
        hi_height,
        hi_width,
        black_level=params.black_level,
        white_level=params.white_level,
    )
    hi_blurred = apply_gaussian_psf(hi_res, params.psf_sigma * supersample)
    binned = bin_down(hi_blurred, supersample)
    sharpened = apply_sharpening(binned, params.sharpen_amount, params.sharpen_sigma)
    final = apply_gamma(sharpened, params.gamma)
    return RenderModelStages(hi_res, hi_blurred, binned, sharpened, final)


def render_with_model(
    H: jax.Array,
    tag_pattern: jax.Array,
    height: int,
    width: int,
    params: RenderModelParams,
    *,
    supersample: int = SUPERSAMPLE,
) -> jax.Array:
    """Render tag through supersample → PSF → bin → sharpen → gamma."""
    return render_with_model_stages(
        H,
        tag_pattern,
        height,
        width,
        params,
        supersample=supersample,
    ).final
