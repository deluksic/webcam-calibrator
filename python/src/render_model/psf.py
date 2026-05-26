"""Separable Gaussian point-spread function (differentiable w.r.t. sigma)."""

import jax
import jax.numpy as jnp

KERNEL_RADIUS = 5


def gaussian_kernel_1d(sigma: jax.Array) -> jax.Array:
    """Normalized 1D Gaussian kernel; static size, differentiable weights."""
    sigma_safe = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float32), jnp.float32(1e-6))
    offsets = jnp.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * (offsets / sigma_safe) ** 2)
    return kernel / kernel.sum()


def _conv1d_horizontal(image: jax.Array, kernel: jax.Array) -> jax.Array:
    x = image[None, :, :, None]
    weights = kernel[None, :, None, None]
    out = jax.lax.conv_general_dilated(
        x,
        weights,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return out[0, :, :, 0]


def _conv1d_vertical(image: jax.Array, kernel: jax.Array) -> jax.Array:
    x = image[None, :, :, None]
    weights = kernel[:, None, None, None]
    out = jax.lax.conv_general_dilated(
        x,
        weights,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return out[0, :, :, 0]


def apply_gaussian_psf(image: jax.Array, sigma: jax.Array) -> jax.Array:
    """Separable 2D Gaussian blur. ``sigma`` is in units of image pixels."""
    kernel = gaussian_kernel_1d(sigma)
    blurred = _conv1d_horizontal(image, kernel)
    return _conv1d_vertical(blurred, kernel)
