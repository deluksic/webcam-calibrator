"""Unsharp-mask sharpening (differentiable w.r.t. amount and sigma)."""

import jax
import jax.numpy as jnp

from render_model.psf import apply_gaussian_psf


def apply_sharpening(
    image: jax.Array,
    amount: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Unsharp mask: ``image + amount * (image - blurred)``."""
    blurred = apply_gaussian_psf(image, sigma)
    return image + amount * (image - blurred)
