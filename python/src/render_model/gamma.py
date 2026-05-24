"""Power-law gamma tone response."""

import jax
import jax.numpy as jnp


def apply_gamma(image: jax.Array, gamma: jax.Array) -> jax.Array:
    """Apply ``x ** (1/gamma)`` with a small floor so zero-crossings stay differentiable."""
    eps = jnp.float32(1e-6)
    x = jnp.clip(image, eps, 1.0)
    return x ** (1.0 / gamma)
