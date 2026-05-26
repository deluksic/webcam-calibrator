"""Power-law gamma tone response."""

import jax
import jax.numpy as jnp


def apply_gamma(image: jax.Array, gamma: jax.Array) -> jax.Array:
    """Apply ``x ** (1/gamma)`` on non-negative linear light (small floor at zero)."""
    eps = jnp.float32(1e-6)
    x = jnp.maximum(image, eps)
    return x ** (1.0 / gamma)


def invert_gamma(display: jax.Array, gamma: jax.Array) -> jax.Array:
    """Map display values back to linear light: ``display ** gamma``."""
    eps = jnp.float32(1e-6)
    x = jnp.maximum(display, eps)
    return x ** gamma
