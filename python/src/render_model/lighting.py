"""Planar multiplicative exposure in tag space (before PSF)."""

import jax.numpy as jnp

from render_model.tag_data import TAG_GRID_SIZE

# Max |α|, |β| so s = 1 + α·u + β·v stays in a modest range for u,v ∈ [-1, 1].
MAX_PLANAR_EXPOSURE_SLOPE = jnp.float32(0.08)
_MIN_EXPOSURE_SCALE = jnp.float32(0.5)
_TAG_CENTER = jnp.float32(TAG_GRID_SIZE) / 2.0
_TAG_HALF_EXTENT = _TAG_CENTER


def tag_normalized_coords(tx: jnp.ndarray, ty: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Centered tag coords in ≈[-1, 1] from inverse-warp centers (8×8 module grid)."""
    u = (tx - _TAG_CENTER) / _TAG_HALF_EXTENT
    v = (ty - _TAG_CENTER) / _TAG_HALF_EXTENT
    return u, v


def planar_exposure_scale(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    grad_u: jnp.ndarray | float,
    grad_v: jnp.ndarray | float,
) -> jnp.ndarray:
    """Multiplicative scale s = 1 + α·u + β·v with bounded α, β."""
    grad_u = jnp.asarray(grad_u, dtype=jnp.float32)
    grad_v = jnp.asarray(grad_v, dtype=jnp.float32)
    u, v = tag_normalized_coords(tx, ty)
    s = 1.0 + grad_u * u + grad_v * v
    return jnp.maximum(s, _MIN_EXPOSURE_SCALE)


def apply_planar_exposure(
    linear: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    grad_u: jnp.ndarray | float,
    grad_v: jnp.ndarray | float,
) -> jnp.ndarray:
    """Apply planar exposure to a linear AA render (same shape as ``linear``)."""
    return linear * planar_exposure_scale(tx, ty, grad_u, grad_v)


def bounded_light_grads(raw_u: jnp.ndarray, raw_v: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map unconstrained optimizer scalars → bounded physical α, β."""
    from jax.nn import tanh

    return (
        MAX_PLANAR_EXPOSURE_SLOPE * tanh(raw_u),
        MAX_PLANAR_EXPOSURE_SLOPE * tanh(raw_v),
    )


def raw_from_light_grads(
    grad_u: jnp.ndarray | float,
    grad_v: jnp.ndarray | float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse of :func:`bounded_light_grads` for optimization packing."""
    scale = MAX_PLANAR_EXPOSURE_SLOPE
    u = jnp.asarray(grad_u, dtype=jnp.float32) / scale
    v = jnp.asarray(grad_v, dtype=jnp.float32) / scale
    return jnp.arctanh(jnp.clip(u, -0.999, 0.999)), jnp.arctanh(jnp.clip(v, -0.999, 0.999))
