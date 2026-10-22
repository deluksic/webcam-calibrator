"""Differentiable full-target rendering in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.ndimage

from render_model.psf import apply_gaussian_psf


def _rotation(yaw: jax.Array, pitch: jax.Array, roll: jax.Array) -> jax.Array:
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    ry = jnp.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = jnp.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def camera_homography(
    *,
    yaw_deg: jax.Array,
    pitch_deg: jax.Array,
    roll_deg: jax.Array,
    distance: jax.Array,
    focal: jax.Array,
    tx: jax.Array,
    ty: jax.Array,
    target_size: tuple[int, int],
    output_size: tuple[int, int],
) -> jax.Array:
    """Homography from target pixel coordinates to output pixel coordinates.

    The target plane is centred at the origin.  Coordinates are normalized by
    target width so that camera distance/focal values are resolution-independent.
    """
    target_w, target_h = target_size
    out_w, out_h = output_size
    rotation = _rotation(
        jnp.deg2rad(jnp.asarray(yaw_deg, dtype=jnp.float32)),
        jnp.deg2rad(jnp.asarray(pitch_deg, dtype=jnp.float32)),
        jnp.deg2rad(jnp.asarray(roll_deg, dtype=jnp.float32)),
    )
    f = jnp.asarray(focal, dtype=jnp.float32) * float(min(out_w, out_h))
    d = jnp.asarray(distance, dtype=jnp.float32)
    cx = out_w / 2.0 + jnp.asarray(tx, dtype=jnp.float32) * out_w
    cy = out_h / 2.0 + jnp.asarray(ty, dtype=jnp.float32) * out_h
    tz = d
    tx3 = jnp.asarray(tx, dtype=jnp.float32) * d
    ty3 = jnp.asarray(ty, dtype=jnp.float32) * d
    r00, r01, r02 = rotation[0, 0], rotation[0, 1], rotation[0, 2]
    r10, r11, r12 = rotation[1, 0], rotation[1, 1], rotation[1, 2]
    r20, r21, r22 = rotation[2, 0], rotation[2, 1], rotation[2, 2]
    h_norm = jnp.array(
        [
            [f * r00 + cx * r20, f * r01 + cx * r21, f * tx3 + cx * tz],
            [f * r10 + cy * r20, f * r11 + cy * r21, f * ty3 + cy * tz],
            [r20, r21, tz],
        ],
        dtype=jnp.float32,
    )
    # target pixels -> normalized target coords
    to_normalized = jnp.array(
        [[1.0 / target_w, 0.0, -0.5], [0.0, 1.0 / target_w, -0.5], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )
    return h_norm @ to_normalized


def apply_homography(h: jax.Array, points: jax.Array) -> jax.Array:
    ones = jnp.ones((points.shape[0], 1), dtype=points.dtype)
    mapped = (h @ jnp.concatenate([points, ones], axis=1).T).T
    return mapped[:, :2] / mapped[:, 2:3]


def render_target(
    h: jax.Array,
    texture: jax.Array,
    *,
    output_size: tuple[int, int],
    supersample: int = 1,
    psf_sigma: jax.Array | float = 0.0,
    background: float = 0.5,
) -> jax.Array:
    """Render the full target through ``h`` with a differentiable bilinear warp."""
    out_w, out_h = output_size
    hi_w, hi_h = out_w * supersample, out_h * supersample
    sx = jnp.array([[float(supersample), 0.0, 0.0], [0.0, float(supersample), 0.0], [0.0, 0.0, 1.0]])
    h_hi = sx @ h
    inv = jnp.linalg.inv(h_hi)
    px = jnp.arange(hi_w, dtype=jnp.float32) + 0.5
    py = jnp.arange(hi_h, dtype=jnp.float32) + 0.5
    px_grid, py_grid = jnp.meshgrid(px, py)
    ones = jnp.ones_like(px_grid)
    mapped = jnp.einsum("ij,jhw->ihw", inv, jnp.stack([px_grid, py_grid, ones], axis=0))
    tx = mapped[0] / mapped[2]
    ty = mapped[1] / mapped[2]
    sampled = jax.scipy.ndimage.map_coordinates(
        texture,
        jnp.stack([ty - 0.5, tx - 0.5], axis=0),
        order=1,
        mode="constant",
        cval=background,
    )
    if supersample > 1:
        sampled = sampled.reshape(out_h, supersample, out_w, supersample).mean(axis=(1, 3))
    sampled = jnp.clip(sampled, 0.0, 1.0)
    return apply_gaussian_psf(sampled, jnp.asarray(psf_sigma, dtype=jnp.float32))


def linear_to_srgb(image: jax.Array) -> jax.Array:
    return jnp.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * jnp.power(jnp.maximum(image, 0.0), 1.0 / 2.4) - 0.055,
    )
