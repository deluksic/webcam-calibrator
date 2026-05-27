"""LM / level-inference region mask from warped tag corners in image space."""

import jax.numpy as jnp

from render_model.renderer import _pixel_centers

# AprilTag 8×8 module grid: one black border cell per side → moat = side / 8.
_APRILTAG_MOAT_SIDE_FRACTION = jnp.float32(1.0 / 8.0)
_MIN_MOAT_PX = jnp.float32(3.0)


def _segment_distance_sq(
    px: jnp.ndarray,
    py: jnp.ndarray,
    x0: jnp.ndarray,
    y0: jnp.ndarray,
    x1: jnp.ndarray,
    y1: jnp.ndarray,
) -> jnp.ndarray:
    """Squared distance from (px, py) to the closed segment (x0, y0)-(x1, y1)."""
    dx = x1 - x0
    dy = y1 - y0
    len_sq = dx * dx + dy * dy + jnp.float32(1e-12)
    t = jnp.clip(((px - x0) * dx + (py - y0) * dy) / len_sq, 0.0, 1.0)
    qx = x0 + t * dx
    qy = y0 + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def _inside_convex_quad(
    px: jnp.ndarray,
    py: jnp.ndarray,
    corners: jnp.ndarray,
) -> jnp.ndarray:
    """True where pixel centers lie inside the convex quad (any consistent winding)."""
    n = corners.shape[0]
    crosses = []
    for i in range(n):
        x0, y0 = corners[i, 0], corners[i, 1]
        x1, y1 = corners[(i + 1) % n, 0], corners[(i + 1) % n, 1]
        crosses.append((px - x0) * (y1 - y0) - (py - y0) * (x1 - x0))
    cross_stack = jnp.stack(crosses, axis=0)
    return jnp.all(cross_stack >= 0.0, axis=0) | jnp.all(cross_stack <= 0.0, axis=0)


def _quad_min_side_length(corners: jnp.ndarray) -> jnp.float32:
    """Shortest edge length of the quad in image space (Euclidean)."""
    n = corners.shape[0]
    lengths = []
    for i in range(n):
        j = (i + 1) % n
        dx = corners[j, 0] - corners[i, 0]
        dy = corners[j, 1] - corners[i, 1]
        lengths.append(jnp.sqrt(dx * dx + dy * dy))
    return jnp.min(jnp.stack(lengths))


def _april_tag_moat_pixels(corners: jnp.ndarray) -> jnp.float32:
    """Inward moat width in pixels: max(3, ⅛ of shortest side)."""
    april_tag = _quad_min_side_length(corners) * _APRILTAG_MOAT_SIDE_FRACTION
    return jnp.maximum(april_tag, _MIN_MOAT_PX)


def _tag_quad_mask(
    corners: jnp.ndarray,
    height: int,
    width: int,
    margin: jnp.ndarray | float,
) -> jnp.ndarray:
    """Per-pixel weights for the warped tag quad plus an edge moat in pixels."""
    cx, cy = _pixel_centers(height, width)
    inside = _inside_convex_quad(cx, cy, corners)
    n = corners.shape[0]
    edge_dists = []
    for i in range(n):
        x0, y0 = corners[i, 0], corners[i, 1]
        x1, y1 = corners[(i + 1) % n, 0], corners[(i + 1) % n, 1]
        edge_dists.append(_segment_distance_sq(cx, cy, x0, y0, x1, y1))
    min_dist = jnp.sqrt(jnp.min(jnp.stack(edge_dists, axis=0), axis=0))
    margin_f = jnp.asarray(margin, dtype=jnp.float32)
    return (inside | (min_dist <= margin_f)).astype(jnp.float32)


def loss_mask_from_corners(
    corners: jnp.ndarray,
    height: int,
    width: int,
) -> jnp.ndarray:
    """Mask for LM loss and level inference: inside the warped tag quad plus edge moat.

    Moat width is max(3 px, ⅛ of shortest side), matching AprilTag border scale on
    large tags and a minimum inset on small ones.
    """
    return _tag_quad_mask(corners, height, width, _april_tag_moat_pixels(corners))
