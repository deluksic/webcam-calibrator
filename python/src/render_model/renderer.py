"""Differentiable tag renderer (inverse warp + Iñigo analytical filtering).

Image coordinates (edge origin):
  (0, 0)     — top-left corner of the image
  (0.5, 0.5) — center of pixel column 0, row 0
  Pixel (px, py) spans [px, px + 1) × [py, py + 1) in (x, y).

Tag coordinates use the same edge convention: cell index i covers [i, i + 1).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from render_model.esf import ESFProfile, ramp_frac_lookup

RenderFn = Callable[[jnp.ndarray, jnp.ndarray, int, int], jnp.ndarray]

# White halo around the tag grid so outer black-border edges AA like interior cells.
TAG_PAD_CELLS = 4


def _pixel_indices(height: int, width: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integer pixel column/row indices (0 … W−1, 0 … H−1)."""
    px = jnp.arange(width, dtype=jnp.float32)
    py = jnp.arange(height, dtype=jnp.float32)
    px_grid, py_grid = jnp.meshgrid(px, py)
    return px_grid, py_grid


def _pixel_centers(height: int, width: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Image (x, y) at the center of each output pixel."""
    px, py = _pixel_indices(height, width)
    return px + 0.5, py + 0.5


def inverse_warp_coords(
    H: jnp.ndarray,
    height: int,
    width: int,
    px_grid: jnp.ndarray | None = None,
    py_grid: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map image (x, y) points to tag-space (tx, ty) via H⁻¹.

    ``px_grid`` / ``py_grid`` are image coordinates in edge space; default is
    each output pixel's center (px + 0.5, py + 0.5).
    """
    if px_grid is None or py_grid is None:
        px_grid, py_grid = _pixel_centers(height, width)

    H_inv = jnp.linalg.inv(H)
    ones = jnp.ones_like(px_grid)
    pts = jnp.stack([px_grid, py_grid, ones], axis=0)
    mapped = jnp.einsum("ij,jhw->ihw", H_inv, pts)
    return mapped[0] / mapped[2], mapped[1] / mapped[2]


def tag_footprint(
    H: jnp.ndarray,
    height: int,
    width: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Tag-space position and screen derivatives at each pixel center."""
    cx, cy = _pixel_centers(height, width)
    tx, ty = inverse_warp_coords(H, height, width, cx, cy)
    tx_dx, ty_dx = inverse_warp_coords(H, height, width, cx + 1.0, cy)
    tx_dy, ty_dy = inverse_warp_coords(H, height, width, cx, cy + 1.0)
    ddx = jnp.stack([tx_dx - tx, ty_dx - ty], axis=0)
    ddy = jnp.stack([tx_dy - tx, ty_dy - ty], axis=0)
    return tx, ty, ddx, ddy


def _iq_footprint(ddx: jnp.ndarray, ddy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Box-filter extent in tag space (Iñigo: w = max(|dpdx|, |dpdy|))."""
    wx = jnp.maximum(jnp.abs(ddx[0]), jnp.abs(ddy[0]))
    wy = jnp.maximum(jnp.abs(ddx[1]), jnp.abs(ddy[1]))
    return wx, wy


def _iq_axis_integral(a: jnp.ndarray, b: jnp.ndarray, w: jnp.ndarray, n: float) -> jnp.ndarray:
    """Axis integral from filterableprocedurals (box filter over N subdivisions per unit)."""
    fa = jnp.mod(a, 1.0)
    fb = jnp.mod(b, 1.0)
    return (
        jnp.floor(a)
        + jnp.minimum(fa * n, 1.0)
        - jnp.floor(b)
        - jnp.minimum(fb * n, 1.0)
    ) / (n * w + 1e-8)


def _iq_integrals(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Global N=2 axis integrals; tag coords ÷2 so one Iq period spans two modules."""
    n = 2.0
    scale = 0.5
    ux, uy = tx * scale, ty * scale
    wx, wy = _iq_footprint(ddx * scale, ddy * scale)
    ax, bx = ux + 0.5 * wx, ux - 0.5 * wx
    ay, by = uy + 0.5 * wy, uy - 0.5 * wy
    return _iq_axis_integral(ax, bx, wx, n), _iq_axis_integral(ay, by, wy, n)


def _esf_axis_integral(
    a: jnp.ndarray,
    b: jnp.ndarray,
    w: jnp.ndarray,
    profile: ESFProfile,
) -> jnp.ndarray:
    """Average white fraction in [b, a] for N=2 period, ESF-blurred (replaces Iñigo box ramp)."""
    n = 2.0
    fa = jnp.mod(a, 1.0)
    fb = jnp.mod(b, 1.0)
    return (
        jnp.floor(a)
        + ramp_frac_lookup(fa, w, profile)
        - jnp.floor(b)
        - ramp_frac_lookup(fb, w, profile)
    ) / (n * w + 1e-8)


def _esf_integrals(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
    profile: ESFProfile,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Global N=2 axis integrals with ESF blur (tag coords ÷2, one period = two modules)."""
    scale = 0.5
    ux, uy = tx * scale, ty * scale
    wx, wy = _iq_footprint(ddx * scale, ddy * scale)
    ax, bx = ux + 0.5 * wx, ux - 0.5 * wx
    ay, by = uy + 0.5 * wy, uy - 0.5 * wy
    return _esf_axis_integral(ax, bx, wx, profile), _esf_axis_integral(ay, by, wy, profile)


def _pad_tag_pattern(
    tag_pattern: jnp.ndarray,
    *,
    pad: int = TAG_PAD_CELLS,
    fill: float = 1.0,
) -> jnp.ndarray:
    """Embed tag in a white halo so exterior edges are grid steps, not hard fills."""
    h, w = tag_pattern.shape
    out = jnp.full((h + 2 * pad, w + 2 * pad), fill, dtype=tag_pattern.dtype)
    return out.at[pad : pad + h, pad : pad + w].set(tag_pattern)


def _sample_tag_periodic(
    tag_pattern: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
    ix: jnp.ndarray,
    iy: jnp.ndarray,
    *,
    pad: int = TAG_PAD_CELLS,
    black_level: jax.Array | float = 0.0,
    white_level: jax.Array | float = 1.0,
    scale_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Closest 2×2 cells with global periodic Iq / ESF weights.

    Tag cells are binary (0/1); each lookup returns ``black_level`` or
    ``white_level`` directly. Edge pixels get the box-filtered blend of those
    levels — not a [0, 1] coverage remapped afterward.
    """
    black_level = jnp.asarray(black_level, dtype=jnp.float32)
    white_level = jnp.asarray(white_level, dtype=jnp.float32)
    padded = _pad_tag_pattern(
        tag_pattern, pad=pad, fill=1.0
    )  # binary; leveled below per cell
    grid_h, grid_w = padded.shape
    i0, j0 = _closest_block_origin(tx, ty)

    out = jnp.zeros_like(tx)
    for di in (0, 1):
        for dj in (0, 1):
            ci = i0 + di
            cj = j0 + dj
            w = _parity_weight(ix, iy, jnp.mod(ci, 2), jnp.mod(cj, 2))
            pi = ci + pad
            pj = cj + pad
            inside = (pi >= 0) & (pi < grid_w) & (pj >= 0) & (pj < grid_h)
            pi_i = jax.lax.stop_gradient(pi)
            pj_i = jax.lax.stop_gradient(pj)
            module = jnp.where(inside, padded[pj_i, pi_i], jnp.float32(1.0))
            level = black_level + module * (white_level - black_level)
            out = out + level * w
    if scale_fn is not None:
        return scale_fn(out)
    return out


def sample_tag_esf(
    tag_pattern: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
    profile: ESFProfile,
    *,
    outside: float = 1.0,
) -> jnp.ndarray:
    """ESF antialiased tag lookup: sub-pixel edge distance via learned ESF, not box filter."""
    ix, iy = _esf_integrals(tx, ty, ddx, ddy, profile)
    return _sample_tag_periodic(
        tag_pattern,
        tx,
        ty,
        ddx,
        ddy,
        ix,
        iy,
        black_level=profile.i_black,
        white_level=profile.i_white,
        scale_fn=None,
    )


def render_tag_esf(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    profile: ESFProfile,
) -> jnp.ndarray:
    """Inverse warp + ESF antialiasing (sub-pixel edge profile, no post-process blur)."""
    tx, ty, ddx, ddy = tag_footprint(H, height, width)
    return sample_tag_esf(tag_pattern, tx, ty, ddx, ddy, profile)


def _closest_block_origin(tx: jnp.ndarray, ty: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-left of the closest 2×2 tag-cell block (pixel-centered, not corner-aligned).

    Cell indices are detached from autodiff — gradients flow only through Iq weights.
    """
    tx_i = jax.lax.stop_gradient(tx)
    ty_i = jax.lax.stop_gradient(ty)
    return jnp.floor(tx_i - 0.5).astype(jnp.int32), jnp.floor(ty_i - 0.5).astype(jnp.int32)


def _parity_weight(ix: jnp.ndarray, iy: jnp.ndarray, pi: jnp.ndarray, pj: jnp.ndarray) -> jnp.ndarray:
    """Global periodic Iq weight for cell parity (pi, pj). Four terms sum to 1."""
    pi = jax.lax.stop_gradient(pi)
    pj = jax.lax.stop_gradient(pj)
    w_tl = (1.0 - ix) * (1.0 - iy)
    w_tr = ix * (1.0 - iy)
    w_bl = (1.0 - ix) * iy
    w_br = ix * iy
    return jnp.where(
        pi == 0,
        jnp.where(pj == 0, w_br, w_tr),
        jnp.where(pj == 0, w_bl, w_tl),
    )


def filtered_cell_coverage(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
    *,
    ci: jnp.ndarray,
    cj: jnp.ndarray,
) -> jnp.ndarray:
    """Box-filtered coverage of tag cell (ci, cj) in the closest 2×2 neighborhood."""
    ix, iy = _iq_integrals(tx, ty, ddx, ddy)
    i0, j0 = _closest_block_origin(tx, ty)
    in_block = (ci >= i0) & (ci <= i0 + 1) & (cj >= j0) & (cj <= j0 + 1)
    return jnp.where(in_block, _parity_weight(ix, iy, jnp.mod(ci, 2), jnp.mod(cj, 2)), 0.0)


def filtered_squares(
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
) -> jnp.ndarray:
    """Box-filtered white top-left cell in each N×N period (Iñigo filterableprocedurals)."""
    ix, iy = _iq_integrals(tx, ty, ddx, ddy)
    return ix * iy


def sample_tag_filtered(
    tag_pattern: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    ddx: jnp.ndarray,
    ddy: jnp.ndarray,
    *,
    black_level: jax.Array | float = 0.0,
    white_level: jax.Array | float = 1.0,
) -> jnp.ndarray:
    """Closest 2×2 tag cells; global periodic Iq weights from each cell's parity.

    Forward pass picks cells via floor(tx, ty); backward pass ignores that branch and
    differentiates only the continuous coverage weights (ix, iy) w.r.t. tag coords.
    """
    ix, iy = _iq_integrals(tx, ty, ddx, ddy)
    return _sample_tag_periodic(
        tag_pattern,
        tx,
        ty,
        ddx,
        ddy,
        ix,
        iy,
        black_level=black_level,
        white_level=white_level,
    )


def render_tag_antialiased(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    black_level: jax.Array | float = 0.0,
    white_level: jax.Array | float = 1.0,
) -> jnp.ndarray:
    """Inverse warp + Iñigo analytical box-filter per tag module."""
    tx, ty, ddx, ddy = tag_footprint(H, height, width)
    return sample_tag_filtered(
        tag_pattern,
        tx,
        ty,
        ddx,
        ddy,
        black_level=black_level,
        white_level=white_level,
    )


def nearest_sample(
    tag_pattern: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    *,
    fill: float = 1.0,
) -> jnp.ndarray:
    """Nearest-neighbor sample: one tag cell per pixel, no interpolation."""
    h, w = tag_pattern.shape
    tx_i = jax.lax.stop_gradient(tx)
    ty_i = jax.lax.stop_gradient(ty)
    ix = jnp.floor(tx_i).astype(jnp.int32)
    iy = jnp.floor(ty_i).astype(jnp.int32)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix_c = jnp.clip(ix, 0, w - 1)
    iy_c = jnp.clip(iy, 0, h - 1)
    sampled = tag_pattern[iy_c, ix_c]
    return jnp.where(inside, sampled, fill)


def bilinear_sample(
    tag_pattern: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    *,
    fill: float = 1.0,
) -> jnp.ndarray:
    """Sample pattern at tag coords; white background outside."""
    coords = jnp.stack([ty, tx], axis=0)
    return map_coordinates(tag_pattern, coords, order=1, mode="constant", cval=fill)


def render_tag(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
) -> jnp.ndarray:
    """Inverse warp + nearest-neighbor tag lookup (hard pixel blocks)."""
    tx, ty = inverse_warp_coords(H, height, width)
    return nearest_sample(tag_pattern, tx, ty)


def render_tag_supersampled(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    samples_per_axis: int = 8,
) -> jnp.ndarray:
    """Reference render: average nearest-neighbor samples over each pixel square."""
    px_base, py_base = _pixel_indices(height, width)
    n = samples_per_axis
    offsets = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n

    acc = jnp.zeros((height, width), dtype=jnp.float32)
    for oy in offsets:
        for ox in offsets:
            tx, ty = inverse_warp_coords(H, height, width, px_base + ox, py_base + oy)
            acc = acc + nearest_sample(tag_pattern, tx, ty)
    return acc / (n * n)


def render_tag_esf_supersampled(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    profile: ESFProfile,
    *,
    samples_per_axis: int = 8,
) -> jnp.ndarray:
    """Reference ESF render: average analytical ESF samples over each pixel square."""
    px_base, py_base = _pixel_indices(height, width)
    _, _, ddx, ddy = tag_footprint(H, height, width)
    n = samples_per_axis
    offsets = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n

    acc = jnp.zeros((height, width), dtype=jnp.float32)
    for oy in offsets:
        for ox in offsets:
            tx, ty = inverse_warp_coords(H, height, width, px_base + ox, py_base + oy)
            acc = acc + sample_tag_esf(tag_pattern, tx, ty, ddx, ddy, profile)
    return acc / (n * n)


def render_tag_bilinear(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
) -> jnp.ndarray:
    """Bilinear tag lookup (wide soft edges; for differentiable experiments)."""
    tx, ty = inverse_warp_coords(H, height, width)
    return bilinear_sample(tag_pattern, tx, ty)


def render_tag_filtered(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    black_level: jax.Array | float = 0.0,
    white_level: jax.Array | float = 1.0,
) -> jnp.ndarray:
    """Alias for :func:`render_tag_antialiased`."""
    return render_tag_antialiased(
        H, tag_pattern, height, width, black_level=black_level, white_level=white_level
    )


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


def quad_mask(
    corners: jnp.ndarray,
    height: int,
    width: int,
    margin: float = 3.5,
) -> jnp.ndarray:
    """Binary mask: tag quad interior plus a ``margin``-px band along its edges.

    Follows the perspective quad (not an axis-aligned bounding box).
    """
    cx, cy = _pixel_centers(height, width)
    inside = _inside_convex_quad(cx, cy, corners)
    n = corners.shape[0]
    edge_dists = []
    for i in range(n):
        x0, y0 = corners[i, 0], corners[i, 1]
        x1, y1 = corners[(i + 1) % n, 0], corners[(i + 1) % n, 1]
        edge_dists.append(_segment_distance_sq(cx, cy, x0, y0, x1, y1))
    min_dist = jnp.sqrt(jnp.min(jnp.stack(edge_dists, axis=0), axis=0))
    margin_f = jnp.float32(margin)
    return (inside | (min_dist <= margin_f)).astype(jnp.float32)


def bbox_mask(
    corners: jnp.ndarray,
    height: int,
    width: int,
    margin: float = 3.5,
) -> jnp.ndarray:
    """Alias for :func:`quad_mask` (perspective quad + edge margin)."""
    return quad_mask(corners, height, width, margin=margin)


def l2_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)
