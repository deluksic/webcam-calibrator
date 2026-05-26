"""Homography utilities (DLT, apply, parameterization)."""

import jax.numpy as jnp
import numpy as np


def apply_homography(H: jnp.ndarray, pts: jnp.ndarray) -> jnp.ndarray:
    """Apply 3x3 homography to Nx2 points. Returns Nx2."""
    ones = jnp.ones((pts.shape[0], 1), dtype=pts.dtype)
    hom = jnp.concatenate([pts, ones], axis=1)
    mapped = (H @ hom.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def dlt_homography_four_point(src: jnp.ndarray, dst: jnp.ndarray) -> jnp.ndarray:
    """Exact DLT for 4 correspondences with ``h22 = 1`` (8×8 linear solve).

    Prefer this over the null-space SVD/eigh path: ``jnp.linalg.solve`` has a stable
    JVP and round-trips ``corners_from_homography`` ↔ ``H`` for consistent corner-space LM.
    """
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src and dst must be 4x2")

    M = jnp.zeros((8, 8), dtype=jnp.float32)
    b = jnp.zeros(8, dtype=jnp.float32)
    for i in range(4):
        x, y = src[i, 0], src[i, 1]
        u, v = dst[i, 0], dst[i, 1]
        M = M.at[2 * i].set(jnp.array([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]))
        b = b.at[2 * i].set(u)
        M = M.at[2 * i + 1].set(jnp.array([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]))
        b = b.at[2 * i + 1].set(v)

    p = jnp.linalg.solve(M, b)
    return jnp.array(
        [
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [p[6], p[7], 1.0],
        ],
        dtype=jnp.float32,
    )


def dlt_homography(src: jnp.ndarray, dst: jnp.ndarray) -> jnp.ndarray:
    """Direct Linear Transform: 4+ point correspondences -> 3x3 H (h22=1)."""
    if src.shape != dst.shape or src.shape[-1] != 2:
        raise ValueError("src and dst must be Nx2 with the same shape")

    n = src.shape[0]
    if n == 4:
        return dlt_homography_four_point(src, dst)

    A = jnp.zeros((2 * n, 9), dtype=jnp.float32)
    for i in range(n):
        x, y = src[i, 0], src[i, 1]
        u, v = dst[i, 0], dst[i, 1]
        A = A.at[2 * i].set([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A = A.at[2 * i + 1].set([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    _, eigvecs = jnp.linalg.eigh(A.T @ A)
    h = eigvecs[:, 0]
    H = h.reshape(3, 3)
    return H / H[2, 2]


def homography_from_corners(
    src_corners: jnp.ndarray,
    dst_corners: jnp.ndarray,
) -> jnp.ndarray:
    """Build H mapping src_corners -> dst_corners (4x2 each, same order)."""
    return dlt_homography(
        jnp.asarray(src_corners, dtype=jnp.float32),
        jnp.asarray(dst_corners, dtype=jnp.float32),
    )


def normalize_homography(H: jnp.ndarray) -> jnp.ndarray:
    """Scale so H[2,2] == 1."""
    return H / H[2, 2]


def homography_to_params(H: jnp.ndarray) -> jnp.ndarray:
    """Flatten 3x3 H to 8-vector (h22 fixed at 1).

    Order (for per-parameter learning-rate scales):

    ======  =========================  ==============================
    Index   Matrix entry               Typical role (near-affine tag)
    ======  =========================  ==============================
    0       h00                        scale / stretch-x
    1       h01                        rotation + shear (off-diagonal)
    2       h02                        translation x
    3       h10                        rotation + shear (off-diagonal)
    4       h11                        scale / stretch-y
    5       h12                        translation y
    6       h20                        perspective
    7       h21                        perspective
    ======  =========================  ==============================

    Rotation and uniform scale are coupled in (0, 1, 3, 4); scales on these
    entries are a heuristic, not a perfect rotation/scale split.
    """
    Hn = normalize_homography(H)
    return jnp.array(
        [
            Hn[0, 0],
            Hn[0, 1],
            Hn[0, 2],
            Hn[1, 0],
            Hn[1, 1],
            Hn[1, 2],
            Hn[2, 0],
            Hn[2, 1],
        ],
        dtype=jnp.float32,
    )


def params_to_homography(params: jnp.ndarray) -> jnp.ndarray:
    """8-vector -> 3x3 H with h22=1."""
    return jnp.array(
        [
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ],
        dtype=params.dtype,
    )


def corners_from_homography(H: jnp.ndarray, src_corners: jnp.ndarray) -> jnp.ndarray:
    """Image-space corners by mapping canonical src corners through H."""
    return apply_homography(H, src_corners)


def corners_from_corner_shifts(
    ref_corners: jnp.ndarray,
    shifts: jnp.ndarray,
) -> jnp.ndarray:
    """Image-space quad ``ref_corners + shifts`` (``shifts`` is length-8 or 4×2)."""
    ref_corners = jnp.asarray(ref_corners, dtype=jnp.float32)
    return ref_corners + jnp.asarray(shifts, dtype=jnp.float32).reshape(4, 2)


def corner_shifts_from_corners(
    ref_corners: jnp.ndarray,
    corners: jnp.ndarray,
) -> jnp.ndarray:
    """Per-corner pixel shifts (8-vector) relative to ``ref_corners``."""
    return (jnp.asarray(corners, dtype=jnp.float32) - jnp.asarray(ref_corners, dtype=jnp.float32)).ravel()


def corner_shifts_from_homography(
    H: jnp.ndarray,
    ref_corners: jnp.ndarray,
    src_corners: jnp.ndarray,
) -> jnp.ndarray:
    """Shifts (8-vector) from ``H`` vs a fixed reference quad."""
    return corner_shifts_from_corners(
        ref_corners, corners_from_homography(H, src_corners)
    )


def image_corners_to_normalized(
    corners: jnp.ndarray,
    width: int | jnp.ndarray,
    height: int | jnp.ndarray,
) -> jnp.ndarray:
    """Map image-space corners (x, y) in pixels to [-1, 1] per axis."""
    w = jnp.asarray(width, dtype=corners.dtype)
    h = jnp.asarray(height, dtype=corners.dtype)
    return jnp.stack(
        [
            2.0 * corners[:, 0] / w - 1.0,
            2.0 * corners[:, 1] / h - 1.0,
        ],
        axis=-1,
    )


def normalized_corners_to_image(
    corners: jnp.ndarray,
    width: int | jnp.ndarray,
    height: int | jnp.ndarray,
) -> jnp.ndarray:
    """Map [-1, 1] normalized corners back to image pixels (x, y)."""
    w = jnp.asarray(width, dtype=corners.dtype)
    h = jnp.asarray(height, dtype=corners.dtype)
    return jnp.stack(
        [
            (corners[:, 0] + 1.0) * w * 0.5,
            (corners[:, 1] + 1.0) * h * 0.5,
        ],
        axis=-1,
    )


def numpy_image_corners_to_normalized(
    corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return np.asarray(
        image_corners_to_normalized(
            jnp.asarray(corners, dtype=jnp.float32), width, height
        ),
        dtype=np.float32,
    )


def numpy_normalized_corners_to_image(
    corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return np.asarray(
        normalized_corners_to_image(
            jnp.asarray(corners, dtype=jnp.float32), width, height
        ),
        dtype=np.float32,
    )


def corner_reprojection_loss(
    H: jnp.ndarray,
    src_corners: jnp.ndarray,
    target_corners: jnp.ndarray,
) -> jnp.ndarray:
    """Mean squared reprojection error for the four tag corners."""
    pred = corners_from_homography(H, src_corners)
    return jnp.mean(jnp.sum((pred - target_corners) ** 2, axis=1))


def corner_rmse(
    H: jnp.ndarray,
    src_corners: jnp.ndarray,
    target_corners: jnp.ndarray,
) -> jnp.ndarray:
    """L2 RMSE between H-mapped src corners and target corners."""
    return jnp.sqrt(corner_reprojection_loss(H, src_corners, target_corners))


def normalized_dlt_homography(src: jnp.ndarray, dst: jnp.ndarray) -> jnp.ndarray:
    """DLT with Hartley normalization on dst (image-space corners).

    Translates dst to its centroid and scales so mean point distance = √2 before
    solving, then denormalizes the result. This keeps the DLT system well-conditioned
    and improves autodiff stability when dst coordinates are large (e.g. 100–1000 px).
    """
    centroid = dst.mean(axis=0)
    diffs = dst - centroid
    scale = jnp.sqrt(2.0) / jnp.maximum(
        jnp.sqrt(jnp.mean(jnp.sum(diffs ** 2, axis=1))), 1e-6
    )
    T = jnp.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    dst_h = jnp.concatenate([dst, jnp.ones((dst.shape[0], 1), dtype=dst.dtype)], axis=1)
    dst_n = (T @ dst_h.T).T[:, :2]
    H_n = dlt_homography(src, dst_n)
    return jnp.linalg.inv(T) @ H_n


def numpy_dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """NumPy DLT for initialization outside JIT."""
    return np.asarray(
        dlt_homography(jnp.asarray(src), jnp.asarray(dst)),
        dtype=np.float64,
    )


def edge_outward_normals(corners: jnp.ndarray) -> jnp.ndarray:
    """Unit outward normals for the four tag edges (TL→TR→BR→BL, CCW boundary).

    Edge ``i`` runs from corner ``i`` to corner ``(i + 1) % 4``. Interior is to the
    left of each directed edge (same convention as :func:`render_model.esf` sampling).
    """
    corners = jnp.asarray(corners, dtype=jnp.float32)
    c0 = corners
    c1 = jnp.roll(corners, shift=-1, axis=0)
    tangent = c1 - c0
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    return jnp.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)


def edge_normal_distances(corners: jnp.ndarray, normals: jnp.ndarray) -> jnp.ndarray:
    """Line constants ``d`` for ``normal · x = d`` using each edge's start corner."""
    return jnp.sum(normals * corners, axis=-1)


def corners_from_normal_offsets(
    base_corners: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """Rebuild quad corners from per-edge shifts along fixed outward normals.

    ``base_corners`` fixes edge orientation (as from a good line fit). Each ``offsets[i]``
    moves edge ``i`` along its outward normal: ``n_i · x = d_i + offsets[i]``. Corner
    ``k`` is the intersection of edges ``(k - 1) % 4`` and ``k`` — the AprilTag model
    (accurate angle, uncertain normal placement).
    """
    base_corners = jnp.asarray(base_corners, dtype=jnp.float32)
    offsets = jnp.asarray(offsets, dtype=jnp.float32)
    normals = edge_outward_normals(base_corners)
    distances = edge_normal_distances(base_corners, normals) + offsets

    def corner_at(k: int) -> jnp.ndarray:
        n0 = normals[(k - 1) % 4]
        n1 = normals[k]
        d0 = distances[(k - 1) % 4]
        d1 = distances[k]
        return jnp.linalg.solve(jnp.stack([n0, n1]), jnp.array([d0, d1]))

    return jnp.stack([corner_at(k) for k in range(4)])


def numpy_corners_from_normal_offsets(
    base_corners: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """NumPy wrapper for :func:`corners_from_normal_offsets`."""
    return np.asarray(
        corners_from_normal_offsets(
            jnp.asarray(base_corners, dtype=jnp.float32),
            jnp.asarray(offsets, dtype=jnp.float32),
        ),
        dtype=np.float32,
    )


def corners_from_uniform_normal_offset(
    base_corners: jnp.ndarray,
    offset: jnp.ndarray | float,
) -> jnp.ndarray:
    """Quad corners with the same normal shift on every edge (1 DOF).

    Same geometry as :func:`corners_from_normal_offsets` but one scalar ``offset``
    applied to all four edges. Models a detector bias that is constant in edge-normal
    distance (e.g. consistent thresholding / ESF bias) independent of edge direction.
    """
    offset = jnp.asarray(offset, dtype=jnp.float32)
    offsets = jnp.broadcast_to(offset, (4,))
    return corners_from_normal_offsets(base_corners, offsets)


def numpy_corners_from_uniform_normal_offset(
    base_corners: np.ndarray,
    offset: float,
) -> np.ndarray:
    """NumPy wrapper for :func:`corners_from_uniform_normal_offset`."""
    return np.asarray(
        corners_from_uniform_normal_offset(
            jnp.asarray(base_corners, dtype=jnp.float32),
            jnp.float32(offset),
        ),
        dtype=np.float32,
    )


def numpy_sample_normal_offset_corners(
    rng: np.random.Generator,
    base_corners: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Perturb ``base_corners`` with i.i.d. per-edge normal offsets (4 DOF, ``sigma`` px)."""
    offsets = rng.normal(0.0, sigma, size=4).astype(np.float32)
    return numpy_corners_from_normal_offsets(base_corners, offsets)


def numpy_sample_uniform_normal_offset_corners(
    rng: np.random.Generator,
    base_corners: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Perturb ``base_corners`` with one shared normal offset (1 DOF, ``sigma`` px)."""
    offset = float(rng.normal(0.0, sigma))
    return numpy_corners_from_uniform_normal_offset(base_corners, offset)
