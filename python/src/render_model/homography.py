"""Homography utilities (DLT, apply, parameterization)."""

import jax.numpy as jnp
import numpy as np


def apply_homography(H: jnp.ndarray, pts: jnp.ndarray) -> jnp.ndarray:
    """Apply 3x3 homography to Nx2 points. Returns Nx2."""
    ones = jnp.ones((pts.shape[0], 1), dtype=pts.dtype)
    hom = jnp.concatenate([pts, ones], axis=1)
    mapped = (H @ hom.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def dlt_homography(src: jnp.ndarray, dst: jnp.ndarray) -> jnp.ndarray:
    """Direct Linear Transform: 4+ point correspondences -> 3x3 H (h22=1)."""
    if src.shape != dst.shape or src.shape[-1] != 2:
        raise ValueError("src and dst must be Nx2 with the same shape")

    n = src.shape[0]
    A = jnp.zeros((2 * n, 9), dtype=jnp.float32)
    for i in range(n):
        x, y = src[i, 0], src[i, 1]
        u, v = dst[i, 0], dst[i, 1]
        A = A.at[2 * i].set([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A = A.at[2 * i + 1].set([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])

    _, _, vh = jnp.linalg.svd(A)
    h = vh[-1, :]
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
