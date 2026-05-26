"""Levenberg–Marquardt optimization over homography and render_model camera params."""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from render_model.camera_params import (
    decode_joint_params_per_step,
    model_params_to_vector,
    vector_to_model_params,
)
from render_model.homography import (
    corners_from_corner_shifts,
    corners_from_homography,
    homography_from_corners,
)
from render_model.pipeline import SUPERSAMPLE, RenderModelParams, render_with_model
from render_model.tag_data import TAG_CANONICAL_CORNERS

HOMOGRAPHY_PARAM_COUNT = 8
CAMERA_PARAM_COUNT = 8
JOINT_PARAM_COUNT = HOMOGRAPHY_PARAM_COUNT + CAMERA_PARAM_COUNT

JOINT_PARAM_NAMES: tuple[str, ...] = (
    "δx0", "δy0", "δx1", "δy1", "δx2", "δy2", "δx3", "δy3",
    "psf σ", "sharpen", "sharpen σ", "γ", "black", "white",
    "light u", "light v",
)

_CANONICAL = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)


def _reference_corners(H_init: jnp.ndarray) -> jnp.ndarray:
    """Fixed image-space quad from ``H_init`` (LM optimizes shifts relative to this)."""
    return corners_from_homography(H_init, _CANONICAL)


def _shifts_params_to_homography(
    shifts: jnp.ndarray,
    ref_corners: jnp.ndarray,
) -> jnp.ndarray:
    """8-vector of per-corner pixel shifts → 3×3 H."""
    corners_px = corners_from_corner_shifts(ref_corners, shifts)
    return homography_from_corners(_CANONICAL, corners_px)


class OptimizationParamTrace(NamedTuple):
    """LM trace; ``params_physical_per_step[:, :8]`` are pixel shifts (not absolute corners)."""

    H: jnp.ndarray
    camera: RenderModelParams
    losses: list[float]
    params_per_step: jnp.ndarray
    params_physical_per_step: jnp.ndarray
    ref_corners: jnp.ndarray


@dataclass(frozen=True)
class OptimizeLMConfig:
    n_steps: int = 30
    initial_damping: float = 1e-2
    damping_factor: float = 10.0
    min_damping: float = 1e-8
    max_damping: float = 1e8
    loss_mask: jnp.ndarray | None = None
    supersample: int = SUPERSAMPLE


def _pack_opt_params(
    h_params: jnp.ndarray,
    camera_vec: jnp.ndarray,
    *,
    optimize_homography: bool,
    optimize_camera: bool,
) -> jnp.ndarray:
    parts: list[jnp.ndarray] = []
    if optimize_homography:
        parts.append(h_params)
    if optimize_camera:
        parts.append(camera_vec)
    if not parts:
        raise ValueError("At least one of optimize_homography or optimize_camera must be True")
    return jnp.concatenate(parts)


def _unpack_opt_params(
    packed: jnp.ndarray,
    *,
    optimize_homography: bool,
    optimize_camera: bool,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    offset = 0
    h_params = camera_vec = None
    if optimize_homography:
        h_params = packed[offset : offset + HOMOGRAPHY_PARAM_COUNT]
        offset += HOMOGRAPHY_PARAM_COUNT
    if optimize_camera:
        camera_vec = packed[offset : offset + CAMERA_PARAM_COUNT]
    return h_params, camera_vec


def make_render_model_residuals(
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    ref_corners: jnp.ndarray,
    optimize_homography: bool = True,
    optimize_camera: bool = True,
    loss_mask: jnp.ndarray | None = None,
    supersample: int = SUPERSAMPLE,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Masked sqrt-weighted pixel residuals; sum(r²) equals masked MSE."""

    def residuals(packed: jnp.ndarray) -> jnp.ndarray:
        shifts, cam = _unpack_opt_params(
            packed, optimize_homography=optimize_homography, optimize_camera=optimize_camera
        )
        assert shifts is not None and cam is not None
        rendered = render_with_model(
            _shifts_params_to_homography(shifts, ref_corners),
            tag_pattern,
            height,
            width,
            vector_to_model_params(cam),
            supersample=supersample,
        )
        diff = rendered - target
        if loss_mask is not None:
            w = jnp.sqrt(loss_mask / jnp.maximum(jnp.sum(loss_mask), 1.0))
            return (diff * w).ravel()
        return (diff / jnp.sqrt(jnp.maximum(diff.size, 1))).ravel()

    return residuals


def compile_lm_run(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    config: OptimizeLMConfig,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Jitted LM loop. First call compiles; later calls are execution only (same shapes)."""
    df, lo, hi = config.damping_factor, config.min_damping, config.max_damping

    @jax.jit
    def run(packed: jnp.ndarray, damping: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def step(carry, _step_idx: jnp.ndarray):
            p, lam = carry
            r = residual_fn(p)
            J = jax.jacfwd(residual_fn)(p)
            hess = J.T @ J + lam * jnp.eye(p.shape[0], dtype=p.dtype)
            delta = jnp.linalg.solve(hess, -(J.T @ r))
            p_try = p + delta
            loss, loss_try = jnp.sum(r * r), jnp.sum(residual_fn(p_try) ** 2)
            accept = loss_try < loss
            p = jnp.where(accept, p_try, p)
            lam = jnp.clip(
                jnp.where(accept, lam / df, lam * df),
                lo,
                hi,
            )
            return (p, lam), (jnp.where(accept, loss_try, loss), p)

        steps = jnp.arange(config.n_steps, dtype=jnp.int32)
        (packed, _), (losses, hist) = jax.lax.scan(step, (packed, damping), steps)
        return packed, losses, hist

    return run


def _optimize_joint_lm(
    H_init: jnp.ndarray,
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    camera_init: RenderModelParams,
    config: OptimizeLMConfig,
) -> tuple[
    jnp.ndarray,
    RenderModelParams,
    list[float],
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    ref_corners = _reference_corners(H_init)
    zero_shifts = jnp.zeros(HOMOGRAPHY_PARAM_COUNT, dtype=jnp.float32)
    packed_init = _pack_opt_params(
        zero_shifts,
        model_params_to_vector(camera_init),
        optimize_homography=True,
        optimize_camera=True,
    )
    residual_fn = make_render_model_residuals(
        target,
        tag_pattern,
        height,
        width,
        ref_corners=ref_corners,
        loss_mask=config.loss_mask,
        supersample=config.supersample,
    )
    init_loss = float(jnp.sum(residual_fn(packed_init) ** 2))
    packed, step_losses, param_hist = compile_lm_run(residual_fn, config)(
        packed_init, jnp.asarray(config.initial_damping, dtype=jnp.float32)
    )
    shifts, cam = _unpack_opt_params(packed, optimize_homography=True, optimize_camera=True)
    assert shifts is not None and cam is not None
    return (
        _shifts_params_to_homography(shifts, ref_corners),
        vector_to_model_params(cam),
        [init_loss, *[float(x) for x in step_losses]],
        packed_init,
        param_hist,
        ref_corners,
    )


def optimize_render_model_lm(
    H_init: jnp.ndarray,
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    camera_init: RenderModelParams | None = None,
    config: OptimizeLMConfig | None = None,
) -> tuple[jnp.ndarray, RenderModelParams, list[float]]:
    from render_model.pipeline import default_params

    H, camera, losses, _, _, _ = _optimize_joint_lm(
        H_init,
        target,
        tag_pattern,
        height,
        width,
        camera_init=camera_init or default_params(),
        config=config or OptimizeLMConfig(),
    )
    return H, camera, losses


def optimize_render_model_lm_with_param_trace(
    H_init: jnp.ndarray,
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    camera_init: RenderModelParams | None = None,
    config: OptimizeLMConfig | None = None,
) -> OptimizationParamTrace:
    from render_model.pipeline import default_params

    H, camera, losses, packed_init, param_hist, ref_corners = _optimize_joint_lm(
        H_init,
        target,
        tag_pattern,
        height,
        width,
        camera_init=camera_init or default_params(),
        config=config or OptimizeLMConfig(),
    )
    params_per_step = jnp.concatenate([packed_init[None, :], param_hist], axis=0)
    return OptimizationParamTrace(
        H,
        camera,
        losses,
        params_per_step,
        decode_joint_params_per_step(params_per_step),
        ref_corners,
    )
