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
from render_model.homography import homography_to_params, params_to_homography
from render_model.pipeline import SUPERSAMPLE, RenderModelParams, render_with_model

HOMOGRAPHY_PARAM_COUNT = 8
CAMERA_PARAM_COUNT = 6
JOINT_PARAM_COUNT = HOMOGRAPHY_PARAM_COUNT + CAMERA_PARAM_COUNT

JOINT_PARAM_NAMES: tuple[str, ...] = (
    "h00",
    "h01",
    "h02",
    "h10",
    "h11",
    "h12",
    "h20",
    "h21",
    "psf σ",
    "sharpen",
    "sharpen σ",
    "γ",
    "black",
    "white",
)


class OptimizationParamTrace(NamedTuple):
    H: jnp.ndarray
    camera: RenderModelParams
    losses: list[float]
    params_per_step: jnp.ndarray
    """Raw packed optimization coordinates (not for display)."""
    params_physical_per_step: jnp.ndarray
    """Physical joint parameters ``(steps, 14)`` for plotting."""


@dataclass(frozen=True)
class OptimizeLMConfig:
    """Levenberg–Marquardt on masked pixel residuals (joint H + camera)."""

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
    h_params = None
    camera_vec = None
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
    optimize_homography: bool = True,
    optimize_camera: bool = True,
    loss_mask: jnp.ndarray | None = None,
    supersample: int = SUPERSAMPLE,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Masked sqrt-weighted pixel residuals; sum(r²) equals masked MSE."""

    def residuals(packed: jnp.ndarray) -> jnp.ndarray:
        h_params, camera_vec = _unpack_opt_params(
            packed,
            optimize_homography=optimize_homography,
            optimize_camera=optimize_camera,
        )
        assert h_params is not None and camera_vec is not None
        H = params_to_homography(h_params)
        camera = vector_to_model_params(camera_vec)
        rendered = render_with_model(
            H, tag_pattern, height, width, camera, supersample=supersample
        )
        diff = rendered - target
        if loss_mask is not None:
            weight = jnp.sqrt(loss_mask / jnp.maximum(jnp.sum(loss_mask), 1.0))
            return (diff * weight).ravel()
        return (diff / jnp.sqrt(jnp.maximum(diff.size, 1))).ravel()

    return residuals


def _compile_lm_optimize(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_steps: int,
    *,
    damping_factor: float,
    min_damping: float,
    max_damping: float,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    @jax.jit
    def run(packed: jnp.ndarray, damping: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray], _: None
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            p, lam = carry
            r = residual_fn(p)
            J = jax.jacfwd(residual_fn)(p)
            g = J.T @ r
            hess = J.T @ J + lam * jnp.eye(p.shape[0], dtype=p.dtype)
            delta = jnp.linalg.solve(hess, -g)
            p_try = p + delta
            r_try = residual_fn(p_try)
            loss = jnp.sum(r * r)
            loss_try = jnp.sum(r_try * r_try)
            accept = loss_try < loss
            p = jnp.where(accept, p_try, p)
            lam = jnp.where(accept, lam / damping_factor, lam * damping_factor)
            lam = jnp.clip(lam, min_damping, max_damping)
            return (p, lam), jnp.where(accept, loss_try, loss)

        (packed, _), losses = jax.lax.scan(body, (packed, damping), None, length=n_steps)
        return packed, losses

    return run


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
    """Joint homography + camera fit via Levenberg–Marquardt on pixel residuals."""
    from render_model.pipeline import default_params

    config = config or OptimizeLMConfig()
    camera_init = camera_init or default_params()
    packed = _pack_opt_params(
        homography_to_params(H_init),
        model_params_to_vector(camera_init),
        optimize_homography=True,
        optimize_camera=True,
    )
    residual_fn = make_render_model_residuals(
        target,
        tag_pattern,
        height,
        width,
        loss_mask=config.loss_mask,
        supersample=config.supersample,
    )
    init_loss = float(jnp.sum(residual_fn(packed) ** 2))
    run = _compile_lm_optimize(
        residual_fn,
        config.n_steps,
        damping_factor=config.damping_factor,
        min_damping=config.min_damping,
        max_damping=config.max_damping,
    )
    packed, step_losses = run(
        packed, jnp.asarray(config.initial_damping, dtype=jnp.float32)
    )
    losses = [init_loss, *[float(x) for x in step_losses]]
    h_out, camera_out = _unpack_opt_params(
        packed, optimize_homography=True, optimize_camera=True
    )
    assert h_out is not None and camera_out is not None
    return (
        params_to_homography(h_out),
        vector_to_model_params(camera_out),
        losses,
    )


def _compile_lm_param_trace(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_steps: int,
    *,
    damping_factor: float,
    min_damping: float,
    max_damping: float,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    @jax.jit
    def run(
        packed: jnp.ndarray, damping: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def body(
            carry: tuple[jnp.ndarray, jnp.ndarray], _: None
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
            p, lam = carry
            r = residual_fn(p)
            J = jax.jacfwd(residual_fn)(p)
            g = J.T @ r
            hess = J.T @ J + lam * jnp.eye(p.shape[0], dtype=p.dtype)
            delta = jnp.linalg.solve(hess, -g)
            p_try = p + delta
            r_try = residual_fn(p_try)
            loss = jnp.sum(r * r)
            loss_try = jnp.sum(r_try * r_try)
            accept = loss_try < loss
            p = jnp.where(accept, p_try, p)
            lam = jnp.where(accept, lam / damping_factor, lam * damping_factor)
            lam = jnp.clip(lam, min_damping, max_damping)
            return (p, lam), (loss, p)

        (packed, _), (losses, param_hist) = jax.lax.scan(
            body, (packed, damping), None, length=n_steps
        )
        return packed, losses, param_hist

    return run


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
    """Joint H + camera LM with per-step packed parameter history."""
    from render_model.pipeline import default_params

    config = config or OptimizeLMConfig()
    camera_init = camera_init or default_params()
    packed_init = _pack_opt_params(
        homography_to_params(H_init),
        model_params_to_vector(camera_init),
        optimize_homography=True,
        optimize_camera=True,
    )
    residual_fn = make_render_model_residuals(
        target,
        tag_pattern,
        height,
        width,
        loss_mask=config.loss_mask,
        supersample=config.supersample,
    )
    init_loss = float(jnp.sum(residual_fn(packed_init) ** 2))
    run = _compile_lm_param_trace(
        residual_fn,
        config.n_steps,
        damping_factor=config.damping_factor,
        min_damping=config.min_damping,
        max_damping=config.max_damping,
    )
    packed, step_losses, param_hist = run(
        packed_init, jnp.asarray(config.initial_damping, dtype=jnp.float32)
    )
    params_per_step = jnp.concatenate([packed_init[None, :], param_hist], axis=0)
    h_out, camera_out = _unpack_opt_params(
        packed, optimize_homography=True, optimize_camera=True
    )
    assert h_out is not None and camera_out is not None
    return OptimizationParamTrace(
        params_to_homography(h_out),
        vector_to_model_params(camera_out),
        [init_loss, *[float(x) for x in step_losses]],
        params_per_step,
        decode_joint_params_per_step(params_per_step),
    )
