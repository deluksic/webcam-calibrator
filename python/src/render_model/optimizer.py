"""Differentiable optimization over homography and render_model camera params."""

from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from render_model.homography import (
    corner_reprojection_loss,
    corner_rmse,
    homography_to_params,
    params_to_homography,
)
from render_model.pipeline import RenderModelParams, render_with_model

HOMOGRAPHY_PARAM_COUNT = 8
CAMERA_PARAM_COUNT = 6

# Relative LR multipliers on homography_to_params order (see homography.py).
# Higher on translation / diagonal (scale); lower on off-diagonals (rotation/shear).
DEFAULT_HOMOGRAPHY_PARAM_LR_SCALES: tuple[float, ...] = (
    3.0,  # h00 scale
    0.25,  # h01 rotation/shear
    1.0,  # h02 translation x
    0.25,  # h10 rotation/shear
    3.0,  # h11 scale
    1.0,  # h12 translation y
    0.5,  # h20 perspective
    0.5,  # h21 perspective
)

JointParams = dict[str, jnp.ndarray]


class OptimizationTrace(NamedTuple):
    H: jnp.ndarray
    camera: RenderModelParams
    losses: list[float]
    corner_rmse_per_step: list[float]


@dataclass(frozen=True)
class OptimizeRenderConfig:
    learning_rate: float = 2e-3
    n_steps: int = 500
    optimizer: str = "adam"
    max_grad_norm: float = 1.0
    loss_mask: jnp.ndarray | None = None
    corner_weight: float = 1.0
    optimize_homography: bool = True
    optimize_camera: bool = False
    camera_learning_rate: float | None = None
    camera_lr_scale: float = 0.1
    homography_param_lr_scales: tuple[float, ...] | None = None
    psf_continuation_start: float | None = None
    psf_continuation_steps: int | None = None
    momentum: float = 0.9


def _psf_continuation_steps(config: OptimizeRenderConfig) -> int:
    if config.psf_continuation_steps is not None:
        return config.psf_continuation_steps
    return config.n_steps


def effective_psf_sigma(
    learned: jax.Array,
    step: jax.Array,
    *,
    start: float,
    n_steps: int,
) -> jax.Array:
    """Linear homotopy from ``start`` (step 0) to ``learned`` (step >= n_steps)."""
    ramp = jnp.minimum(1.0, step / jnp.maximum(n_steps, 1))
    start_val = jnp.asarray(start, dtype=learned.dtype)
    return start_val * (1.0 - ramp) + learned * ramp


def apply_psf_continuation_to_camera(
    camera: RenderModelParams,
    step: jax.Array | int,
    config: OptimizeRenderConfig,
) -> RenderModelParams:
    if config.psf_continuation_start is None:
        return camera
    step_arr = jnp.asarray(step, dtype=camera.psf_sigma.dtype)
    return replace(
        camera,
        psf_sigma=effective_psf_sigma(
            camera.psf_sigma,
            step_arr,
            start=config.psf_continuation_start,
            n_steps=_psf_continuation_steps(config),
        ),
    )


def model_params_to_vector(params: RenderModelParams) -> jnp.ndarray:
    return jnp.stack(
        [
            params.psf_sigma,
            params.sharpen_amount,
            params.sharpen_sigma,
            params.gamma,
            params.black_level,
            params.white_level,
        ]
    )


def vector_to_model_params(values: jnp.ndarray) -> RenderModelParams:
    return RenderModelParams(
        psf_sigma=values[0],
        sharpen_amount=values[1],
        sharpen_sigma=values[2],
        gamma=values[3],
        black_level=values[4],
        white_level=values[5],
    )


def _camera_learning_rate(config: OptimizeRenderConfig) -> float:
    if config.camera_learning_rate is not None:
        return config.camera_learning_rate
    return config.learning_rate * config.camera_lr_scale


def homography_param_lr_scales(config: OptimizeRenderConfig) -> jnp.ndarray:
    """Per-entry multipliers for ``homography_to_params`` (effective lr = base × scale)."""
    scales = config.homography_param_lr_scales
    if scales is None:
        scales = DEFAULT_HOMOGRAPHY_PARAM_LR_SCALES
    if len(scales) != HOMOGRAPHY_PARAM_COUNT:
        raise ValueError(
            f"homography_param_lr_scales must have length {HOMOGRAPHY_PARAM_COUNT}, got {len(scales)}"
        )
    return jnp.asarray(scales, dtype=jnp.float32)


def _lr_scale_pytree(
    params: jnp.ndarray | JointParams,
    config: OptimizeRenderConfig,
) -> jnp.ndarray | JointParams:
    """Gradient multipliers applied before the Adam (etc.) step."""
    h_scales = homography_param_lr_scales(config)
    if isinstance(params, dict):
        return {"h": h_scales, "camera": jnp.ones(CAMERA_PARAM_COUNT, dtype=jnp.float32)}
    if params.shape[0] == HOMOGRAPHY_PARAM_COUNT:
        return h_scales
    if params.shape[0] == HOMOGRAPHY_PARAM_COUNT + CAMERA_PARAM_COUNT:
        camera_scale = jnp.float32(_camera_learning_rate(config) / config.learning_rate)
        return jnp.concatenate(
            [h_scales, jnp.full(CAMERA_PARAM_COUNT, camera_scale, dtype=jnp.float32)]
        )
    raise ValueError(f"Unexpected packed parameter length {params.shape[0]}")


def _optimizer_transform(
    learning_rate: float,
    name: str,
    *,
    momentum: float = 0.9,
) -> optax.GradientTransformation:
    if name == "adam":
        return optax.adam(learning_rate)
    if name == "adamw":
        return optax.adamw(learning_rate)
    if name == "rmsprop":
        return optax.rmsprop(learning_rate)
    if name == "sgd_momentum":
        return optax.sgd(learning_rate, momentum=momentum)
    if name == "sgd":
        return optax.sgd(learning_rate)
    raise ValueError(f"Unknown optimizer: {name!r}")


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


def make_render_model_loss(
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    fixed_camera: RenderModelParams,
    loss_mask: jnp.ndarray | None = None,
    src_corners: jnp.ndarray | None = None,
    target_corners: jnp.ndarray | None = None,
    corner_weight: float = 0.0,
    optimize_homography: bool = True,
    optimize_camera: bool = False,
    config: OptimizeRenderConfig | None = None,
) -> Callable[[jnp.ndarray, jax.Array], jnp.ndarray]:
    """Pixel MSE (+ optional corner loss) through ``render_with_model``."""
    use_corners = (
        corner_weight > 0.0
        and src_corners is not None
        and target_corners is not None
        and optimize_homography
    )

    def loss(packed: jnp.ndarray, step: jax.Array) -> jnp.ndarray:
        h_params, camera_vec = _unpack_opt_params(
            packed,
            optimize_homography=optimize_homography,
            optimize_camera=optimize_camera,
        )
        assert h_params is not None
        H = params_to_homography(h_params)
        camera = (
            vector_to_model_params(camera_vec)
            if optimize_camera and camera_vec is not None
            else fixed_camera
        )
        if config is not None and optimize_camera:
            camera = apply_psf_continuation_to_camera(camera, step, config)
        rendered = render_with_model(H, tag_pattern, height, width, camera)
        diff2 = (rendered - target) ** 2
        if loss_mask is not None:
            pixel_loss = jnp.sum(diff2 * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        else:
            pixel_loss = jnp.mean(diff2)

        if not use_corners:
            return pixel_loss

        corner_loss = corner_reprojection_loss(
            params_to_homography(h_params), src_corners, target_corners
        )
        return pixel_loss + corner_weight * corner_loss

    return loss


def make_joint_render_model_loss(
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    loss_mask: jnp.ndarray | None = None,
    src_corners: jnp.ndarray | None = None,
    target_corners: jnp.ndarray | None = None,
    corner_weight: float = 0.0,
    config: OptimizeRenderConfig | None = None,
) -> Callable[[JointParams, jax.Array], jnp.ndarray]:
    """Joint H + camera loss on a labeled parameter dict."""
    use_corners = (
        corner_weight > 0.0 and src_corners is not None and target_corners is not None
    )

    def loss(params: JointParams, step: jax.Array) -> jnp.ndarray:
        H = params_to_homography(params["h"])
        camera = vector_to_model_params(params["camera"])
        if config is not None:
            camera = apply_psf_continuation_to_camera(camera, step, config)
        rendered = render_with_model(H, tag_pattern, height, width, camera)
        diff2 = (rendered - target) ** 2
        if loss_mask is not None:
            pixel_loss = jnp.sum(diff2 * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        else:
            pixel_loss = jnp.mean(diff2)

        if not use_corners:
            return pixel_loss

        corner_loss = corner_reprojection_loss(H, src_corners, target_corners)
        return pixel_loss + corner_weight * corner_loss

    return loss


def make_render_model_loss_fixed_h(
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    H: jnp.ndarray,
    *,
    loss_mask: jnp.ndarray | None = None,
    config: OptimizeRenderConfig | None = None,
) -> Callable[[jnp.ndarray, jax.Array], jnp.ndarray]:
    """Camera-only loss with fixed homography."""

    def loss(camera_vec: jnp.ndarray, step: jax.Array) -> jnp.ndarray:
        camera = vector_to_model_params(camera_vec)
        if config is not None:
            camera = apply_psf_continuation_to_camera(camera, step, config)
        rendered = render_with_model(H, tag_pattern, height, width, camera)
        diff2 = (rendered - target) ** 2
        if loss_mask is not None:
            return jnp.sum(diff2 * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        return jnp.mean(diff2)

    return loss


def _make_optimizer(
    params: jnp.ndarray | JointParams,
    config: OptimizeRenderConfig,
) -> optax.GradientTransformation:
    clip = optax.clip_by_global_norm(config.max_grad_norm)
    lr_scales = _lr_scale_pytree(params, config)

    if isinstance(params, dict):
        camera_lr = _camera_learning_rate(config)
        transforms = {
            "h": optax.chain(
                optax.scale(lr_scales["h"]),
                _optimizer_transform(
                    config.learning_rate, config.optimizer, momentum=config.momentum
                ),
            ),
            "camera": _optimizer_transform(
                camera_lr, config.optimizer, momentum=config.momentum
            ),
        }
        base = optax.multi_transform(transforms, {"h": "h", "camera": "camera"})
        return optax.chain(clip, base)

    base = optax.chain(
        optax.scale(lr_scales),
        _optimizer_transform(
            config.learning_rate, config.optimizer, momentum=config.momentum
        ),
    )
    return optax.chain(clip, base)


def _compile_optimize(
    loss_fn: Callable[..., jnp.ndarray],
    opt: optax.GradientTransformation,
    n_steps: int,
    *,
    corner_rmse_fn: Callable[[Any], jnp.ndarray] | None = None,
) -> Callable[..., Any]:
    """JIT-compiled scan over the full optimization loop."""

    if corner_rmse_fn is None:

        @jax.jit
        def run(params: Any, opt_state: optax.OptState) -> tuple[Any, jnp.ndarray]:
            steps = jnp.arange(n_steps, dtype=jnp.int32)

            def body(
                carry: tuple[Any, optax.OptState], step: jax.Array
            ) -> tuple[tuple[Any, optax.OptState], jnp.ndarray]:
                p, state = carry
                loss, grads = jax.value_and_grad(lambda packed: loss_fn(packed, step))(p)
                updates, state = opt.update(grads, state, p)
                p = optax.apply_updates(p, updates)
                return (p, state), loss

            (params, _), losses = jax.lax.scan(body, (params, opt_state), steps)
            return params, losses

        return run

    @jax.jit
    def run_traced(
        params: Any, opt_state: optax.OptState
    ) -> tuple[Any, jnp.ndarray, jnp.ndarray]:
        steps = jnp.arange(n_steps, dtype=jnp.int32)

        def body(
            carry: tuple[Any, optax.OptState], step: jax.Array
        ) -> tuple[tuple[Any, optax.OptState], tuple[jnp.ndarray, jnp.ndarray]]:
            p, state = carry
            loss, grads = jax.value_and_grad(lambda packed: loss_fn(packed, step))(p)
            updates, state = opt.update(grads, state, p)
            p = optax.apply_updates(p, updates)
            rmse = corner_rmse_fn(p)
            return (p, state), (loss, rmse)

        (params, _), (losses, rmses) = jax.lax.scan(body, (params, opt_state), steps)
        return params, losses, rmses

    return run_traced


def _run_optax_loop(
    params: jnp.ndarray | JointParams,
    loss_fn: Callable[..., jnp.ndarray],
    config: OptimizeRenderConfig,
    *,
    corner_rmse_fn: Callable[[Any], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray | JointParams, list[float], list[float] | None]:
    opt = _make_optimizer(params, config)
    opt_state = opt.init(params)
    run = _compile_optimize(
        loss_fn, opt, config.n_steps, corner_rmse_fn=corner_rmse_fn
    )
    if corner_rmse_fn is None:
        params, losses = run(params, opt_state)
        return params, [float(x) for x in losses], None

    init_rmse = float(corner_rmse_fn(params))
    params, losses, rmses = run(params, opt_state)
    corner_history = [init_rmse, *[float(x) for x in rmses]]
    return params, [float(x) for x in losses], corner_history


def first_step_corner_rmse_below(
    corner_rmse_per_step: list[float],
    threshold: float,
) -> int | None:
    """Return the first index (0 = init) where corner RMSE is below ``threshold``."""
    for index, value in enumerate(corner_rmse_per_step):
        if value < threshold:
            return index
    return None


def optimize_render_model_with_trace(
    H_init: jnp.ndarray,
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    camera_init: RenderModelParams | None = None,
    config: OptimizeRenderConfig | None = None,
    src_corners: jnp.ndarray | None = None,
    target_corners: jnp.ndarray | None = None,
) -> OptimizationTrace:
    """Like ``optimize_render_model`` but records corner RMSE after each step (and init)."""
    from render_model.pipeline import default_params

    if src_corners is None or target_corners is None:
        raise ValueError("src_corners and target_corners are required for tracing")

    config = config or OptimizeRenderConfig()
    camera_init = camera_init or default_params()
    camera_vec = model_params_to_vector(camera_init)
    h_params = homography_to_params(H_init)

    def _corner_rmse_joint(params: JointParams) -> jnp.ndarray:
        return corner_rmse(
            params_to_homography(params["h"]), src_corners, target_corners
        )

    if not (config.optimize_homography and config.optimize_camera):
        raise ValueError("trace optimization requires joint H + camera optimization")

    joint_params: JointParams = {"h": h_params, "camera": camera_vec}
    loss_fn = make_joint_render_model_loss(
        target,
        tag_pattern,
        height,
        width,
        loss_mask=config.loss_mask,
        src_corners=src_corners,
        target_corners=target_corners,
        corner_weight=config.corner_weight,
        config=config,
    )
    joint_params, losses, corner_history = _run_optax_loop(
        joint_params,
        loss_fn,
        config,
        corner_rmse_fn=_corner_rmse_joint,
    )
    assert corner_history is not None
    return OptimizationTrace(
        params_to_homography(joint_params["h"]),
        vector_to_model_params(joint_params["camera"]),
        losses,
        corner_history,
    )


def optimize_render_model(
    H_init: jnp.ndarray,
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    camera_init: RenderModelParams | None = None,
    config: OptimizeRenderConfig | None = None,
    src_corners: jnp.ndarray | None = None,
    target_corners: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, RenderModelParams, list[float]]:
    """Optimize homography and/or camera params against a target image."""
    from render_model.pipeline import default_params

    config = config or OptimizeRenderConfig()
    camera_init = camera_init or default_params()
    camera_vec = model_params_to_vector(camera_init)
    h_params = homography_to_params(H_init)

    if config.optimize_camera and not config.optimize_homography:
        loss_fn = make_render_model_loss_fixed_h(
            target,
            tag_pattern,
            height,
            width,
            params_to_homography(h_params),
            loss_mask=config.loss_mask,
            config=config,
        )
        camera_config = replace(config, learning_rate=_camera_learning_rate(config))
        packed, losses, _ = _run_optax_loop(camera_vec, loss_fn, camera_config)
        return (
            params_to_homography(h_params),
            vector_to_model_params(packed),
            losses,
        )

    if config.optimize_homography and config.optimize_camera:
        joint_params: JointParams = {"h": h_params, "camera": camera_vec}
        loss_fn = make_joint_render_model_loss(
            target,
            tag_pattern,
            height,
            width,
            loss_mask=config.loss_mask,
            src_corners=src_corners,
            target_corners=target_corners,
            corner_weight=config.corner_weight,
            config=config,
        )
        joint_params, losses, _ = _run_optax_loop(joint_params, loss_fn, config)
        return (
            params_to_homography(joint_params["h"]),
            vector_to_model_params(joint_params["camera"]),
            losses,
        )

    packed = _pack_opt_params(
        h_params,
        camera_vec,
        optimize_homography=config.optimize_homography,
        optimize_camera=config.optimize_camera,
    )
    loss_fn = make_render_model_loss(
        target,
        tag_pattern,
        height,
        width,
        fixed_camera=camera_init,
        loss_mask=config.loss_mask,
        src_corners=src_corners,
        target_corners=target_corners,
        corner_weight=config.corner_weight,
        optimize_homography=config.optimize_homography,
        optimize_camera=config.optimize_camera,
        config=config,
    )
    packed, losses, _ = _run_optax_loop(packed, loss_fn, config)

    h_out, camera_out = _unpack_opt_params(
        packed,
        optimize_homography=config.optimize_homography,
        optimize_camera=config.optimize_camera,
    )
    H_opt = params_to_homography(h_out) if h_out is not None else params_to_homography(h_params)
    if config.optimize_camera:
        assert camera_out is not None
        camera_final = vector_to_model_params(camera_out)
    else:
        camera_final = camera_init
    return H_opt, camera_final, losses
