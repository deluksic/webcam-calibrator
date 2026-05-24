"""Homography optimizer via differentiable rendering."""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optax

from render_model.camera import CameraModel, LinearCamera
from render_model.esf import ESFProfile, esf_params_to_profile, esf_profile_to_params
from render_model.homography import (
    corner_reprojection_loss,
    homography_to_params,
    params_to_homography,
)
from render_model.renderer import RenderFn, render_tag_antialiased, render_tag_esf

HOMOGRAPHY_PARAM_COUNT = 8


@dataclass(frozen=True)
class ESFConfig:
    """Joint ESF + homography optimization settings."""

    profile: ESFProfile
    optimize_levels: bool = True  # i_black, i_white
    optimize_shape: bool = False  # esf sample deltas (ramp LUT fixed at init)


@dataclass(frozen=True)
class OptimizeConfig:
    learning_rate: float = 2e-4
    n_steps: int = 800
    optimizer: str = "sgd"
    max_grad_norm: float = 1.0
    loss_mask: jnp.ndarray | None = None
    corner_weight: float = 1.0
    esf: ESFConfig | None = None
    fixed_esf_profile: ESFProfile | None = None


def _unpack_params(
    params: jnp.ndarray,
    esf_config: ESFConfig | None,
) -> tuple[jnp.ndarray, ESFProfile | None]:
    h_params = params[:HOMOGRAPHY_PARAM_COUNT]
    if esf_config is None:
        return h_params, None
    esf_params = params[HOMOGRAPHY_PARAM_COUNT:]
    profile = esf_params_to_profile(esf_params, esf_config.profile)
    return h_params, profile


def _init_params(H_init: jnp.ndarray, esf_config: ESFConfig | None) -> jnp.ndarray:
    params = homography_to_params(H_init)
    if esf_config is None:
        return params
    return jnp.concatenate([params, esf_profile_to_params(esf_config.profile)])


def make_render_loss(
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    render_fn: RenderFn = render_tag_antialiased,
    camera: CameraModel | None = None,
    loss_mask: jnp.ndarray | None = None,
    src_corners: jnp.ndarray | None = None,
    target_corners: jnp.ndarray | None = None,
    corner_weight: float = 0.0,
    esf_config: ESFConfig | None = None,
    fixed_esf_profile: ESFProfile | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Combined pixel + corner loss; optional learnable or fixed ESF."""
    camera = camera or LinearCamera()
    use_corners = corner_weight > 0.0 and src_corners is not None and target_corners is not None
    use_learnable_esf = esf_config is not None
    use_fixed_esf = fixed_esf_profile is not None and not use_learnable_esf

    def loss(params: jnp.ndarray) -> jnp.ndarray:
        if use_learnable_esf:
            h_params, profile = _unpack_params(params, esf_config)
            H = params_to_homography(h_params)
            rendered = render_tag_esf(H, tag_pattern, height, width, profile)
        else:
            H = params_to_homography(params)
            if use_fixed_esf:
                rendered = render_tag_esf(H, tag_pattern, height, width, fixed_esf_profile)
            else:
                rendered = render_fn(H, tag_pattern, height, width)
        observed = camera.apply(rendered)
        diff2 = (observed - target) ** 2
        if loss_mask is not None:
            pixel_loss = jnp.sum(diff2 * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        else:
            pixel_loss = jnp.mean(diff2)

        if not use_corners:
            return pixel_loss

        corner_loss = corner_reprojection_loss(H, src_corners, target_corners)
        return pixel_loss + corner_weight * corner_loss

    return loss


def optimize_homography(
    H_init: jnp.ndarray,
    target: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    config: OptimizeConfig | None = None,
    render_fn: RenderFn = render_tag_antialiased,
    camera: CameraModel | None = None,
    src_corners: jnp.ndarray | None = None,
    target_corners: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, ESFProfile | None, list[float]]:
    """Gradient descent on homography (+ optional ESF levels); returns (H_opt, esf_opt, losses)."""
    config = config or OptimizeConfig()
    learn_esf = config.esf is not None and (
        config.esf.optimize_levels or config.esf.optimize_shape
    )
    params = _init_params(H_init, config.esf if learn_esf else None)
    loss_fn = make_render_loss(
        target,
        tag_pattern,
        height,
        width,
        render_fn=render_fn,
        camera=camera,
        loss_mask=config.loss_mask,
        src_corners=src_corners,
        target_corners=target_corners,
        corner_weight=config.corner_weight,
        esf_config=config.esf if learn_esf else None,
        fixed_esf_profile=config.fixed_esf_profile,
    )

    if config.optimizer == "adam":
        base = optax.adam(config.learning_rate)
    else:
        base = optax.sgd(config.learning_rate)
    opt = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), base)

    opt_state = opt.init(params)
    step_fn = jax.jit(lambda p: jax.value_and_grad(loss_fn)(p))

    losses: list[float] = []
    for _ in range(config.n_steps):
        loss, grads = step_fn(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))

    h_params, esf_opt = _unpack_params(params, config.esf if learn_esf else None)
    return params_to_homography(h_params), esf_opt, losses
