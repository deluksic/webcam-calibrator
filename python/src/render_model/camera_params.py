"""Bounded pack/unpack for ``RenderModelParams`` (optimizer internal coordinates)."""

import jax
import jax.numpy as jnp
from jax.nn import sigmoid, softplus

from render_model.lighting import bounded_light_grads, raw_from_light_grads
from render_model.pipeline import LEVEL_GAP, RenderModelParams, sanitize_levels

GAMMA_MIN = 1.0
GAMMA_MAX = 4.0
PSF_SIGMA_MIN = 0.05
SHARPEN_SIGMA_MIN = 0.3
SHARPEN_SIGMA_MAX = 2.0


def _logit(p: jnp.ndarray) -> jnp.ndarray:
    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    return jnp.log(p / (1.0 - p))


def _softplus_inv(y: jnp.ndarray) -> jnp.ndarray:
    y = jnp.maximum(y, 0.0)
    return jnp.where(y > 1e-6, jnp.log(jnp.expm1(y)), jnp.float32(-20.0))


def model_params_to_vector(params: RenderModelParams) -> jnp.ndarray:
    """Map physical camera params → unconstrained optimization vector."""
    black, white = sanitize_levels(params.black_level, params.white_level)
    gamma = jnp.clip(params.gamma, GAMMA_MIN, GAMMA_MAX)
    psf = jnp.maximum(params.psf_sigma, PSF_SIGMA_MIN)
    sharpen_sigma = jnp.clip(params.sharpen_sigma, SHARPEN_SIGMA_MIN, SHARPEN_SIGMA_MAX)
    span = white - black
    span_denom = jnp.maximum(1.0 - black - LEVEL_GAP, LEVEL_GAP)
    raw_u, raw_v = raw_from_light_grads(params.light_grad_u, params.light_grad_v)
    return jnp.stack(
        [
            _softplus_inv(psf - PSF_SIGMA_MIN),
            params.sharpen_amount,
            _logit(
                (sharpen_sigma - SHARPEN_SIGMA_MIN)
                / (SHARPEN_SIGMA_MAX - SHARPEN_SIGMA_MIN)
            ),
            _logit((gamma - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)),
            _logit(black / jnp.maximum(1.0 - LEVEL_GAP, LEVEL_GAP)),
            _logit((span - LEVEL_GAP) / span_denom),
            raw_u,
            raw_v,
        ]
    )


def camera_params_physical_vector(params: RenderModelParams) -> jnp.ndarray:
    """Physical camera values for display (not the internal optimization vector)."""
    black, white = sanitize_levels(params.black_level, params.white_level)
    return jnp.stack(
        [
            params.psf_sigma,
            params.sharpen_amount,
            params.sharpen_sigma,
            params.gamma,
            black,
            white,
            params.light_grad_u,
            params.light_grad_v,
        ]
    )


def decode_joint_params_per_step(packed: jnp.ndarray) -> jnp.ndarray:
    """Decode packed LM history to plottable physical units ``(steps, 16)``.

    First 8 entries are corner shifts in pixels (same as the packed LM vector).
    Remaining 8 are decoded camera parameters (psf, sharpen, …, light u/v).
    """

    def decode_row(row: jnp.ndarray) -> jnp.ndarray:
        cam = camera_params_physical_vector(vector_to_model_params(row[8:16]))
        return jnp.concatenate([row[:8], cam])

    return jax.vmap(decode_row)(packed)


def vector_to_model_params(values: jnp.ndarray) -> RenderModelParams:
    """Map optimization vector → physical camera params (bounded)."""
    psf_sigma = PSF_SIGMA_MIN + softplus(values[0])
    sharpen_amount = values[1]
    sharpen_sigma = SHARPEN_SIGMA_MIN + (SHARPEN_SIGMA_MAX - SHARPEN_SIGMA_MIN) * sigmoid(
        values[2]
    )
    gamma = GAMMA_MIN + (GAMMA_MAX - GAMMA_MIN) * sigmoid(values[3])
    black_level = (1.0 - LEVEL_GAP) * sigmoid(values[4])
    span = LEVEL_GAP + (1.0 - black_level - LEVEL_GAP) * sigmoid(values[5])
    white_level = black_level + span
    light_grad_u, light_grad_v = bounded_light_grads(values[6], values[7])
    return RenderModelParams(
        psf_sigma=psf_sigma,
        sharpen_amount=sharpen_amount,
        sharpen_sigma=sharpen_sigma,
        gamma=gamma,
        black_level=black_level,
        white_level=white_level,
        light_grad_u=light_grad_u,
        light_grad_v=light_grad_v,
    )
