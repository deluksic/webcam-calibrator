"""Tiny fully convolutional U-Net for single-pixel keypoint heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class UNetConfig:
    filters: int = 5
    epochs: int = 25
    batch_size: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    max_positive_weight: float = 1000.0
    focal_gamma: float = 2.0
    dice_weight: float = 0.5
    seed: int = 0


@dataclass
class TrainedUNet:
    params: dict
    history: dict
    config: UNetConfig

    def score(self, images: np.ndarray) -> np.ndarray:
        return predict_heatmap(self.params, images)


def _conv(x: jax.Array, w: jax.Array, b: jax.Array, stride: int = 1) -> jax.Array:
    y = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + b.reshape(1, 1, 1, -1)


def _init(key: jax.Array, shape: tuple[int, ...], fan_in: int) -> jax.Array:
    return jax.random.normal(key, shape, dtype=jnp.float32) * jnp.sqrt(2.0 / max(1, fan_in))


def init_unet(key: jax.Array, filters: int = 5) -> dict:
    f = filters
    keys = jax.random.split(key, 14)
    return {
        "e1a": _init(keys[0], (3, 3, 1, f), 9),
        "e1b": _init(keys[1], (3, 3, f, f), 9 * f),
        "e2a": _init(keys[2], (3, 3, f, 2 * f), 9 * f),
        "e2b": _init(keys[3], (3, 3, 2 * f, 2 * f), 9 * 2 * f),
        "e3a": _init(keys[4], (3, 3, 2 * f, 4 * f), 9 * 2 * f),
        "e3b": _init(keys[5], (3, 3, 4 * f, 4 * f), 9 * 4 * f),
        "d2": _init(keys[6], (3, 3, 6 * f, 2 * f), 9 * 6 * f),
        "d1": _init(keys[7], (3, 3, 3 * f, f), 9 * 3 * f),
        "out": _init(keys[8], (1, 1, f, 1), f),
        "b1a": jnp.zeros((f,), dtype=jnp.float32),
        "b1b": jnp.zeros((f,), dtype=jnp.float32),
        "b2a": jnp.zeros((2 * f,), dtype=jnp.float32),
        "b2b": jnp.zeros((2 * f,), dtype=jnp.float32),
        "b3a": jnp.zeros((4 * f,), dtype=jnp.float32),
        "b3b": jnp.zeros((4 * f,), dtype=jnp.float32),
        "bd2": jnp.zeros((2 * f,), dtype=jnp.float32),
        "bd1": jnp.zeros((f,), dtype=jnp.float32),
        "bout": jnp.zeros((1,), dtype=jnp.float32),
    }


def _pool2(x: jax.Array) -> jax.Array:
    return jax.lax.reduce_window(
        x,
        -jnp.inf,
        jax.lax.max,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="SAME",
    )


def _resize_to(x: jax.Array, target: jax.Array) -> jax.Array:
    return jax.image.resize(
        x,
        (x.shape[0], target.shape[1], target.shape[2], x.shape[3]),
        method="bilinear",
    )


def forward(params: dict, images: jax.Array) -> jax.Array:
    e1 = jax.nn.relu(_conv(images, params["e1a"], params["b1a"]))
    e1 = jax.nn.relu(_conv(e1, params["e1b"], params["b1b"]))
    e2 = _pool2(e1)
    e2 = jax.nn.relu(_conv(e2, params["e2a"], params["b2a"]))
    e2 = jax.nn.relu(_conv(e2, params["e2b"], params["b2b"]))
    e3 = _pool2(e2)
    e3 = jax.nn.relu(_conv(e3, params["e3a"], params["b3a"]))
    e3 = jax.nn.relu(_conv(e3, params["e3b"], params["b3b"]))

    d2 = _resize_to(e3, e2)
    d2 = jnp.concatenate([d2, e2], axis=-1)
    d2 = jax.nn.relu(_conv(d2, params["d2"], params["bd2"]))
    d1 = _resize_to(d2, e1)
    d1 = jnp.concatenate([d1, e1], axis=-1)
    d1 = jax.nn.relu(_conv(d1, params["d1"], params["bd1"]))
    return _conv(d1, params["out"], params["bout"])


def predict_heatmap(params: dict, images: np.ndarray) -> np.ndarray:
    logits = forward(params, jnp.asarray(images, dtype=jnp.float32))
    return np.asarray(jax.nn.sigmoid(logits[..., 0]), dtype=np.float32)


def _loss(params: dict, images: jax.Array, targets: jax.Array, config: UNetConfig) -> jax.Array:
    logits = forward(params, images)[..., 0]
    probs = jax.nn.sigmoid(logits)
    eps = 1e-6
    positive_count = jnp.sum(targets)
    positive_weight = jnp.minimum(
        config.max_positive_weight,
        (targets.size - positive_count) / (positive_count + 1.0),
    )
    bce = -(targets * jnp.log(probs + eps) + (1.0 - targets) * jnp.log(1.0 - probs + eps))
    # Focal modulation stops the huge background from dominating the gradient.
    pt = targets * probs + (1.0 - targets) * (1.0 - probs)
    focal = jnp.power(1.0 - pt, config.focal_gamma) * bce
    weighted = (1.0 + positive_weight * targets) * focal
    dice = 1.0 - (2.0 * jnp.sum(probs * targets) + 1.0) / (jnp.sum(probs) + jnp.sum(targets) + 1.0)
    return jnp.mean(weighted) + config.dice_weight * dice


def _init_adam(params: dict) -> dict:
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": jnp.asarray(0, dtype=jnp.int32),
    }


def _adam_step(params: dict, state: dict, grads: dict, lr: float, weight_decay: float):
    t = state["t"] + 1
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mh = jax.tree_util.tree_map(lambda m: m / (1 - b1**t), m)
    vh = jax.tree_util.tree_map(lambda v: v / (1 - b2**t), v)
    updated = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + 1e-8) - lr * weight_decay * p,
        params,
        mh,
        vh,
    )
    return updated, {"m": m, "v": v, "t": t}


def train_unet(
    train_images: np.ndarray,
    train_targets: np.ndarray,
    validation_images: np.ndarray,
    validation_targets: np.ndarray,
    config: UNetConfig | None = None,
    on_epoch: Callable[[int, int, float, float], None] | None = None,
) -> TrainedUNet:
    config = config or UNetConfig()
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    params = init_unet(init_key, config.filters)
    state = _init_adam(params)
    train_x = jnp.asarray(train_images, dtype=jnp.float32)
    train_y = jnp.asarray(train_targets[..., 0], dtype=jnp.float32)
    val_x = jnp.asarray(validation_images, dtype=jnp.float32)
    val_y = jnp.asarray(validation_targets[..., 0], dtype=jnp.float32)

    @jax.jit
    def step(params, state, images, targets):
        loss, grads = jax.value_and_grad(_loss)(params, images, targets, config)
        params, state = _adam_step(params, state, grads, config.learning_rate, config.weight_decay)
        return params, state, loss

    rng = np.random.default_rng(config.seed)
    history = {"train_loss": [], "val_loss": []}
    for _epoch in range(config.epochs):
        order = rng.permutation(len(train_x))
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            params, state, loss = step(params, state, train_x[batch], train_y[batch])
            losses.append(float(loss))
        val_loss = float(_loss(params, val_x, val_y, config))
        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["val_loss"].append(val_loss)
        if on_epoch is not None:
            on_epoch(len(history["train_loss"]), config.epochs, history["train_loss"][-1], val_loss)
    return TrainedUNet(params, history, config)


def parameter_count(params: dict) -> int:
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params)))
