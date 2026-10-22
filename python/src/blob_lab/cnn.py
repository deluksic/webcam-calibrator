"""Tiny JAX CNN patch classifier with a manual Adam optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from blob_lab.metrics import best_threshold, binary_metrics, roc_auc


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 7
    seed: int = 0


@dataclass
class TrainedCnn:
    params: dict
    history: dict
    threshold: float
    metrics: dict
    config: TrainingConfig

    def score(self, images: np.ndarray) -> np.ndarray:
        return np.asarray(predict_proba(self.params, images), dtype=np.float32)


def _init_layer(key: jax.Array, shape: tuple[int, ...], fan_in: int) -> jax.Array:
    return jax.random.normal(key, shape, dtype=jnp.float32) * jnp.sqrt(2.0 / max(1, fan_in))


def init_cnn(key: jax.Array, patch_size: int = 32, filters: int = 5) -> dict:
    key, k1, k2, k3 = jax.random.split(key, 4)
    pooled = patch_size // 2
    return {
        "conv_w": _init_layer(k1, (3, 3, 1, filters), 9),
        "conv_b": jnp.zeros((filters,), dtype=jnp.float32),
        "dense_w": _init_layer(k2, (filters * pooled * pooled, 1), filters * pooled * pooled),
        "dense_b": jnp.zeros((1,), dtype=jnp.float32),
    }


def forward(params: dict, images: jax.Array) -> jax.Array:
    """Return logits for ``images`` shaped (N, H, W, 1)."""
    x = jax.lax.conv_general_dilated(
        images,
        params["conv_w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = jax.nn.relu(x + params["conv_b"].reshape(1, 1, 1, -1))
    n, h, w, c = x.shape
    x = x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))
    x = x.reshape(n, -1)
    return x @ params["dense_w"] + params["dense_b"]


def predict_proba(params: dict, images: np.ndarray) -> np.ndarray:
    logits = forward(params, jnp.asarray(images, dtype=jnp.float32))
    return np.asarray(jax.nn.sigmoid(logits[:, 0]), dtype=np.float32)


def _binary_loss(params: dict, images: jax.Array, labels: jax.Array) -> jax.Array:
    logits = forward(params, images)[:, 0]
    return jnp.mean(jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def _init_adam(params: dict) -> dict:
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": jnp.asarray(0, dtype=jnp.int32),
    }


def _adam_step(params: dict, state: dict, grads: dict, learning_rate: float, weight_decay: float) -> tuple[dict, dict]:
    t = state["t"] + 1
    beta1, beta2 = 0.9, 0.999
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * (g * g), state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta2**t), v)
    updated = jax.tree_util.tree_map(
        lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + 1e-8) - learning_rate * weight_decay * p,
        params,
        m_hat,
        v_hat,
    )
    return updated, {"m": m, "v": v, "t": t}


def train_cnn(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    config: TrainingConfig | None = None,
    *,
    filters: int = 5,
) -> TrainedCnn:
    config = config or TrainingConfig()
    patch_size = int(train_images.shape[1])
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    params = init_cnn(init_key, patch_size, filters)
    state = _init_adam(params)

    train_images_j = jnp.asarray(train_images, dtype=jnp.float32)
    train_labels_j = jnp.asarray(train_labels, dtype=jnp.float32)
    val_images_j = jnp.asarray(validation_images, dtype=jnp.float32)
    val_labels_j = jnp.asarray(validation_labels, dtype=jnp.float32)

    @jax.jit
    def train_step(params, state, images, labels):
        loss, grads = jax.value_and_grad(_binary_loss)(params, images, labels)
        params, state = _adam_step(params, state, grads, config.learning_rate, config.weight_decay)
        return params, state, loss

    rng = np.random.default_rng(config.seed)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_params = params
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(config.epochs):
        order = rng.permutation(len(train_images_j))
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            params, state, loss = train_step(
                params,
                state,
                train_images_j[batch],
                train_labels_j[batch],
            )
            losses.append(float(loss))
        val_logits = forward(params, val_images_j)[:, 0]
        val_loss = float(_binary_loss(params, val_images_j, val_labels_j))
        val_pred = np.asarray(jax.nn.sigmoid(val_logits) >= 0.5, dtype=np.float32)
        val_accuracy = float(np.mean(val_pred == np.asarray(val_labels_j)))
        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break

    val_scores = np.asarray(jax.nn.sigmoid(forward(best_params, val_images_j)[:, 0]), dtype=np.float32)
    threshold, metrics = best_threshold(np.asarray(val_labels_j), val_scores)
    metrics = dict(metrics)
    metrics["auc"] = roc_auc(np.asarray(val_labels_j), val_scores)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return TrainedCnn(best_params, history, threshold, metrics, config)


def parameter_count(params: dict) -> int:
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params)))


def save_cnn(path: str, model: TrainedCnn) -> None:
    arrays = {f"param_{key}": np.asarray(value) for key, value in model.params.items()}
    arrays["threshold"] = np.asarray(model.threshold)
    np.savez(path, **arrays)


def load_cnn(path: str, patch_size: int = 32, filters: int = 5) -> TrainedCnn:
    data = np.load(path)
    params = {
        "conv_w": jnp.asarray(data["param_conv_w"]),
        "conv_b": jnp.asarray(data["param_conv_b"]),
        "dense_w": jnp.asarray(data["param_dense_w"]),
        "dense_b": jnp.asarray(data["param_dense_b"]),
    }
    return TrainedCnn(params, {}, float(data["threshold"]), binary_metrics(
        np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), float(data["threshold"])
    ), TrainingConfig())
