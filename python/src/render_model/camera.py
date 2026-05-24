"""Camera models: linear, gamma, sharpening, noise, composable pipeline."""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp


class CameraModel(Protocol):
    def apply(self, image: jnp.ndarray, *, key: jax.Array | None = None) -> jnp.ndarray:
        """Transform rendered linear image."""
        ...


@dataclass(frozen=True)
class LinearCamera:
    """Identity camera (pure linear response)."""

    def apply(self, image: jnp.ndarray, *, key: jax.Array | None = None) -> jnp.ndarray:
        return image


@dataclass(frozen=True)
class GammaCamera:
    gamma: float = 2.2

    def apply(self, image: jnp.ndarray, *, key: jax.Array | None = None) -> jnp.ndarray:
        return jnp.clip(image, 0.0, 1.0) ** (1.0 / self.gamma)


@dataclass(frozen=True)
class SharpeningCamera:
    """Unsharp mask: image + amount * (image - blurred)."""

    sigma: float = 1.0
    amount: float = 0.5

    def apply(self, image: jnp.ndarray, *, key: jax.Array | None = None) -> jnp.ndarray:
        # 3x3 box blur via pad+slice (no conv; JAX 0.4 lacks gaussian_filter).
        p = jnp.pad(image, 1, mode="edge")
        blurred = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
        sharp = image + self.amount * (image - blurred)
        return jnp.clip(sharp, 0.0, 1.0)


@dataclass(frozen=True)
class GaussianNoiseCamera:
    std: float = 0.02

    def apply(self, image: jnp.ndarray, *, key: jax.Array | None = None) -> jnp.ndarray:
        if key is None:
            return image
        noise = jax.random.normal(key, image.shape, dtype=image.dtype) * self.std
        return jnp.clip(image + noise, 0.0, 1.0)


@dataclass(frozen=True)
class CameraPipeline:
    """Compose camera effects in order."""

    stages: tuple[CameraModel, ...]

    def apply(self, image: jnp.ndarray, *, key: jax.Array | None = None) -> jnp.ndarray:
        out = image
        subkeys = jax.random.split(key, len(self.stages)) if key is not None else [None] * len(self.stages)
        for stage, subkey in zip(self.stages, subkeys, strict=True):
            out = stage.apply(out, key=subkey)
        return out
