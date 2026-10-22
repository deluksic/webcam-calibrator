"""Full-target perspective dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from blob_lab.camera import CameraRanges, random_pose
from blob_lab.render import apply_homography, camera_homography, render_target
from blob_lab.target_layout import HexTargetLayout, build_target_texture


@dataclass(frozen=True)
class FullTargetDatasetConfig:
    patch_size: int = 32
    output_width: int = 640
    output_height: int = 480
    samples: int = 400
    patches_per_scene: int = 8
    positive_fraction: float = 0.5
    supersample: int = 1
    camera: CameraRanges = field(default_factory=CameraRanges)
    blur_sigma: tuple[float, float] = (0.0, 1.2)
    noise_sigma: tuple[float, float] = (0.0, 0.025)
    contrast: tuple[float, float] = (0.7, 1.0)
    background_linear: float = 0.5
    seed: int = 0


@dataclass
class PatchDataset:
    images: np.ndarray
    labels: np.ndarray
    metadata: list[dict[str, Any]]
    config: FullTargetDatasetConfig

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def split(self, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)):
        n = len(self)
        n_train = int(n * fractions[0])
        n_val = int(n * fractions[1])
        return (
            self.subset(slice(0, n_train)),
            self.subset(slice(n_train, n_train + n_val)),
            self.subset(slice(n_train + n_val, n)),
        )

    def subset(self, selection) -> "PatchDataset":
        return PatchDataset(
            images=self.images[selection],
            labels=self.labels[selection],
            metadata=[self.metadata[i] for i in np.arange(len(self))[selection]],
            config=self.config,
        )


def _crop(image: np.ndarray, center: tuple[float, float], size: int) -> np.ndarray:
    patch = cv2.getRectSubPix(image.astype(np.float32), (size, size), center)
    if patch.ndim == 2:
        patch = patch[:, :, None]
    return patch.astype(np.float32)


def _project(layout: HexTargetLayout, h: jax.Array, points) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    arr = jnp.asarray([[point.x, point.y] for point in points], dtype=jnp.float32)
    return np.asarray(apply_homography(h, arr), dtype=np.float32)


def generate_full_target_dataset(
    config: FullTargetDatasetConfig,
    layout: HexTargetLayout,
    *,
    count: int | None = None,
) -> PatchDataset:
    rng = np.random.default_rng(config.seed)
    count = config.samples if count is None else count
    if count <= 0:
        return PatchDataset(
            images=np.zeros((0, config.patch_size, config.patch_size, 1), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.float32),
            metadata=[],
            config=config,
        )

    texture = jnp.asarray(build_target_texture(layout, config.background_linear), dtype=jnp.float32)
    render_fn = jax.jit(
        lambda h, sigma: render_target(
            h,
            texture,
            output_size=(config.output_width, config.output_height),
            supersample=config.supersample,
            psf_sigma=sigma,
            background=config.background_linear,
        )
    )

    square_points = layout.square_centers
    junction_points = layout.junction_points
    images: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []
    attempts = 0
    positives_target = int(round(count * config.positive_fraction))
    negatives_target = count - positives_target
    positives_done = 0
    negatives_done = 0
    while len(images) < count and attempts < count * 80:
        attempts += 1
        pose = random_pose(rng, config.camera)
        h = camera_homography(
            yaw_deg=pose.yaw_deg,
            pitch_deg=pose.pitch_deg,
            roll_deg=pose.roll_deg,
            distance=pose.distance,
            focal=pose.focal,
            tx=pose.tx,
            ty=pose.ty,
            target_size=(layout.width_px, layout.height_px),
            output_size=(config.output_width, config.output_height),
        )
        det = float(jnp.linalg.det(h))
        if not np.isfinite(det) or abs(det) < 1e-8:
            continue
        sigma = float(rng.uniform(*config.blur_sigma))
        rendered = np.asarray(render_fn(h, sigma), dtype=np.float32)
        if rendered.shape != (config.output_height, config.output_width):
            rendered = rendered.reshape(config.output_height, config.output_width)
        if float(rendered.std()) < 0.02:
            continue

        for positive in (True, False):
            remaining_class = (
                positives_target - positives_done
                if positive
                else negatives_target - negatives_done
            )
            if remaining_class <= 0 or len(images) >= count:
                continue
            candidates = square_points if positive else junction_points
            projected = _project(layout, h, candidates)
            if projected.shape[0] == 0:
                continue
            margin = config.patch_size / 2 + 1
            valid = np.where(
                (projected[:, 0] >= margin)
                & (projected[:, 0] < config.output_width - margin)
                & (projected[:, 1] >= margin)
                & (projected[:, 1] < config.output_height - margin)
            )[0]
            if valid.size == 0:
                continue
            order = rng.permutation(valid)
            take = min(config.patches_per_scene, remaining_class, count - len(images), len(order))
            for selected in order[:take]:
                center = tuple(float(value) for value in projected[int(selected)])
                patch = _crop(rendered, center, config.patch_size)
                contrast = float(rng.uniform(*config.contrast))
                patch = (patch - 0.5) * contrast + 0.5
                noise = float(rng.uniform(*config.noise_sigma))
                if noise > 0:
                    patch = patch + rng.normal(0.0, noise, size=patch.shape).astype(np.float32)
                patch = np.clip(patch, 0.0, 1.0)
                images.append(patch)
                labels.append(1.0 if positive else 0.0)
                if positive:
                    positives_done += 1
                else:
                    negatives_done += 1
                metadata.append(
                    {
                        "positive": bool(positive),
                        "center": center,
                        "class": "square" if positive else "junction",
                        "pose": {
                            "yaw_deg": pose.yaw_deg,
                            "pitch_deg": pose.pitch_deg,
                            "roll_deg": pose.roll_deg,
                            "distance": pose.distance,
                            "focal": pose.focal,
                        },
                        "blur_sigma": sigma,
                        "contrast": contrast,
                        "noise_sigma": noise,
                    }
                )
                if len(images) >= count:
                    break

    if not images:
        raise RuntimeError("Could not generate any dataset patches; check target/layout/camera ranges.")
    return PatchDataset(
        images=np.stack(images, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        metadata=metadata,
        config=config,
    )


def preview_grid(dataset: PatchDataset, rows: int = 4, cols: int = 6) -> np.ndarray:
    """Return an RGB tile grid for the marimo preview."""
    count = min(len(dataset), rows * cols)
    tiles: list[np.ndarray] = []
    for index in range(count):
        patch = dataset.images[index, :, :, 0]
        rgb = np.repeat(patch[:, :, None], 3, axis=2)
        rgb[0, 0] = [1.0, 0.0, 0.0] if dataset.labels[index] > 0.5 else [0.0, 0.0, 1.0]
        tiles.append(rgb)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((dataset.config.patch_size, dataset.config.patch_size, 3), dtype=np.float32))
    return np.concatenate(
        [np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1) for row in range(rows)],
        axis=0,
    )
