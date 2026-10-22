"""Full-target single-pixel keypoint heatmap dataset."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from pathlib import Path
from math import ceil, exp

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from blob_lab.camera import CameraRanges, random_pose
from blob_lab.dataset import _crop
from blob_lab.render import apply_homography, camera_homography, render_target
from blob_lab.target_layout import HexTargetLayout, build_target_texture


@dataclass(frozen=True)
class HeatmapDatasetConfig:
    output_width: int = 640
    output_height: int = 480
    samples: int = 80
    supersample: int = 1
    camera: CameraRanges = field(default_factory=CameraRanges)
    blur_sigma: tuple[float, float] = (0.0, 1.2)
    noise_sigma: tuple[float, float] = (0.0, 0.025)
    contrast: tuple[float, float] = (0.75, 1.0)
    background_linear: float = 0.5
    natural_fraction: float = 0.25
    natural_dir: str | None = None
    randomize_colors: bool = True
    feature_sigma: float = 1.2
    variance_window: int = 5
    keypoint_sigma: float = 0.8
    seed: int = 0


@dataclass
class HeatmapDataset:
    images: np.ndarray
    features: np.ndarray
    heatmaps: np.ndarray
    masks: np.ndarray
    junctions: np.ndarray
    metadata: list[dict[str, Any]]
    config: HeatmapDatasetConfig

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def split(self, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)):
        n = len(self)
        n_train = int(n * fractions[0])
        n_val = int(n * fractions[1])
        return (
            self.subset(slice(0, n_train)),
            self.subset(slice(n_train, n_train + n_val)),
            self.subset(slice(n_train + n_val, n)),
        )

    def subset(self, selection) -> "HeatmapDataset":
        return HeatmapDataset(
            images=self.images[selection],
            features=self.features[selection],
            heatmaps=self.heatmaps[selection],
            masks=self.masks[selection],
            junctions=self.junctions[selection],
            metadata=[self.metadata[i] for i in np.arange(len(self))[selection]],
            config=self.config,
        )


def _add_gaussian(heatmap: np.ndarray, x: float, y: float, sigma: float) -> None:
    """Add a subpixel-centered Gaussian to a single-channel target."""
    if sigma <= 0:
        ix = int(np.floor(x))
        iy = int(np.floor(y))
        if 0 <= ix < heatmap.shape[1] and 0 <= iy < heatmap.shape[0]:
            heatmap[iy, ix, 0] = 1.0
        return
    radius = max(1, int(ceil(3.0 * sigma)))
    ix = int(np.floor(x))
    iy = int(np.floor(y))
    height, width = heatmap.shape[:2]
    for py in range(max(0, iy - radius), min(height, iy + radius + 1)):
        for px in range(max(0, ix - radius), min(width, ix + radius + 1)):
            dx = (px + 0.5) - x
            dy = (py + 0.5) - y
            value = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            if value > heatmap[py, px, 0]:
                heatmap[py, px, 0] = value


def _project_points(layout: HexTargetLayout, h: jax.Array) -> np.ndarray:
    points = layout.square_centers
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    arr = jnp.asarray([[p.x, p.y] for p in points], dtype=jnp.float32)
    return np.asarray(apply_homography(h, arr), dtype=np.float32)


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _procedural_natural(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Smooth, non-target-like texture used when no photo directory is available."""
    coarse = rng.random((max(4, height // 24), max(4, width // 24))).astype(np.float32)
    image = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    for octave in (8, 16, 32):
        small = rng.random((max(3, height // octave), max(3, width // octave))).astype(np.float32)
        image += 0.35 * cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    image += 0.15 * (xx / max(1, width - 1) - 0.5) + 0.15 * (yy / max(1, height - 1) - 0.5)
    image = (image - image.min()) / max(1e-6, image.max() - image.min())
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _load_natural_image(rng: np.random.Generator, config: HeatmapDatasetConfig) -> np.ndarray:
    height, width = config.output_height, config.output_width
    paths: list[Path] = []
    if config.natural_dir:
        root = Path(config.natural_dir)
        if root.is_dir():
            paths = sorted(
                path
                for path in root.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            )
    if paths:
        path = paths[int(rng.integers(0, len(paths)))]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return _procedural_natural(rng, height, width)
        image = image.astype(np.float32) / 255.0
    else:
        image = _procedural_natural(rng, height, width)
    # Cover-resize and crop to the requested output size.
    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(width, int(round(image.shape[1] * scale))), max(height, int(round(image.shape[0] * scale)))))
    top = max(0, (resized.shape[0] - height) // 2)
    left = max(0, (resized.shape[1] - width) // 2)
    return _srgb_to_linear(resized[top : top + height, left : left + width])


def _normalize_map(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value - value.mean()) / max(1e-5, float(value.std()))


def extract_features(image: np.ndarray, config: HeatmapDatasetConfig) -> np.ndarray:
    """Four fixed input channels: gray, gradient magnitude, variance, DoH."""
    gray = np.asarray(image, dtype=np.float32)
    sigma = max(0.3, float(config.feature_sigma))
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    window = max(3, int(config.variance_window) | 1)
    local_mean = cv2.blur(gray, (window, window))
    local_mean_sq = cv2.blur(gray * gray, (window, window))
    variance = np.maximum(0.0, local_mean_sq - local_mean * local_mean)
    hxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    hyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    hxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    dog = hxx * hyy - hxy * hxy
    channels = [
        _normalize_map(gray),
        _normalize_map(gradient),
        _normalize_map(variance),
        _normalize_map(dog),
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def generate_heatmap_dataset(
    config: HeatmapDatasetConfig,
    layout: HexTargetLayout,
    *,
    count: int | None = None,
) -> HeatmapDataset:
    rng = np.random.default_rng(config.seed)
    count = config.samples if count is None else count
    if count <= 0:
        return HeatmapDataset(
            np.zeros((0, config.output_height, config.output_width, 1), dtype=np.float32),
            np.zeros((0, config.output_height, config.output_width, 4), dtype=np.float32),
            np.zeros((0, config.output_height, config.output_width, 1), dtype=np.float32),
            np.zeros((0, config.output_height, config.output_width, 1), dtype=np.float32),
            np.zeros((0, config.output_height, config.output_width, 1), dtype=np.float32),
            [],
            config,
        )

    texture = jnp.asarray(build_target_texture(layout, config.background_linear), dtype=jnp.float32)
    render_fn = jax.jit(
        lambda h, target_texture, sigma: render_target(
            h,
            target_texture,
            output_size=(config.output_width, config.output_height),
            supersample=config.supersample,
            psf_sigma=sigma,
            background=config.background_linear,
        )
    )

    images: list[np.ndarray] = []
    features: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    junctions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    attempts = 0
    natural_target = int(round(count * max(0.0, min(1.0, config.natural_fraction))))
    natural_done = 0
    target_target = count - natural_target
    target_done = 0
    while len(images) < count and attempts < count * 80:
        attempts += 1
        need_natural = natural_done < natural_target and (
            target_done >= target_target or rng.random() < config.natural_fraction
        )
        if need_natural:
            image = _load_natural_image(rng, config)
            contrast = float(rng.uniform(*config.contrast))
            image = (image - 0.5) * contrast + 0.5
            noise = float(rng.uniform(*config.noise_sigma))
            if noise > 0:
                image = image + rng.normal(0.0, noise, size=image.shape).astype(np.float32)
            image = np.clip(image, 0.0, 1.0)[:, :, None]
            images.append(image)
            features.append(extract_features(image[:, :, 0], config))
            heatmaps.append(np.zeros((config.output_height, config.output_width, 1), dtype=np.float32))
            masks.append(np.zeros((config.output_height, config.output_width, 1), dtype=np.float32))
            junctions.append(np.zeros((config.output_height, config.output_width, 1), dtype=np.float32))
            metadata.append({"points": [], "junctions": [], "natural": True})
            natural_done += 1
            continue

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
        if not np.isfinite(float(jnp.linalg.det(h))) or abs(float(jnp.linalg.det(h))) < 1e-8:
            continue
        sigma = float(rng.uniform(*config.blur_sigma))
        if config.randomize_colors:
            _colors = tuple(int(value) for value in rng.integers(0, 2, size=len(layout.quads)))
            _sample_layout = replace(
                layout,
                quads=tuple(
                    replace(quad, color=color)
                    for quad, color in zip(layout.quads, _colors)
                ),
                colors=_colors,
            )
            _sample_texture = jnp.asarray(
                build_target_texture(_sample_layout, config.background_linear),
                dtype=jnp.float32,
            )
        else:
            _sample_texture = texture
        image = np.asarray(render_fn(h, _sample_texture, sigma), dtype=np.float32)
        if image.shape != (config.output_height, config.output_width):
            image = image.reshape(config.output_height, config.output_width)
        if float(image.std()) < 0.02:
            continue

        projected = _project_points(layout, h)
        if projected.shape[0] == 0:
            continue
        heatmap = np.zeros((config.output_height, config.output_width, 1), dtype=np.float32)
        kept: list[tuple[float, float]] = []
        for x, y in projected:
            ix = int(np.floor(float(x)))
            iy = int(np.floor(float(y)))
            if 0 <= ix < config.output_width and 0 <= iy < config.output_height:
                _add_gaussian(heatmap, float(x), float(y), config.keypoint_sigma)
                kept.append((float(x), float(y)))
        if not kept:
            continue

        contrast = float(rng.uniform(*config.contrast))
        image = (image - 0.5) * contrast + 0.5
        noise = float(rng.uniform(*config.noise_sigma))
        if noise > 0:
            image = image + rng.normal(0.0, noise, size=image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)[:, :, None]

        _junction_source = layout.three_way_junction_points or layout.junction_points
        _junction_arr = np.asarray([[p.x, p.y] for p in _junction_source], dtype=np.float32)
        _projected_junctions = (
            np.asarray(apply_homography(h, jnp.asarray(_junction_arr)), dtype=np.float32)
            if _junction_arr.size
            else np.zeros((0, 2), dtype=np.float32)
        )
        junction_heatmap = np.zeros((config.output_height, config.output_width, 1), dtype=np.float32)
        kept_junctions: list[tuple[float, float]] = []
        for x, y in _projected_junctions:
            ix = int(np.floor(float(x)))
            iy = int(np.floor(float(y)))
            if 0 <= ix < config.output_width and 0 <= iy < config.output_height:
                _add_gaussian(junction_heatmap, float(x), float(y), config.keypoint_sigma)
                kept_junctions.append((float(x), float(y)))
        _mask = np.zeros((config.output_height, config.output_width), dtype=np.uint8)
        if layout.cell_polygons:
            for _polygon in layout.cell_polygons:
                _polygon_arr = np.asarray([[p.x, p.y] for p in _polygon], dtype=np.float32)
                _projected_polygon = np.asarray(
                    apply_homography(h, jnp.asarray(_polygon_arr)), dtype=np.float32
                )
                cv2.fillConvexPoly(
                    _mask,
                    np.round(_projected_polygon).astype(np.int32),
                    1,
                )
        else:
            _corners = np.asarray(
                [[0.0, 0.0], [layout.width_px, 0.0], [layout.width_px, layout.height_px], [0.0, layout.height_px]],
                dtype=np.float32,
            )
            _projected_corners = np.asarray(
                apply_homography(h, jnp.asarray(_corners)), dtype=np.float32
            )
            cv2.fillConvexPoly(_mask, np.round(_projected_corners).astype(np.int32), 1)
        images.append(image)
        features.append(extract_features(image[:, :, 0], config))
        heatmaps.append(heatmap)
        masks.append(_mask[:, :, None].astype(np.float32))
        junctions.append(junction_heatmap)
        target_done += 1
        metadata.append(
            {
                "points": kept,
                "junctions": kept_junctions,
                "pose": {
                    "yaw_deg": pose.yaw_deg,
                    "pitch_deg": pose.pitch_deg,
                    "roll_deg": pose.roll_deg,
                    "distance": pose.distance,
                    "focal": pose.focal,
                },
                "blur_sigma": sigma,
            }
        )

    if not images:
        raise RuntimeError("Could not generate any heatmap samples; check target/camera ranges.")
    return HeatmapDataset(
        images=np.stack(images, axis=0).astype(np.float32),
        features=np.stack(features, axis=0).astype(np.float32),
        heatmaps=np.stack(heatmaps, axis=0).astype(np.float32),
        masks=np.stack(masks, axis=0).astype(np.float32),
        junctions=np.stack(junctions, axis=0).astype(np.float32),
        metadata=metadata,
        config=config,
    )
