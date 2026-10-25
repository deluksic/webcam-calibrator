"""Synthetic office composites for target-area (ROI) segmentation.

The target is rendered with the full browser canvas, including the same
10% blank padding as the Hex Target page, then composited over an office
photograph as a floating card with a soft drop shadow.

The ROI task is deliberately low-frequency: the mask is the general target
area (canvas, marker hull, or honeycomb cell union), not individual markers.
The naive full-resolution peak finder runs inside this mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from blob_lab.augment import (
    CameraSharpenConfig,
    SharpenParams,
    apply_camera_sharpen,
    sample_augmentation_rng,
    sample_sharpen_params,
)
from blob_lab.camera import (
    CameraRanges,
    PoseAugmentConfig,
    pose_tilt_deg,
    random_pose,
)
from blob_lab.heatmap_dataset import HeatmapDatasetConfig, _load_natural_image, _srgb_to_linear
from blob_lab.render import apply_homography, camera_homography, render_target
from blob_lab.target_layout import (
    DEFAULT_PADDING_FRACTION,
    HexTargetLayout,
    build_hex_target_layout,
    build_target_texture,
)

# The Hex Target page draws a 0.5 gray canvas with a 10% blank border and
# renders the canvas at 1:1 device pixels. Keep the composite on the same
# canvas coordinates; do not crop to the marker bounds.
BROWSER_TARGET_BACKGROUND = 0.5
BROWSER_TARGET_PADDING = DEFAULT_PADDING_FRACTION


@dataclass(frozen=True)
class RoiDatasetConfig:
    output_width: int = 640
    output_height: int = 480
    samples: int = 120
    natural_fraction: float = 0.40
    # Negative-only images (walls, art). Positives still use background_dir.
    negative_dir: str | None = None
    natural_dir: str | None = None
    background_dir: str | None = None
    # Draw pure negatives from both the wall/art pool and the office/user
    # background pool, so every photo can appear as a presence-negative.
    negative_include_backgrounds: bool = True
    target_width_fraction: tuple[float, float] = (0.15, 0.72)
    # A separate close-up branch: the target fills most of the frame, as when
    # the phone is held near the webcam.
    closeup_fraction: float = 0.30
    closeup_width_fraction: tuple[float, float] = (0.70, 0.90)
    # Targets near or past the frame edge. The mask is clipped, but the model
    # must still segment the visible part instead of behaving as if the border
    # were a guaranteed background margin.
    cropped_fraction: float = 0.30
    cropped_width_fraction: tuple[float, float] = (0.45, 0.90)
    cropped_min_visible_fraction: float = 0.08
    # Randomize the rendered canvas aspect so the model cannot key on a 4:3
    # card boundary. A phone screen is usually 16:9 or wider.
    canvas_aspects: tuple[float, ...] = (4.0 / 3.0,)
    center_margin: float = 0.04
    camera: CameraRanges = field(default_factory=CameraRanges)
    pose_augment: PoseAugmentConfig = field(default_factory=PoseAugmentConfig)
    sharpen: CameraSharpenConfig = field(default_factory=CameraSharpenConfig)
    blur_sigma: tuple[float, float] = (0.2, 0.8)
    # Out-of-focus close-ups are common on webcams, but far targets are
    # already only tens of pixels wide. Blur is scaled by target width below.
    heavy_blur_fraction: float = 0.20
    heavy_blur_sigma: tuple[float, float] = (0.6, 1.8)
    frame_blur_fraction: float = 0.15
    frame_blur_sigma: tuple[float, float] = (0.3, 1.2)
    supersample: int = 1
    shadow_strength: float = 0.35
    shadow_offset: tuple[float, float] = (0.006, 0.010)
    shadow_blur_fraction: float = 0.012
    mask_mode: str = "hull"  # canvas | hull | cells
    mask_margin: float = 0.02
    min_mask_area_fraction: float = 0.006
    min_visible_fraction: float = 0.35
    noise_sigma: tuple[float, float] = (0.0, 0.02)
    contrast: tuple[float, float] = (0.85, 1.05)
    # The browser target is a 0.5 gray canvas, but a real screen/print in an
    # office is rarely at the ideal exposure. Vary the board itself slightly.
    target_gray_range: tuple[float, float] = (0.38, 0.62)
    randomize_colors: bool = True
    seed: int = 0


@dataclass
class RoiDataset:
    images: np.ndarray  # (N, H, W, 1), float16 linear light
    masks: np.ndarray  # (N, H, W, 1), uint8 0/1 target area
    presence: np.ndarray  # (N,), 1 target present
    metadata: list[dict[str, Any]]
    config: RoiDatasetConfig

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def repeated(self, count: int) -> "RoiDataset":
        count = max(1, int(count))
        return RoiDataset(
            images=np.concatenate([self.images] * count, axis=0),
            masks=np.concatenate([self.masks] * count, axis=0),
            presence=np.concatenate([self.presence] * count, axis=0),
            metadata=[meta for _ in range(count) for meta in self.metadata],
            config=self.config,
        )

    def concatenated(self, other: "RoiDataset") -> "RoiDataset":
        return RoiDataset(
            images=np.concatenate([self.images, other.images], axis=0),
            masks=np.concatenate([self.masks, other.masks], axis=0),
            presence=np.concatenate([self.presence, other.presence], axis=0),
            metadata=[*self.metadata, *other.metadata],
            config=self.config,
        )

    def split(self, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)):
        n = len(self)
        n_train = int(n * fractions[0])
        n_val = int(n * fractions[1])
        return (
            self.subset(slice(0, n_train)),
            self.subset(slice(n_train, n_train + n_val)),
            self.subset(slice(n_train + n_val, n)),
        )

    def subset(self, selection) -> "RoiDataset":
        return RoiDataset(
            images=self.images[selection],
            masks=self.masks[selection],
            presence=self.presence[selection],
            metadata=[self.metadata[i] for i in np.arange(len(self))[selection]],
            config=self.config,
        )

    def shuffled(self, seed: int = 0) -> "RoiDataset":
        order = np.random.default_rng(seed).permutation(len(self))
        return self.subset(order)

    def flipped(self, *, horizontal: bool = True, vertical: bool = False) -> "RoiDataset":
        if not horizontal and not vertical:
            return self
        height, width = self.images.shape[1:3]
        images = self.images
        masks = self.masks
        if horizontal:
            images = np.flip(images, axis=2)
            masks = np.flip(masks, axis=2)
        if vertical:
            images = np.flip(images, axis=1)
            masks = np.flip(masks, axis=1)

        def flip_point(point):
            x, y = float(point[0]), float(point[1])
            if horizontal:
                x = (width - 1) - x
            if vertical:
                y = (height - 1) - y
            return (x, y)

        flipped_meta: list[dict[str, Any]] = []
        for meta in self.metadata:
            entry = dict(meta)
            if entry.get("points"):
                entry["points"] = [flip_point(point) for point in entry["points"]]
            if entry.get("junctions"):
                entry["junctions"] = [flip_point(point) for point in entry["junctions"]]
            if entry.get("canvas_corners"):
                entry["canvas_corners"] = [
                    list(flip_point(point)) for point in entry["canvas_corners"]
                ]
            if entry.get("H"):
                transform = np.eye(3, dtype=np.float32)
                if horizontal:
                    transform = (
                        np.asarray(
                            [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            dtype=np.float32,
                        )
                        @ transform
                    )
                if vertical:
                    transform = (
                        np.asarray(
                            [[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]],
                            dtype=np.float32,
                        )
                        @ transform
                    )
                entry["H"] = (
                    transform @ np.asarray(entry["H"], dtype=np.float32)
                ).tolist()
            flipped_meta.append(entry)
        return RoiDataset(
            images=images,
            masks=masks,
            presence=self.presence.copy(),
            metadata=flipped_meta,
            config=self.config,
        )

    def augmented(self, *, horizontal: bool = True, vertical: bool = False) -> "RoiDataset":
        copies = [self]
        if horizontal:
            copies.append(self.flipped(horizontal=True))
        if vertical:
            copies.append(self.flipped(vertical=True))
        if horizontal and vertical:
            copies.append(self.flipped(horizontal=True, vertical=True))
        if len(copies) == 1:
            return self
        return RoiDataset(
            images=np.concatenate([copy.images for copy in copies], axis=0),
            masks=np.concatenate([copy.masks for copy in copies], axis=0),
            presence=np.concatenate([copy.presence for copy in copies], axis=0),
            metadata=[meta for copy in copies for meta in copy.metadata],
            config=self.config,
        )


def _project(h: jax.Array | np.ndarray, points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(apply_homography(jnp.asarray(h), jnp.asarray(arr)), dtype=np.float32)


def _canvas_corners(layout: HexTargetLayout) -> np.ndarray:
    w, h = float(layout.width_px), float(layout.height_px)
    return np.asarray([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)


def _fit_homography_to_box(
    h: np.ndarray,
    target_size: tuple[int, int],
    output_size: tuple[int, int],
    width_fraction: float,
    center: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale/translate an existing camera homography to a chosen frame box.

    The transform is an output-space similarity, so the perspective of the
    target is preserved while its apparent size and position are controlled.
    """
    target_w, target_h = target_size
    out_w, out_h = output_size
    corners = _canvas_corners_size(target_w, target_h)
    projected = _project(h, corners)
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    bbox_w = max(1e-3, float(hi[0] - lo[0]))
    bbox_h = max(1e-3, float(hi[1] - lo[1]))
    desired_w = out_w * width_fraction
    desired_h = desired_w * (target_h / max(1.0, float(target_w)))
    scale = min(desired_w / bbox_w, desired_h / bbox_h)
    center_src = (lo + hi) * 0.5
    tx = float(center[0]) - float(center_src[0]) * scale
    ty = float(center[1]) - float(center_src[1]) * scale
    transform = np.asarray(
        [[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    fitted = transform @ np.asarray(h, dtype=np.float32)
    return fitted.astype(np.float32), _project(fitted, corners), float(scale)


def _canvas_corners_size(target_w: int, target_h: int) -> np.ndarray:
    w, h = float(target_w), float(target_h)
    return np.asarray([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)


def _fill_polygon(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.shape[0] >= 3:
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
    return mask


def _target_mask(
    layout: HexTargetLayout,
    h: np.ndarray,
    frame_shape: tuple[int, int],
    config: RoiDatasetConfig,
    target_width_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the ROI mask from the projected target geometry.

    ``hull`` is the convex hull of all projected marker corners: a compact,
    hexagon-like region around the pattern. ``canvas`` is the full browser
    canvas including the 10% blank padding. ``cells`` is the union of the
    projected honeycomb cells.
    """
    mode = config.mask_mode.lower()
    corners = _project(h, _canvas_corners(layout))
    if mode == "canvas":
        polygon = corners
    elif mode == "cells" and layout.cell_polygons:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        for cell in layout.cell_polygons:
            projected = _project(h, np.asarray([[p.x, p.y] for p in cell], dtype=np.float32))
            if projected.shape[0] >= 3:
                cv2.fillConvexPoly(mask, np.round(projected).astype(np.int32), 1)
        polygon = corners
        # Keep the cell union as the base mask; the polygon is only metadata.
        base = mask
    else:
        marker_corners: list[np.ndarray] = []
        for quad in layout.quads:
            marker_corners.append(
                _project(
                    h,
                    np.asarray(
                        [
                            [quad.left, quad.top],
                            [quad.right, quad.top],
                            [quad.right, quad.bottom],
                            [quad.left, quad.bottom],
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        points = np.concatenate(marker_corners, axis=0) if marker_corners else corners
        polygon = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)

    if mode != "cells":
        base = _fill_polygon(frame_shape, polygon)

    margin_px = max(0, int(round(config.mask_margin * max(1.0, target_width_px))))
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        base = cv2.dilate(base, kernel, iterations=1)
    return (base > 0).astype(np.float32), polygon


def _composite_target(
    background: np.ndarray,
    target: np.ndarray,
    alpha: np.ndarray,
    *,
    target_width_px: float,
    sample_index: int,
    config: RoiDatasetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, SharpenParams | None]:
    """Composite a premultiplied target over a background with a soft shadow."""
    h, w = background.shape
    if config.shadow_strength > 0.0:
        dx = float(config.shadow_offset[0]) * w
        dy = float(config.shadow_offset[1]) * h
        shifted = cv2.warpAffine(
            alpha,
            np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
        sigma = max(1.0, float(config.shadow_blur_fraction) * max(1.0, target_width_px))
        shadow = cv2.GaussianBlur(shifted, (0, 0), sigma) * float(config.shadow_strength)
        background = background * (1.0 - np.clip(shadow, 0.0, 1.0))
    # render_target writes premultiplied values (outside the canvas is zero),
    # so use the "over" operator directly instead of multiplying by alpha again.
    image = target + background * (1.0 - alpha)
    contrast = float(rng.uniform(*config.contrast))
    image = (image - 0.5) * contrast + 0.5
    noise = float(rng.uniform(*config.noise_sigma))
    if noise > 0.0:
        image = image + rng.normal(0.0, noise, size=image.shape).astype(np.float32)
    frame_blur_sigma = 0.0
    if rng.random() < config.frame_blur_fraction:
        frame_blur_sigma = float(rng.uniform(*config.frame_blur_sigma)) * min(
            1.0, max(0.25, target_width_px / 320.0)
        )
        image = cv2.GaussianBlur(
            image, (0, 0), frame_blur_sigma, borderType=cv2.BORDER_REPLICATE
        )
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    sharpen_params = sample_sharpen_params(
        config.sharpen,
        sample_augmentation_rng(config.seed, sample_index),
    )
    image = apply_camera_sharpen(image, sharpen_params)
    return image, frame_blur_sigma, sharpen_params


def _project_visible_points(
    layout: HexTargetLayout,
    h: np.ndarray,
    width: int,
    height: int,
    points_attr: str,
) -> list[tuple[float, float]]:
    source = getattr(layout, points_attr)
    if not source:
        return []
    projected = _project(h, np.asarray([[p.x, p.y] for p in source], dtype=np.float32))
    kept: list[tuple[float, float]] = []
    for x, y in projected:
        if 0.0 <= float(x) < width and 0.0 <= float(y) < height:
            kept.append((float(x), float(y)))
    return kept


def _texture_for_sample(
    layout: HexTargetLayout,
    rng: np.random.Generator,
    config: RoiDatasetConfig,
    static_texture: jax.Array,
    board_gray: float,
) -> jax.Array:
    if not config.randomize_colors:
        if abs(board_gray - BROWSER_TARGET_BACKGROUND) < 1e-6:
            return static_texture
        return jnp.asarray(build_target_texture(layout, board_gray), dtype=jnp.float32)
    colors = tuple(int(value) for value in rng.integers(0, 2, size=len(layout.quads)))
    sample_layout = replace(
        layout,
        quads=tuple(replace(quad, color=color) for quad, color in zip(layout.quads, colors)),
        colors=colors,
    )
    return jnp.asarray(
        build_target_texture(sample_layout, board_gray), dtype=jnp.float32
    )


def _load_background(rng: np.random.Generator, config: RoiDatasetConfig) -> np.ndarray:
    bg_config = HeatmapDatasetConfig(
        output_width=config.output_width,
        output_height=config.output_height,
        natural_dir=config.background_dir or config.natural_dir,
    )
    return _load_natural_image(rng, bg_config).astype(np.float32)


def _load_negative(rng: np.random.Generator, config: RoiDatasetConfig) -> np.ndarray:
    directories = [
        directory
        for directory in (
            config.negative_dir,
            config.background_dir or config.natural_dir,
        )
        if directory
    ]
    if not config.negative_include_backgrounds:
        directories = directories[:1]
    if not directories:
        directories = [config.natural_dir or config.background_dir]
    directory = directories[int(rng.integers(0, len(directories)))]
    bg_config = HeatmapDatasetConfig(
        output_width=config.output_width,
        output_height=config.output_height,
        natural_dir=directory,
    )
    return _load_natural_image(rng, bg_config).astype(np.float32)


def load_real_positives(
    directory: str | Path,
    config: RoiDatasetConfig,
) -> RoiDataset | None:
    """Load real camera frames with hand-authored masks.

    Layout: ``directory/<stem>.jpg`` and ``directory/<stem>_mask.png``.
    The mask is thresholded at 127 and cover-resized with nearest-neighbor.
    """
    root = Path(directory)
    if not root.is_dir():
        return None
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for image_path in sorted(root.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        if image_path.stem.endswith("_mask"):
            continue
        mask_path = root / f"{image_path.stem}_mask.png"
        if not mask_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        scale = max(
            config.output_width / image.shape[1],
            config.output_height / image.shape[0],
        )
        width = max(config.output_width, int(round(image.shape[1] * scale)))
        height = max(config.output_height, int(round(image.shape[0] * scale)))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        left = max(0, (width - config.output_width) // 2)
        top = max(0, (height - config.output_height) // 2)
        image = image[
            top : top + config.output_height, left : left + config.output_width
        ]
        mask = mask[
            top : top + config.output_height, left : left + config.output_width
        ]
        linear = _srgb_to_linear(image.astype(np.float32) / 255.0)
        sharpen_params = sample_sharpen_params(
            config.sharpen,
            sample_augmentation_rng(config.seed, len(images)),
            probability=config.sharpen.real_positive_probability,
        )
        linear = apply_camera_sharpen(linear, sharpen_params)
        images.append(linear[:, :, None])
        masks.append((mask >= 127).astype(np.uint8)[:, :, None])
        metadata.append(
            {
                "natural": False,
                "presence": 1,
                "real_positive": True,
                "source": str(image_path),
                "pose_branch": "real",
                "tilt_deg": None,
                "sharpen": None
                if sharpen_params is None
                else sharpen_params.as_dict(),
            }
        )
    if not images:
        return None
    return RoiDataset(
        images=np.stack(images, axis=0).astype(np.float16),
        masks=np.stack(masks, axis=0).astype(np.uint8),
        presence=np.ones((len(images),), dtype=np.float32),
        metadata=metadata,
        config=config,
    )


def _empty_dataset(config: RoiDatasetConfig) -> RoiDataset:
    return RoiDataset(
        images=np.zeros(
            (0, config.output_height, config.output_width, 1), dtype=np.float16
        ),
        masks=np.zeros(
            (0, config.output_height, config.output_width, 1), dtype=np.uint8
        ),
        presence=np.zeros((0,), dtype=np.float32),
        metadata=[],
        config=config,
    )


def generate_roi_dataset(
    config: RoiDatasetConfig,
    layout: HexTargetLayout,
    *,
    count: int | None = None,
) -> RoiDataset:
    """Generate office composites with exact target-area masks.

    Positive scenes render the full browser target canvas (including the 10%
    padding) through a random perspective and composite it over an office
    photograph with a soft drop shadow. Negative scenes are plain office
    photographs with an all-zero mask.
    """
    rng = np.random.default_rng(config.seed)
    count = config.samples if count is None else int(count)
    if count <= 0:
        return _empty_dataset(config)

    aspect_layouts: list[HexTargetLayout] = []
    for aspect in config.canvas_aspects:
        if abs(aspect - layout.width_px / max(1, layout.height_px)) < 1e-3:
            aspect_layouts.append(layout)
            continue
        width = 1280
        height = max(2, int(round(width / max(0.1, aspect))))
        aspect_layouts.append(
            build_hex_target_layout(
                width_px=width,
                height_px=height,
                columns=layout.columns,
                points_per_edge=layout.points_per_edge,
                marker_scale=layout.marker_scale,
                color_seed=layout.color_seed,
                padding_fraction=BROWSER_TARGET_PADDING,
            )
        )
    if not aspect_layouts:
        aspect_layouts = [layout]
    static_textures = [
        jnp.asarray(build_target_texture(item, BROWSER_TARGET_BACKGROUND), dtype=jnp.float32)
        for item in aspect_layouts
    ]
    ones_textures = [jnp.ones_like(item) for item in static_textures]

    @jax.jit
    def _render_pair(
        h: jax.Array,
        target_texture: jax.Array,
        alpha_texture: jax.Array,
        sigma: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        target = render_target(
            h,
            target_texture,
            output_size=(config.output_width, config.output_height),
            supersample=config.supersample,
            psf_sigma=sigma,
            background=0.0,
        )
        alpha = render_target(
            h,
            alpha_texture,
            output_size=(config.output_width, config.output_height),
            supersample=config.supersample,
            psf_sigma=sigma,
            background=0.0,
        )
        return target, alpha

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    presence: list[float] = []
    metadata: list[dict[str, Any]] = []

    natural_target = int(round(count * max(0.0, min(1.0, config.natural_fraction))))
    natural_done = 0
    target_target = count - natural_target
    target_done = 0
    attempts = 0
    max_attempts = max(64, count * 80)

    while len(images) < count and attempts < max_attempts:
        attempts += 1
        need_natural = natural_done < natural_target and (
            target_done >= target_target or rng.random() < config.natural_fraction
        )
        if need_natural:
            background = _load_negative(rng, config)
            sharpen_params = sample_sharpen_params(
                config.sharpen,
                sample_augmentation_rng(config.seed, len(images)),
            )
            background = apply_camera_sharpen(background, sharpen_params)
            images.append(background[:, :, None])
            masks.append(
                np.zeros(
                    (config.output_height, config.output_width, 1), dtype=np.float32
                )
            )
            presence.append(0.0)
            metadata.append(
                {
                    "natural": True,
                    "presence": 0,
                    "points": [],
                    "junctions": [],
                    "canvas_corners": [],
                    "mask_area": 0,
                    "pose_branch": "negative",
                    "tilt_deg": None,
                    "sharpen": None
                    if sharpen_params is None
                    else sharpen_params.as_dict(),
                }
            )
            natural_done += 1
            continue

        aspect_index = int(rng.integers(0, len(aspect_layouts)))
        sample_layout = aspect_layouts[aspect_index]
        pose = random_pose(rng, config.camera, config.pose_augment)
        h = camera_homography(
            yaw_deg=pose.yaw_deg,
            pitch_deg=pose.pitch_deg,
            roll_deg=pose.roll_deg,
            distance=pose.distance,
            focal=pose.focal,
            tx=pose.tx,
            ty=pose.ty,
            target_size=(sample_layout.width_px, sample_layout.height_px),
            output_size=(config.output_width, config.output_height),
        )
        if not np.isfinite(float(jnp.linalg.det(h))) or abs(float(jnp.linalg.det(h))) < 1e-8:
            continue

        cropped = rng.random() < config.cropped_fraction
        closeup = rng.random() < config.closeup_fraction
        if cropped:
            width_range = config.cropped_width_fraction
        elif closeup:
            width_range = config.closeup_width_fraction
        else:
            width_range = config.target_width_fraction
        width_fraction = float(rng.uniform(*width_range))
        height_fraction = width_fraction * (
            sample_layout.height_px / max(1.0, float(sample_layout.width_px))
        )
        margin = float(config.center_margin)
        if cropped:
            # Push one side of the target past the frame, then jitter the other
            # axis. The mask/visibility checks below reject mostly-empty cases.
            side = int(rng.integers(0, 4))
            cx = 0.5 * config.output_width
            cy = 0.5 * config.output_height
            if side == 0:
                cx = float(rng.uniform(-0.10, 0.25) * config.output_width)
                cy += float(rng.uniform(-0.20, 0.20) * config.output_height)
            elif side == 1:
                cx = float(rng.uniform(0.75, 1.10) * config.output_width)
                cy += float(rng.uniform(-0.20, 0.20) * config.output_height)
            elif side == 2:
                cy = float(rng.uniform(-0.10, 0.25) * config.output_height)
                cx += float(rng.uniform(-0.20, 0.20) * config.output_width)
            else:
                cy = float(rng.uniform(0.75, 1.10) * config.output_height)
                cx += float(rng.uniform(-0.20, 0.20) * config.output_width)
            center = (cx, cy)
        else:
            cx_lo = 0.5 * width_fraction + margin
            cx_hi = 1.0 - cx_lo
            cy_lo = 0.5 * height_fraction + margin
            cy_hi = 1.0 - cy_lo
            if cx_hi <= cx_lo or cy_hi <= cy_lo:
                continue
            center = (
                float(rng.uniform(cx_lo, cx_hi) * config.output_width),
                float(rng.uniform(cy_lo, cy_hi) * config.output_height),
            )
        fitted_h, canvas_projected, _scale = _fit_homography_to_box(
            np.asarray(h, dtype=np.float32),
            (sample_layout.width_px, sample_layout.height_px),
            (config.output_width, config.output_height),
            width_fraction,
            center,
        )
        if not np.isfinite(fitted_h).all() or abs(float(np.linalg.det(fitted_h))) < 1e-10:
            continue

        canvas_lo = canvas_projected.min(axis=0)
        canvas_hi = canvas_projected.max(axis=0)
        target_width_px = float(max(1.0, canvas_hi[0] - canvas_lo[0]))
        target_height_px = float(max(1.0, canvas_hi[1] - canvas_lo[1]))

        heavy_blur = rng.random() < config.heavy_blur_fraction
        # Keep the PSF proportional to the target size: a 3 px blur destroys a
        # 100 px-wide far target but is mild on a 500 px close-up.
        blur_scale = min(1.0, max(0.25, target_width_px / 320.0))
        sigma = float(
            rng.uniform(*(config.heavy_blur_sigma if heavy_blur else config.blur_sigma))
        ) * blur_scale
        board_gray = float(rng.uniform(*config.target_gray_range))
        sample_texture = _texture_for_sample(
            sample_layout,
            rng,
            config,
            static_textures[aspect_index],
            board_gray,
        )
        target, alpha = _render_pair(
            jnp.asarray(fitted_h, dtype=jnp.float32),
            sample_texture,
            ones_textures[aspect_index],
            jnp.asarray(sigma, dtype=jnp.float32),
        )
        target_np = np.asarray(target, dtype=np.float32)
        alpha_np = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
        if float(alpha_np.sum()) <= 0.0:
            continue
        background = _load_background(rng, config)
        image, frame_blur_sigma, sharpen_params = _composite_target(
            background,
            target_np,
            alpha_np,
            target_width_px=target_width_px,
            sample_index=len(images),
            config=config,
            rng=rng,
        )
        mask, _polygon = _target_mask(
            sample_layout,
            fitted_h,
            (config.output_height, config.output_width),
            config,
            target_width_px,
        )
        mask_area_fraction = float(mask.mean())
        if mask_area_fraction < config.min_mask_area_fraction:
            continue

        points = _project_visible_points(
            sample_layout, fitted_h, config.output_width, config.output_height, "square_centers"
        )
        junctions = _project_visible_points(
            sample_layout,
            fitted_h,
            config.output_width,
            config.output_height,
            "three_way_junction_points",
        )
        if not points:
            continue
        # The mask is deliberately inclusive; a point on the boundary still
        # counts for containment, so check the dilated mask.
        contained = 0
        for x, y in points:
            ix = int(np.clip(round(x), 0, config.output_width - 1))
            iy = int(np.clip(round(y), 0, config.output_height - 1))
            contained += int(mask[iy, ix] > 0.5)
        contained_fraction = contained / max(1, len(points))
        visible_fraction = len(points) / max(1, len(sample_layout.square_centers))
        min_visible = (
            config.cropped_min_visible_fraction
            if cropped
            else config.min_visible_fraction
        )
        if visible_fraction < min_visible:
            continue

        images.append(image[:, :, None])
        masks.append(mask[:, :, None])
        presence.append(1.0)
        metadata.append(
            {
                "natural": False,
                "presence": 1,
                "pose": {
                    "yaw_deg": pose.yaw_deg,
                    "pitch_deg": pose.pitch_deg,
                    "roll_deg": pose.roll_deg,
                    "distance": pose.distance,
                    "focal": pose.focal,
                },
                "pose_branch": pose.pose_branch,
                "tilt_deg": float(pose_tilt_deg(pose)),
                "H": fitted_h.tolist(),
                "points": points,
                "junctions": junctions,
                "canvas_corners": canvas_projected.tolist(),
                "target_width_px": target_width_px,
                "target_height_px": target_height_px,
                "mask_area": float(mask.sum()),
                "mask_area_fraction": mask_area_fraction,
                "contained_fraction": contained_fraction,
                "visible_fraction": visible_fraction,
                "cropped": cropped,
                "blur_sigma": sigma,
                "blur_scale": blur_scale,
                "heavy_blur": heavy_blur,
                "frame_blur_sigma": frame_blur_sigma,
                "closeup": closeup,
                "canvas_aspect": sample_layout.width_px / max(1, sample_layout.height_px),
                "board_gray": board_gray,
                "mask_mode": config.mask_mode,
                "sharpen": None
                if sharpen_params is None
                else sharpen_params.as_dict(),
            }
        )
        target_done += 1

    if not images:
        raise RuntimeError(
            "Could not generate any ROI samples; check target/camera/background settings."
        )
    return RoiDataset(
        # Half-precision images and byte masks keep the larger training sets
        # in memory; training converts to fp32 after normalization.
        images=np.stack(images, axis=0).astype(np.float16),
        masks=np.stack(masks, axis=0).astype(np.uint8),
        presence=np.asarray(presence, dtype=np.float32),
        metadata=metadata,
        config=config,
    )
