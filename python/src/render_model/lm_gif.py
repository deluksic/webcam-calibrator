"""GIF frames from LM optimization traces (render per step)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_model.homography import corners_from_corner_shifts, homography_from_corners
from render_model.optimizer import OptimizationParamTrace
from render_model.pipeline import SUPERSAMPLE, RenderModelParams, render_with_model
from render_model.tag_data import TAG_CANONICAL_CORNERS

_CANONICAL = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)


def camera_from_physical_row(row: jnp.ndarray) -> RenderModelParams:
    """Build ``RenderModelParams`` from ``decode_joint_params_per_step`` camera columns."""
    return RenderModelParams(
        psf_sigma=row[8],
        sharpen_amount=row[9],
        sharpen_sigma=row[10],
        gamma=row[11],
        black_level=row[12],
        white_level=row[13],
        light_grad_u=row[14],
        light_grad_v=row[15],
    )


def homography_from_physical_row(
    row: jnp.ndarray,
    ref_corners: jnp.ndarray,
) -> jnp.ndarray:
    """3×3 H from corner shifts (first 8 entries) relative to ``ref_corners``."""
    corners_px = corners_from_corner_shifts(ref_corners, row[:8])
    return homography_from_corners(_CANONICAL, corners_px)


def render_frame_at_step(
    row: jnp.ndarray,
    ref_corners: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    supersample: int = SUPERSAMPLE,
) -> jnp.ndarray:
    """Single display image ``(H, W)`` for one physical-parameter row."""
    H = homography_from_physical_row(row, ref_corners)
    params = camera_from_physical_row(row)
    return render_with_model(
        H,
        tag_pattern,
        height,
        width,
        params,
        supersample=supersample,
    )


def renders_from_physical_params(
    params_physical: jnp.ndarray,
    ref_corners: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    step_stride: int = 1,
    supersample: int = SUPERSAMPLE,
) -> np.ndarray:
    """Stack of renders ``(n_frames, H, W)`` in float32 ``[0, 1]``."""
    rows = jnp.asarray(params_physical)[::step_stride]
    render_step = jax.jit(
        lambda row: render_frame_at_step(
            row,
            ref_corners,
            tag_pattern,
            height,
            width,
            supersample=supersample,
        )
    )
    frames = jax.vmap(render_step)(rows)
    return np.asarray(frames, dtype=np.float32)


def renders_from_lm_trace(
    trace: OptimizationParamTrace,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    step_stride: int = 1,
    supersample: int = SUPERSAMPLE,
) -> np.ndarray:
    """Renders for each LM step in ``trace.params_physical_per_step``."""
    return renders_from_physical_params(
        trace.params_physical_per_step,
        trace.ref_corners,
        tag_pattern,
        height,
        width,
        step_stride=step_stride,
        supersample=supersample,
    )


def _to_uint8_gray(image: np.ndarray, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    arr = np.clip(np.asarray(image, dtype=np.float32), vmin, vmax)
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    return (arr * 255.0).astype(np.uint8)


def _scale_nearest(gray: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return gray
    return np.repeat(np.repeat(gray, factor, axis=0), factor, axis=1)


def _annotate_frame(
    gray: np.ndarray,
    *,
    step: int,
    loss: float | None,
    label: str,
) -> Image.Image:
    img = Image.fromarray(gray, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    lines = [label, f"step {step}"]
    if loss is not None:
        lines.append(f"MSE {loss:.5f}")
    text = "\n".join(lines)
    margin = 4
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle(
        (margin, margin, margin + tw + 6, margin + th + 6),
        fill=(0, 0, 0),
    )
    draw.text((margin + 3, margin + 3), text, fill=(255, 255, 255), font=font)
    return img


def _compose_row(
    panels: Sequence[np.ndarray],
    *,
    gap: int = 2,
) -> np.ndarray:
    """Horizontal concat of equally tall uint8 grayscale panels."""
    arrays = [np.asarray(p, dtype=np.uint8) for p in panels]
    h = max(a.shape[0] for a in arrays)
    padded = []
    for arr in arrays:
        if arr.shape[0] < h:
            pad = np.zeros((h - arr.shape[0], arr.shape[1]), dtype=np.uint8)
            arr = np.concatenate([arr, pad], axis=0)
        padded.append(arr)
    if len(padded) == 1:
        return padded[0]
    sep = np.full((h, gap), 32, dtype=np.uint8)
    out = padded[0]
    for arr in padded[1:]:
        out = np.concatenate([out, sep, arr], axis=1)
    return out


def write_optimization_gif(
    path: str | Path,
    params_physical: jnp.ndarray,
    ref_corners: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    target: jnp.ndarray | None = None,
    losses: Sequence[float] | None = None,
    step_stride: int = 1,
    duration_ms: int = 120,
    scale: int = 4,
    show_target: bool = True,
    supersample: int = SUPERSAMPLE,
) -> Path:
    """Write a GIF of ``render_with_model`` at each LM step (optional target column)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    renders = renders_from_physical_params(
        params_physical,
        ref_corners,
        tag_pattern,
        height,
        width,
        step_stride=step_stride,
        supersample=supersample,
    )
    target_u8 = None
    if show_target and target is not None:
        target_u8 = _scale_nearest(_to_uint8_gray(np.asarray(target)), scale)

    pil_frames: list[Image.Image] = []
    stride = max(1, int(step_stride))
    for i, frame in enumerate(renders):
        step = i * stride
        loss = None
        if losses is not None and step < len(losses):
            loss = float(losses[step])
        render_u8 = _scale_nearest(_to_uint8_gray(frame), scale)
        panels = []
        if target_u8 is not None:
            panels.append(target_u8)
        panels.append(render_u8)
        row = _compose_row(panels)
        pil_frames.append(
            _annotate_frame(row, step=step, loss=loss, label="render")
        )

    if not pil_frames:
        raise ValueError("no frames to write")

    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out_path


def write_lm_trace_gif(
    path: str | Path,
    trace: OptimizationParamTrace,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    *,
    target: jnp.ndarray | None = None,
    step_stride: int = 1,
    duration_ms: int = 120,
    scale: int = 4,
    show_target: bool = True,
    supersample: int = SUPERSAMPLE,
) -> Path:
    """``write_optimization_gif`` using an :class:`OptimizationParamTrace`."""
    return write_optimization_gif(
        path,
        trace.params_physical_per_step,
        trace.ref_corners,
        tag_pattern,
        height,
        width,
        target=target,
        losses=trace.losses,
        step_stride=step_stride,
        duration_ms=duration_ms,
        scale=scale,
        show_target=show_target,
        supersample=supersample,
    )
