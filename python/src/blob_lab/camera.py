"""Perspective camera helpers for the differentiable target renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, degrees, radians

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    distance: float
    focal: float
    tx: float = 0.0
    ty: float = 0.0
    # Which sampler produced the pose: "uniform" or "oblique". Stored in
    # dataset metadata so the next training/evaluation pass can stratify.
    pose_branch: str = "uniform"


@dataclass(frozen=True)
class CameraRanges:
    yaw_deg: tuple[float, float] = (-45.0, 45.0)
    pitch_deg: tuple[float, float] = (-35.0, 35.0)
    roll_deg: tuple[float, float] = (-25.0, 25.0)
    distance: tuple[float, float] = (2.8, 3.8)
    focal: tuple[float, float] = (2.0, 2.7)
    tx: tuple[float, float] = (-0.08, 0.08)
    ty: tuple[float, float] = (-0.08, 0.08)


@dataclass(frozen=True)
class PoseAugmentConfig:
    """Mixture of the usual uniform poses and an oblique-heavy branch.

    High tilt is where the current detector fails, so training/evaluation data
    intentionally over-samples it. Set ``oblique_fraction=0`` to recover the
    old uniform behaviour.
    """

    oblique_fraction: float = 0.5
    oblique_yaw_deg: tuple[float, float] = (35.0, 65.0)
    oblique_pitch_deg: tuple[float, float] = (20.0, 45.0)
    min_oblique_tilt_deg: float = 40.0
    max_attempts: int = 8


# Case-level tilt is roll-independent: the target normal in camera coordinates
# is (sin(yaw)cos(pitch), -sin(pitch), cos(yaw)cos(pitch)), so its angle to the
# optical axis is acos(|cos(yaw)cos(pitch)|).
def pose_tilt_deg(pose: CameraPose) -> float:
    product = abs(cos(radians(pose.yaw_deg)) * cos(radians(pose.pitch_deg)))
    return degrees(acos(min(1.0, max(0.0, product))))


def random_pose(
    rng: np.random.Generator,
    ranges: CameraRanges,
    augment: PoseAugmentConfig | None = None,
) -> CameraPose:
    def uniform(pair: tuple[float, float]) -> float:
        low, high = pair
        return float(rng.uniform(low, high))

    def sample_pose(yaw: float, pitch: float, branch: str) -> CameraPose:
        return CameraPose(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=uniform(ranges.roll_deg),
            distance=uniform(ranges.distance),
            focal=uniform(ranges.focal),
            tx=uniform(ranges.tx),
            ty=uniform(ranges.ty),
            pose_branch=branch,
        )

    if (
        augment is not None
        and augment.oblique_fraction > 0.0
        and rng.random() < min(1.0, augment.oblique_fraction)
    ):
        for _ in range(max(1, int(augment.max_attempts))):
            yaw = uniform(augment.oblique_yaw_deg) * (1.0 if rng.random() < 0.5 else -1.0)
            pitch = uniform(augment.oblique_pitch_deg) * (
                1.0 if rng.random() < 0.5 else -1.0
            )
            pose = sample_pose(yaw, pitch, "oblique")
            if pose_tilt_deg(pose) >= augment.min_oblique_tilt_deg:
                return pose
        # If the requested ranges cannot produce enough tilt, fall through to
        # the usual uniform branch rather than returning a degenerate pose.

    return sample_pose(
        uniform(ranges.yaw_deg),
        uniform(ranges.pitch_deg),
        "uniform",
    )


def homography_jacobian(
    matrix: np.ndarray | list[list[float]],
    target_x: float,
    target_y: float,
) -> np.ndarray:
    """2x2 Jacobian of a homography at a target-space point.

    The target point is mapped to image coordinates ``(u, v, w)``; the local
    affine approximation is ``J / w`` with the standard perspective correction.
    """
    h = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    x = float(target_x)
    y = float(target_y)
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(w) < 1e-12:
        raise ValueError("homography is singular at the requested target point")
    u = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
    v = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
    return np.asarray(
        [
            [(h[0, 0] - u * h[2, 0]) / w, (h[0, 1] - u * h[2, 1]) / w],
            [(h[1, 0] - v * h[2, 0]) / w, (h[1, 1] - v * h[2, 1]) / w],
        ],
        dtype=np.float64,
    )


def local_foreshortening(
    matrix: np.ndarray | list[list[float]],
    target_point: tuple[float, float] | np.ndarray,
) -> tuple[float, float]:
    """Return ``(min/max singular value ratio, equivalent tilt in degrees)``.

    A pure rotation/scale has ratio 1 and tilt 0; an affine compression by
    ``cos(tilt)`` has ratio ``cos(tilt)``. This is the per-point analogue of the
    case-level pose tilt and is what the evaluation bins use.
    """
    jacobian = homography_jacobian(matrix, float(target_point[0]), float(target_point[1]))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    largest = max(float(singular[0]), 1e-12)
    ratio = min(1.0, max(0.0, float(singular[-1]) / largest))
    return ratio, degrees(acos(ratio))


TILT_BINS: tuple[tuple[str, float, float], ...] = (
    ("frontal", 0.0, 25.0),
    ("moderate", 25.0, 40.0),
    ("oblique", 40.0, 55.0),
    ("steep", 55.0, 180.0),
)


def tilt_bin(tilt_deg: float | None) -> str:
    if tilt_deg is None:
        return "unknown"
    value = float(tilt_deg)
    for name, lower, upper in TILT_BINS:
        if lower <= value < upper:
            return name
    return "steep"
