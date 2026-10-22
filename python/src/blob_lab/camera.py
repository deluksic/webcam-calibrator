"""Perspective camera helpers for the differentiable target renderer."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class CameraRanges:
    yaw_deg: tuple[float, float] = (-45.0, 45.0)
    pitch_deg: tuple[float, float] = (-35.0, 35.0)
    roll_deg: tuple[float, float] = (-25.0, 25.0)
    distance: tuple[float, float] = (2.8, 3.8)
    focal: tuple[float, float] = (2.0, 2.7)
    tx: tuple[float, float] = (-0.08, 0.08)
    ty: tuple[float, float] = (-0.08, 0.08)


def random_pose(rng: np.random.Generator, ranges: CameraRanges) -> CameraPose:
    def uniform(pair: tuple[float, float]) -> float:
        low, high = pair
        return float(rng.uniform(low, high))

    return CameraPose(
        yaw_deg=uniform(ranges.yaw_deg),
        pitch_deg=uniform(ranges.pitch_deg),
        roll_deg=uniform(ranges.roll_deg),
        distance=uniform(ranges.distance),
        focal=uniform(ranges.focal),
        tx=uniform(ranges.tx),
        ty=uniform(ranges.ty),
    )
