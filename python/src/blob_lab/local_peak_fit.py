"""Local extrema + adaptive cubic-surface fitting for hex-target markers.

This is the experimental replacement for the histogram-mask -> CCL -> centroid
chain.  It deliberately does not use a segmentation mask: markers are the local
maxima (+) of the smoothed luma for white markers and the local maxima of the
negated luma for black markers.  Around each extremum a bivariate cubic surface
is fitted over a support region that stops before the gray background or a
neighbouring marker, and the fitted stationary point is used as the subpixel
center.

The module also exposes the fitted-shape descriptors used by the offline
shape-cluster analysis.  It is pure NumPy/OpenCV and has no dataset or plotting
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import cv2
import numpy as np
from scipy.spatial import cKDTree

# Coordinate convention:
#   * image arrays are indexed by integer pixel indices (x, y);
#   * a pixel index (x, y) is centered at continuous image (x + 0.5, y + 0.5);
#   * all public LocalPeak.x / LocalPeak.y values are continuous image coords.
# The fitting math is done in pixel-index coordinates because the sampled
# response is defined there.

BORDER_MARGIN = 8
PROMINENCE_WINDOW = 11
RING_ANGLES = 24
CUBIC_TERMS = 10


@dataclass(frozen=True)
class LocalPeakCandidate:
    """A local extremum before subpixel fitting (index coordinates)."""

    x: int
    y: int
    polarity: int  # 0 = black/dark, 1 = white/bright
    peak: float  # signed response value: +luma for white, -luma for black
    valley: float  # local background in the signed response
    prominence: float

    @property
    def continuous_x(self) -> float:
        return float(self.x) + 0.5

    @property
    def continuous_y(self) -> float:
        return float(self.y) + 0.5


@dataclass(frozen=True)
class CurvatureData:
    """Precomputed Hessian sign masks and the Hessian components."""

    white_concave: np.ndarray
    black_concave: np.ndarray
    hxx: np.ndarray
    hxy: np.ndarray
    hyy: np.ndarray


@dataclass(frozen=True)
class SupportRegion:
    """Selected support pixels and their normalized coordinate basis."""

    offsets: np.ndarray  # (N, 2) integer (dx, dy)
    semi_axes: tuple[float, float]
    basis: np.ndarray  # (2, 2), columns are the ellipse axes
    kind: str
    used_curvature_mask: bool

    @property
    def radius(self) -> float:
        return float(max(self.semi_axes))

    @property
    def pixel_count(self) -> int:
        return int(self.offsets.shape[0])


@dataclass(frozen=True)
class LocalPeak:
    """A fitted (or fallback) subpixel peak."""

    x: float
    y: float
    polarity: int
    candidate_x: float
    candidate_y: float
    peak: float
    prominence: float
    support_kind: str
    support_radius: float
    support_axes: tuple[float, float]
    support_pixels: int
    used_curvature_mask: bool
    fit_ok: bool
    fallback: bool
    eigenvalues: tuple[float, float]
    anisotropy: float
    orientation_deg: float
    residual_rms: float
    iterations: int
    offset_x: float
    offset_y: float

    def descriptor_dict(self) -> dict[str, object]:
        return {
            "polarity": self.polarity,
            "candidate_x": self.candidate_x,
            "candidate_y": self.candidate_y,
            "fit_x": self.x,
            "fit_y": self.y,
            "peak": self.peak,
            "prominence": self.prominence,
            "support_kind": self.support_kind,
            "support_radius": self.support_radius,
            "support_axis_1": self.support_axes[0],
            "support_axis_2": self.support_axes[1],
            "support_pixels": self.support_pixels,
            "used_curvature_mask": self.used_curvature_mask,
            "fit_ok": self.fit_ok,
            "fallback": self.fallback,
            "eigenvalue_1": self.eigenvalues[0],
            "eigenvalue_2": self.eigenvalues[1],
            "anisotropy": self.anisotropy,
            "orientation_deg": self.orientation_deg,
            "residual_rms": self.residual_rms,
            "iterations": self.iterations,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
        }


def smooth_luma(luma: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    image = np.asarray(luma, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    if sigma <= 0.0:
        return image.copy()
    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )


def _find_maxima(
    response: np.ndarray,
    *,
    min_prominence: float,
    window: int,
    border: int,
) -> list[tuple[int, int, float, float, float]]:
    """One representative per local-maximum plateau, with peak-to-valley."""
    height, width = response.shape
    dilated = cv2.dilate(response, np.ones((3, 3), dtype=np.uint8))
    mask = response >= dilated - 1e-7
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []

    # Collapse flat plateaus to one representative without a full CCL: a pixel
    # is rejected if an earlier raster-order neighbour has exactly the same
    # value and is also a local maximum.
    keep = np.ones(xs.size, dtype=bool)
    for index in range(xs.size):
        x = int(xs[index])
        y = int(ys[index])
        value = response[y, x]
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and mask[ny, nx]
                and abs(float(response[ny, nx]) - float(value)) <= 1e-7
            ):
                keep[index] = False
                break
    xs = xs[keep]
    ys = ys[keep]
    if xs.size == 0:
        return []

    radius = max(1, int(window) // 2)
    candidates: list[tuple[int, int, float, float, float]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        peak = float(response[y, x])
        valley = float(response[y0:y1, x0:x1].min())
        prominence = peak - valley
        if prominence >= min_prominence:
            candidates.append((x, y, peak, valley, prominence))
    return candidates


def detect_local_peaks(
    smoothed: np.ndarray,
    *,
    min_prominence: float = 0.02,
    prominence_window: int = PROMINENCE_WINDOW,
    border: int = BORDER_MARGIN,
    max_candidates: int = 20000,
) -> list[LocalPeakCandidate]:
    """Find white maxima and black minima on the smoothed luma."""
    image = np.asarray(smoothed, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]

    white = _find_maxima(
        image,
        min_prominence=min_prominence,
        window=prominence_window,
        border=border,
    )
    black = _find_maxima(
        -image,
        min_prominence=min_prominence,
        window=prominence_window,
        border=border,
    )

    merged: dict[tuple[int, int], tuple[int, int, int, float, float, float]] = {}
    for polarity, entries in ((1, white), (0, black)):
        for x, y, peak, valley, prominence in entries:
            key = (x, y)
            existing = merged.get(key)
            if existing is None or prominence > existing[5]:
                merged[key] = (x, y, polarity, float(peak), float(valley), float(prominence))

    ordered = sorted(
        merged.values(),
        key=lambda item: (-item[5], item[2], item[1], item[0]),
    )
    if len(ordered) > max_candidates:
        ordered = ordered[:max_candidates]
    return [
        LocalPeakCandidate(
            x=int(x),
            y=int(y),
            polarity=int(polarity),
            peak=float(peak),
            valley=float(valley),
            prominence=float(prominence),
        )
        for x, y, polarity, peak, valley, prominence in ordered
    ]



def refresh_candidate(
    response: np.ndarray,
    candidate: LocalPeakCandidate,
    *,
    window: int = 11,
) -> LocalPeakCandidate:
    """Recompute peak/valley/prominence on the response used for fitting.

    Detection is intentionally done on a more smoothed image than the fit (a
    small-scale prominence gate must reject flat-top sensor noise), so the
    candidate's response values have to be refreshed in the fit image before
    support selection.
    """
    image = np.asarray(response, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    signed = image if candidate.polarity == 1 else -image
    height, width = signed.shape
    radius = max(1, int(window) // 2)
    x0 = max(0, candidate.x - radius)
    x1 = min(width, candidate.x + radius + 1)
    y0 = max(0, candidate.y - radius)
    y1 = min(height, candidate.y + radius + 1)
    peak = float(signed[candidate.y, candidate.x])
    valley = float(signed[y0:y1, x0:x1].min())
    return LocalPeakCandidate(
        x=candidate.x,
        y=candidate.y,
        polarity=candidate.polarity,
        peak=peak,
        valley=valley,
        prominence=peak - valley,
    )

def estimate_spacing(
    candidates: list[LocalPeakCandidate],
    *,
    fallback: float = 10.0,
) -> float:
    if len(candidates) < 4:
        return float(fallback)
    points = np.asarray([[item.x, item.y] for item in candidates], dtype=np.float64)
    # The prominence gate normally leaves a few hundred candidates; cap the
    # O(n^2) fallback so a pathological frame cannot allocate gigabytes.
    if points.shape[0] > 2000:
        points = points[:2000]
    try:
        delta = points[:, None, :] - points[None, :, :]
        distances = np.sqrt((delta * delta).sum(axis=-1))
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
        spacing = float(np.median(nearest))
    except (MemoryError, ValueError):
        return float(fallback)
    if not np.isfinite(spacing) or spacing < 2.0 or spacing > 128.0:
        return float(fallback)
    return spacing


def support_radii(
    spacing: float,
    *,
    max_radius: float | None = None,
    compact_radius: float | None = None,
) -> tuple[float, float]:
    if max_radius is None:
        max_radius = float(np.clip(0.30 * float(spacing), 2.5, 7.0))
    max_radius = float(max(2.0, max_radius))
    if compact_radius is None:
        compact_radius = max(2.0, 0.60 * max_radius)
    compact_radius = float(np.clip(compact_radius, 2.0, max_radius))
    return compact_radius, max_radius



def support_radii_for_candidates(
    candidates: list[LocalPeakCandidate],
    *,
    fallback_spacing: float = 10.0,
    neighbours: int = 4,
) -> list[tuple[float, float]]:
    """Per-candidate support radii from the local candidate neighbourhood.

    The earlier version used one global median spacing for the whole image.
    That is wrong under perspective: the compressed side of the target has a
    much smaller local pitch and needs a smaller support than the stretched
    side.  The k-nearest candidate distances give a local scale estimate that
    follows the image content; the curvature/ring stops still decide the final
    extent.
    """
    if len(candidates) < 4:
        compact, maximum = support_radii(fallback_spacing)
        return [(compact, maximum) for _ in candidates]
    points = np.asarray([[item.x, item.y] for item in candidates], dtype=np.float64)
    try:
        tree = cKDTree(points)
        k = min(max(2, int(neighbours) + 1), len(candidates))
        distances, _indices = tree.query(points, k=k)
        if distances.ndim == 1:
            distances = distances[:, None]
        nearest = distances[:, 1:]
        local_spacing = np.median(nearest, axis=1)
    except Exception:
        compact, maximum = support_radii(fallback_spacing)
        return [(compact, maximum) for _ in candidates]
    local_spacing = np.where(np.isfinite(local_spacing), local_spacing, fallback_spacing)
    maximum = np.clip(0.45 * local_spacing, 2.5, 7.0)
    compact = np.maximum(2.0, 0.60 * maximum)
    compact = np.minimum(compact, maximum)
    return [(float(compact[i]), float(maximum[i])) for i in range(len(candidates))]

def curvature_maps(smoothed: np.ndarray, *, epsilon: float = 1e-6) -> CurvatureData:
    image = np.asarray(smoothed, dtype=np.float64)
    if image.ndim == 3:
        image = image[..., 0]
    gy, gx = np.gradient(image)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    hxy = 0.5 * (gxy + gyx)
    det = gxx * gyy - hxy * hxy
    white = (gxx < -epsilon) & (gyy < -epsilon) & (det > epsilon * epsilon)
    black = (gxx > epsilon) & (gyy > epsilon) & (det > epsilon * epsilon)
    return CurvatureData(
        white_concave=white,
        black_concave=black,
        hxx=gxx,
        hxy=hxy,
        hyy=gyy,
    )


def _bilinear_many(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    valid = (xs >= 0.0) & (xs <= width - 1.0) & (ys >= 0.0) & (ys <= height - 1.0)
    x_clipped = np.clip(xs, 0.0, width - 1.0)
    y_clipped = np.clip(ys, 0.0, height - 1.0)
    x0 = np.floor(x_clipped).astype(np.int64)
    y0 = np.floor(y_clipped).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x_clipped - x0
    wy = y_clipped - y0
    values = (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    return values, valid


def _ring_values(
    response: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * pi, RING_ANGLES, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return _bilinear_many(response, xs, ys)


def _ring_accepts(
    response: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    previous_radius: float,
    peak: float,
    valley: float,
    prominence: float,
) -> bool:
    middle, valid = _ring_values(response, cx, cy, radius)
    if int(valid.sum()) < 8:
        return False
    previous, prev_valid = _ring_values(response, cx, cy, previous_radius)
    if int(prev_valid.sum()) < 8:
        return False
    plus, plus_valid = _ring_values(response, cx, cy, radius + 0.4)
    minus, minus_valid = _ring_values(response, cx, cy, radius - 0.4)
    if int(plus_valid.sum()) < 8 or int(minus_valid.sum()) < 8:
        return False
    common = valid & plus_valid & minus_valid
    if int(common.sum()) < 8:
        return False

    current_median = float(np.median(middle[valid]))
    previous_median = float(np.median(previous[prev_valid]))
    if current_median >= previous_median - 1e-5:
        return False
    if current_median <= valley + 0.10 * prominence:
        return False
    if float(middle[valid].max()) > peak + 1e-5:
        return False

    slope = plus[common] - minus[common]
    if float(np.mean(slope < 0.0)) < 0.80:
        return False
    second = plus[common] - 2.0 * middle[common] + minus[common]
    if float(np.mean(second <= 0.0)) < 0.60:
        return False
    return True


def _ray_stop(
    response: np.ndarray,
    cx: float,
    cy: float,
    direction: np.ndarray,
    *,
    max_radius: float,
    peak: float,
    valley: float,
    prominence: float,
    floor_fraction: float = 0.20,
) -> float:
    """Distance along a ray until the peak has fallen into the local gray.

    A local prominence floor is more stable than a finite-difference inflection
    test at these marker scales: the second derivative of a 2-4 px blurred
    marker is dominated by pixel quantisation.  The floor is still local to the
    candidate (a fraction of its own peak-to-valley), not a global threshold.
    """
    floor = valley + float(floor_fraction) * prominence
    last = 2.0
    previous = peak
    step = 0.5
    radius = step
    while radius <= max_radius + 1e-9:
        x = cx + float(direction[0]) * radius
        y = cy + float(direction[1]) * radius
        value, valid = _bilinear_many(response, np.asarray([x]), np.asarray([y]))
        if not bool(valid[0]):
            break
        current = float(value[0])
        if current >= previous - 1e-5 or current > peak + 1e-5:
            break
        if current <= floor:
            break
        last = radius
        previous = current
        radius += step
    return float(min(max_radius, max(2.0, last)))


def _disk_offsets(radius: float) -> np.ndarray:
    limit = int(np.ceil(radius))
    dx, dy = np.meshgrid(
        np.arange(-limit, limit + 1, dtype=np.int32),
        np.arange(-limit, limit + 1, dtype=np.int32),
        indexing="xy",
    )
    mask = dx * dx + dy * dy <= radius * radius + 1e-9
    return np.stack([dx[mask], dy[mask]], axis=1).astype(np.int32)


def _ellipse_offsets(
    semi_axes: tuple[float, float],
    basis: np.ndarray,
) -> np.ndarray:
    limit = int(np.ceil(max(semi_axes)))
    dx, dy = np.meshgrid(
        np.arange(-limit, limit + 1, dtype=np.int32),
        np.arange(-limit, limit + 1, dtype=np.int32),
        indexing="xy",
    )
    u = dx * basis[0, 0] + dy * basis[1, 0]
    v = dx * basis[0, 1] + dy * basis[1, 1]
    a1 = max(1e-6, float(semi_axes[0]))
    a2 = max(1e-6, float(semi_axes[1]))
    mask = (u * u) / (a1 * a1) + (v * v) / (a2 * a2) <= 1.0 + 1e-9
    return np.stack([dx[mask], dy[mask]], axis=1).astype(np.int32)


def _filter_support(
    offsets: np.ndarray,
    candidate: LocalPeakCandidate,
    curvature: CurvatureData | None,
    *,
    use_curvature_mask: bool,
) -> tuple[np.ndarray, bool]:
    if not use_curvature_mask or curvature is None:
        return offsets, False
    mask = (
        curvature.white_concave
        if candidate.polarity == 1
        else curvature.black_concave
    )
    height, width = mask.shape
    keep = []
    for dx, dy in offsets:
        x = candidate.x + int(dx)
        y = candidate.y + int(dy)
        if 0 <= x < width and 0 <= y < height and bool(mask[y, x]):
            keep.append((int(dx), int(dy)))
    if len(keep) < CUBIC_TERMS:
        return offsets, False
    return np.asarray(keep, dtype=np.int32), True



def _connected_curvature_region(
    response: np.ndarray,
    candidate: LocalPeakCandidate,
    curvature: CurvatureData,
    max_radius: float,
    valley: float,
    prominence: float,
    *,
    floor_fraction: float = 0.10,
) -> np.ndarray:
    """Grow the connected same-sign-curvature region around a candidate.

    This is the actual "curvature support" rather than intersecting a fixed
    disk with a sign mask: the region is connected, follows the local
    positive-curvature pixels, stops at the local gray floor, and is bounded by
    the per-candidate local radius.
    """
    mask = (
        curvature.white_concave
        if candidate.polarity == 1
        else curvature.black_concave
    )
    height, width = mask.shape
    floor = float(valley) + float(floor_fraction) * float(prominence)
    limit = int(np.ceil(max_radius))
    radius_sq = float(max_radius) * float(max_radius)
    visited: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = [(int(candidate.x), int(candidate.y))]
    offsets: list[tuple[int, int, float]] = []
    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        dx = x - int(candidate.x)
        dy = y - int(candidate.y)
        if dx * dx + dy * dy > radius_sq:
            continue
        if not (0 <= x < width and 0 <= y < height):
            continue
        is_center = dx == 0 and dy == 0
        if not is_center and not bool(mask[y, x]):
            continue
        if not is_center and float(response[y, x]) <= floor:
            continue
        offsets.append((dx, dy, float(response[y, x])))
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if ox == 0 and oy == 0:
                    continue
                nx = x + ox
                ny = y + oy
                if abs(nx - int(candidate.x)) > limit or abs(ny - int(candidate.y)) > limit:
                    continue
                if (nx, ny) not in visited:
                    stack.append((nx, ny))
    if not offsets:
        return np.zeros((0, 2), dtype=np.int32)
    ordered = sorted(offsets, key=lambda item: (item[1], item[0]))
    return np.asarray([[dx, dy] for dx, dy, _value in ordered], dtype=np.int32)

def select_support(
    smoothed: np.ndarray,
    candidate: LocalPeakCandidate,
    curvature: CurvatureData | None,
    *,
    compact_radius: float,
    max_radius: float,
    mode: str,
) -> SupportRegion:
    response = smoothed if candidate.polarity == 1 else -smoothed
    basis = np.eye(2, dtype=np.float64)
    use_curvature_mask = mode in {"adaptive_circle_concave", "adaptive_ellipse_concave"}

    if mode == "curvature_region":
        if curvature is None:
            return select_support(
                smoothed,
                candidate,
                curvature,
                compact_radius=compact_radius,
                max_radius=max_radius,
                mode="adaptive_circle_concave",
            )
        offsets = _connected_curvature_region(
            response,
            candidate,
            curvature,
            max_radius,
            candidate.valley,
            candidate.prominence,
        )
        if offsets.shape[0] < CUBIC_TERMS:
            return select_support(
                smoothed,
                candidate,
                curvature,
                compact_radius=compact_radius,
                max_radius=max_radius,
                mode="adaptive_circle_concave",
            )
        scale = max(
            2.0,
            float(
                np.sqrt(
                    np.mean(
                        offsets[:, 0].astype(np.float64) ** 2
                        + offsets[:, 1].astype(np.float64) ** 2
                    )
                )
            ),
        )
        return SupportRegion(
            offsets=offsets,
            semi_axes=(scale, scale),
            basis=basis,
            kind="region",
            used_curvature_mask=True,
        )

    if mode in {"compact", "full", "adaptive_circle", "adaptive_circle_concave"}:
        radius = compact_radius if mode == "compact" else max_radius
        if mode in {"adaptive_circle", "adaptive_circle_concave"}:
            current = compact_radius
            while current < max_radius - 1e-9:
                next_radius = min(max_radius, current + 0.5)
                if not _ring_accepts(
                    response,
                    float(candidate.x),
                    float(candidate.y),
                    next_radius,
                    previous_radius=current,
                    peak=candidate.peak,
                    valley=candidate.valley,
                    prominence=candidate.prominence,
                ):
                    break
                current = next_radius
            radius = current
        offsets = _disk_offsets(radius)
        offsets, used_mask = _filter_support(
            offsets,
            candidate,
            curvature,
            use_curvature_mask=use_curvature_mask,
        )
        return SupportRegion(
            offsets=offsets,
            semi_axes=(float(radius), float(radius)),
            basis=basis,
            kind="circle",
            used_curvature_mask=used_mask,
        )

    if mode != "adaptive_ellipse_concave":
        raise ValueError(f"unknown support mode: {mode}")

    # Elliptical support aligned with the local Hessian. The axes are the
    # minimum of the opposite-side stop distances so a neighbour on one side
    # cannot stretch the support in that direction.
    if curvature is None:
        offsets = _disk_offsets(compact_radius)
        return SupportRegion(
            offsets=offsets,
            semi_axes=(float(compact_radius), float(compact_radius)),
            basis=basis,
            kind="circle",
            used_curvature_mask=False,
        )
    h = np.asarray(
        [
            [curvature.hxx[candidate.y, candidate.x], curvature.hxy[candidate.y, candidate.x]],
            [curvature.hxy[candidate.y, candidate.x], curvature.hyy[candidate.y, candidate.x]],
        ],
        dtype=np.float64,
    )
    if candidate.polarity == 0:
        h = -h
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(h)
    except np.linalg.LinAlgError:
        eigenvalues = np.asarray([0.0, 0.0])
        eigenvectors = np.eye(2)
    if eigenvalues[1] >= -1e-9:
        offsets = _disk_offsets(compact_radius)
        offsets, used_mask = _filter_support(
            offsets,
            candidate,
            curvature,
            use_curvature_mask=True,
        )
        return SupportRegion(
            offsets=offsets,
            semi_axes=(float(compact_radius), float(compact_radius)),
            basis=basis,
            kind="circle",
            used_curvature_mask=used_mask,
        )
    # ``eigh`` returns ascending eigenvalues: index 0 is the narrow (more
    # negative) direction, index 1 the broad direction.  Peak width scales as
    # 1/sqrt(|lambda|), so the curvature ratio directly gives the ellipse
    # aspect.  The broad-axis extent comes from the local falloff; the narrow
    # axis is additionally capped by the narrow-side stop so a compressed
    # neighbour cannot be pulled into the fit.
    axis_narrow = eigenvectors[:, 0]
    axis_broad = eigenvectors[:, 1]
    narrow_stop = min(
        _ray_stop(
            response,
            float(candidate.x),
            float(candidate.y),
            axis_narrow,
            max_radius=max_radius,
            peak=candidate.peak,
            valley=candidate.valley,
            prominence=candidate.prominence,
        ),
        _ray_stop(
            response,
            float(candidate.x),
            float(candidate.y),
            -axis_narrow,
            max_radius=max_radius,
            peak=candidate.peak,
            valley=candidate.valley,
            prominence=candidate.prominence,
        ),
    )
    # The broad axis is the stretched (well-separated) lattice direction, so
    # taking the farther side captures the marker's true extent; the narrow
    # axis keeps the minimum of the two sides so a compressed neighbour is not
    # pulled into the fit.
    broad_stop = max(
        _ray_stop(
            response,
            float(candidate.x),
            float(candidate.y),
            axis_broad,
            max_radius=max_radius,
            peak=candidate.peak,
            valley=candidate.valley,
            prominence=candidate.prominence,
        ),
        _ray_stop(
            response,
            float(candidate.x),
            float(candidate.y),
            -axis_broad,
            max_radius=max_radius,
            peak=candidate.peak,
            valley=candidate.valley,
            prominence=candidate.prominence,
        ),
    )
    ratio = float(
        np.clip(
            np.sqrt(
                max(0.0, abs(float(eigenvalues[1])))
                / max(1e-12, abs(float(eigenvalues[0])))
            ),
            0.25,
            1.0,
        )
    )
    # Near-isotropic peaks do not benefit from an ellipse; the compact/adaptive
    # circle is less prone to including background than a broad-axis fit.
    if ratio >= 0.85:
        return select_support(
            smoothed,
            candidate,
            curvature,
            compact_radius=compact_radius,
            max_radius=max_radius,
            mode="adaptive_circle_concave",
        )
    broad = float(np.clip(broad_stop, 2.0, max_radius))
    narrow = float(np.clip(min(broad * ratio, narrow_stop), 2.0, max_radius))
    broad = max(broad, narrow)
    basis = np.stack([axis_narrow, axis_broad], axis=1).astype(np.float64)
    offsets = _ellipse_offsets((narrow, broad), basis)
    a1, a2 = narrow, broad
    offsets, used_mask = _filter_support(
        offsets,
        candidate,
        curvature,
        use_curvature_mask=True,
    )
    return SupportRegion(
        offsets=offsets,
        semi_axes=(a1, a2),
        basis=basis,
        kind="ellipse",
        used_curvature_mask=used_mask,
    )


def _design_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.ones_like(u),
            u,
            v,
            u * u,
            u * v,
            v * v,
            u * u * u,
            u * u * v,
            u * v * v,
            v * v * v,
        ],
        axis=1,
    )


def _surface_value(coefficients: np.ndarray, u: float, v: float) -> float:
    design = _design_matrix(np.asarray([u]), np.asarray([v]))[0]
    return float(design @ coefficients)


def _solve_cubic_vertex(
    coefficients: np.ndarray,
    *,
    max_iterations: int = 25,
) -> tuple[float, float, np.ndarray, int] | None:
    c = np.asarray(coefficients, dtype=np.float64)
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = c.tolist()
    u = 0.0
    v = 0.0
    for iteration in range(1, max_iterations + 1):
        gx = c1 + 2.0 * c3 * u + c4 * v + 3.0 * c6 * u * u + 2.0 * c7 * u * v + c8 * v * v
        gy = c2 + c4 * u + 2.0 * c5 * v + c7 * u * u + 2.0 * c8 * u * v + 3.0 * c9 * v * v
        hxx = 2.0 * c3 + 6.0 * c6 * u + 2.0 * c7 * v
        hxy = c4 + 2.0 * c7 * u + 2.0 * c8 * v
        hyy = 2.0 * c5 + 2.0 * c8 * u + 6.0 * c9 * v
        hessian = np.asarray([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
        det = float(np.linalg.det(hessian))
        if not np.isfinite(det) or abs(det) < 1e-12:
            return None
        step = np.linalg.solve(hessian, -np.asarray([gx, gy], dtype=np.float64))
        if not np.isfinite(step).all():
            return None
        scale = 1.0
        if float(np.linalg.norm(step)) > 0.5:
            scale = 0.5 / float(np.linalg.norm(step))
        for _ in range(10):
            candidate_u = u + step[0] * scale
            candidate_v = v + step[1] * scale
            if abs(candidate_u) <= 1.5 and abs(candidate_v) <= 1.5:
                break
            scale *= 0.5
        u += step[0] * scale
        v += step[1] * scale
        if float(np.linalg.norm(step * scale)) < 1e-8:
            break
    else:
        return None
    hxx = 2.0 * c3 + 6.0 * c6 * u + 2.0 * c7 * v
    hxy = c4 + 2.0 * c7 * u + 2.0 * c8 * v
    hyy = 2.0 * c5 + 2.0 * c8 * u + 6.0 * c9 * v
    hessian = np.asarray([[hxx, hxy], [hxy, hyy]], dtype=np.float64)
    try:
        eigenvalues = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError:
        return None
    if eigenvalues[1] >= -1e-10:
        return None
    return u, v, hessian, iteration


def fit_peak(
    smoothed: np.ndarray,
    candidate: LocalPeakCandidate,
    curvature: CurvatureData | None,
    *,
    compact_radius: float,
    max_radius: float,
    mode: str,
) -> LocalPeak:
    response = smoothed if candidate.polarity == 1 else -smoothed
    support = select_support(
        smoothed,
        candidate,
        curvature,
        compact_radius=compact_radius,
        max_radius=max_radius,
        mode=mode,
    )
    offsets = support.offsets
    if offsets.shape[0] < CUBIC_TERMS:
        return _fallback_peak(candidate, support, reason="support")

    dx = offsets[:, 0].astype(np.float64)
    dy = offsets[:, 1].astype(np.float64)
    if support.kind == "ellipse":
        basis = support.basis
        a1 = max(1e-6, float(support.semi_axes[0]))
        a2 = max(1e-6, float(support.semi_axes[1]))
        u = (dx * basis[0, 0] + dy * basis[1, 0]) / a1
        v = (dx * basis[0, 1] + dy * basis[1, 1]) / a2
    else:
        radius = max(1e-6, float(support.semi_axes[0]))
        u = dx / radius
        v = dy / radius

    ys = candidate.y + offsets[:, 1]
    xs = candidate.x + offsets[:, 0]
    height, width = response.shape
    if (
        int(xs.min()) < 0
        or int(ys.min()) < 0
        or int(xs.max()) >= width
        or int(ys.max()) >= height
    ):
        return _fallback_peak(candidate, support, reason="bounds")
    values = response[ys, xs].astype(np.float64)
    design = _design_matrix(u, v)
    try:
        normal = design.T @ design
        ridge = 1e-6 * float(np.trace(normal)) / float(CUBIC_TERMS)
        gram = normal + ridge * np.eye(CUBIC_TERMS, dtype=np.float64)
        coefficients = np.linalg.solve(gram, design.T @ values)
    except np.linalg.LinAlgError:
        return _fallback_peak(candidate, support, reason="singular")
    if not np.isfinite(coefficients).all():
        return _fallback_peak(candidate, support, reason="nonfinite")

    solved = _solve_cubic_vertex(coefficients)
    if solved is None:
        return _fallback_peak(candidate, support, reason="vertex")
    u_star, v_star, hessian, iterations = solved
    if u_star * u_star + v_star * v_star > 1.0 + 1e-6:
        return _fallback_peak(candidate, support, reason="outside")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[1] >= -1e-10:
        return _fallback_peak(candidate, support, reason="hessian")

    if support.kind == "ellipse":
        basis = support.basis
        a1 = float(support.semi_axes[0])
        a2 = float(support.semi_axes[1])
        x_index = (
            float(candidate.x)
            + u_star * a1 * float(basis[0, 0])
            + v_star * a2 * float(basis[0, 1])
        )
        y_index = (
            float(candidate.y)
            + u_star * a1 * float(basis[1, 0])
            + v_star * a2 * float(basis[1, 1])
        )
    else:
        radius = float(support.semi_axes[0])
        x_index = float(candidate.x) + u_star * radius
        y_index = float(candidate.y) + v_star * radius

    fitted = _surface_value(coefficients, u_star, v_star)
    residual = float(np.sqrt(np.mean((values - design @ coefficients) ** 2)))
    largest = max(abs(float(eigenvalues[0])), 1e-12)
    anisotropy = float(np.sqrt(max(0.0, abs(float(eigenvalues[1])) / largest)))
    orientation = (
        float(np.degrees(np.arctan2(support.basis[1, 0], support.basis[0, 0])))
        if support.kind == "ellipse"
        else 0.0
    )
    return LocalPeak(
        x=x_index + 0.5,
        y=y_index + 0.5,
        polarity=candidate.polarity,
        candidate_x=float(candidate.x),
        candidate_y=float(candidate.y),
        peak=float(fitted),
        prominence=float(candidate.prominence),
        support_kind=support.kind,
        support_radius=support.radius,
        support_axes=support.semi_axes,
        support_pixels=support.pixel_count,
        used_curvature_mask=support.used_curvature_mask,
        fit_ok=True,
        fallback=False,
        eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
        anisotropy=anisotropy,
        orientation_deg=orientation,
        residual_rms=residual,
        iterations=int(iterations),
        offset_x=float(x_index - candidate.x),
        offset_y=float(y_index - candidate.y),
    )


def _fallback_peak(
    candidate: LocalPeakCandidate,
    support: SupportRegion,
    *,
    reason: str,
) -> LocalPeak:
    _ = reason
    orientation = (
        float(np.degrees(np.arctan2(support.basis[1, 0], support.basis[0, 0])))
        if support.kind == "ellipse"
        else 0.0
    )
    return LocalPeak(
        x=candidate.continuous_x,
        y=candidate.continuous_y,
        polarity=candidate.polarity,
        candidate_x=float(candidate.x),
        candidate_y=float(candidate.y),
        peak=float(candidate.peak),
        prominence=float(candidate.prominence),
        support_kind=support.kind,
        support_radius=support.radius,
        support_axes=support.semi_axes,
        support_pixels=support.pixel_count,
        used_curvature_mask=support.used_curvature_mask,
        fit_ok=False,
        fallback=True,
        eigenvalues=(float("nan"), float("nan")),
        anisotropy=float("nan"),
        orientation_deg=orientation,
        residual_rms=float("nan"),
        iterations=0,
        offset_x=0.0,
        offset_y=0.0,
    )


def fit_peaks(
    smoothed: np.ndarray,
    candidates: list[LocalPeakCandidate],
    *,
    spacing: float,
    mode: str,
    compact_radius: float | None = None,
    max_radius: float | None = None,
    curvature: CurvatureData | None = None,
    local_radii: bool = False,
) -> list[LocalPeak]:
    if local_radii:
        radii = support_radii_for_candidates(
            candidates,
            fallback_spacing=spacing,
        )
    else:
        compact, maximum = support_radii(
            spacing,
            max_radius=max_radius,
            compact_radius=compact_radius,
        )
        radii = [(compact, maximum) for _ in candidates]
    return [
        fit_peak(
            smoothed,
            candidate,
            curvature,
            compact_radius=radii[index][0],
            max_radius=radii[index][1],
            mode=mode,
        )
        for index, candidate in enumerate(candidates)
    ]
