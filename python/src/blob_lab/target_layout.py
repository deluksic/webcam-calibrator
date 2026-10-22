"""Exact Python port of the TypeScript hex-target layout.

The target page exports quads in pixel coordinates.  The same pattern and
layout code is reproduced here so the Python renderer can be used with or
without an exported JSON file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, floor, isfinite, pi, sin, sqrt
from typing import Iterable


DESIGN_WIDTH = 720.0
DESIGN_HEIGHT = 540.0
COLUMN_PITCH = 1.5
ROW_PITCH = sqrt(3.0)
CORNER_COUNT = 6
DUPLICATE_DISTANCE = 0.5
DEFAULT_PADDING_FRACTION = 0.1
DEFAULT_MARKER_SCALE = 0.45
MIN_PADDING_PX = 8
MIN_GAP_FRACTION = 0.06


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float


@dataclass(frozen=True)
class PatternPoint:
    x: float
    y: float
    index: int


@dataclass(frozen=True)
class PatternGuide:
    a: Vec2
    b: Vec2


@dataclass(frozen=True)
class HexPattern:
    points: tuple[PatternPoint, ...]
    guides: tuple[PatternGuide, ...]
    bounds: tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    cells: int
    candidate_edges: int
    raw_points: int
    cell_polygons: tuple[tuple[Vec2, ...], ...] = ()


@dataclass(frozen=True)
class HexTargetQuad:
    design_id: int
    color: int
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class HexTargetLayout:
    width_px: int
    height_px: int
    padding_x_px: int
    padding_y_px: int
    columns: int
    points_per_edge: int
    marker_scale: float
    color_seed: int
    spacing_px: int
    side_px: int
    quads: tuple[HexTargetQuad, ...]
    pattern_points: tuple[PatternPoint, ...]
    guides: tuple[PatternGuide, ...]
    junction_points: tuple[Vec2, ...]
    three_way_junction_points: tuple[Vec2, ...] = field(default_factory=tuple)
    colors: tuple[int, ...] = field(default_factory=tuple)
    # Hexagon cell outlines in target pixel coordinates.
    cell_polygons: tuple[tuple[Vec2, ...], ...] = field(default_factory=tuple)

    @property
    def square_centers(self) -> tuple[Vec2, ...]:
        return tuple(
            Vec2((quad.left + quad.right) / 2.0, (quad.top + quad.bottom) / 2.0)
            for quad in self.quads
        )


def _i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _imul(a: int, b: int) -> int:
    return _i32((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF))


def _ushr(value: int, count: int) -> int:
    return (value & 0xFFFFFFFF) >> count


def _js_round(value: float) -> int:
    return floor(value + 0.5)


def mulberry32(seed: int):
    state = seed & 0xFFFFFFFF

    def random() -> float:
        nonlocal state
        state = _i32(state + 0x6D2B79F5)
        value = _imul(state ^ _ushr(state, 15), 1 | state)
        value = _i32((value + _imul(value ^ _ushr(value, 7), 61 | value)) ^ value)
        return _ushr(value ^ _ushr(value, 14), 0) / 4294967296.0

    return random


def random_binary_colors(count: int, seed: int) -> tuple[int, ...]:
    random = mulberry32(seed)
    return tuple(0 if random() < 0.5 else 1 for _ in range(count))


def _distance_sq(a: Vec2, b: Vec2) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy


def _vertex_key(point: Vec2) -> str:
    return f"{point.x:.4f}:{point.y:.4f}"


def _edge_key(a: Vec2, b: Vec2) -> str:
    first = _vertex_key(a)
    second = _vertex_key(b)
    return f"{first}|{second}" if first < second else f"{second}|{first}"


def _remove_near_duplicates(points: Iterable[Vec2]) -> list[Vec2]:
    kept: list[Vec2] = []
    threshold = DUPLICATE_DISTANCE * DUPLICATE_DISTANCE
    for point in points:
        if all(_distance_sq(point, other) >= threshold for other in kept):
            kept.append(point)
    return kept


def _build_cells(columns: int, radius: float) -> list[tuple[Vec2, int, int]]:
    column_pitch = COLUMN_PITCH * radius
    row_pitch = ROW_PITCH * radius
    rows = max(2, _js_round((DESIGN_HEIGHT - 2.0 * radius) / row_pitch))
    cells: list[tuple[Vec2, int, int]] = []
    for column in range(columns):
        offset = 0.0 if column % 2 == 0 else row_pitch / 2.0
        for row in range(rows):
            cells.append((Vec2(column * column_pitch, row * row_pitch + offset), column, row))
    return cells


def _build_cell_edges(cells: Iterable[tuple[Vec2, int, int]], radius: float) -> list[tuple[Vec2, Vec2]]:
    edges: dict[str, tuple[Vec2, Vec2]] = {}
    for center, _column, _row in cells:
        corners = [
            Vec2(center.x + cos(i * pi / 3.0) * radius, center.y + sin(i * pi / 3.0) * radius)
            for i in range(CORNER_COUNT)
        ]
        for corner in range(CORNER_COUNT):
            a = corners[corner]
            b = corners[(corner + 1) % CORNER_COUNT]
            key = _edge_key(a, b)
            if key not in edges:
                edges[key] = (a, b)
    return list(edges.values())


def generate_hex_pattern(columns: int = 8, points_per_edge: int = 4) -> HexPattern:
    columns = max(2, _js_round(columns))
    points_per_edge = max(1, _js_round(points_per_edge))
    radius = DESIGN_WIDTH / (COLUMN_PITCH * (columns - 1) + 2.0)
    cells = _build_cells(columns, radius)
    candidate_edges = _build_cell_edges(cells, radius)

    guides: list[PatternGuide] = []
    raw_points: list[Vec2] = []
    for a, b in candidate_edges:
        guides.append(PatternGuide(a, b))
        dx = b.x - a.x
        dy = b.y - a.y
        for step in range(points_per_edge):
            t = (step + 0.5) / points_per_edge
            raw_points.append(Vec2(a.x + dx * t, a.y + dy * t))

    deduped = _remove_near_duplicates(raw_points)
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for point in [*deduped, *(point for guide in guides for point in (guide.a, guide.b))]:
        min_x = min(min_x, point.x)
        max_x = max(max_x, point.x)
        min_y = min(min_y, point.y)
        max_y = max(max_y, point.y)

    has_bounds = isfinite(min_x) and isfinite(min_y)
    center_x = (min_x + max_x) / 2.0 if has_bounds else 0.0
    center_y = (min_y + max_y) / 2.0 if has_bounds else 0.0

    centered_points = sorted(
        (Vec2(point.x - center_x, point.y - center_y) for point in deduped),
        key=lambda point: (point.y, point.x),
    )
    points = [PatternPoint(point.x, point.y, index) for index, point in enumerate(centered_points)]
    centered_guides = tuple(
        PatternGuide(
            Vec2(guide.a.x - center_x, guide.a.y - center_y),
            Vec2(guide.b.x - center_x, guide.b.y - center_y),
        )
        for guide in guides
    )
    bounds = (
        (min_x - center_x, max_x - center_x, min_y - center_y, max_y - center_y)
        if has_bounds
        else (-1.0, 1.0, -1.0, 1.0)
    )
    centered_cells = tuple(
        tuple(
            Vec2(
                center.x + cos(corner * pi / 3.0) * radius - center_x,
                center.y + sin(corner * pi / 3.0) * radius - center_y,
            )
            for corner in range(CORNER_COUNT)
        )
        for center, _column, _row in cells
    )
    return HexPattern(
        points=tuple(points),
        guides=centered_guides,
        bounds=bounds,
        cells=len(cells),
        candidate_edges=len(candidate_edges),
        raw_points=len(raw_points),
        cell_polygons=centered_cells,
    )


def unique_points(points: Iterable[Vec2], tolerance: float = 1e-6) -> tuple[Vec2, ...]:
    kept: list[Vec2] = []
    threshold = tolerance * tolerance
    for point in points:
        if all(_distance_sq(point, other) >= threshold for other in kept):
            kept.append(point)
    return tuple(kept)


def build_hex_target_layout(
    *,
    width_px: int,
    height_px: int,
    columns: int = 8,
    points_per_edge: int = 4,
    marker_scale: float = DEFAULT_MARKER_SCALE,
    color_seed: int = 1,
    padding_fraction: float = DEFAULT_PADDING_FRACTION,
) -> HexTargetLayout:
    width_px = max(0, floor(width_px))
    height_px = max(0, floor(height_px))
    columns = max(2, _js_round(columns))
    points_per_edge = max(1, _js_round(points_per_edge))
    marker_scale = min(0.7, max(0.2, marker_scale))
    padding_fraction = min(0.4, max(0.0, padding_fraction))
    padding_x = max(MIN_PADDING_PX, _js_round(width_px * padding_fraction))
    padding_y = max(MIN_PADDING_PX, _js_round(height_px * padding_fraction))

    empty = HexTargetLayout(
        width_px=width_px,
        height_px=height_px,
        padding_x_px=padding_x,
        padding_y_px=padding_y,
        columns=columns,
        points_per_edge=points_per_edge,
        marker_scale=marker_scale,
        color_seed=color_seed,
        spacing_px=0,
        side_px=0,
        quads=(),
        pattern_points=(),
        guides=(),
        junction_points=(),
        three_way_junction_points=(),
        colors=(),
        cell_polygons=(),
    )
    if width_px < 2 or height_px < 2 or width_px - padding_x * 2 < 2 or height_px - padding_y * 2 < 2:
        return empty

    pattern = generate_hex_pattern(columns, points_per_edge)
    if not pattern.points:
        return empty
    pattern_width = pattern.bounds[1] - pattern.bounds[0]
    pattern_height = pattern.bounds[3] - pattern.bounds[2]
    if pattern_width <= 0 or pattern_height <= 0:
        return empty

    scale = min(
        (width_px - 2 * padding_x) / pattern_width,
        (height_px - 2 * padding_y) / pattern_height,
    )
    offset_x = width_px / 2.0 - ((pattern.bounds[0] + pattern.bounds[1]) / 2.0) * scale
    offset_y = height_px / 2.0 - ((pattern.bounds[2] + pattern.bounds[3]) / 2.0) * scale
    centers = [
        (point.x * scale + offset_x, point.y * scale + offset_y, point.index)
        for point in pattern.points
    ]

    min_pair_distance = float("inf")
    nearest_distances: list[float] = []
    for first, (ax, ay, _) in enumerate(centers):
        nearest = float("inf")
        for second in range(first + 1, len(centers)):
            bx, by, _ = centers[second]
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            nearest = min(nearest, distance)
            min_pair_distance = min(min_pair_distance, distance)
        if isfinite(nearest):
            nearest_distances.append(nearest)
    if not nearest_distances or not isfinite(min_pair_distance):
        return empty
    nearest_distances.sort()
    median_nearest = nearest_distances[len(nearest_distances) // 2]
    min_gap = max(1, _js_round(median_nearest * MIN_GAP_FRACTION))
    desired_side = max(1, floor(median_nearest * marker_scale))
    side_value = max(1, min(float(desired_side), min_pair_distance - min_gap * 2))
    side_px = max(1, floor(side_value))
    if side_px < 1 or min_pair_distance < min_gap * 2 + 2:
        return empty

    colors = random_binary_colors(len(pattern.points), color_seed)
    quads: list[HexTargetQuad] = []
    for x, y, source_index in centers:
        left = _js_round(x) - side_px // 2
        top = _js_round(y) - side_px // 2
        right = left + side_px
        bottom = top + side_px
        if left < 0 or top < 0 or right > width_px or bottom > height_px:
            continue
        quads.append(HexTargetQuad(source_index, colors[source_index], left, top, right, bottom))

    target_guides = tuple(
        PatternGuide(
            Vec2(guide.a.x * scale + offset_x, guide.a.y * scale + offset_y),
            Vec2(guide.b.x * scale + offset_x, guide.b.y * scale + offset_y),
        )
        for guide in pattern.guides
    )
    junction_points = unique_points(
        (point for guide in target_guides for point in (guide.a, guide.b)),
        tolerance=1e-4,
    )
    _degree: dict[str, int] = {}
    _by_key: dict[str, Vec2] = {}
    for guide in target_guides:
        for point in (guide.a, guide.b):
            key = _vertex_key(point)
            _degree[key] = _degree.get(key, 0) + 1
            _by_key[key] = point
    three_way_junction_points = tuple(
        _by_key[key] for key, degree in _degree.items() if degree >= 3
    )
    target_cell_polygons = tuple(
        tuple(Vec2(point.x * scale + offset_x, point.y * scale + offset_y) for point in polygon)
        for polygon in pattern.cell_polygons
    )

    return HexTargetLayout(
        width_px=width_px,
        height_px=height_px,
        padding_x_px=padding_x,
        padding_y_px=padding_y,
        columns=columns,
        points_per_edge=points_per_edge,
        marker_scale=marker_scale,
        color_seed=color_seed,
        spacing_px=_js_round(median_nearest),
        side_px=side_px,
        quads=tuple(quads),
        pattern_points=pattern.points,
        guides=target_guides,
        junction_points=junction_points,
        three_way_junction_points=three_way_junction_points,
        colors=colors,
        cell_polygons=target_cell_polygons,
    )


def build_target_texture(layout: HexTargetLayout, background: float = 0.5):
    """Build the static linear-light canonical target texture."""
    import numpy as np

    texture = np.full((layout.height_px, layout.width_px), background, dtype=np.float32)
    for quad in layout.quads:
        value = 0.0 if quad.color == 0 else 1.0
        texture[quad.top:quad.bottom, quad.left:quad.right] = value
    return texture
