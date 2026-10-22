"""Load the target-page JSON export and validate layout parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blob_lab.target_layout import (
    DEFAULT_MARKER_SCALE,
    HexTargetLayout,
    HexTargetQuad,
    build_hex_target_layout,
)


class LayoutMismatchError(ValueError):
    """The exported quads do not match the reconstructed Python layout."""


def layout_from_params(
    *,
    width_px: int,
    height_px: int,
    columns: int = 8,
    points_per_edge: int = 4,
    marker_scale: float = DEFAULT_MARKER_SCALE,
    color_seed: int = 1,
    padding_fraction: float = 0.1,
) -> HexTargetLayout:
    return build_hex_target_layout(
        width_px=width_px,
        height_px=height_px,
        columns=columns,
        points_per_edge=points_per_edge,
        marker_scale=marker_scale,
        color_seed=color_seed,
        padding_fraction=padding_fraction,
    )


def _quad_tuple(quad: HexTargetQuad) -> tuple[int, int, int, int, int, int]:
    return (quad.design_id, quad.color, quad.left, quad.top, quad.right, quad.bottom)


def load_target_export(source: str | Path | dict[str, Any], *, strict: bool = True) -> HexTargetLayout:
    if isinstance(source, dict):
        data = source
    else:
        data = json.loads(Path(source).read_text())

    if data.get("version") != 1:
        raise ValueError(f"Unsupported target export version: {data.get('version')!r}")

    resolution = data["resolution"]
    width_px = int(resolution["widthPx"])
    height_px = int(resolution["heightPx"])
    padding = data.get("padding", {})
    padding_fraction = (
        float(padding["xPx"]) / width_px if width_px else 0.1
    )

    layout = layout_from_params(
        width_px=width_px,
        height_px=height_px,
        columns=int(data["columns"]),
        points_per_edge=int(data["pointsPerEdge"]),
        marker_scale=float(data["markerScale"]),
        color_seed=int(data["colorSeed"]),
        padding_fraction=padding_fraction,
    )

    exported = tuple(
        HexTargetQuad(
            design_id=int(quad["designId"]),
            color=int(quad["color"]),
            left=int(quad["left"]),
            top=int(quad["top"]),
            right=int(quad["right"]),
            bottom=int(quad["bottom"]),
        )
        for quad in data.get("quads", [])
    )

    if strict and exported:
        python_quads = tuple(_quad_tuple(quad) for quad in layout.quads)
        export_quads = tuple(_quad_tuple(quad) for quad in exported)
        if python_quads != export_quads:
            mismatch = next(
                (
                    (index, python_quads[index], export_quads[index])
                    for index in range(min(len(python_quads), len(export_quads)))
                    if python_quads[index] != export_quads[index]
                ),
                (
                    min(len(python_quads), len(export_quads)),
                    python_quads[len(python_quads) - 1] if python_quads else None,
                    export_quads[len(export_quads) - 1] if export_quads else None,
                ),
            )
            raise LayoutMismatchError(
                "Exported target quads do not match the Python layout. "
                f"First mismatch at index {mismatch[0]}: python={mismatch[1]!r}, export={mismatch[2]!r}"
            )

    if exported and not layout.quads:
        return layout

    # Keep the reconstructed pattern/guides, but use exported quads as the
    # authoritative render geometry when they are identical.
    return layout
