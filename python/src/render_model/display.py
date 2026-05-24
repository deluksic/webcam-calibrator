"""Marimo / matplotlib display helpers for pixel-accurate image preview."""

import io
from typing import Any

import numpy as np

IMSHOW_KWARGS: dict[str, Any] = {
    "interpolation": "none",
    "resample": False,
}

PIXELATED_STYLE: dict[str, str] = {
    "image-rendering": "pixelated",
    "width": "100%",
    "height": "100%",
    "object-fit": "contain",
}


def imshow_extent(width: int, height: int) -> tuple[float, float, float, float]:
    """Matplotlib ``imshow`` extent for edge-origin image coordinates."""
    return (0.0, float(width), float(height), 0.0)


def centered_diff_limits(diff: np.ndarray, *, floor: float = 1e-6) -> float:
    """Symmetric ±limit for a diverging colormap on ``target - pred``."""
    arr = np.asarray(diff, dtype=np.float64)
    return float(max(np.max(np.abs(arr)), floor))


def configure_matplotlib_image_display() -> None:
    """Disable matplotlib resampling so ``imshow`` stays nearest-neighbor."""
    import matplotlib as mpl

    mpl.rcParams["image.interpolation"] = "none"
    mpl.rcParams["image.resample"] = False


def pixelated_image(
    image: np.ndarray,
    *,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    caption: str | None = None,
) -> Any:
    """Render a numeric image for fluid layouts (fills parent, keeps aspect)."""
    import marimo as mo

    return mo.image(
        np.asarray(image),
        vmin=vmin,
        vmax=vmax,
        caption=caption,
        style=PIXELATED_STYLE,
    )


def _grid_cell(scope: str, inner_html: str) -> str:
    return (
        f'<div class="{scope}-cell">'
        f'<div class="{scope}-tile">{inner_html}</div>'
        f"</div>"
    )


def pixelated_grid(
    panels: list[tuple[str, np.ndarray]],
    *,
    columns: int = 2,
    cell_min_height: str = "min(45vh, 520px)",
) -> Any:
    """Grid of pixelated images; each cell sizes the image with preserved aspect."""
    import marimo as mo
    from marimo._output.formatting import as_html

    scope = "render-model-grid"
    cells_html = []
    for title, image in panels:
        arr = np.asarray(image)
        caption = f"{title}  ({arr.shape[1]}×{arr.shape[0]})"
        tile = as_html(pixelated_image(arr, caption=caption)).text
        cells_html.append(_grid_cell(scope, tile))

    grid_style = (
        "display:grid;"
        f"grid-template-columns:repeat({columns},minmax(0,1fr));"
        "gap:0.75rem;"
        "width:100%;"
    )
    cell_css = (
        f".{scope}-cell{{min-height:{cell_min_height};display:flex;"
        f"flex-direction:column;overflow:hidden}}"
        f".{scope}-tile{{flex:1;min-height:0;display:flex;flex-direction:column}}"
        f".{scope}-tile figure{{flex:1;min-height:0;display:flex;"
        f"flex-direction:column;margin:0}}"
        f".{scope}-tile img{{flex:1;min-height:0;width:100%;height:100%;"
        f"object-fit:contain;image-rendering:pixelated}}"
    )

    return mo.Html(
        f"<style>{cell_css}</style>"
        f'<div class="{scope}" style="{grid_style}">'
        f'{"".join(cells_html)}'
        f"</div>"
    )


def pixelated_figure(fig: Any) -> Any:
    """Render a matplotlib figure in marimo with inline nearest-neighbor scaling."""
    import marimo as mo

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    return mo.image(buf, style=PIXELATED_STYLE)
