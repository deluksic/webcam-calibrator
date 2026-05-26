"""Diagnostics for how black/white levels affect the render and masked LM loss."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from render_model.pipeline import RenderModelParams, render_with_model
from render_model.renderer import render_tag_antialiased


class BlackWhiteGradientMaps(NamedTuple):
    """Spatial Jacobians and LM loss contributions for black / white levels."""

    render: jnp.ndarray
    residual: jnp.ndarray
    d_render_d_black: jnp.ndarray
    d_render_d_white: jnp.ndarray
    loss_contrib_black: jnp.ndarray
    loss_contrib_white: jnp.ndarray
    d_aa_d_black: jnp.ndarray | None
    d_aa_d_white: jnp.ndarray | None


def _masked_residual_weights(
    loss_mask: jnp.ndarray | None,
    height: int,
    width: int,
) -> jnp.ndarray | None:
    if loss_mask is None:
        return None
    return jnp.sqrt(loss_mask / jnp.maximum(jnp.sum(loss_mask), 1.0))


def _params_with_levels(
    params: RenderModelParams,
    *,
    black_level: jax.Array,
    white_level: jax.Array,
) -> RenderModelParams:
    return RenderModelParams(
        psf_sigma=params.psf_sigma,
        sharpen_amount=params.sharpen_amount,
        sharpen_sigma=params.sharpen_sigma,
        gamma=params.gamma,
        black_level=black_level,
        white_level=white_level,
        light_grad_u=params.light_grad_u,
        light_grad_v=params.light_grad_v,
    )


def compute_black_white_gradient_maps(
    H: jnp.ndarray,
    tag_pattern: jnp.ndarray,
    height: int,
    width: int,
    params: RenderModelParams,
    target: jnp.ndarray,
    *,
    loss_mask: jnp.ndarray | None = None,
    include_aa_stage: bool = True,
) -> BlackWhiteGradientMaps:
    """Jacobian images ``∂render/∂black``, ``∂render/∂white`` and per-pixel LM contributions.

    For masked MSE with residuals ``r = w * (render - target)`` and ``loss = Σ r²``:

    ``loss_contrib_black = 2 * r * (∂render/∂black)`` (summed over pixels → ``d loss / d black``).

    ``include_aa_stage`` adds pre-PSF AA Jacobians (where ``level = black + module * (white - black)``).
    """
    weights = _masked_residual_weights(loss_mask, height, width)

    def render_at_levels(black_level: jax.Array, white_level: jax.Array) -> jnp.ndarray:
        return render_with_model(
            H,
            tag_pattern,
            height,
            width,
            _params_with_levels(params, black_level=black_level, white_level=white_level),
        )

    def aa_at_levels(black_level: jax.Array, white_level: jax.Array) -> jnp.ndarray:
        return render_tag_antialiased(
            H,
            tag_pattern,
            height,
            width,
            black_level=black_level,
            white_level=white_level,
        )

    black = params.black_level
    white = params.white_level
    render = render_at_levels(black, white)
    d_render_d_black = jax.jacfwd(lambda b: render_at_levels(b, white))(black)
    d_render_d_white = jax.jacfwd(lambda w: render_at_levels(black, w))(white)

    if weights is None:
        residual = (render - target) / jnp.sqrt(jnp.maximum(render.size, 1))
    else:
        residual = (render - target) * weights

    loss_contrib_black = 2.0 * residual * d_render_d_black
    loss_contrib_white = 2.0 * residual * d_render_d_white

    d_aa_d_black = d_aa_d_white = None
    if include_aa_stage:
        aa = aa_at_levels(black, white)
        d_aa_d_black = jax.jacfwd(lambda b: aa_at_levels(b, white))(black)
        d_aa_d_white = jax.jacfwd(lambda w: aa_at_levels(black, w))(white)

    return BlackWhiteGradientMaps(
        render=render,
        residual=residual,
        d_render_d_black=d_render_d_black,
        d_render_d_white=d_render_d_white,
        loss_contrib_black=loss_contrib_black,
        loss_contrib_white=loss_contrib_white,
        d_aa_d_black=d_aa_d_black,
        d_aa_d_white=d_aa_d_white,
    )


def _nan_outside_mask(img: np.ndarray, loss_mask: np.ndarray | None) -> np.ndarray:
    """Hide pixels outside the training mask (NaN → transparent in diverging cmap)."""
    if loss_mask is None:
        return img
    out = np.asarray(img, dtype=np.float64).copy()
    out[np.asarray(loss_mask, dtype=np.float32) <= 0.0] = np.nan
    return out


def plot_black_white_gradient_maps(
    maps: BlackWhiteGradientMaps,
    *,
    width: int,
    height: int,
    extent: tuple[float, float, float, float] | None = None,
    loss_mask: jnp.ndarray | None = None,
    title: str = "Black / white level gradients",
) -> object:
    """Matplotlib figure: render, residuals, and gradient / contribution heatmaps.

    **Colormap (``RdBu_r``):** dark red = large **positive** value; dark blue = large
    **negative**; white ≈ 0. Symmetric limits ±``lim`` per row.

    **``∂render/∂·`` rows** show the full-frame Jacobian — every output pixel changes
    when black/white change (PSF spreads sensitivity outside the loss mask).

    **``2r·∂render/∂·`` rows** are masked in the plot (NaN outside ``loss_mask``); only
    masked pixels enter ``d(loss)/d(level)`` because ``r = w·(render − target)`` and
    ``w = 0`` outside the mask.
    """
    import matplotlib.pyplot as plt

    from render_model.display import centered_diff_limits, imshow_extent

    if extent is None:
        extent = imshow_extent(width, height)
    mask_np = None if loss_mask is None else np.asarray(loss_mask, dtype=np.float32)

    render = np.asarray(maps.render)
    residual = _nan_outside_mask(np.asarray(maps.residual), mask_np)
    d_rb = np.asarray(maps.d_render_d_black)
    d_rw = np.asarray(maps.d_render_d_white)
    c_b = _nan_outside_mask(np.asarray(maps.loss_contrib_black), mask_np)
    c_w = _nan_outside_mask(np.asarray(maps.loss_contrib_white), mask_np)
    d_aa_b = None if maps.d_aa_d_black is None else np.asarray(maps.d_aa_d_black)
    d_aa_w = None if maps.d_aa_d_white is None else np.asarray(maps.d_aa_d_white)

    has_aa = d_aa_b is not None
    nrows = 4 if has_aa else 3
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 3.0 * nrows), squeeze=False)
    cbar_label = "RdBu: red +, blue −"

    def _show(
        ax,
        img,
        *,
        cmap: str,
        panel_title: str,
        sym: bool = False,
        lim: float | None = None,
        cbar: bool = False,
    ):
        vmin = vmax = None
        if sym:
            finite = np.isfinite(img)
            lim = lim if lim is not None else centered_diff_limits(
                img[finite] if np.any(finite) else img
            )
            vmin, vmax = -lim, lim
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color="0.15")
        im = ax.imshow(
            img,
            cmap=cm,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        ax.set_title(panel_title, fontsize=9)
        ax.axis("off")
        if cbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        return im

    lim_full = centered_diff_limits(
        np.maximum(np.abs(d_rb), np.abs(d_rw)), mask=mask_np
    )

    _show(axes[0, 0], render, cmap="gray", panel_title="Render (full pipeline)")
    _show(axes[0, 1], residual, cmap="RdBu_r", panel_title="r (masked, outside=blank)", sym=True)
    if mask_np is not None:
        _show(axes[0, 2], mask_np, cmap="gray", panel_title="Loss mask", sym=False)
    else:
        axes[0, 2].axis("off")

    _show(
        axes[1, 0],
        d_rb,
        cmap="RdBu_r",
        panel_title="∂render/∂black (full frame)",
        sym=True,
        lim=lim_full,
        cbar=True,
    )
    _show(axes[1, 1], d_rw, cmap="RdBu_r", panel_title="∂render/∂white (full frame)", sym=True, lim=lim_full)
    _show(
        axes[1, 2],
        c_b,
        cmap="RdBu_r",
        panel_title="2r·∂render/∂black (in mask)",
        sym=True,
        lim=centered_diff_limits(c_b, mask=mask_np),
    )

    _show(
        axes[2, 0],
        c_w,
        cmap="RdBu_r",
        panel_title="2r·∂render/∂white (in mask)",
        sym=True,
        lim=centered_diff_limits(c_w, mask=mask_np),
    )
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    c_b_raw = np.asarray(maps.loss_contrib_black)
    c_w_raw = np.asarray(maps.loss_contrib_white)
    if mask_np is not None:
        in_mask = mask_np > 0
        sum_cb = float(np.sum(c_b_raw[in_mask]))
        sum_cw = float(np.sum(c_w_raw[in_mask]))
    else:
        sum_cb = float(np.sum(c_b_raw))
        sum_cw = float(np.sum(c_w_raw))
    axes[2, 1].text(
        0.02,
        0.55,
        (
            f"Σ 2r·∂/∂black (mask) = {sum_cb:.4e}\n"
            f"Σ 2r·∂/∂white (mask) = {sum_cw:.4e}\n"
            f"Σ ∂render/∂black (all px) = {float(np.sum(d_rb)):.4e}\n"
            f"Σ ∂render/∂white (all px) = {float(np.sum(d_rw)):.4e}"
        ),
        transform=axes[2, 1].transAxes,
        fontsize=9,
        va="center",
    )

    if has_aa and d_aa_w is not None:
        lim_aa = centered_diff_limits(
            np.maximum(np.abs(d_aa_b), np.abs(d_aa_w)), mask=mask_np
        )
        _show(axes[3, 0], d_aa_b, cmap="RdBu_r", panel_title="∂AA/∂black (full frame)", sym=True, lim=lim_aa)
        _show(axes[3, 1], d_aa_w, cmap="RdBu_r", panel_title="∂AA/∂white (full frame)", sym=True, lim=lim_aa)
        axes[3, 2].axis("off")
        axes[3, 2].text(
            0.02,
            0.5,
            "∂render rows: sensitivity of every output pixel.\n"
            "Loss uses r = w·(render−target); w=0 outside mask.\n"
            "PSF/sharpen spread ∂render outside the quad.",
            transform=axes[3, 2].transAxes,
            fontsize=9,
            va="center",
        )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig
