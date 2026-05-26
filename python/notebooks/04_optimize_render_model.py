"""Optimize homography + camera (LM) against a render_model target."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from render_model import (
        JOINT_PARAM_NAMES,
        OptimizeLMConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        loss_mask_from_corners,
        build_tag_pattern,
        centered_diff_limits,
        corner_rmse,
        corners_from_homography,
        imshow_extent,
        numpy_dlt_homography,
        numpy_sample_normal_offset_corners,
        numpy_sample_uniform_normal_offset_corners,
        optimize_render_model_lm_with_param_trace,
        camera_with_inferred_levels,
        compute_black_white_gradient_maps,
        plot_black_white_gradient_maps,
        render_with_model,
        write_lm_trace_gif,
    )

    return (
        JOINT_PARAM_NAMES,
        OptimizeLMConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        build_tag_pattern,
        camera_with_inferred_levels,
        centered_diff_limits,
        compute_black_white_gradient_maps,
        corner_rmse,
        corners_from_homography,
        imshow_extent,
        jax,
        jnp,
        loss_mask_from_corners,
        np,
        numpy_dlt_homography,
        numpy_sample_normal_offset_corners,
        numpy_sample_uniform_normal_offset_corners,
        optimize_render_model_lm_with_param_trace,
        plot_black_white_gradient_maps,
        plt,
        render_with_model,
        write_lm_trace_gif,
    )


@app.cell
def _(mo):
    mo.md("""
    # Optimize homography with render_model

    1. **Target** — GT homography + GT camera (black=0.1, white=0.9)
    2. **Init** — DLT from GT + corner error model (see `corner_init_model`), camera init
    3. **Optimize** — joint H + camera (can it recover contrast?)
    """)
    return


@app.cell
def _(
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    build_tag_pattern,
    jnp,
    np,
    numpy_dlt_homography,
):
    image_width = 80
    image_height = 60
    tag_pattern = build_tag_pattern(0)
    src_corners = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)

    gt_corners_ref = np.array(
        [[80.0, 60.0], [260.0, 45.0], [270.0, 190.0], [70.0, 200.0]],
        dtype=np.float64,
    )
    ref_size = np.array([320.0, 240.0], dtype=np.float64)
    gt_corners = (
        gt_corners_ref / ref_size * np.array([image_width, image_height], dtype=np.float64)
    ).astype(np.float32)

    H_gt = jnp.asarray(numpy_dlt_homography(src_corners, gt_corners), dtype=jnp.float32)
    camera_gt = RenderModelParams(
        psf_sigma=jnp.float32(0.8),
        sharpen_amount=jnp.float32(1),
        sharpen_sigma=jnp.float32(0.8),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.85),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )
    camera_init_seed = RenderModelParams(
        psf_sigma=jnp.float32(1.5),
        sharpen_amount=jnp.float32(0.5),
        sharpen_sigma=jnp.float32(1.5),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.0),
        white_level=jnp.float32(1.0),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )
    return (
        H_gt,
        camera_gt,
        camera_init_seed,
        gt_corners,
        image_height,
        image_width,
        src_corners,
        tag_pattern,
    )


@app.cell
def _(
    H_gt,
    camera_gt,
    corners_from_homography,
    image_height,
    image_width,
    loss_mask_from_corners,
    render_with_model,
    src_corners,
    tag_pattern,
):
    target = render_with_model(H_gt, tag_pattern, image_height, image_width, camera_gt)
    gt_image_corners = corners_from_homography(H_gt, src_corners)
    loss_mask = loss_mask_from_corners(gt_image_corners, image_height, image_width)
    return gt_image_corners, loss_mask, target


@app.cell
def _(
    gt_corners,
    jnp,
    np,
    numpy_dlt_homography,
    numpy_sample_normal_offset_corners,
    numpy_sample_uniform_normal_offset_corners,
    src_corners,
):
    # uniform_normal (1 DOF) | per_edge_normal (4) | isotropic (8)
    corner_init_model = "uniform_normal"
    corner_normal_sigma_px = 1.0
    rng = np.random.default_rng(0)
    if corner_init_model == "uniform_normal":
        noisy_corners = numpy_sample_uniform_normal_offset_corners(
            rng, gt_corners, sigma=corner_normal_sigma_px
        )
    elif corner_init_model == "per_edge_normal":
        noisy_corners = numpy_sample_normal_offset_corners(
            rng, gt_corners, sigma=corner_normal_sigma_px
        )
    elif corner_init_model == "isotropic":
        noisy_corners = gt_corners + 1
    else:
        raise ValueError(f"unknown corner_init_model: {corner_init_model!r}")
    H_init = jnp.asarray(numpy_dlt_homography(src_corners, noisy_corners), dtype=jnp.float32)
    return (H_init,)


@app.cell
def _(
    H_init,
    camera_init_seed,
    camera_with_inferred_levels,
    image_height,
    image_width,
    loss_mask,
    tag_pattern,
    target,
):
    camera_init = camera_with_inferred_levels(
        camera_init_seed,
        target,
        H_init,
        tag_pattern,
        image_height,
        image_width,
        loss_mask=loss_mask,
    )
    return (camera_init,)


@app.cell
def _(mo):
    mo.md("""
    ### Black / white level gradient maps

    **Colormap:** ``RdBu`` — **dark red = large positive**, dark blue = large negative, white ≈ 0.

    **``∂render/∂black`` / ``∂render/∂white``** are computed for **every** image pixel (full frame).
    Black/white still affect background/halo pixels and PSF spreads sensitivity **outside** the loss mask.

    **``2r·∂render/∂·``** is what actually drives levels in LM (``r = 0`` outside the mask). Those panels
    hide out-of-mask pixels. The loss mask is shown top-right.
    """)
    return


@app.cell
def _(
    H_gt,
    H_init,
    camera_gt,
    camera_init,
    compute_black_white_gradient_maps,
    image_height,
    image_width,
    imshow_extent,
    loss_mask,
    plot_black_white_gradient_maps,
    tag_pattern,
    target,
):
    _extent = imshow_extent(image_width, image_height)
    _maps_init = compute_black_white_gradient_maps(
        H_init,
        tag_pattern,
        image_height,
        image_width,
        camera_init,
        target,
        loss_mask=loss_mask,
    )
    _fig_init = plot_black_white_gradient_maps(
        _maps_init,
        width=image_width,
        height=image_height,
        extent=_extent,
        loss_mask=loss_mask,
        title="Init camera — black/white gradients",
    )
    _maps_gt = compute_black_white_gradient_maps(
        H_gt,
        tag_pattern,
        image_height,
        image_width,
        camera_gt,
        target,
        loss_mask=loss_mask,
    )
    _fig_gt = plot_black_white_gradient_maps(
        _maps_gt,
        width=image_width,
        height=image_height,
        extent=_extent,
        loss_mask=loss_mask,
        title="GT camera — black/white gradients",
    )
    _fig_init, _fig_gt
    return


@app.cell
def _(
    H_gt,
    H_init,
    OptimizeLMConfig,
    camera_init,
    corner_rmse,
    gt_corners,
    image_height,
    image_width,
    jax,
    loss_mask,
    optimize_render_model_lm_with_param_trace,
    src_corners,
    tag_pattern,
    target,
):
    config = OptimizeLMConfig(n_steps=50, loss_mask=loss_mask)
    lm_trace = optimize_render_model_lm_with_param_trace(
        H_init,
        target,
        tag_pattern,
        image_height,
        image_width,
        camera_init=camera_init,
        config=config,
    )
    jax.block_until_ready(lm_trace.H)
    H_opt = lm_trace.H
    camera_opt = lm_trace.camera
    losses = lm_trace.losses
    params_lm = lm_trace.params_physical_per_step
    init_corner_err = float(corner_rmse(H_init, src_corners, gt_corners))
    opt_corner_err = float(corner_rmse(H_opt, src_corners, gt_corners))
    gt_corner_err = float(corner_rmse(H_gt, src_corners, gt_corners))
    return (
        H_opt,
        camera_opt,
        gt_corner_err,
        init_corner_err,
        lm_trace,
        losses,
        opt_corner_err,
        params_lm,
    )


@app.cell
def _(
    camera_gt,
    camera_init,
    camera_opt,
    gt_corner_err,
    init_corner_err,
    losses,
    mo,
    opt_corner_err,
):
    def _fmt_camera(name, p):
        return (
            f"**{name}:** psf={float(p.psf_sigma):.3f}, "
            f"sharpen={float(p.sharpen_amount):.3f}, "
            f"sharpen σ={float(p.sharpen_sigma):.3f}, γ={float(p.gamma):.3f}, "
            f"black={float(p.black_level):.3f}, "
            f"white={float(p.white_level):.3f}"
        )

    mo.vstack([
        mo.md("### Results"),
        mo.md(
            f"""
            | Stage | Corner RMSE (px) |
            |-------|------------------|
            | Init (noisy DLT) | {init_corner_err:.4f} |
            | After optimize | {opt_corner_err:.4f} |
            | GT | {gt_corner_err:.4f} |

            **Final masked MSE:** {losses[-1]:.6f} ({len(losses) - 1} LM steps)

            {_fmt_camera("Camera GT", camera_gt)}

            {_fmt_camera("Camera init", camera_init)}

            {_fmt_camera("After optimize", camera_opt)}
            """
        ),
    ])
    return


@app.cell
def _(losses, plt):
    _fig_loss, (_ax_lin, _ax_log) = plt.subplots(1, 2, figsize=(10, 3))
    _steps = range(len(losses))

    _ax_lin.plot(_steps, losses, color="tab:orange")
    _ax_lin.set_xlabel("step (0 = init)")
    _ax_lin.set_ylabel("masked MSE")
    _ax_lin.set_title("Linear")
    _ax_lin.grid(True, alpha=0.3)

    _ax_log.semilogy(_steps, losses, color="tab:orange")
    _ax_log.set_xlabel("step (0 = init)")
    _ax_log.set_ylabel("masked MSE")
    _ax_log.set_title("Log")
    _ax_log.grid(True, alpha=0.3, which="both")

    _fig_loss.suptitle("Joint H + camera (LM)", fontsize=12)
    _fig_loss.tight_layout()
    _fig_loss
    return


@app.cell
def _(JOINT_PARAM_NAMES, np, params_lm, plt):
    _lm = np.asarray(params_lm)
    _fig, _axes = plt.subplots(4, 4, figsize=(14, 10), squeeze=False)

    for _i, _name in enumerate(JOINT_PARAM_NAMES):
        _ax = _axes[_i // 4, _i % 4]
        _y = _lm[:, _i]
        _ax.plot(_y, color="C0", alpha=0.9, lw=1.5)
        if _name == "sharpen":
            _ax.set_yscale("symlog", linthresh=1e-4)
        _ax.set_title(_name, fontsize=9)
        _ax.grid(True, alpha=0.25)

    for _j in range(len(JOINT_PARAM_NAMES), 16):
        _axes[_j // 4, _j % 4].axis("off")

    _fig.suptitle(f"LM parameter trajectories ({len(_lm) - 1} steps)", y=1.01)
    _fig.supxlabel("step (0 = init)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    image_height,
    image_width,
    lm_trace,
    mo,
    tag_pattern,
    target,
    write_lm_trace_gif,
):
    from pathlib import Path as _Path

    _gif_dir = _Path(__file__).parent / "_lm_gifs"
    _gif_path = write_lm_trace_gif(
        _gif_dir / "04_optimize_render.gif",
        lm_trace,
        tag_pattern,
        image_height,
        image_width,
        target=target,
        step_stride=2,
        duration_ms=100,
        scale=6,
    )
    mo.vstack([
        mo.md("### LM render animation (target | render per step)"),
        mo.image(_gif_path, width="100%"),
        mo.md(f"Saved to `{_gif_path}`"),
    ])
    return


@app.cell
def _(
    H_init,
    H_opt,
    corners_from_homography,
    gt_image_corners,
    image_height,
    image_width,
    imshow_extent,
    np,
    plt,
    src_corners,
    target,
):
    def _plot_quad(ax, corners, *, color, label, linewidth=1.5, linestyle="-"):
        ax.plot(
            np.r_[corners[:, 0], corners[0, 0]],
            np.r_[corners[:, 1], corners[0, 1]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=5,
        )

    _pred_init = np.asarray(corners_from_homography(H_init, src_corners))
    _pred_opt = np.asarray(corners_from_homography(H_opt, src_corners))
    _gt_np = np.asarray(gt_image_corners)

    _fig_corners, (_ax_true, _ax_err) = plt.subplots(1, 2, figsize=(11, 4.5))
    for _ax, _title in [(_ax_true, "Corner quads (pixels)"), (_ax_err, "Init / opt error (×20)")]:
        _ax.imshow(
            np.asarray(target),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=imshow_extent(image_width, image_height),
            origin="upper",
        )
        _ax.set_xlim(0.0, float(image_width))
        _ax.set_ylim(float(image_height), 0.0)
        _ax.set_aspect("equal")
        _ax.set_title(_title)
        _ax.axis("off")

    _plot_quad(_ax_true, _gt_np, color="lime", label="GT")
    _plot_quad(_ax_true, _pred_init, color="red", label="Init", linestyle="--")
    _plot_quad(_ax_true, _pred_opt, color="cyan", label="Optimized", linestyle="--")
    _ax_true.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8)

    _exag = 20.0
    _plot_quad(
        _ax_err,
        _gt_np + _exag * (_pred_init - _gt_np),
        color="red",
        label=f"Init error ×{_exag:.0f}",
    )
    _plot_quad(
        _ax_err,
        _gt_np + _exag * (_pred_opt - _gt_np),
        color="cyan",
        label=f"Opt error ×{_exag:.0f}",
    )
    _plot_quad(_ax_err, _gt_np, color="lime", label="GT", linewidth=1.0, linestyle=":")
    _ax_err.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8)

    _fig_corners.tight_layout()
    _fig_corners
    return


@app.cell
def _(
    H_init,
    H_opt,
    camera_init,
    camera_opt,
    centered_diff_limits,
    image_height,
    image_width,
    imshow_extent,
    np,
    plt,
    render_with_model,
    tag_pattern,
    target,
):
    _target_np = np.asarray(target)
    _render_init = np.asarray(
        render_with_model(H_init, tag_pattern, image_height, image_width, camera_init)
    )
    _render_opt = np.asarray(
        render_with_model(H_opt, tag_pattern, image_height, image_width, camera_opt)
    )
    _diff = _target_np - _render_opt
    _lim = centered_diff_limits(_diff)
    _extent = imshow_extent(image_width, image_height)

    _fig_cmp, _axes_cmp = plt.subplots(1, 4, figsize=(13, 3.5))
    _panels = [
        ("Initial (wrong camera)", _render_init, "gray", 0.0, 1.0),
        ("Optimized", _render_opt, "gray", 0.0, 1.0),
        ("Target", _target_np, "gray", 0.0, 1.0),
        ("Target − Optimized", _diff, "RdBu_r", -_lim, _lim),
    ]
    for _ax, (_title, _img, _cmap, _vmin, _vmax) in zip(_axes_cmp, _panels, strict=True):
        _im = _ax.imshow(
            _img,
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
            interpolation="nearest",
            extent=_extent,
            origin="upper",
        )
        _ax.set_title(_title)
        _ax.axis("off")
        if _title.startswith("Target −"):
            plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)

    _fig_cmp.tight_layout()
    _fig_cmp
    return


if __name__ == "__main__":
    app.run()
