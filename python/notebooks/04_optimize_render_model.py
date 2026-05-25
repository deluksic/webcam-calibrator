"""Optimize homography (+ optional camera) against a render_model target."""

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
        OptimizeRenderConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        centered_diff_limits,
        corner_rmse,
        corners_from_homography,
        imshow_extent,
        model_params_to_vector,
        numpy_dlt_homography,
        numpy_sample_normal_offset_corners,
        numpy_sample_uniform_normal_offset_corners,
        optimize_render_model,
        render_with_model,
        vector_to_model_params,
    )

    return (
        OptimizeRenderConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        centered_diff_limits,
        corner_rmse,
        corners_from_homography,
        imshow_extent,
        jnp,
        model_params_to_vector,
        np,
        numpy_dlt_homography,
        numpy_sample_normal_offset_corners,
        numpy_sample_uniform_normal_offset_corners,
        optimize_render_model,
        plt,
        render_with_model,
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
        psf_sigma=jnp.float32(0.9),
        sharpen_amount=jnp.float32(0.6),
        sharpen_sigma=jnp.float32(1.1),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.85),
    )
    camera_init = RenderModelParams(
        psf_sigma=jnp.float32(1.1),
        sharpen_amount=jnp.float32(0.5),
        sharpen_sigma=jnp.float32(1.5),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.05),
        white_level=jnp.float32(1.0),
    )
    return (
        H_gt,
        camera_gt,
        camera_init,
        gt_corners,
        image_height,
        image_width,
        src_corners,
        tag_pattern,
    )


@app.cell
def _(
    H_gt,
    bbox_mask,
    camera_gt,
    corners_from_homography,
    image_height,
    image_width,
    render_with_model,
    src_corners,
    tag_pattern,
):
    target = render_with_model(H_gt, tag_pattern, image_height, image_width, camera_gt)
    gt_image_corners = corners_from_homography(H_gt, src_corners)
    loss_mask = bbox_mask(gt_image_corners, image_height, image_width, margin=3.5)
    return loss_mask, target


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
    H_gt,
    H_init,
    OptimizeRenderConfig,
    camera_init,
    corner_rmse,
    gt_corners,
    image_height,
    image_width,
    loss_mask,
    optimize_render_model,
    src_corners,
    tag_pattern,
    target,
):
    config = OptimizeRenderConfig(
        learning_rate=5e-4,
        n_steps=1000,
        optimizer="adam",
        corner_weight=1.0,
        loss_mask=loss_mask,
        optimize_homography=True,
        optimize_camera=True,
        camera_lr_scale=5.0,
    )
    H_opt, camera_opt, losses = optimize_render_model(
        H_init,
        target,
        tag_pattern,
        image_height,
        image_width,
        camera_init=camera_init,
        config=config,
        src_corners=src_corners,
        target_corners=gt_corners,
    )
    init_corner_err = float(corner_rmse(H_init, src_corners, gt_corners))
    opt_corner_err = float(corner_rmse(H_opt, src_corners, gt_corners))
    gt_corner_err = float(corner_rmse(H_gt, src_corners, gt_corners))
    return (
        H_opt,
        camera_opt,
        gt_corner_err,
        init_corner_err,
        losses,
        opt_corner_err,
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
    model_params_to_vector,
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

            **Final loss:** {losses[-1]:.6f} ({len(losses)} steps)

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
    _ax_lin.set_xlabel("step")
    _ax_lin.set_ylabel("combined loss")
    _ax_lin.set_title("Linear")
    _ax_lin.grid(True, alpha=0.3)

    _ax_log.semilogy(_steps, losses, color="tab:orange")
    _ax_log.set_xlabel("step")
    _ax_log.set_ylabel("combined loss")
    _ax_log.set_title("Log")
    _ax_log.grid(True, alpha=0.3, which="both")

    _fig_loss.suptitle("Joint H + camera optimization", fontsize=12)
    _fig_loss.tight_layout()
    _fig_loss
    return


@app.cell
def _(
    H_init,
    H_opt,
    corners_from_homography,
    gt_corners,
    image_height,
    image_width,
    imshow_extent,
    np,
    plt,
    src_corners,
    target,
):
    _pred_init = np.asarray(corners_from_homography(H_init, src_corners))
    _pred_opt = np.asarray(corners_from_homography(H_opt, src_corners))
    _gt_np = np.asarray(gt_corners)
    _corner_exaggeration = 50.0
    _vis_init = _gt_np + _corner_exaggeration * (_pred_init - _gt_np)
    _vis_opt = _gt_np + _corner_exaggeration * (_pred_opt - _gt_np)
    _extent = imshow_extent(image_width, image_height)

    _fig_corners, _ax_corners = plt.subplots(figsize=(6, 4.5))
    _ax_corners.imshow(
        np.asarray(target),
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=_extent,
        origin="upper",
    )
    for _corners, _color, _label in [
        (_gt_np, "lime", "GT corners"),
        (_vis_init, "red", f"Init error ×{_corner_exaggeration:.0f}"),
        (_vis_opt, "cyan", f"Optimized error ×{_corner_exaggeration:.0f}"),
    ]:
        _ax_corners.plot(
            np.r_[_corners[:, 0], _corners[0, 0]],
            np.r_[_corners[:, 1], _corners[0, 1]],
            color=_color,
            linewidth=1.5,
            label=_label,
            zorder=5,
        )
    _ax_corners.set_title("Corner reprojection (on target)")
    _ax_corners.axis("off")
    _ax_corners.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8)
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
