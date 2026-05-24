"""Compare optimizers: step count to reach corner RMSE < threshold."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from render_model import (
        OptimizeRenderConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        corners_from_homography,
        default_params,
        first_step_corner_rmse_below,
        numpy_dlt_homography,
        numpy_sample_normal_offset_corners,
        numpy_sample_uniform_normal_offset_corners,
        optimize_render_model_with_trace,
        render_with_model,
    )

    return (
        OptimizeRenderConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        corners_from_homography,
        default_params,
        first_step_corner_rmse_below,
        jnp,
        np,
        numpy_dlt_homography,
        numpy_sample_normal_offset_corners,
        numpy_sample_uniform_normal_offset_corners,
        optimize_render_model_with_trace,
        plt,
        render_with_model,
    )


@app.cell
def _(mo):
    mo.md("""
    # Optimizer comparison

    Same init and target as `04_optimize_render_model` (40×30). For each optimizer, report
    raw numbers and overlay loss curves.
    """)
    return


@app.cell
def _(
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    bbox_mask,
    build_tag_pattern,
    corners_from_homography,
    default_params,
    jnp,
    np,
    numpy_dlt_homography,
    numpy_sample_normal_offset_corners,
    numpy_sample_uniform_normal_offset_corners,
    render_with_model,
):
    image_width = 40
    image_height = 30
    n_steps = 500
    corner_rmse_threshold_px = 0.05
    learning_rate = 2e-3
    camera_lr_scale = 0.1
    corner_weight = 0.5
    psf_continuation_start = 1.8

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
        black_level=jnp.float32(0.0),
        white_level=jnp.float32(1.0),
    )
    camera_init = default_params()

    target = render_with_model(H_gt, tag_pattern, image_height, image_width, camera_gt)
    gt_image_corners = corners_from_homography(H_gt, TAG_CANONICAL_CORNERS)
    loss_mask = bbox_mask(gt_image_corners, image_height, image_width, margin=2.0)

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
    else:
        noisy_corners = gt_corners + rng.uniform(
            -corner_normal_sigma_px,
            corner_normal_sigma_px,
            gt_corners.shape,
        ).astype(np.float32)
    H_init = jnp.asarray(numpy_dlt_homography(src_corners, noisy_corners), dtype=jnp.float32)
    return (
        H_init,
        camera_init,
        camera_lr_scale,
        corner_rmse_threshold_px,
        corner_weight,
        gt_corners,
        image_height,
        image_width,
        learning_rate,
        loss_mask,
        n_steps,
        psf_continuation_start,
        src_corners,
        tag_pattern,
        target,
    )


@app.cell
def _(
    H_init,
    OptimizeRenderConfig,
    camera_init,
    camera_lr_scale,
    corner_rmse_threshold_px,
    corner_weight,
    first_step_corner_rmse_below,
    gt_corners,
    image_height,
    image_width,
    learning_rate,
    loss_mask,
    n_steps,
    optimize_render_model_with_trace,
    psf_continuation_start,
    src_corners,
    tag_pattern,
    target,
):
    _variants = [
        ("adam + psf continuation", "adam", psf_continuation_start),
        ("adam", "adam", None),
        ("adamw + psf continuation", "adamw", psf_continuation_start),
        ("rmsprop + psf continuation", "rmsprop", psf_continuation_start),
        ("sgd + psf continuation", "sgd", psf_continuation_start),
        ("sgd_momentum + psf continuation", "sgd_momentum", psf_continuation_start),
    ]

    comparison_rows: list[dict[str, str | float | int]] = []
    comparison_traces: list[tuple[str, list[float], list[float]]] = []
    for _label, _optimizer, _psf_start in _variants:
        _config = OptimizeRenderConfig(
            learning_rate=learning_rate,
            n_steps=n_steps,
            optimizer=_optimizer,
            corner_weight=corner_weight,
            loss_mask=loss_mask,
            optimize_homography=True,
            optimize_camera=True,
            camera_lr_scale=camera_lr_scale,
            psf_continuation_start=_psf_start,
        )
        _trace = optimize_render_model_with_trace(
            H_init,
            target,
            tag_pattern,
            image_height,
            image_width,
            camera_init=camera_init,
            config=_config,
            src_corners=src_corners,
            target_corners=gt_corners,
        )
        _init_rmse = _trace.corner_rmse_per_step[0]
        _final_rmse = _trace.corner_rmse_per_step[-1]
        _final_loss = _trace.losses[-1]
        _first_below = first_step_corner_rmse_below(
            _trace.corner_rmse_per_step, corner_rmse_threshold_px
        )
        comparison_rows.append({
            "optimizer": _label,
            "init_corner_rmse_px": _init_rmse,
            "final_corner_rmse_px": _final_rmse,
            "final_loss": _final_loss,
            "first_step_index_below_0.05px": (
                "never" if _first_below is None else _first_below
            ),
            "n_steps": n_steps,
        })
        comparison_traces.append((
            _label,
            _trace.losses,
            _trace.corner_rmse_per_step,
        ))
    return comparison_rows, comparison_traces


@app.cell
def _(
    comparison_rows: list[dict[str, str | float | int]],
    corner_rmse_threshold_px,
    mo,
):
    _header = (
        "| optimizer | init RMSE (px) | final RMSE (px) | final loss | "
        "first step < 0.05 px | n_steps |\n"
        "|-----------|----------------|-----------------|------------|"
        "--------------------|---------|\n"
    )
    _body = ""
    for _row in comparison_rows:
        _first = _row["first_step_index_below_0.05px"]
        _body += (
            f"| {_row['optimizer']} "
            f"| {_row['init_corner_rmse_px']:.6f} "
            f"| {_row['final_corner_rmse_px']:.6f} "
            f"| {_row['final_loss']:.8f} "
            f"| {_first} "
            f"| {_row['n_steps']} |\n"
        )

    mo.md(
        f"""
        Threshold: corner RMSE **< {corner_rmse_threshold_px} px** (step index 0 = init, before updates).

        {_header}{_body}
        """
    )
    return


@app.cell
def _(comparison_traces: list[tuple[str, list[float], list[float]]], plt):
    _fig, (_ax_lin, _ax_log) = plt.subplots(1, 2, figsize=(12, 4))

    for _label, _losses, _ in comparison_traces:
        _steps = range(len(_losses))
        _ax_lin.plot(_steps, _losses, label=_label, linewidth=1.5)
        _ax_log.semilogy(_steps, _losses, label=_label, linewidth=1.5)

    _ax_lin.set_xlabel("step")
    _ax_lin.set_ylabel("combined loss")
    _ax_lin.set_title("Linear")
    _ax_lin.grid(True, alpha=0.3)
    _ax_lin.legend(fontsize=7, loc="upper right")

    _ax_log.set_xlabel("step")
    _ax_log.set_ylabel("combined loss")
    _ax_log.set_title("Log")
    _ax_log.grid(True, alpha=0.3, which="both")
    _ax_log.legend(fontsize=7, loc="upper right")

    _fig.suptitle("Optimizer comparison — loss", fontsize=12)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(comparison_traces: list[tuple[str, list[float], list[float]]], plt):
    _fig, (_ax_lin, _ax_log) = plt.subplots(1, 2, figsize=(12, 4))

    for _label, _, _corner_rmses in comparison_traces:
        _steps = range(len(_corner_rmses))
        _ax_lin.plot(_steps, _corner_rmses, label=_label, linewidth=1.5)
        _ax_log.semilogy(_steps, _corner_rmses, label=_label, linewidth=1.5)

    _ax_lin.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="0.05 px")
    _ax_log.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="0.05 px")

    _ax_lin.set_xlabel("step index (0 = init)")
    _ax_lin.set_ylabel("corner RMSE (px)")
    _ax_lin.set_title("Linear")
    _ax_lin.grid(True, alpha=0.3)
    _ax_lin.legend(fontsize=7, loc="upper right")

    _ax_log.set_xlabel("step index (0 = init)")
    _ax_log.set_ylabel("corner RMSE (px)")
    _ax_log.set_title("Log")
    _ax_log.grid(True, alpha=0.3, which="both")
    _ax_log.legend(fontsize=7, loc="upper right")

    _fig.suptitle("Optimizer comparison — corner RMSE", fontsize=12)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
