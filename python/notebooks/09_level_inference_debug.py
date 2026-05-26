"""Debug black/white level inference: masks, target means, pipeline vs linear levels."""

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
        TAG_CANONICAL_CORNERS,
        RenderModelParams,
        apply_gamma,
        apply_gaussian_psf,
        apply_sharpening,
        build_tag_pattern,
        configure_matplotlib_image_display,
        corners_from_corner_shifts,
        corners_from_homography,
        corner_shifts_from_homography,
        homography_from_corners,
        imshow_extent,
        invert_gamma,
        numpy_dlt_homography,
        render_tag_antialiased,
        render_with_model_stages,
        tag_cell_modules_at_pixels,
        loss_mask_from_corners,
    )
    from render_model.level_inference import (
        _LINEAR_BLACK_MAX,
        _LINEAR_WHITE_MIN,
        aa_black_white_sample_masks,
        camera_with_inferred_levels,
        infer_black_white_from_target,
    )

    configure_matplotlib_image_display()
    return (
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        aa_black_white_sample_masks,
        apply_gamma,
        apply_gaussian_psf,
        apply_sharpening,
        build_tag_pattern,
        camera_with_inferred_levels,
        corner_shifts_from_homography,
        corners_from_homography,
        imshow_extent,
        infer_black_white_from_target,
        invert_gamma,
        jnp,
        loss_mask_from_corners,
        np,
        numpy_dlt_homography,
        plt,
        render_tag_antialiased,
        render_with_model_stages,
        tag_cell_modules_at_pixels,
    )


@app.cell
def _(mo):
    mo.md("""
    # Level inference debug

    Same synthetic setup as notebook **04**. Masks: PSF-blurred AA render (fixed
    **σ = 0.8**), **loss_mask** interior (same as LM), linear **&lt; 0.2** → black,
    **&gt; 0.8** → white.
    Inference averages **inv-γ(target)** per pixel (current **γ**), then mean.

    Remaining gaps: sharpen (not inverted), wrong **H**, AA render still uses current
    black/white for the classification image.
    """)
    return


@app.cell
def _(
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    build_tag_pattern,
    corners_from_homography,
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
        sharpen_amount=jnp.float32(1.0),
        sharpen_sigma=jnp.float32(0.8),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.85),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )
    camera_init = RenderModelParams(
        psf_sigma=jnp.float32(1.5),
        sharpen_amount=jnp.float32(0.5),
        sharpen_sigma=jnp.float32(1.5),
        gamma=jnp.float32(2.2),
        black_level=jnp.float32(0.1),
        white_level=jnp.float32(0.9),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )
    ref_corners = corners_from_homography(H_gt, src_corners)
    return (
        H_gt,
        camera_gt,
        camera_init,
        gt_corners,
        image_height,
        image_width,
        ref_corners,
        src_corners,
        tag_pattern,
    )


@app.cell
def _(
    H_gt,
    camera_gt,
    corner_shifts_from_homography,
    gt_corners,
    image_height,
    image_width,
    jnp,
    np,
    numpy_dlt_homography,
    ref_corners,
    render_with_model_stages,
    src_corners,
    tag_pattern,
):
    # Init H: small corner noise like notebook 04
    rng = np.random.default_rng(0)
    noisy = gt_corners + rng.normal(0, 1.0, gt_corners.shape).astype(np.float32)
    H_init = jnp.asarray(
        numpy_dlt_homography(src_corners, noisy), dtype=jnp.float32
    )
    shifts_init = corner_shifts_from_homography(H_init, ref_corners, src_corners)

    stages_gt = render_with_model_stages(
        H_gt, tag_pattern, image_height, image_width, camera_gt
    )
    target = stages_gt.final
    binned_gt = stages_gt.sharpened  # linear before gamma; sharpened input to gamma
    return H_init, stages_gt, target


@app.cell
def _(
    H_init,
    corners_from_homography,
    image_height,
    image_width,
    loss_mask_from_corners,
    src_corners,
):
    # Same loss region as notebooks 04 / 08 (init quad in image space).
    init_image_corners = corners_from_homography(H_init, src_corners)
    loss_mask = loss_mask_from_corners(init_image_corners, image_height, image_width)
    return (loss_mask,)


@app.cell
def _(mo):
    level_mode = mo.ui.dropdown(
        options=["init (optimizer)", "GT (oracle masks)"],
        value="init (optimizer)",
        label="Levels used for AA render (classification)",
    )
    homography_mode = mo.ui.dropdown(
        options=["H_init", "H_gt"],
        value="H_init",
        label="Homography for warp / masks",
    )
    mo.vstack([level_mode, homography_mode])
    return homography_mode, level_mode


@app.cell
def _(
    H_gt,
    H_init,
    aa_black_white_sample_masks,
    apply_gaussian_psf,
    camera_gt,
    camera_init,
    homography_mode,
    image_height,
    image_width,
    infer_black_white_from_target,
    invert_gamma,
    jnp,
    level_mode,
    loss_mask,
    np,
    render_tag_antialiased,
    tag_cell_modules_at_pixels,
    tag_pattern,
    target,
):
    H = H_init if homography_mode.value == "H_init" else H_gt
    cam = camera_init if level_mode.value == "init (optimizer)" else camera_gt
    black = cam.black_level
    white = cam.white_level

    is_black, is_white = aa_black_white_sample_masks(
        H,
        tag_pattern,
        image_height,
        image_width,
        black,
        white,
        loss_mask=loss_mask,
    )
    inf_b, inf_w = infer_black_white_from_target(
        target, is_black, is_white, cam.gamma
    )

    aa = render_tag_antialiased(
        H,
        tag_pattern,
        image_height,
        image_width,
        black_level=black,
        white_level=white,
    )
    psf_aa = apply_gaussian_psf(aa, jnp.float32(0.8))

    _, valid = tag_cell_modules_at_pixels(H, tag_pattern, image_height, image_width)
    weight = valid * loss_mask

    def masked_mean(img, mask):
        s = float(jnp.sum(mask))
        if s < 1:
            return float("nan")
        return float(jnp.sum(img * mask) / s)

    target_np = np.asarray(target)
    target_linear = np.asarray(invert_gamma(target, cam.gamma))

    rows = {
        "GT linear levels": (float(camera_gt.black_level), float(camera_gt.white_level)),
        "Inferred (inv-γ, current masks)": (float(inf_b), float(inf_w)),
        "Mean display target @ is_black": (
            masked_mean(target, is_black),
            masked_mean(target, is_white),
        ),
        "Mean inv-γ(target) @ masks": (
            masked_mean(target_linear, is_black),
            masked_mean(target_linear, is_white),
        ),
    }
    return (
        aa,
        inf_b,
        inf_w,
        is_black,
        is_white,
        psf_aa,
        rows,
        target_np,
        weight,
    )


@app.cell
def _(camera_gt, mo, rows):
    summary_md_lines = [
        "| metric | black | white |",
        "| --- | ---: | ---: |",
    ]
    for metric_name, (black_val, white_val) in rows.items():
        summary_md_lines.append(
            f"| {metric_name} | {black_val:.4f} | {white_val:.4f} |"
        )
    summary_md_lines.append(
        f"\n**GT** black={float(camera_gt.black_level):.3f}, white={float(camera_gt.white_level):.3f}"
    )
    mo.md("\n".join(summary_md_lines))
    return


@app.cell
def _(
    aa,
    camera_gt,
    image_height,
    image_width,
    imshow_extent,
    inf_b,
    inf_w,
    is_black,
    is_white,
    np,
    plt,
    psf_aa,
    target_np,
    weight,
):
    mask_fig, mask_axes = plt.subplots(2, 4, figsize=(14, 7))
    extent = imshow_extent(image_width, image_height)
    panels = [
        ("target (γ)", target_np, "gray"),
        ("AA linear", np.asarray(aa), "gray"),
        ("PSF(AA)", np.asarray(psf_aa), "gray"),
        ("loss_mask × valid", np.asarray(weight), "gray"),
        ("is_black mask", np.asarray(is_black), "gray"),
        ("is_white mask", np.asarray(is_white), "gray"),
        ("target (black samples)", np.where(np.asarray(is_black) > 0, target_np, np.nan), "gray"),
        ("target (white samples)", np.where(np.asarray(is_white) > 0, target_np, np.nan), "gray"),
    ]
    for panel_ax, (title, img, cmap) in zip(mask_axes.ravel(), panels):
        panel_im = panel_ax.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=extent)
        panel_ax.set_title(title)
        panel_ax.set_aspect("equal")
        plt.colorbar(panel_im, ax=panel_ax, fraction=0.046)
    mask_fig.suptitle(
        f"inferred b={inf_b:.3f} w={inf_w:.3f}  "
        f"(GT b={float(camera_gt.black_level):.3f} w={float(camera_gt.white_level):.3f})  "
        f"PSF mask linear <{float(_LINEAR_BLACK_MAX)} / >{float(_LINEAR_WHITE_MIN)}",
        fontsize=11,
    )
    plt.tight_layout()
    mask_fig
    return


@app.cell
def _(is_black, is_white, np, plt, target_np):
    hist_fig, hist_axes = plt.subplots(1, 2, figsize=(10, 3.5))
    black_samples = target_np[np.asarray(is_black) > 0]
    white_samples = target_np[np.asarray(is_white) > 0]
    hist_axes[0].hist(black_samples, bins=40, color="k", alpha=0.7)
    hist_axes[0].set_title(f"target on is_black (n={black_samples.size})")
    hist_axes[0].set_xlabel("display value")
    hist_axes[1].hist(white_samples, bins=40, color="w", edgecolor="k", alpha=0.7)
    hist_axes[1].set_title(f"target on is_white (n={white_samples.size})")
    hist_axes[1].set_xlabel("display value")
    plt.tight_layout()
    hist_fig
    return


@app.cell
def _(
    apply_gamma,
    apply_gaussian_psf,
    apply_sharpening,
    camera_gt,
    image_height,
    image_width,
    is_black,
    is_white,
    jnp,
    np,
    stages_gt,
    target_np,
):
    """Pipeline-stage means on the *same* is_black/is_white masks (from prior cell)."""

    def mean_on(mask, arr):
        m = np.asarray(mask) > 0
        if not np.any(m):
            return float("nan")
        return float(np.mean(np.asarray(arr)[m]))

    stage_names = ["aa (hi)", "hi_blurred", "binned", "sharpened", "final (target)"]
    stage_arrays = [
        stages_gt.hi_res,
        stages_gt.hi_blurred,
        stages_gt.binned,
        stages_gt.sharpened,
        stages_gt.final,
    ]
    black_means = [mean_on(is_black, s) for s in stage_arrays]
    white_means = [mean_on(is_white, s) for s in stage_arrays]

    gamma = float(camera_gt.gamma)
    inv_gamma = gamma  # display = linear ** (1/gamma)  =>  linear = display ** gamma

    def inv_display(x):
        return np.power(np.maximum(x, 1e-6), inv_gamma)

    target_lin_black = mean_on(is_black, inv_display(target_np))
    target_lin_white = mean_on(is_white, inv_display(target_np))

    def flat_to_display(level):
        hi = jnp.broadcast_to(level, (image_height, image_width))
        blurred = apply_gaussian_psf(hi, camera_gt.psf_sigma)
        binned = blurred  # supersample=1
        sharp = apply_sharpening(binned, camera_gt.sharpen_amount, camera_gt.sharpen_sigma)
        return float(apply_gamma(sharp, camera_gt.gamma)[0, 0])

    flat_display_b = flat_to_display(camera_gt.black_level)
    flat_display_w = flat_to_display(camera_gt.white_level)

    pipeline_table = {
        "stage": stage_names + ["inv_γ(target) mean", "flat field → display"],
        "mean @ is_black": black_means + [target_lin_black, flat_display_b],
        "mean @ is_white": white_means + [target_lin_white, flat_display_w],
    }
    return (pipeline_table,)


@app.cell
def _(camera_gt, mo, pipeline_table):
    pipeline_md_lines = [
        "### Means on current `is_black` / `is_white` masks (GT render stages)",
        "",
        "| stage | mean @ black mask | mean @ white mask |",
        "| --- | ---: | ---: |",
    ]
    for stage_idx, stage_name in enumerate(pipeline_table["stage"]):
        black_mean = pipeline_table["mean @ is_black"][stage_idx]
        white_mean = pipeline_table["mean @ is_white"][stage_idx]
        pipeline_md_lines.append(
            f"| {stage_name} | {black_mean:.4f} | {white_mean:.4f} |"
        )
    pipeline_md_lines.append(
        f"\nGT **linear** levels: {float(camera_gt.black_level):.4f} / {float(camera_gt.white_level):.4f}"
    )
    mo.md("\n".join(pipeline_md_lines))
    return


@app.cell
def _(
    H_init,
    camera_init,
    camera_with_inferred_levels,
    image_height,
    image_width,
    loss_mask,
    tag_pattern,
    target,
):
    camera_inferred = camera_with_inferred_levels(
        camera_init,
        target,
        H_init,
        tag_pattern,
        image_height,
        image_width,
        loss_mask=loss_mask,
    )
    infer_packed_b = float(camera_inferred.black_level)
    infer_packed_w = float(camera_inferred.white_level)
    return infer_packed_b, infer_packed_w


@app.cell
def _(camera_gt, infer_packed_b, infer_packed_w, mo):
    mo.md(f"""
    ### `camera_with_inferred_levels` (H_init + camera_init)

    | | black | white |
    | --- | ---: | ---: |
    | GT | {float(camera_gt.black_level):.4f} | {float(camera_gt.white_level):.4f} |
    | Inferred init | {infer_packed_b:.4f} | {infer_packed_w:.4f} |
    """)
    return


if __name__ == "__main__":
    app.run()
