"""Fit real tag photos with Levenberg–Marquardt (saved corners only)."""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    from render_model import (
        JOINT_PARAM_NAMES,
        OptimizeLMConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        loss_mask_from_corners,
        build_tag_pattern,
        camera_params_physical_vector,
        centered_diff_limits,
        configure_matplotlib_image_display,
        corners_from_corner_shifts,
        corners_from_homography,
        homography_from_corners,
        imshow_extent,
        OptimizationParamTrace,
        camera_with_inferred_levels,
        compile_lm_run,
        decode_joint_params_per_step,
        make_render_model_residuals,
        model_params_to_vector,
        render_with_model,
        vector_to_model_params,
        write_lm_trace_gif,
    )

    configure_matplotlib_image_display()
    return (
        Image,
        JOINT_PARAM_NAMES,
        OptimizationParamTrace,
        OptimizeLMConfig,
        Path,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        build_tag_pattern,
        camera_with_inferred_levels,
        centered_diff_limits,
        compile_lm_run,
        corners_from_corner_shifts,
        corners_from_homography,
        decode_joint_params_per_step,
        homography_from_corners,
        imshow_extent,
        jax,
        jnp,
        json,
        loss_mask_from_corners,
        make_render_model_residuals,
        model_params_to_vector,
        np,
        plt,
        render_with_model,
        time,
        vector_to_model_params,
        write_lm_trace_gif,
    )


@app.cell
def _(Path):
    test_images_dir = Path(__file__).parent / "test_images"

    def corners_json_path(image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}.corners.json")

    def tag_id_from_filename(name: str) -> int:
        _stem = Path(name).stem
        if _stem.endswith("_1x"):
            _stem = _stem[: -len("_1x")]
        if not _stem.startswith("image_"):
            raise ValueError(f"Expected image_<id>_1x.png, got {name!r}")
        _tag_part = _stem.removeprefix("image_")
        if not _tag_part.isdigit():
            raise ValueError(f"Expected numeric tag id, got {name!r}")
        return int(_tag_part)

    _exts = {".png", ".jpg", ".jpeg", ".webp"}
    _all_paths = sorted(
        p for p in test_images_dir.iterdir() if p.suffix.lower() in _exts
    )
    ready_image_paths = [
        p for p in _all_paths if corners_json_path(p).is_file()
    ]
    if not ready_image_paths:
        raise FileNotFoundError(
            f"No images with .corners.json in {test_images_dir}"
        )
    return (
        corners_json_path,
        ready_image_paths,
        tag_id_from_filename,
        test_images_dir,
    )


@app.cell
def _(mo, ready_image_paths):
    _default_name = next(
        (p.name for p in ready_image_paths if p.name.endswith("_1x.png")),
        ready_image_paths[0].name,
    )
    image_selector = mo.ui.dropdown(
        options={p.name: p.name for p in ready_image_paths},
        value=_default_name,
        label="Image (saved corners only)",
        full_width=True,
    )
    run_lm = mo.ui.run_button(label="Run LM optimization")
    mo.vstack([
        image_selector,
        run_lm,
        mo.md(
            "Changing the image re-runs the **compile** cell (JAX JIT). "
            "Click **Run** to execute LM (timed in the run cell)."
        ),
    ])
    return image_selector, run_lm


@app.cell
def _(
    Image,
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    build_tag_pattern,
    corners_from_homography,
    corners_json_path,
    homography_from_corners,
    image_selector,
    jnp,
    json,
    loss_mask_from_corners,
    mo,
    np,
    ready_image_paths,
    tag_id_from_filename,
    test_images_dir,
):
    _image_name = image_selector.value
    _image_path = test_images_dir / _image_name
    if _image_path not in ready_image_paths:
        mo.stop(True, mo.md(f"No saved corners for `{_image_name}` — pick another image."))

    _corners_path = corners_json_path(_image_path)
    _photo = np.asarray(Image.open(_image_path).convert("L"), dtype=np.float32) / 255.0
    _height, _width = _photo.shape
    _init_corners_px = np.asarray(json.loads(_corners_path.read_text()), dtype=np.float32)
    _src_corners = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)
    _H_init = homography_from_corners(_src_corners, jnp.asarray(_init_corners_px))
    _tag_id = tag_id_from_filename(_image_name)
    _tag_pattern = build_tag_pattern(_tag_id)
    _init_corners = corners_from_homography(_H_init, _src_corners)
    _loss_mask = loss_mask_from_corners(_init_corners, _height, _width)
    _target = jnp.asarray(_photo, dtype=jnp.float32)
    camera_init_seed = RenderModelParams(
        psf_sigma=jnp.float32(1.5),
        sharpen_amount=jnp.float32(1.0),
        sharpen_sigma=jnp.float32(0.6),
        gamma=jnp.float32(1.75),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.8),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )

    image_name = _image_name
    photo = _photo
    image_height = _height
    image_width = _width
    H_init = _H_init
    tag_id = _tag_id
    tag_pattern = _tag_pattern
    loss_mask = _loss_mask
    target = _target
    return (
        H_init,
        camera_init_seed,
        image_height,
        image_name,
        image_width,
        loss_mask,
        photo,
        tag_id,
        tag_pattern,
        target,
    )


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
def _(
    H_init,
    OptimizeLMConfig,
    TAG_CANONICAL_CORNERS,
    camera_init,
    corners_from_homography,
    image_height,
    image_width,
    jnp,
    loss_mask,
    make_render_model_residuals,
    model_params_to_vector,
    tag_pattern,
    target,
):
    lm_config = OptimizeLMConfig(n_steps=20, loss_mask=loss_mask)
    ref_corners = corners_from_homography(H_init, TAG_CANONICAL_CORNERS)
    packed_init = jnp.concatenate(
        [
            jnp.zeros(8, dtype=jnp.float32),
            model_params_to_vector(camera_init),
        ]
    )
    residual_fn = make_render_model_residuals(
        target,
        tag_pattern,
        image_height,
        image_width,
        ref_corners=ref_corners,
        loss_mask=loss_mask,
    )
    lm_damping = jnp.asarray(lm_config.initial_damping, dtype=jnp.float32)
    init_loss = float(jnp.sum(residual_fn(packed_init) ** 2))
    return (
        init_loss,
        lm_config,
        lm_damping,
        packed_init,
        ref_corners,
        residual_fn,
    )


@app.cell
def _(compile_lm_run, jax, lm_config, lm_damping, packed_init, residual_fn):
    lm_run = compile_lm_run(residual_fn, lm_config)
    _warm_packed, _, _ = lm_run(packed_init, lm_damping)
    jax.block_until_ready(_warm_packed)
    return (lm_run,)


@app.cell
def _(
    OptimizationParamTrace,
    TAG_CANONICAL_CORNERS,
    corners_from_corner_shifts,
    decode_joint_params_per_step,
    homography_from_corners,
    init_loss,
    jax,
    jnp,
    lm_damping,
    lm_run,
    mo,
    packed_init,
    ref_corners,
    run_lm,
    time,
    vector_to_model_params,
):
    mo.stop(not run_lm.value, mo.md("Click **Run LM optimization**."))
    _t0 = time.perf_counter()
    packed, step_losses, param_hist = lm_run(packed_init, lm_damping)
    jax.block_until_ready(packed)
    lm_run_s = time.perf_counter() - _t0
    corners_px = corners_from_corner_shifts(ref_corners, packed[:8])
    H_lm = homography_from_corners(jnp.asarray(TAG_CANONICAL_CORNERS), corners_px)
    camera_lm = vector_to_model_params(packed[8:16])
    params_per_step = jnp.concatenate([packed_init[None, :], param_hist], axis=0)
    params_lm = decode_joint_params_per_step(params_per_step)
    losses_lm = [init_loss, *[float(x) for x in step_losses]]
    lm_trace = OptimizationParamTrace(
        H_lm,
        camera_lm,
        losses_lm,
        params_per_step,
        params_lm,
        ref_corners,
    )
    return H_lm, camera_lm, lm_run_s, lm_trace, losses_lm, params_lm


@app.cell
def _(camera_init, camera_lm, image_name, lm_run_s, losses_lm, mo, tag_id):
    def _fmt_cam(_label: str, _p) -> str:
        return (
            f"**{_label}:** psf={float(_p.psf_sigma):.3f}, "
            f"sharpen={float(_p.sharpen_amount):.3f}, "
            f"sharpen σ={float(_p.sharpen_sigma):.3f}, "
            f"γ={float(_p.gamma):.3f}, black={float(_p.black_level):.3f}, "
            f"white={float(_p.white_level):.3f}"
        )

    mo.md(
        f"""
        ### `{image_name}` (tag {tag_id})

        - **Steps:** {len(losses_lm) - 1}
        - **Final loss:** {losses_lm[-1]:.6f} (init {losses_lm[0]:.6f})
        - **Run (execute):** {lm_run_s:.2f}s (compile time = marimo runtime on the cell above)

        {_fmt_cam("Camera init", camera_init)}
        {_fmt_cam("LM", camera_lm)}
        """
    )
    return


@app.cell
def _(losses_lm, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(losses_lm, color="C0", alpha=0.9)
    _ax.set_yscale("log")
    _ax.set_xlabel("step (0 = init)")
    _ax.set_ylabel("masked MSE")
    _ax.set_title(f"LM loss ({len(losses_lm) - 1} steps)")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
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
    Path,
    image_height,
    image_name,
    image_width,
    lm_trace,
    mo,
    tag_pattern,
    target,
    write_lm_trace_gif,
):
    _gif_dir = Path(__file__).parent / "_lm_gifs"
    _stem = Path(image_name).stem
    _gif_path = write_lm_trace_gif(
        _gif_dir / f"08_{_stem}.gif",
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
        mo.md(f"### LM render animation — `{image_name}` (photo | render per step)"),
        mo.image(_gif_path, width="100%"),
        mo.md(f"Saved to `{_gif_path}`"),
    ])
    return


@app.cell(hide_code=True)
def _(
    H_init,
    H_lm,
    TAG_CANONICAL_CORNERS,
    corners_from_homography,
    image_height,
    image_width,
    imshow_extent,
    jnp,
    np,
    photo,
    plt,
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

    _src = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)
    _corners_init = np.asarray(corners_from_homography(H_init, _src))
    _corners_final = np.asarray(corners_from_homography(H_lm, _src))

    _fig_corners, _ax_corners = plt.subplots(figsize=(5.5, 4.5))
    _ax_corners.imshow(
        photo,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=imshow_extent(image_width, image_height),
        origin="upper",
    )
    _ax_corners.set_xlim(0.0, float(image_width))
    _ax_corners.set_ylim(float(image_height), 0.0)
    _ax_corners.set_aspect("equal")
    _ax_corners.set_title("Corners (pixels): initial vs final")
    _ax_corners.axis("off")
    _plot_quad(_ax_corners, _corners_init, color="red", label="Initial", linestyle="--")
    _plot_quad(_ax_corners, _corners_final, color="cyan", label="Final", linestyle="--")
    _ax_corners.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=8)
    _fig_corners.tight_layout()
    _fig_corners
    return


@app.cell(hide_code=True)
def _(
    H_init,
    H_lm,
    camera_init,
    camera_lm,
    centered_diff_limits,
    image_height,
    image_width,
    imshow_extent,
    loss_mask,
    np,
    photo,
    plt,
    render_with_model,
    tag_pattern,
):
    _init_render = np.asarray(
        render_with_model(H_init, tag_pattern, image_height, image_width, camera_init)
    )
    _lm_render = np.asarray(
        render_with_model(H_lm, tag_pattern, image_height, image_width, camera_lm)
    )
    _mask = np.asarray(loss_mask, dtype=bool)
    _diff_raw = photo - _lm_render
    _diff = np.where(_mask, _diff_raw, np.nan)
    _vdiff = centered_diff_limits(_diff_raw, mask=_mask)

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for _ax, _img, _title in [
        (_axes[0], _init_render, "Init render"),
        (_axes[1], _lm_render, "LM render"),
        (_axes[2], photo, "Photo"),
        (_axes[3], _diff, "Photo − LM"),
    ]:
        if _title.startswith("Photo −"):
            _im = _ax.imshow(
                _img,
                cmap="RdBu_r",
                vmin=-_vdiff,
                vmax=_vdiff,
                extent=imshow_extent(image_width, image_height),
            )
            _cb = _fig.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
            _cb.set_label(f"photo − LM  [−{_vdiff:.3f}, +{_vdiff:.3f}] (masked)")
        else:
            _ax.imshow(
                _img,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                extent=imshow_extent(image_width, image_height),
            )
        _ax.set_title(_title)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
