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
        bbox_mask,
        build_tag_pattern,
        camera_params_physical_vector,
        centered_diff_limits,
        configure_matplotlib_image_display,
        corners_from_homography,
        homography_from_corners,
        imshow_extent,
        optimize_render_model_lm,
        optimize_render_model_lm_with_param_trace,
        render_with_model,
        vector_to_model_params,
    )

    configure_matplotlib_image_display()
    return (
        Image,
        JOINT_PARAM_NAMES,
        OptimizeLMConfig,
        Path,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        centered_diff_limits,
        corners_from_homography,
        homography_from_corners,
        imshow_extent,
        jax,
        jnp,
        json,
        np,
        optimize_render_model_lm,
        optimize_render_model_lm_with_param_trace,
        plt,
        render_with_model,
        time,
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
    compile_jit = mo.ui.run_button(label="Compile (warmup JIT)")
    run_optimization = mo.ui.run_button(label="Run LM optimization")
    mo.vstack([
        image_selector,
        compile_jit,
        run_optimization,
        mo.md(
            "1. Pick an image · 2. **Compile** (JIT, untimed) · "
            "3. **Run LM optimization** (execution only)"
        ),
    ])
    return compile_jit, image_selector, run_optimization


@app.cell
def _(
    Image,
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    bbox_mask,
    build_tag_pattern,
    corners_from_homography,
    corners_json_path,
    homography_from_corners,
    image_selector,
    jnp,
    json,
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
    _loss_mask = bbox_mask(_init_corners, _height, _width, margin=3.5)
    _target = jnp.asarray(_photo, dtype=jnp.float32)
    _camera_init = RenderModelParams(
        psf_sigma=jnp.float32(0.5),
        sharpen_amount=jnp.float32(1.0),
        sharpen_sigma=jnp.float32(1.0),
        gamma=jnp.float32(2.0),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.8),
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
    camera_init = _camera_init
    return (
        H_init,
        camera_init,
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
def _(image_name, mo):
    get_warm_key, set_warm_key = mo.state("")

    if get_warm_key() and get_warm_key() != image_name:
        set_warm_key("")
    return get_warm_key, set_warm_key


@app.cell
def _(OptimizeLMConfig, loss_mask):
    lm_config = OptimizeLMConfig(
        n_steps=100,
        loss_mask=loss_mask,
    )
    return (lm_config,)


@app.cell
def _(
    H_init,
    camera_init,
    compile_jit,
    get_warm_key,
    image_height,
    image_name,
    image_width,
    jax,
    lm_config,
    mo,
    optimize_render_model_lm,
    set_warm_key,
    tag_pattern,
    target,
    time,
):
    _status = mo.md("Click **Compile (warmup JIT)** before running LM optimization.")

    if compile_jit.value:
        _t0 = time.perf_counter()
        _H_lm, _camera_lm, _losses_lm = optimize_render_model_lm(
            H_init,
            target,
            tag_pattern,
            image_height,
            image_width,
            camera_init=camera_init,
            config=lm_config,
        )
        jax.block_until_ready(_H_lm)
        _compile_s = time.perf_counter() - _t0
        set_warm_key(image_name)
        _status = mo.md(
            f"**Compiled** for `{image_name}` — LM JIT: {_compile_s:.2f}s\n\n"
            "Now click **Run LM optimization**."
        )
    elif get_warm_key() == image_name:
        _status = mo.md(f"**Already compiled** for `{image_name}` — ready to run.")

    _status
    return


@app.cell
def _(get_warm_key, image_name, mo, run_optimization):
    mo.stop(
        get_warm_key() != image_name,
        mo.md(f"Compile JIT for `{image_name}` first."),
    )
    mo.stop(
        not run_optimization.value,
        mo.md("Click **Run LM optimization**."),
    )
    return


@app.cell
def _(
    H_init,
    camera_init,
    image_height,
    image_width,
    jax,
    lm_config,
    optimize_render_model_lm_with_param_trace,
    tag_pattern,
    target,
    time,
):
    _t0 = time.perf_counter()
    _lm_trace = optimize_render_model_lm_with_param_trace(
        H_init,
        target,
        tag_pattern,
        image_height,
        image_width,
        camera_init=camera_init,
        config=lm_config,
    )
    jax.block_until_ready(_lm_trace.H)
    lm_wall_s = time.perf_counter() - _t0
    H_lm = _lm_trace.H
    camera_lm = _lm_trace.camera
    losses_lm = _lm_trace.losses
    params_lm = _lm_trace.params_physical_per_step
    return H_lm, camera_lm, lm_wall_s, losses_lm, params_lm


@app.cell
def _(camera_init, camera_lm, image_name, lm_wall_s, losses_lm, mo, tag_id):
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
        - **Exec time (post-JIT):** {lm_wall_s:.2f}s

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
