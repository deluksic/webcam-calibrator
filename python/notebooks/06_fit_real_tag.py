"""Fit render_model to a real photo — clicks are init only, no corner ground truth."""

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
    from pathlib import Path

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    from PIL import Image

    from render_model import (
        OptimizeRenderConfig,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        centered_diff_limits,
        configure_matplotlib_image_display,
        corners_from_homography,
        homography_from_corners,
        imshow_extent,
        model_params_to_vector,
        optimize_render_model,
        render_with_model,
    )

    configure_matplotlib_image_display()
    return (
        Image,
        OptimizeRenderConfig,
        Path,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        bbox_mask,
        build_tag_pattern,
        centered_diff_limits,
        corners_from_homography,
        go,
        homography_from_corners,
        imshow_extent,
        jnp,
        json,
        model_params_to_vector,
        np,
        optimize_render_model,
        plt,
        render_with_model,
    )


@app.cell
def _(Path):
    test_images_dir = Path(__file__).parent / "test_images"

    def corners_json_path(image_path: Path) -> Path:
        return image_path.with_name(f"{image_path.stem}.corners.json")

    def tag_id_from_filename(name: str) -> int:
        stem = Path(name).stem
        if stem.endswith("_1x"):
            stem = stem[: -len("_1x")]
        if not stem.startswith("image_"):
            raise ValueError(f"Expected image_<id>.png or image_<id>_1x.png, got {name!r}")
        tag_part = stem.removeprefix("image_")
        if not tag_part.isdigit():
            raise ValueError(f"Expected image_<id>.png or image_<id>_1x.png, got {name!r}")
        return int(tag_part)

    _image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    test_image_paths = sorted(
        p for p in test_images_dir.iterdir() if p.suffix.lower() in _image_exts
    )
    if not test_image_paths:
        raise FileNotFoundError(f"No images in {test_images_dir}")
    return (
        corners_json_path,
        tag_id_from_filename,
        test_image_paths,
        test_images_dir,
    )


@app.cell
def _(mo, test_image_paths):
    _default_name = next(
        (p.name for p in test_image_paths if p.name.endswith("_1x.png")),
        test_image_paths[0].name,
    )
    image_selector = mo.ui.dropdown(
        options={p.name: p.name for p in test_image_paths},
        value=_default_name,
        label="Test image",
        full_width=True,
    )
    image_selector
    return (image_selector,)


@app.cell
def _(
    Image,
    TAG_CANONICAL_CORNERS,
    build_tag_pattern,
    corners_json_path,
    image_selector,
    jnp,
    np,
    tag_id_from_filename,
    test_images_dir,
):
    image_name = image_selector.value
    image_path = test_images_dir / image_name
    corners_path = corners_json_path(image_path)
    tag_id = tag_id_from_filename(image_name)

    photo = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    image_height, image_width = photo.shape
    tag_pattern = build_tag_pattern(tag_id)
    src_corners = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)
    return (
        corners_path,
        image_height,
        image_name,
        image_width,
        photo,
        src_corners,
        tag_id,
        tag_pattern,
    )


@app.cell
def _(image_name, mo, tag_id):
    mo.md(f"""
    # Fit render_model to real photo

    **Image:** `{image_name}` · **Tag id:** {tag_id}

    Clicks are **initialization only** (not ground truth, not in the loss).

    1. Pick an image from `test_images/` (use `*_1x.png` for Retina captures)
    2. **Drag a small box** on each corner — TL → TR → BR → BL (Plotly heatmap)
    3. Check the preview, then click **Run optimization**
    """)
    return


@app.cell
def _(corners_path, json, mo):
    entries = None
    if corners_path.is_file():
        entries = json.loads(corners_path.read_text())

    use_saved = mo.ui.checkbox(
        value=entries is not None,
        label=(
            f"Use saved corners ({corners_path.name})"
            if entries is not None
            else f"No saved corners yet ({corners_path.name})"
        ),
    )
    mo.vstack([
        use_saved,
        mo.md(
            "Uncheck to click fresh corners. Saved to "
            "`<image_stem>.corners.json` next to the image."
        ),
    ])
    return entries, use_saved


@app.cell
def _(image_name, mo):
    get_corners, set_corners = mo.state([])
    get_last_sel, set_last_sel = mo.state(None)
    get_reset_clicks, set_reset_clicks = mo.state(0)
    get_corners_image, set_corners_image = mo.state("")

    if get_corners_image() != image_name:
        set_corners_image(image_name)
        set_corners([])
        set_last_sel(None)
        set_reset_clicks(0)
    return (
        get_corners,
        get_last_sel,
        get_reset_clicks,
        set_corners,
        set_last_sel,
        set_reset_clicks,
    )


@app.cell
def _(entries, use_saved):
    show_picker = not (use_saved.value and entries is not None)
    return (show_picker,)


@app.cell
def _(get_corners, go, image_height, image_width, mo, photo, show_picker):
    corner_picker = None
    reset_corners = None
    _picker_ui = mo.md("*Using saved corners — picker skipped.*")

    if show_picker:
        _corners = list(get_corners())

        _fig = go.Figure()
        _fig.add_trace(
            go.Heatmap(
                z=photo.tolist(),
                x=list(range(image_width)),
                y=list(range(image_height)),
                colorscale=[[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]],
                showscale=False,
                hoverinfo="skip",
            )
        )
        _fig.update_layout(
            title=f"Drag a box on each corner ({len(_corners)}/4)",
            xaxis=dict(range=[-0.5, image_width - 0.5], constrain="domain"),
            yaxis=dict(
                range=[image_height - 0.5, -0.5],
                scaleanchor="x",
                scaleratio=1,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=520,
            clickmode="event+select",
        )

        corner_picker = mo.ui.plotly(_fig)
        reset_corners = mo.ui.button(
            label="Reset corners",
            value=0,
            on_click=lambda n: n + 1,
        )
        _picker_ui = mo.vstack([
            corner_picker,
            reset_corners,
            mo.md(
                f"**{len(_corners)}/4** — drag a **small box** on the outer black "
                "border (TL → TR → BR → BL)."
            ),
        ])

    _picker_ui
    return corner_picker, reset_corners


@app.cell
def _(
    corner_picker,
    corners_path,
    get_corners,
    get_last_sel,
    get_reset_clicks,
    image_height,
    image_width,
    mo,
    reset_corners,
    set_corners,
    set_last_sel,
    set_reset_clicks,
    show_picker,
    use_saved,
):
    if not use_saved.value and show_picker:
        if corner_picker is None or reset_corners is None:
            mo.stop(True, mo.md("Uncheck **Use saved corners** to pick corners on the plot."))

        _corners = list(get_corners())

        if reset_corners.value > get_reset_clicks():
            set_reset_clicks(reset_corners.value)
            set_corners([])
            if corner_picker.points:
                _pt = corner_picker.points[0]
                set_last_sel((
                    tuple(corner_picker.indices) if corner_picker.indices else (),
                    _pt.get("x"),
                    _pt.get("y"),
                ))
            else:
                set_last_sel(None)
            if corners_path.is_file():
                corners_path.unlink()
        elif corner_picker.points:
            _pt = corner_picker.points[0]
            _sel_key = (
                tuple(corner_picker.indices) if corner_picker.indices else (),
                _pt.get("x"),
                _pt.get("y"),
            )
            if _sel_key != get_last_sel():
                set_last_sel(_sel_key)
                _x = int(round(float(_pt["x"])))
                _y = int(round(float(_pt["y"])))
                _x = max(0, min(image_width - 1, _x))
                _y = max(0, min(image_height - 1, _y))
                _new = [float(_x), float(_y)]
                if len(_corners) < 4 and (not _corners or _corners[-1] != _new):
                    set_corners(_corners + [_new])

        _n = len(get_corners())
        if _n < 4:
            mo.stop(
                True,
                mo.md(f"**{_n}/4** — drag a small box on each corner (TL → TR → BR → BL)."),
            )
    return


@app.cell
def _(get_corners, imshow_extent, np, photo, plt, show_picker, use_saved):
    _preview = None
    if not use_saved.value and show_picker and len(get_corners()) >= 4:
        _labels = ["TL", "TR", "BR", "BL"]
        _fig, _ax = plt.subplots(figsize=(6, 5))
        _ax.imshow(
            photo,
            cmap="gray",
            vmin=0,
            vmax=1.0,
            extent=imshow_extent(photo.shape[1], photo.shape[0]),
        )
        _arr = np.asarray(list(get_corners()), dtype=np.float32)
        _ax.plot(
            np.r_[_arr[:, 0], _arr[0, 0]],
            np.r_[_arr[:, 1], _arr[0, 1]],
            color="cyan",
            lw=1.5,
        )
        _ax.scatter(_arr[:, 0], _arr[:, 1], c="cyan", s=40, zorder=3)
        for _i, (_x, _y) in enumerate(_arr):
            _ax.text(
                _x, _y, _labels[_i], color="yellow", fontsize=9, ha="center", va="bottom"
            )
        _ax.set_xlim(0, photo.shape[1])
        _ax.set_ylim(photo.shape[0], 0)
        _ax.set_title("Corner preview (before optimization)")
        _ax.axis("off")
        _fig.tight_layout()
        _preview = _fig

    _preview
    return


@app.cell
def _(mo):
    run_optimization = mo.ui.run_button(label="Run optimization")
    run_optimization
    return (run_optimization,)


@app.cell
def _(
    corners_path,
    entries,
    get_corners,
    json,
    mo,
    np,
    run_optimization,
    use_saved,
):
    _status = mo.md("")

    if use_saved.value and entries is not None:
        init_corners_px = np.asarray(entries, dtype=np.float32)
        _status = mo.md(f"Using saved corners.\n\n```\n{init_corners_px.tolist()}\n```")
    else:
        mo.stop(
            not run_optimization.value,
            mo.md("**4/4 corners set** — click **Run optimization** when ready."),
        )
        init_corners_px = np.asarray(list(get_corners()), dtype=np.float32)
        corners_path.write_text(json.dumps(init_corners_px.tolist(), indent=2))
        _status = mo.md(
            f"Saved `{corners_path.name}`.\n\n```\n{init_corners_px.tolist()}\n```"
        )

    _status
    return (init_corners_px,)


@app.cell
def _(homography_from_corners, init_corners_px, jnp, src_corners):
    H_init = homography_from_corners(src_corners, jnp.asarray(init_corners_px, dtype=jnp.float32))
    return (H_init,)


@app.cell
def _(
    H_init,
    RenderModelParams,
    bbox_mask,
    corners_from_homography,
    image_height,
    image_width,
    jnp,
    photo,
    src_corners,
):
    target = jnp.asarray(photo, dtype=jnp.float32)
    init_image_corners = corners_from_homography(H_init, src_corners)
    loss_mask = bbox_mask(init_image_corners, image_height, image_width, margin=3.5)

    camera_init = RenderModelParams(
        psf_sigma=jnp.float32(1.0),
        sharpen_amount=jnp.float32(0.4),
        sharpen_sigma=jnp.float32(1.0),
        gamma=jnp.float32(2.0),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.8),
    )
    return camera_init, loss_mask, target


@app.cell
def _(
    H_init,
    OptimizeRenderConfig,
    camera_init,
    image_height,
    image_width,
    loss_mask,
    optimize_render_model,
    tag_pattern,
    target,
):
    config = OptimizeRenderConfig(
        learning_rate=2e-3,
        n_steps=400,
        corner_weight=0.0,
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
    )
    return H_opt, camera_opt, config, losses


@app.cell
def _(
    camera_init,
    camera_opt,
    config,
    image_name,
    losses,
    mo,
    model_params_to_vector,
    tag_id,
):
    def _fmt_cam(name, p):
        v = model_params_to_vector(p)
        return (
            f"**{name}:** psf={float(v[0]):.3f}, sharpen={float(v[1]):.3f}, "
            f"sharpen σ={float(v[2]):.3f}, γ={float(v[3]):.3f}, "
            f"black={float(v[4]):.3f}, white={float(v[5]):.3f}"
        )

    mo.md(
        f"""
        ### Results — `{image_name}` (tag {tag_id}, pixel-only loss, {config.n_steps} steps)

        - **Final loss:** {losses[-1]:.6f} (init {losses[0]:.6f})
        - {_fmt_cam("Camera init", camera_init)}
        - {_fmt_cam("Camera opt", camera_opt)}

        Visual overlap is the metric — no corner GT.
        """
    )
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
    photo,
    plt,
    render_with_model,
    tag_pattern,
):
    init_render = np.asarray(
        render_with_model(H_init, tag_pattern, image_height, image_width, camera_init)
    )
    opt_render = np.asarray(
        render_with_model(H_opt, tag_pattern, image_height, image_width, camera_opt)
    )
    diff = photo - opt_render
    vdiff = centered_diff_limits(diff)

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for _ax, _img, _title in [
        (_axes[0], init_render, "Init render"),
        (_axes[1], opt_render, "Optimized render"),
        (_axes[2], photo, "Photo"),
        (_axes[3], diff, "Photo − optimized"),
    ]:
        _kw = dict(cmap="gray", vmin=0.0, vmax=1.0, extent=imshow_extent(image_width, image_height))
        if _title.startswith("Photo −"):
            _ax.imshow(_img, cmap="RdBu_r", vmin=-vdiff, vmax=vdiff, extent=_kw["extent"])
        else:
            _ax.imshow(_img, **_kw)
        _ax.set_title(_title)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    H_init,
    H_opt,
    corners_from_homography,
    imshow_extent,
    init_corners_px,
    np,
    photo,
    plt,
    src_corners,
):
    _pred_init = np.asarray(corners_from_homography(H_init, src_corners))
    _pred_opt = np.asarray(corners_from_homography(H_opt, src_corners))
    _fig, _ax = plt.subplots(figsize=(8, 6))
    _ax.imshow(photo, cmap="gray", vmin=0, vmax=1.0, extent=imshow_extent(photo.shape[1], photo.shape[0]))
    for _pts, _color, _label in [
        (init_corners_px, "red", "Your clicks (init)"),
        (_pred_init, "orange", "H_init corners"),
        (_pred_opt, "cyan", "H_opt corners"),
    ]:
        _ax.plot(
            np.r_[_pts[:, 0], _pts[0, 0]],
            np.r_[_pts[:, 1], _pts[0, 1]],
            color=_color,
            lw=1.5,
            label=_label,
        )
    _ax.set_xlim(0, photo.shape[1])
    _ax.set_ylim(photo.shape[0], 0)
    _ax.legend(loc="upper right", fontsize=8)
    _ax.set_title("Corner drift (clicks were starting point only)")
    _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
