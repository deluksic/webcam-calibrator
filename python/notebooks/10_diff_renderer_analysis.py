"""Analyze calibration exports with per-tag differential renderer optimization."""

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
    import plotly.graph_objects as go
    from PIL import Image

    from render_model import (
        JOINT_PARAM_NAMES,
        TAG_CANONICAL_CORNERS,
        OptimizeLMConfig,
        RenderModelParams,
        build_tag_pattern,
        camera_with_inferred_levels,
        compile_lm_run,
        corners_from_corner_shifts,
        homography_from_corners,
        loss_mask_from_corners,
        make_render_model_residuals,
        model_params_to_vector,
        vector_to_model_params,
    )

    return (
        JOINT_PARAM_NAMES,
        Image,
        OptimizeLMConfig,
        Path,
        RenderModelParams,
        TAG_CANONICAL_CORNERS,
        build_tag_pattern,
        camera_with_inferred_levels,
        compile_lm_run,
        corners_from_corner_shifts,
        go,
        homography_from_corners,
        jax,
        jnp,
        json,
        loss_mask_from_corners,
        make_render_model_residuals,
        model_params_to_vector,
        np,
        plt,
        time,
        vector_to_model_params,
    )


@app.cell
def _(mo):
    json_file = mo.ui.file(filetypes=[".json"], label="Calibration export JSON")
    image_dir = mo.ui.text(
        value=".",
        label="Directory containing PNG images (same folder as JSON)",
        full_width=True,
    )
    mo.vstack([
        mo.md("## Load calibration export"),
        json_file,
        image_dir,
        mo.md("Upload the JSON export file and set the directory containing the frame PNGs."),
    ])
    return image_dir, json_file


@app.cell
def _(Path, image_dir, json, json_file, mo, np):
    mo.stop(not json_file.value, mo.md("Upload a JSON file to begin."))

    _contents = json_file.contents()
    _data = json.loads(_contents) if isinstance(_contents, str) else json.loads(
        _contents.decode("utf-8")
    )

    if _data.get("kind") != "ok":
        mo.stop(True, mo.md(f"Not a calibration-ok export (kind={_data.get('kind')!r})."))

    _frame_images = _data.get("frameImages", {})
    _observations = _data.get("observations", [])
    _image_dir = Path(image_dir.value)

    if not _observations:
        mo.stop(True, mo.md("Export contains no `observations` — re-export with frame images."))

    # Build frame list: only frames that have both observations and an available image
    _frames = []
    for _obs in _observations:
        _fid = _obs["frameId"]
        _img_file = _obs.get("imageFile", "")
        _img_path = _image_dir / _img_file if _img_file else None
        if _img_path and _img_path.is_file():
            _img = np.asarray(Image.open(_img_path).convert("L"), dtype=np.float32) / 255.0
            _h, _w = _img.shape
            _frames.append({
                "frameId": _fid,
                "image": _img,
                "width": _w,
                "height": _h,
                "tags": _obs.get("tags", []),
                "imageFile": _img_file,
            })

    if not _frames:
        mo.stop(
            True,
            mo.md(
                f"No frame images found in `{_image_dir}`. "
                "Place the PNG files in the same directory as the JSON."
            ),
        )

    export_data = _data
    frames = _frames
    image_dir_path = _image_dir

    mo.md(f"Loaded **{len(frames)}** frames with images from `{_image_dir}`.")
    return export_data, frames, image_dir_path


@app.cell
def _(frames, mo):
    _options = {
        f"Frame {f['frameId']} ({len(f['tags'])} tags)": i
        for i, f in enumerate(frames)
    }
    _default_key = next(iter(_options.keys()))
    frame_selector = mo.ui.dropdown(
        options=_options,
        value=_default_key,
        label="Select frame to analyze",
        full_width=True,
    )
    run_button = mo.ui.run_button(label="Run per-tag LM optimization")
    mo.vstack([frame_selector, run_button])
    return frame_selector, run_button


@app.cell
def _(
    OptimizeLMConfig,
    RenderModelParams,
    TAG_CANONICAL_CORNERS,
    build_tag_pattern,
    camera_with_inferred_levels,
    compile_lm_run,
    corners_from_corner_shifts,
    frame_selector,
    frames,
    homography_from_corners,
    jnp,
    loss_mask_from_corners,
    make_render_model_residuals,
    model_params_to_vector,
    mo,
    np,
    run_button,
    time,
    vector_to_model_params,
):
    mo.stop(not run_button.value, mo.md("Click **Run** to start per-tag optimization."))

    _frame = frames[frame_selector.value]
    _img = _frame["image"]
    _full_h, _full_w = _img.shape
    _tags = _frame["tags"]

    mo.md(
        f"### Frame {_frame['frameId']} — {_full_w}×{_full_h}, {len(_tags)} tags"
    )

    # -- compute uniform crop size from max tag extent (corners + moat + margin) --

    _tag_infos = []  # (tag_id, init_corners_np, bbox_x0, y0, x1, y1)
    _max_crop_w = 0
    _max_crop_h = 0

    for _tag in _tags:
        _corners_raw = _tag["corners"]
        _init_corners_np = np.array(
            [[c["x"], c["y"]] for c in _corners_raw], dtype=np.float32
        )
        _init_corners_j = jnp.asarray(_init_corners_np)

        # Moat = 1/8 of shortest side (matches loss_mask padding)
        _sides = []
        for _i in range(4):
            _j = (_i + 1) % 4
            _dx = _init_corners_np[_j, 0] - _init_corners_np[_i, 0]
            _dy = _init_corners_np[_j, 1] - _init_corners_np[_i, 1]
            _sides.append(np.sqrt(_dx * _dx + _dy * _dy))
        _moat = np.min(_sides) / 8.0

        _x0 = int(np.floor(np.min(_init_corners_np[:, 0]) - _moat))
        _y0 = int(np.floor(np.min(_init_corners_np[:, 1]) - _moat))
        _x1 = int(np.ceil(np.max(_init_corners_np[:, 0]) + _moat))
        _y1 = int(np.ceil(np.max(_init_corners_np[:, 1]) + _moat))

        _tag_infos.append({
            "tagId": _tag["tagId"],
            "initCorners": _init_corners_np,
            "x0": _x0,
            "y0": _y0,
            "x1": _x1,
            "y1": _y1,
        })

        _max_crop_w = max(_max_crop_w, _x1 - _x0)
        _max_crop_h = max(_max_crop_h, _y1 - _y0)

    _margin = 6  # extra pixels for safety
    _crop_w = _max_crop_w + _margin
    _crop_h = _max_crop_h + _margin

    mo.md(
        f"Tag crop size: **{_crop_w}×{_crop_h}** px "
        f"(max tag extent + {_margin}px margin, from {_full_w}×{_full_h} full frame)"
    )

    # -- per-tag optimization on cropped regions --

    _results = []
    _lm_steps = 12
    _seed = RenderModelParams(
        psf_sigma=jnp.float32(1.5),
        sharpen_amount=jnp.float32(1.0),
        sharpen_sigma=jnp.float32(0.6),
        gamma=jnp.float32(1.75),
        black_level=jnp.float32(0.2),
        white_level=jnp.float32(0.8),
        light_grad_u=jnp.float32(0.0),
        light_grad_v=jnp.float32(0.0),
    )

    for _info in _tag_infos:
        _tag_id = _info["tagId"]
        _init_corners_full = jnp.asarray(_info["initCorners"])
        _x0, _y0 = _info["x0"], _info["y0"]
        _x1, _y1 = _info["x1"], _info["y1"]

        # Center crop within the uniform-size window
        _tag_w = _x1 - _x0
        _tag_h = _y1 - _y0
        _off_x = (_crop_w - _tag_w) // 2
        _off_y = (_crop_h - _tag_h) // 2
        _crop_x0 = _x0 - _off_x
        _crop_y0 = _y0 - _off_y

        # Clamp crop to image bounds
        _crop_x0 = max(0, _crop_x0)
        _crop_y0 = max(0, _crop_y0)
        _crop_x1 = min(_full_w, _crop_x0 + _crop_w)
        _crop_y1 = min(_full_h, _crop_y0 + _crop_h)

        # Extract crop and build target
        _crop = _img[_crop_y0:_crop_y1, _crop_x0:_crop_x1]
        _ch = _crop.shape[0]
        _cw = _crop.shape[1]
        _target = jnp.asarray(_crop, dtype=jnp.float32)

        # Adjust corners to crop coordinates
        _init_corners_crop = _init_corners_full - jnp.array([_crop_x0, _crop_y0], dtype=jnp.float32)

        _pattern = build_tag_pattern(_tag_id)
        _H_init = homography_from_corners(
            jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32),
            _init_corners_crop,
        )
        _mask = loss_mask_from_corners(_init_corners_crop, _ch, _cw)

        _cam_init = camera_with_inferred_levels(
            _seed, _target, _H_init, _pattern, _ch, _cw, loss_mask=_mask,
        )

        _config = OptimizeLMConfig(n_steps=_lm_steps, loss_mask=_mask)
        _residual_fn = make_render_model_residuals(
            _target, _pattern, _ch, _cw,
            ref_corners=_init_corners_crop, loss_mask=_mask,
        )

        _packed_init = jnp.concatenate(
            [jnp.zeros(8, dtype=jnp.float32), model_params_to_vector(_cam_init)]
        )

        _lm_run = compile_lm_run(_residual_fn, _config)
        _damping = jnp.asarray(_config.initial_damping, dtype=jnp.float32)
        _t0 = time.perf_counter()
        _packed, _step_losses, _param_hist = _lm_run(_packed_init, _damping)
        jax.block_until_ready(_packed)
        _elapsed = time.perf_counter() - _t0

        # Convert corners back to full-image coordinates
        _final_corners_crop = corners_from_corner_shifts(_init_corners_crop, _packed[:8])
        _final_corners_full = _final_corners_crop + jnp.array([_crop_x0, _crop_y0], dtype=jnp.float32)
        _final_cam = vector_to_model_params(_packed[8:16])

        _init_mse = float(_step_losses[0]) if len(_step_losses) > 0 else float(
            jnp.sum(_residual_fn(_packed_init) ** 2)
        )
        _final_mse = float(_step_losses[-1])

        _init_np = np.asarray(_init_corners_full)
        _final_np = np.asarray(_final_corners_full)
        _corner_deltas = _final_np - _init_np
        _corner_movements = np.sqrt(
            _corner_deltas[:, 0] ** 2 + _corner_deltas[:, 1] ** 2
        )

        _results.append({
            "tagId": _tag_id,
            "initCorners": _init_np,
            "finalCorners": _final_np,
            "initMse": _init_mse,
            "finalMse": _final_mse,
            "cornerMovements": _corner_movements,
            "meanMovement": float(np.mean(_corner_movements)),
            "maxMovement": float(np.max(_corner_movements)),
            "camera": _final_cam,
            "elapsed": _elapsed,
        })

    results = sorted(_results, key=lambda r: r["tagId"])

    mo.md(
        f"Optimized **{len(results)}** tags "
        f"({_lm_steps} LM steps each). "
        f"Total time: {sum(r['elapsed'] for r in results):.1f}s"
    )
    return results


@app.cell
def _(frame_selector, frames, go, mo, np, results):
    mo.stop(not results, mo.md("Run optimization first."))

    _frame = frames[frame_selector.value]
    _img = _frame["image"]
    _h, _w = _img.shape

    _fig = go.Figure()

    _fig.add_trace(
        go.Heatmap(
            z=_img,
            colorscale="gray",
            showscale=False,
            zmin=0.0,
            zmax=1.0,
            hovertemplate="(%{x}, %{y})<extra></extra>",
        )
    )

    _colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    ]

    for _ri, _r in enumerate(results):
        _color = _colors[_ri % len(_colors)]

        _ic = _r["initCorners"]
        _fc = _r["finalCorners"]

        _fig.add_trace(
            go.Scatter(
                x=list(_ic[:, 0]) + [_ic[0, 0]],
                y=list(_ic[:, 1]) + [_ic[0, 1]],
                mode="lines",
                line=dict(color=_color, dash="dash", width=1.5),
                name=f"Tag {_r['tagId']} init",
                legendgroup=f"tag{_r['tagId']}",
                hovertext=f"Tag {_r['tagId']} init",
                hoverinfo="text",
            )
        )

        _fig.add_trace(
            go.Scatter(
                x=list(_fc[:, 0]) + [_fc[0, 0]],
                y=list(_fc[:, 1]) + [_fc[0, 1]],
                mode="lines",
                line=dict(color=_color, dash="solid", width=2),
                name=f"Tag {_r['tagId']} opt",
                legendgroup=f"tag{_r['tagId']}",
                hovertext=(
                    f"Tag {_r['tagId']}<br>"
                    f"MSE: {_r['finalMse']:.4f}<br>"
                    f"Mean &Delta;: {_r['meanMovement']:.2f} px<br>"
                    f"Max &Delta;: {_r['maxMovement']:.2f} px"
                ),
                hoverinfo="text",
            )
        )

    _fig.update_layout(
        xaxis=dict(
            range=[0, _w],
            constrain="domain",
            showgrid=False,
            zeroline=False,
            title="x (pixels)",
        ),
        yaxis=dict(
            range=[_h, 0],
            constrain="domain",
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            title="y (pixels)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=40),
        height=700,
        title=f"Frame {_frame['frameId']} — dashed=init, solid=optimized",
    )

    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, results):
    mo.stop(not results, mo.md("Run optimization first."))

    _rows = []
    for _r in results:
        _cam = _r["camera"]
        _rows.append({
            "Tag ID": _r["tagId"],
            "Init MSE": f"{_r['initMse']:.4f}",
            "Final MSE": f"{_r['finalMse']:.4f}",
            "Mean Δ (px)": f"{_r['meanMovement']:.2f}",
            "Max Δ (px)": f"{_r['maxMovement']:.2f}",
            "PSF σ": f"{float(_cam.psf_sigma):.2f}",
            "Gamma": f"{float(_cam.gamma):.2f}",
            "Black": f"{float(_cam.black_level):.3f}",
            "White": f"{float(_cam.white_level):.3f}",
            "Light ∇u": f"{float(_cam.light_grad_u):.3f}",
            "Light ∇v": f"{float(_cam.light_grad_v):.3f}",
        })

    mo.md("## Per-tag results")
    mo.ui.table(
        _rows,
        selection=None,
        page_size=20,
    )
    return


@app.cell
def _(mo, np, plt, results):
    mo.stop(not results, mo.md("Run optimization first."))

    _all_movements = np.concatenate([r["cornerMovements"] for r in results])

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(_all_movements, bins=30, color="C0", alpha=0.8, edgecolor="white")
    _ax.axvline(np.mean(_all_movements), color="C3", linestyle="--", label=f"Mean: {np.mean(_all_movements):.2f} px")
    _ax.axvline(np.median(_all_movements), color="C2", linestyle="--", label=f"Median: {np.median(_all_movements):.2f} px")
    _ax.set_xlabel("Corner movement (pixels)")
    _ax.set_ylabel("Count")
    _ax.set_title(f"Corner movement distribution ({len(_all_movements)} corners, {len(results)} tags)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
