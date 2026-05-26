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
        RenderModelParams,
        build_tag_pattern,
        camera_with_inferred_levels,
        corners_from_corner_shifts,
        homography_from_corners,
        loss_mask_from_corners,
        model_params_to_vector,
        vector_to_model_params,
    )
    from render_model.pipeline import SUPERSAMPLE, render_with_model

    CANONICAL = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)

    # Canonical corners are cyclic: TL, TR, BR, BL.
    # GPU export corners are strip: TL, TR, BL, BR.
    # Reorder image corners to cyclic before any render_model call.
    STRIP_TO_CYCLIC_4 = jnp.array([0, 1, 3, 2], dtype=jnp.int32)
    # Same but with closing vertex for Plotly polygon outlines.
    STRIP_TO_CYCLIC_5 = [0, 1, 3, 2, 0]

    return (
        CANONICAL,
        JOINT_PARAM_NAMES,
        Image,
        Path,
        RenderModelParams,
        STRIP_TO_CYCLIC_4,
        STRIP_TO_CYCLIC_5,
        SUPERSAMPLE,
        build_tag_pattern,
        camera_with_inferred_levels,
        corners_from_corner_shifts,
        go,
        homography_from_corners,
        jax,
        jnp,
        json,
        loss_mask_from_corners,
        model_params_to_vector,
        np,
        plt,
        render_with_model,
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
    CANONICAL,
    RenderModelParams,
    STRIP_TO_CYCLIC_4,
    SUPERSAMPLE,
    build_tag_pattern,
    camera_with_inferred_levels,
    corners_from_corner_shifts,
    frame_selector,
    frames,
    homography_from_corners,
    jax,
    jnp,
    loss_mask_from_corners,
    model_params_to_vector,
    mo,
    np,
    render_with_model,
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

    _tag_infos = []
    _max_crop_w = 0
    _max_crop_h = 0

    for _tag in _tags:
        _corners_raw = _tag["corners"]
        _init_corners_np = np.array(
            [[c["x"], c["y"]] for c in _corners_raw], dtype=np.float32
        )
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
            "x0": _x0, "y0": _y0, "x1": _x1, "y1": _y1,
        })

        _max_crop_w = max(_max_crop_w, _x1 - _x0)
        _max_crop_h = max(_max_crop_h, _y1 - _y0)

    _margin = 6
    _crop_w = _max_crop_w + _margin
    _crop_h = _max_crop_h + _margin

    mo.md(
        f"Tag crop size: **{_crop_w}×{_crop_h}** px "
        f"(max tag extent + {_margin}px margin, from {_full_w}×{_full_h} full frame)"
    )

    # -- single-compiled LM for the uniform crop size --

    _lm_steps = 12
    _df = jnp.float32(10.0)
    _lo = jnp.float32(1e-8)
    _hi = jnp.float32(1e8)
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

    # Generic residual: all tag-specific data is explicit → JIT-compiled once.
    @jax.jit
    def _lm_run(packed_init, damping, target, tag_pattern, ref_corners, loss_mask):
        def _residual(packed):
            shifts = packed[:8]
            cam_vec = packed[8:16]
            _corners = corners_from_corner_shifts(ref_corners, shifts)
            _H = homography_from_corners(CANONICAL, _corners)
            rendered = render_with_model(
                _H, tag_pattern, _crop_h, _crop_w,
                vector_to_model_params(cam_vec),
                supersample=SUPERSAMPLE,
            )
            diff = rendered - target
            w = jnp.sqrt(loss_mask / jnp.maximum(jnp.sum(loss_mask), 1.0))
            return (diff * w).ravel()

        def _step(carry, _step_idx):
            p, lam = carry
            r = _residual(p)
            J = jax.jacfwd(_residual)(p)
            hess = J.T @ J + lam * jnp.eye(16, dtype=p.dtype)
            delta = jnp.linalg.solve(hess, -(J.T @ r))
            p_try = p + delta
            loss = jnp.sum(r * r)
            loss_try = jnp.sum(_residual(p_try) ** 2)
            accept = loss_try < loss
            p = jnp.where(accept, p_try, p)
            lam = jnp.clip(jnp.where(accept, lam / _df, lam * _df), _lo, _hi)
            return (p, lam), (jnp.where(accept, loss_try, loss), p)

        steps = jnp.arange(_lm_steps, dtype=jnp.int32)
        (packed, _), (losses, hist) = jax.lax.scan(
            _step, (packed_init, damping), steps,
        )
        return packed, losses, hist

    # Warm-up: compile once with the first tag's data
    _first = _tag_infos[0]
    _t0 = time.perf_counter()
    _init_c = jnp.asarray(_first["initCorners"])
    _x0, _y0 = _first["x0"], _first["y0"]
    _off_x = (_crop_w - (_first["x1"] - _x0)) // 2
    _off_y = (_crop_h - (_first["y1"] - _y0)) // 2
    _crop_x0_warm = max(0, _x0 - _off_x)
    _crop_y0_warm = max(0, _y0 - _off_y)
    _crop_warm = np.pad(
        _img[_crop_y0_warm:_crop_y0_warm + _crop_h, _crop_x0_warm:_crop_x0_warm + _crop_w],
        ((0, max(0, _crop_h - (min(_full_h, _crop_y0_warm + _crop_h) - _crop_y0_warm))),
         (0, max(0, _crop_w - (min(_full_w, _crop_x0_warm + _crop_w) - _crop_x0_warm)))),
        mode='constant',
    )
    _ch_warm, _cw_warm = _crop_warm.shape
    _target_warm = jnp.asarray(_crop_warm, dtype=jnp.float32)
    # Reorder strip→cyclic to match CANONICAL ordering
    _corners_crop_warm = (_init_c - jnp.array([_crop_x0_warm, _crop_y0_warm], dtype=jnp.float32))[STRIP_TO_CYCLIC_4]
    _H_warm = homography_from_corners(CANONICAL, _corners_crop_warm)
    _pat_warm = build_tag_pattern(_first["tagId"])
    _mask_warm = loss_mask_from_corners(_corners_crop_warm, _ch_warm, _cw_warm)
    _cam_warm = camera_with_inferred_levels(
        _seed, _target_warm, _H_warm, _pat_warm, _ch_warm, _cw_warm, loss_mask=_mask_warm,
    )
    _packed_warm = jnp.concatenate(
        [jnp.zeros(8, dtype=jnp.float32), model_params_to_vector(_cam_warm)]
    )
    _damping_warm = jnp.float32(1e-2)
    _ = _lm_run(_packed_warm, _damping_warm, _target_warm, _pat_warm, _corners_crop_warm, _mask_warm)
    jax.block_until_ready(_)
    _compile_s = time.perf_counter() - _t0
    mo.md(f"LM compiled in **{_compile_s:.1f}s** (warm-up with first tag)")

    # -- run LM for every tag using the same compiled function --

    _results = []
    _total_s = 0.0

    for _info in _tag_infos:
        _tag_id = _info["tagId"]
        _init_c = jnp.asarray(_info["initCorners"])
        _x0, _y0 = _info["x0"], _info["y0"]
        _x1, _y1 = _info["x1"], _info["y1"]

        _off_x = (_crop_w - (_x1 - _x0)) // 2
        _off_y = (_crop_h - (_y1 - _y0)) // 2
        _crop_x0 = max(0, _x0 - _off_x)
        _crop_y0 = max(0, _y0 - _off_y)

        # Extract exact-size crop, padding at image edges if needed
        _raw = _img[_crop_y0:_crop_y0 + _crop_h, _crop_x0:_crop_x0 + _crop_w]
        _pad_bottom = _crop_h - _raw.shape[0]
        _pad_right = _crop_w - _raw.shape[1]
        if _pad_bottom > 0 or _pad_right > 0:
            _raw = np.pad(_raw, ((0, _pad_bottom), (0, _pad_right)), mode='constant')
        _target = jnp.asarray(_raw, dtype=jnp.float32)

        # Reorder strip→cyclic to match CANONICAL ordering
        _corners_crop = (_init_c - jnp.array([_crop_x0, _crop_y0], dtype=jnp.float32))[STRIP_TO_CYCLIC_4]
        _pattern = build_tag_pattern(_tag_id)
        _H_init = homography_from_corners(CANONICAL, _corners_crop)
        _mask = loss_mask_from_corners(_corners_crop, _crop_h, _crop_w)
        _cam_init = camera_with_inferred_levels(
            _seed, _target, _H_init, _pattern, _crop_h, _crop_w, loss_mask=_mask,
        )
        _packed_init = jnp.concatenate(
            [jnp.zeros(8, dtype=jnp.float32), model_params_to_vector(_cam_init)]
        )
        _damping = jnp.float32(1e-2)

        _t0 = time.perf_counter()
        _packed, _step_losses, _hist = _lm_run(
            _packed_init, _damping, _target, _pattern, _corners_crop, _mask,
        )
        jax.block_until_ready(_packed)
        _elapsed = time.perf_counter() - _t0
        _total_s += _elapsed

        _final_corners_crop = corners_from_corner_shifts(_corners_crop, _packed[:8])
        # Reorder cyclic→strip for result storage (STRIP_TO_CYCLIC_4 is self-inverse)
        _final_corners_full = _final_corners_crop[STRIP_TO_CYCLIC_4] + jnp.array(
            [_crop_x0, _crop_y0], dtype=jnp.float32
        )
        _final_cam = vector_to_model_params(_packed[8:16])

        _init_mse = float(_step_losses[0])
        _final_mse = float(_step_losses[-1])

        _init_np = np.asarray(_init_c)
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
        f"(compile {_compile_s:.1f}s, run {_total_s:.1f}s, "
        f"{_lm_steps} LM steps each)."
    )
    return results


@app.cell
def _(STRIP_TO_CYCLIC_5, frame_selector, frames, go, mo, np, results):
    mo.stop(not results, mo.md("Run optimization first."))

    _frame = frames[frame_selector.value]
    _img = _frame["image"]
    _h, _w = _img.shape

    _fig = go.Figure()

    _fig.add_trace(
        go.Heatmap(
            z=_img,
            x0=0.5, dx=1.0,
            y0=0.5, dy=1.0,
            colorscale="gray",
            showscale=False,
            zmin=0.0,
            zmax=1.0,
            hovertemplate="(%{x:.1f}, %{y:.1f})<extra></extra>",
        )
    )

    _colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    ]

    for _ri, _r in enumerate(results):
        _color = _colors[_ri % len(_colors)]

        # Corners are in strip order (TL, TR, BL, BR).
        # Reorder to cyclic (TL, TR, BR, BL) for closed polygon outlines.
        _ic = _r["initCorners"][STRIP_TO_CYCLIC_5]
        _fc = _r["finalCorners"][STRIP_TO_CYCLIC_5]

        _fig.add_trace(
            go.Scatter(
                x=list(_ic[:, 0]),
                y=list(_ic[:, 1]),
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
                x=list(_fc[:, 0]),
                y=list(_fc[:, 1]),
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
        dragmode="pan",
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
        height=min(900, max(700, _h)),
        title=f"Frame {_frame['frameId']} — dashed=init, solid=optimized",
    )

    mo.ui.plotly(_fig, config={"scrollZoom": True, "displayModeBar": True})
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
