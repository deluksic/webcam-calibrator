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
def _(mo):
    run_all_button = mo.ui.run_button(label="Run per-tag LM on ALL frames")
    mo.vstack([run_all_button, mo.md("Runs differential-renderer LM optimization on every tag in every frame.")])
    return (run_all_button,)


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
        label="Select frame to inspect",
        full_width=True,
    )
    mo.vstack([frame_selector])
    return (frame_selector,)


@app.cell
def _(
    CANONICAL,
    RenderModelParams,
    STRIP_TO_CYCLIC_4,
    SUPERSAMPLE,
    build_tag_pattern,
    camera_with_inferred_levels,
    corners_from_corner_shifts,
    frames,
    homography_from_corners,
    jax,
    jnp,
    loss_mask_from_corners,
    model_params_to_vector,
    mo,
    np,
    render_with_model,
    run_all_button,
    time,
    vector_to_model_params,
):
    mo.stop(not run_all_button.value, mo.md("Click **Run** to start per-tag optimization on all frames."))

    _lm_steps = 30
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

    _all_results = {}  # frame_index → sorted list of tag result dicts
    _compile_total = 0.0
    _run_total = 0.0
    _total_tags = 0

    for _fi, _frame in enumerate(frames):
        _img = _frame["image"]
        _full_h, _full_w = _img.shape
        _tags = _frame["tags"]

        # -- compute uniform crop size from max tag extent --
        _tag_infos = []
        _max_crop_w = 0
        _max_crop_h = 0
        for _tag in _tags:
            _corners_raw = _tag["corners"]
            _init_corners_np = np.array(
                [[c["x"], c["y"]] for c in _corners_raw], dtype=np.float32
            )
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

        # Per-frame JIT-compiled LM
        @jax.jit
        def _lm_run(packed_init, damping, target, tag_pattern, ref_corners, loss_mask):
            def _residual(packed):
                shifts = packed[:8]
                cam_vec = packed[8:16]
                _c = corners_from_corner_shifts(ref_corners, shifts)
                _H = homography_from_corners(CANONICAL, _c)
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

        # Warm-up
        _first = _tag_infos[0]
        _t0 = time.perf_counter()
        _init_c = jnp.asarray(_first["initCorners"])
        _x0, _y0 = _first["x0"], _first["y0"]
        _off_x = (_crop_w - (_first["x1"] - _x0)) // 2
        _off_y = (_crop_h - (_first["y1"] - _y0)) // 2
        _cx = max(0, _x0 - _off_x)
        _cy = max(0, _y0 - _off_y)
        _warm = np.pad(
            _img[_cy:_cy + _crop_h, _cx:_cx + _crop_w],
            ((0, max(0, _crop_h - min(_full_h, _cy + _crop_h) + _cy)),
             (0, max(0, _crop_w - min(_full_w, _cx + _crop_w) + _cx))),
            mode='constant',
        )
        _ch, _cw = _warm.shape
        _tw = jnp.asarray(_warm, dtype=jnp.float32)
        _cc = (_init_c - jnp.array([_cx, _cy], dtype=jnp.float32))[STRIP_TO_CYCLIC_4]
        _Hw = homography_from_corners(CANONICAL, _cc)
        _pw = build_tag_pattern(_first["tagId"])
        _mw = loss_mask_from_corners(_cc, _ch, _cw)
        _cw2 = camera_with_inferred_levels(_seed, _tw, _Hw, _pw, _ch, _cw, loss_mask=_mw)
        _ = _lm_run(
            jnp.concatenate([jnp.zeros(8, dtype=jnp.float32), model_params_to_vector(_cw2)]),
            jnp.float32(1e-3), _tw, _pw, _cc, _mw,
        )
        jax.block_until_ready(_)
        _compile_total += time.perf_counter() - _t0

        # Run every tag in this frame
        _fr = []
        for _info in _tag_infos:
            _tid = _info["tagId"]
            _ic = jnp.asarray(_info["initCorners"])
            _x0, _y0 = _info["x0"], _info["y0"]
            _x1, _y1 = _info["x1"], _info["y1"]
            _ox = (_crop_w - (_x1 - _x0)) // 2
            _oy = (_crop_h - (_y1 - _y0)) // 2
            _cx = max(0, _x0 - _ox)
            _cy = max(0, _y0 - _oy)
            _raw = _img[_cy:_cy + _crop_h, _cx:_cx + _crop_w]
            _pb = _crop_h - _raw.shape[0]
            _pr = _crop_w - _raw.shape[1]
            if _pb > 0 or _pr > 0:
                _raw = np.pad(_raw, ((0, _pb), (0, _pr)), mode='constant')
            _targ = jnp.asarray(_raw, dtype=jnp.float32)
            _cc2 = (_ic - jnp.array([_cx, _cy], dtype=jnp.float32))[STRIP_TO_CYCLIC_4]
            _pat = build_tag_pattern(_tid)
            _Hi = homography_from_corners(CANONICAL, _cc2)
            _mask = loss_mask_from_corners(_cc2, _crop_h, _crop_w)
            _ci = camera_with_inferred_levels(_seed, _targ, _Hi, _pat, _crop_h, _crop_w, loss_mask=_mask)
            _pi = jnp.concatenate([jnp.zeros(8, dtype=jnp.float32), model_params_to_vector(_ci)])
            _t0 = time.perf_counter()
            _p, _sl, _h = _lm_run(_pi, jnp.float32(1e-3), _targ, _pat, _cc2, _mask)
            jax.block_until_ready(_p)
            _elapsed = time.perf_counter() - _t0
            _run_total += _elapsed
            _fcc = corners_from_corner_shifts(_cc2, _p[:8])
            _ff = _fcc[STRIP_TO_CYCLIC_4] + jnp.array([_cx, _cy], dtype=jnp.float32)
            _fc = vector_to_model_params(_p[8:16])
            _in = np.asarray(_ic)
            _fn = np.asarray(_ff)
            _deltas = _fn - _in
            _mov = np.sqrt(_deltas[:, 0] ** 2 + _deltas[:, 1] ** 2)
            _fr.append({
                "tagId": _tid,
                "initCorners": _in,
                "finalCorners": _fn,
                "initMse": float(_sl[0]),
                "finalMse": float(_sl[-1]),
                "cornerMovements": _mov,
                "meanMovement": float(np.mean(_mov)),
                "maxMovement": float(np.max(_mov)),
                "camera": _fc,
                "elapsed": _elapsed,
            })

        _all_results[_fi] = sorted(_fr, key=lambda r: r["tagId"])
        _total_tags += len(_fr)

    all_results = _all_results

    mo.md(
        f"Optimized **{_total_tags}** tags across **{len(frames)}** frames "
        f"(compile {_compile_total:.1f}s, run {_run_total:.1f}s, "
        f"{_lm_steps} LM steps each)."
    )
    return (all_results,)


@app.cell
def _(STRIP_TO_CYCLIC_5, all_results, frame_selector, frames, go, mo, np):
    mo.stop(not all_results, mo.md("Run optimization first."))

    _fi = frame_selector.value
    _frame = frames[_fi]
    _img = _frame["image"]
    _h, _w = _img.shape
    _results = all_results.get(_fi, [])

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

    for _ri, _r in enumerate(_results):
        _color = _colors[_ri % len(_colors)]
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
        showlegend=False,
        height=min(900, max(700, _h)),
        title=f"Frame {_frame['frameId']} — dashed=init, solid=optimized",
    )

    mo.ui.plotly(_fig, config={"scrollZoom": True, "displayModeBar": True})
    return


@app.cell
def _(all_results, frame_selector, mo):
    mo.stop(not all_results, mo.md("Run optimization first."))

    _results = all_results.get(frame_selector.value, [])
    _rows = []
    for _r in _results:
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
def _(all_results, frames, mo, np):
    """Per-frame aggregate: mean corner movement and MSE improvement."""
    mo.stop(not all_results, mo.md("Run optimization first."))

    _rows = []
    for _fi, _frame in enumerate(frames):
        _fr = all_results.get(_fi, [])
        if not _fr:
            continue
        _movements = np.concatenate([r["cornerMovements"] for r in _fr])
        _init_mse = np.mean([r["initMse"] for r in _fr])
        _final_mse = np.mean([r["finalMse"] for r in _fr])
        _rows.append({
            "Frame": _frame["frameId"],
            "Tags": len(_fr),
            "Mean Δ (px)": f"{np.mean(_movements):.3f}",
            "Max Δ (px)": f"{np.max(_movements):.3f}",
            "Mean init MSE": f"{_init_mse:.4f}",
            "Mean final MSE": f"{_final_mse:.4f}",
            "MSE improved": f"{_init_mse - _final_mse:+.4f}",
        })

    _all_mov = np.concatenate([
        np.concatenate([r["cornerMovements"] for r in all_results.get(_fi, [])])
        for _fi in range(len(frames))
    ])

    mo.md("## Aggregate per-frame summary")
    mo.ui.table(_rows, selection=None, page_size=20)
    mo.md(
        f"**All frames:** {len(_rows)} frames,  "
        f"mean Δ **{np.mean(_all_mov):.3f}** px,  "
        f"median Δ {np.median(_all_mov):.3f} px,  "
        f"max Δ {np.max(_all_mov):.3f} px"
    )
    return


@app.cell
def _(all_results, export_data, frames, mo, np):
    """Re-run calibrateCameraRO comparing init vs refined corners."""
    mo.stop(not all_results, mo.md("Run optimization first."))

    import cv2

    _K = export_data.get("K")
    _img_size = export_data.get("imageSize")

    mo.stop(
        not _K or not _img_size,
        mo.md("Export missing K/imageSize — skipping."),
    )

    _w, _h = _img_size["width"], _img_size["height"]

    # Learn layout from first frame's init corners, matching TypeScript
    # learnLayoutFromFrame. UNIT_SQUARE and initCorners are both strip order.
    _first_results = all_results.get(0, [])
    mo.stop(
        len(_first_results) < 2,
        mo.md(f"Need >=2 tags in first frame (have {len(_first_results)})."),
    )

    _sorted = sorted(_first_results, key=lambda r: r["tagId"])
    _anchor = _sorted[0]
    _unit_square = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64
    )  # strip: TL, TR, BL, BR

    _H_anchor, _ = cv2.findHomography(_unit_square, _anchor["initCorners"], method=0)
    _H_inv = np.linalg.inv(_H_anchor)

    # Map each tag's strip-ordered init corners through H_inv → object space
    _layout = {}
    for _r in _sorted:
        _obj = cv2.perspectiveTransform(
            _r["initCorners"].astype(np.float64).reshape(1, 4, 2), _H_inv
        ).reshape(4, 2)
        _layout[_r["tagId"]] = _obj

    # Zero-mean in xy
    _all = np.concatenate(list(_layout.values()), axis=0)
    _mean = _all.mean(axis=0)
    for _tid in _layout:
        _layout[_tid] = _layout[_tid] - _mean

    # Build object-point template (strip order, sorted by tagId)
    _obj_template = []
    for _tid in sorted(_layout):
        _oc = _layout[_tid]
        for _ci in range(4):
            _obj_template.append((_tid, _ci, [float(_oc[_ci, 0]), float(_oc[_ci, 1]), 0.0]))

    # Build per-frame image points for shared tags
    _obj_pts, _init_img_pts, _ref_img_pts = [], [], []
    _frame_ids = []

    for _fi, _frame in enumerate(frames):
        _fr = all_results.get(_fi)
        if not _fr:
            continue
        _res_by_tag = {r["tagId"]: r for r in _fr}
        _oi, _ii, _ri = [], [], []
        for _tid, _ci, _obj in _obj_template:
            _r = _res_by_tag.get(_tid)
            if _r is None:
                _oi.clear()
                break
            _oi.append(_obj)
            _ii.append(_r["initCorners"][_ci].tolist())
            _ri.append(_r["finalCorners"][_ci].tolist())
        if len(_oi) < 6:
            continue
        _obj_pts.append(np.array(_oi, dtype=np.float32))
        _init_img_pts.append(np.array(_ii, dtype=np.float32))
        _ref_img_pts.append(np.array(_ri, dtype=np.float32))
        _frame_ids.append(_frame["frameId"])

    mo.stop(
        len(_obj_pts) < 3,
        mo.md(f"Need >=3 frames with shared tags (have {len(_obj_pts)})."),
    )

    _n_frames = len(_obj_pts)
    _n_corners = _obj_pts[0].shape[0]

    # iFixedPoint: anchor tag's TR corner (= corner 1 in strip order)
    _anchor_tr_idx = None
    for _i, (_tid, _ci, _obj) in enumerate(_obj_template):
        if _tid == _anchor["tagId"] and _ci == 1:
            _anchor_tr_idx = _i
            break

    _K_mat = np.array(
        [[_K["fx"], 0, _K["cx"]], [0, _K["fy"], _K["cy"]], [0, 0, 1]], dtype=np.float64
    )
    _dist = np.array(
        export_data.get("distortion", [0, 0, 0, 0, 0, 0, 0, 0]), dtype=np.float64
    )

    _flags = cv2.CALIB_USE_INTRINSIC_GUESS
    _criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10)

    _t0 = cv2.calibrateCameraRO(
        _obj_pts, _init_img_pts, (_w, _h), _anchor_tr_idx, _K_mat.copy(), _dist.copy(),
        flags=_flags, criteria=_criteria,
    )
    _rms_init, _K_init, _dist_init, _rvecs_init, _tvecs_init, _ = _t0

    _t1 = cv2.calibrateCameraRO(
        _obj_pts, _ref_img_pts, (_w, _h), _anchor_tr_idx, _K_mat.copy(), _dist.copy(),
        flags=_flags, criteria=_criteria,
    )
    _rms_ref, _K_ref, _dist_ref, _rvecs_ref, _tvecs_ref, _ = _t1

    # Per-view reprojection errors
    _per_view_init = []
    _per_view_ref = []
    for _vi in range(_n_frames):
        _pi, _ = cv2.projectPoints(
            _obj_pts[_vi], _rvecs_init[_vi], _tvecs_init[_vi], _K_init, _dist_init,
        )
        _pr, _ = cv2.projectPoints(
            _obj_pts[_vi], _rvecs_ref[_vi], _tvecs_ref[_vi], _K_ref, _dist_ref,
        )
        _per_view_init.append(
            float(np.sqrt(np.mean((_pi.reshape(-1, 2) - _init_img_pts[_vi]) ** 2)))
        )
        _per_view_ref.append(
            float(np.sqrt(np.mean((_pr.reshape(-1, 2) - _ref_img_pts[_vi]) ** 2)))
        )

    _per_view_rows = []
    for _vi in range(_n_frames):
        _per_view_rows.append({
            "Frame": _frame_ids[_vi],
            "Init RMS": f"{_per_view_init[_vi]:.4f}",
            "Refined RMS": f"{_per_view_ref[_vi]:.4f}",
            "Δ": f"{_per_view_init[_vi] - _per_view_ref[_vi]:+.4f}",
        })

    _corner_errs = {0: [], 1: [], 2: [], 3: []}
    for _vi in range(_n_frames):
        _proj, _ = cv2.projectPoints(
            _obj_pts[_vi], _rvecs_ref[_vi], _tvecs_ref[_vi], _K_ref, _dist_ref,
        )
        _proj = _proj.reshape(-1, 2)
        _img = _ref_img_pts[_vi]
        for _ci in range(_n_corners):
            _corner_errs[_ci % 4].append(
                float(np.sqrt(np.sum((_proj[_ci] - _img[_ci]) ** 2)))
            )
    _corner_names = ["TL", "TR", "BR", "BL"]
    _corner_rows = []
    for _ci in range(4):
        _errs = np.array(_corner_errs[_ci])
        _corner_rows.append({
            "Corner": _corner_names[_ci],
            "Mean (px)": f"{np.mean(_errs):.4f}",
            "Median (px)": f"{np.median(_errs):.4f}",
            "Max (px)": f"{np.max(_errs):.4f}",
            "Count": len(_errs),
        })

    mo.vstack([
        mo.md(
            f"## calibrateCameraRO ({_n_frames} frames, {_n_corners} corners)\n\n"
            f"| | Init | Refined | Δ |\n"
            f"|---|---|---|---|\n"
            f"| **RMS (px)** | {_rms_init:.4f} | {_rms_ref:.4f} | {_rms_init - _rms_ref:+.4f} |\n"
            f"| **fx** | {_K_init[0,0]:.2f} | {_K_ref[0,0]:.2f} | {_K_init[0,0] - _K_ref[0,0]:+.2f} |\n"
            f"| **fy** | {_K_init[1,1]:.2f} | {_K_ref[1,1]:.2f} | {_K_init[1,1] - _K_ref[1,1]:+.2f} |\n"
            f"| **cx** | {_K_init[0,2]:.2f} | {_K_ref[0,2]:.2f} | {_K_init[0,2] - _K_ref[0,2]:+.2f} |\n"
            f"| **cy** | {_K_init[1,2]:.2f} | {_K_ref[1,2]:.2f} | {_K_init[1,2] - _K_ref[1,2]:+.2f} |\n"
        ),
        mo.md("### Per-frame RMS"),
        mo.ui.table(_per_view_rows, selection=None, page_size=20),
        mo.md(
            f"Export calibration overall RMS: **{export_data.get('rmsPx', '?')}** px  "
            f"(for reference — may differ due to frame/tag subset)"
        ),
        mo.md("### Refined per-corner-index RMS"),
        mo.ui.table(_corner_rows, selection=None, page_size=10),
    ])


@app.cell
def _(all_results, frame_selector, frames, go, mo, np):
    """Quiver plot: direction and magnitude of LM corner movement (first frame)."""
    mo.stop(not all_results, mo.md("Run optimization first."))

    _fi = frame_selector.value
    _results = all_results.get(_fi, [])
    mo.stop(not _results, mo.md("No results for selected frame."))

    _frame = frames[_fi]
    _img = _frame["image"]
    _h, _w = _img.shape

    _fig = go.Figure()
    _fig.add_trace(
        go.Heatmap(
            z=_img, x0=0.5, dx=1.0, y0=0.5, dy=1.0,
            colorscale="gray", showscale=False, zmin=0.0, zmax=1.0,
        )
    )

    for _r in _results:
        _ic = _r["initCorners"]
        _fc = _r["finalCorners"]
        for _ci in range(4):
            _fig.add_trace(
                go.Scatter(
                    x=[_ic[_ci, 0], _fc[_ci, 0]],
                    y=[_ic[_ci, 1], _fc[_ci, 1]],
                    mode="lines+markers",
                    line=dict(color="red", width=1),
                    marker=dict(size=4, color=["red", "cyan"]),
                    showlegend=False,
                    hovertext=f"Tag {_r['tagId']} corner {_ci}: "
                              f"{np.linalg.norm(_fc[_ci] - _ic[_ci]):.2f} px",
                    hoverinfo="text",
                )
            )

    _fig.update_layout(
        dragmode="pan",
        xaxis=dict(range=[0, _w], constrain="domain", showgrid=False, zeroline=False),
        yaxis=dict(range=[_h, 0], constrain="domain", scaleanchor="x", scaleratio=1,
                   showgrid=False, zeroline=False),
        showlegend=False,
        height=min(900, max(700, _h)),
        title=f"Frame {_frame['frameId']} — corner movement direction (red→cyan)",
    )
    mo.ui.plotly(_fig, config={"scrollZoom": True, "displayModeBar": True})
    return


@app.cell
def _(all_results, mo, np, plt):
    mo.stop(not all_results, mo.md("Run optimization first."))

    _movements = []
    _tag_count = 0
    for _fr in all_results.values():
        for _r in _fr:
            _movements.extend(_r["cornerMovements"].tolist())
            _tag_count += 1
    _all_movements = np.array(_movements)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(_all_movements, bins=30, color="C0", alpha=0.8, edgecolor="white")
    _ax.axvline(np.mean(_all_movements), color="C3", linestyle="--", label=f"Mean: {np.mean(_all_movements):.2f} px")
    _ax.axvline(np.median(_all_movements), color="C2", linestyle="--", label=f"Median: {np.median(_all_movements):.2f} px")
    _ax.set_xlabel("Corner movement (pixels)")
    _ax.set_ylabel("Count")
    _ax.set_title(f"Corner movement distribution ({len(_all_movements)} corners, {_tag_count} tags, all frames)")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
