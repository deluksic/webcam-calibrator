"""Compare SUPERSAMPLE=4 vs native-res rendering across real tag photos."""

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
    from dataclasses import replace
    from pathlib import Path

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
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
        homography_from_corners,
        imshow_extent,
        jnp,
        json,
        np,
        optimize_render_model,
        plt,
        render_with_model,
        replace,
        time,
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
            raise ValueError(f"Expected image_<id>_1x.png, got {name!r}")
        tag_part = stem.removeprefix("image_")
        if not tag_part.isdigit():
            raise ValueError(f"Expected numeric tag id, got {name!r}")
        return int(tag_part)

    return corners_json_path, tag_id_from_filename, test_images_dir


@app.cell
def _(mo):
    mo.md("""
    # Supersample experiment

    Compare **ss4** (4× supersample + bin), **ss1_psf** (native res + PSF),
    and **ss1_nopsf** (native res, PSF init 0) on all `*_1x` images with saved corners.

    Same `n_steps` and init for fair quality comparison; wall time varies.
    """)
    return


@app.cell
def _(
    Image,
    TAG_CANONICAL_CORNERS,
    bbox_mask,
    build_tag_pattern,
    corners_from_homography,
    corners_json_path,
    homography_from_corners,
    jnp,
    json,
    np,
    tag_id_from_filename,
    test_images_dir,
):
    src_corners = jnp.asarray(TAG_CANONICAL_CORNERS, dtype=jnp.float32)
    _exts = {".png", ".jpg", ".jpeg", ".webp"}
    samples = []

    for _image_path in sorted(test_images_dir.iterdir()):
        if _image_path.suffix.lower() not in _exts:
            continue
        if not _image_path.name.endswith("_1x.png"):
            continue
        _corners_path = corners_json_path(_image_path)
        if not _corners_path.is_file():
            continue

        _photo = np.asarray(Image.open(_image_path).convert("L"), dtype=np.float32) / 255.0
        _height, _width = _photo.shape
        _init_corners_px = np.asarray(json.loads(_corners_path.read_text()), dtype=np.float32)
        _H_init = homography_from_corners(src_corners, jnp.asarray(_init_corners_px))
        _tag_id = tag_id_from_filename(_image_path.name)
        _tag_pattern = build_tag_pattern(_tag_id)
        _init_image_corners = corners_from_homography(_H_init, src_corners)
        _loss_mask = bbox_mask(_init_image_corners, _height, _width, margin=3.5)

        samples.append(
            {
                "name": _image_path.name,
                "photo": _photo,
                "height": _height,
                "width": _width,
                "H_init": _H_init,
                "tag_pattern": _tag_pattern,
                "loss_mask": _loss_mask,
                "tag_id": _tag_id,
            }
        )

    if not samples:
        raise FileNotFoundError(
            f"No *_1x.png with .corners.json in {test_images_dir}"
        )
    return (samples,)


@app.cell
def _(OptimizeRenderConfig, RenderModelParams, jnp, samples):
    n_steps = 800
    learning_rate = 2e-3
    camera_lr_scale = 5.0

    def _camera_init(psf_sigma: float) -> RenderModelParams:
        return RenderModelParams(
            psf_sigma=jnp.float32(psf_sigma),
            sharpen_amount=jnp.float32(0.4),
            sharpen_sigma=jnp.float32(1.0),
            gamma=jnp.float32(2.0),
            black_level=jnp.float32(0.2),
            white_level=jnp.float32(0.8),
        )

    variant_specs = [
        ("ss4", 4, 1.0),
        ("ss1_psf", 1, 1.0),
        ("ss1_nopsf", 1, 0.0),
    ]

    variants = []
    for _vlabel, _vss, _psf_init in variant_specs:
        variants.append(
            (
                _vlabel,
                _vss,
                _camera_init(_psf_init),
                OptimizeRenderConfig(
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    corner_weight=0.0,
                    optimize_homography=True,
                    optimize_camera=True,
                    camera_lr_scale=camera_lr_scale,
                    supersample=_vss,
                ),
            )
        )

    sample_names = [s["name"] for s in samples]
    return sample_names, variants


@app.cell
def _(
    jnp,
    mo,
    np,
    optimize_render_model,
    render_with_model,
    replace,
    samples,
    time,
    variants,
):
    experiment_rows = []

    for _sample in samples:
        _target = jnp.asarray(_sample["photo"], dtype=jnp.float32)
        _height = _sample["height"]
        _width = _sample["width"]
        _loss_mask = _sample["loss_mask"]

        for _vlabel, _vss, _camera_init, _base_config in variants:
            _config = replace(_base_config, loss_mask=_loss_mask)
            _t0 = time.perf_counter()
            _H_opt, _camera_opt, _losses = optimize_render_model(
                _sample["H_init"],
                _target,
                _sample["tag_pattern"],
                _height,
                _width,
                camera_init=_camera_init,
                config=_config,
            )
            _wall_s = time.perf_counter() - _t0
            _opt_render = np.asarray(
                render_with_model(
                    _H_opt,
                    _sample["tag_pattern"],
                    _height,
                    _width,
                    _camera_opt,
                    supersample=_vss,
                )
            )
            experiment_rows.append(
                {
                    "image": _sample["name"],
                    "variant": _vlabel,
                    "supersample": _vss,
                    "wall_s": _wall_s,
                    "losses": _losses,
                    "final_loss": float(_losses[-1]),
                    "init_loss": float(_losses[0]),
                    "H_opt": _H_opt,
                    "camera_opt": _camera_opt,
                    "opt_render": _opt_render,
                    "photo": _sample["photo"],
                }
            )

    mo.md(f"Completed **{len(experiment_rows)}** runs ({len(samples)} images × {len(variants)} variants).")
    return (experiment_rows,)


@app.cell
def _(experiment_rows, mo, sample_names, variants):
    variant_labels = [v[0] for v in variants]
    _header = "| Image | " + " | ".join(variant_labels) + " |"
    _sep = "| --- | " + " | ".join(["---"] * len(variant_labels)) + " |"
    _lines = [_header, _sep]

    for _img_name in sample_names:
        _cells = []
        for _vlabel in variant_labels:
            _row = next(
                r for r in experiment_rows if r["image"] == _img_name and r["variant"] == _vlabel
            )
            _cells.append(f"{_row['final_loss']:.5f} / {_row['wall_s']:.1f}s")
        _lines.append("| " + _img_name + " | " + " | ".join(_cells) + " |")

    mo.md(
        "### Summary (final loss / wall time)\n\n" + "\n".join(_lines)
    )
    return (variant_labels,)


@app.cell
def _(experiment_rows, plt, sample_names, variant_labels):
    _n_images = len(sample_names)
    _fig, _axes = plt.subplots(
        _n_images, 1, figsize=(8, 2.5 * _n_images), squeeze=False
    )
    for _i, _img_name in enumerate(sample_names):
        _ax = _axes[_i, 0]
        for _vlabel in variant_labels:
            _row = next(
                r for r in experiment_rows if r["image"] == _img_name and r["variant"] == _vlabel
            )
            _ax.plot(_row["losses"], label=_vlabel, alpha=0.9)
        _ax.set_yscale("log")
        _ax.set_title(_img_name)
        _ax.set_xlabel("step")
        _ax.set_ylabel("loss")
        _ax.legend(fontsize=8)
        _ax.grid(True, alpha=0.3)
    _fig.suptitle("Loss curves by variant", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    centered_diff_limits,
    experiment_rows,
    imshow_extent,
    plt,
    sample_names,
    variant_labels,
):
    _n_images = len(sample_names)
    _n_variants = len(variant_labels)
    _fig, _axes = plt.subplots(
        _n_images,
        _n_variants,
        figsize=(3.2 * _n_variants, 3.0 * _n_images),
        squeeze=False,
    )

    for i, _img_name in enumerate(sample_names):
        for j, _vlabel in enumerate(variant_labels):
            _row = next(
                r for r in experiment_rows if r["image"] == _img_name and r["variant"] == _vlabel
            )
            _diff = _row["photo"] - _row["opt_render"]
            _vdiff = centered_diff_limits(_diff)
            _ax = _axes[i, j]
            _ax.imshow(
                _diff,
                cmap="RdBu_r",
                vmin=-_vdiff,
                vmax=_vdiff,
                extent=imshow_extent(_row["photo"].shape[1], _row["photo"].shape[0]),
            )
            _ax.set_title(f"{_img_name}\n{_vlabel}", fontsize=8)
            _ax.axis("off")

    _fig.suptitle("Photo − optimized (per variant)", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(experiment_rows, mo, sample_names, variant_labels):
    _header = "| Image | Variant | psf σ | sharpen | sharpen σ | γ | black | white |"
    _sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    _lines = [_header, _sep]

    for _img_name in sample_names:
        for _vlabel in variant_labels:
            _row = next(
                r for r in experiment_rows if r["image"] == _img_name and r["variant"] == _vlabel
            )
            _c = _row["camera_opt"]
            _lines.append(
                f"| {_img_name} | {_vlabel} | {float(_c.psf_sigma):.4f} | "
                f"{float(_c.sharpen_amount):.4f} | {float(_c.sharpen_sigma):.4f} | "
                f"{float(_c.gamma):.4f} | "
                f"{float(_c.black_level):.4f} | {float(_c.white_level):.4f} |"
            )

    mo.md("### Converged camera params\n\n" + "\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
