"""Full-target keypoint heatmap detector lab."""

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

    from blob_lab import (
        HeatmapDatasetConfig,
        TinyUNet,
        TrainedUNetTorch,
        UNetTorchConfig,
        build_hex_target_layout,
        build_target_texture,
        detect_peaks,
        evaluate_heatmaps,
        generate_heatmap_dataset,
        load_target_export,
        load_unet_checkpoint,
        overlay_peaks,
        predict_heatmap_torch,
        save_unet_checkpoint,
        torch_unet_parameter_count,
        resolve_device,
        train_unet_torch,
    )

    return (
        HeatmapDatasetConfig,
        Path,
        TinyUNet,
        TrainedUNetTorch,
        UNetTorchConfig,
        build_hex_target_layout,
        build_target_texture,
        detect_peaks,
        evaluate_heatmaps,
        generate_heatmap_dataset,
        json,
        load_target_export,
        load_unet_checkpoint,
        np,
        overlay_peaks,
        plt,
        resolve_device,
        save_unet_checkpoint,
        torch_unet_parameter_count,
        train_unet_torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # Keypoint heatmap detector lab

    Renders the full hex target under perspective, trains a tiny fully convolutional
    U-Net to produce a **single-pixel heatmap at each square center**, and runs a
    separate peak detector to extract candidate points.

    Junctions and square interiors are not predicted as regions. The only positive
    pixels are the projected centers of the squares.
    """)
    return


@app.cell
def _(mo):
    target_json = mo.ui.file(filetypes=[".json"], label="Target export JSON (optional)")
    columns = mo.ui.slider(4, 16, value=7, step=1, label="Columns")
    points_per_edge = mo.ui.slider(4, 6, value=4, step=1, label="Points per edge")
    marker_scale = mo.ui.slider(0.25, 0.65, value=0.45, step=0.01, label="Marker scale")
    color_seed = mo.ui.number(value=1, label="Color seed")
    target_width = mo.ui.slider(640, 1536, value=1280, step=32, label="Target width")
    target_height = mo.ui.slider(480, 1152, value=960, step=24, label="Target height")
    mo.vstack(
        [
            mo.md("## Target layout"),
            target_json,
            mo.hstack([columns, points_per_edge]),
            mo.hstack([marker_scale, color_seed]),
            mo.hstack([target_width, target_height]),
        ]
    )
    return (
        color_seed,
        columns,
        marker_scale,
        points_per_edge,
        target_height,
        target_json,
        target_width,
    )


@app.cell
def _(
    build_hex_target_layout,
    build_target_texture,
    color_seed,
    columns,
    json,
    load_target_export,
    marker_scale,
    mo,
    np,
    points_per_edge,
    target_height,
    target_json,
    target_width,
):
    if target_json.value:
        _raw = target_json.contents()
        _data = json.loads(_raw) if isinstance(_raw, str) else json.loads(_raw.decode("utf-8"))
        layout = load_target_export(_data)
    else:
        layout = build_hex_target_layout(
            width_px=target_width.value,
            height_px=target_height.value,
            columns=columns.value,
            points_per_edge=points_per_edge.value,
            marker_scale=marker_scale.value,
            color_seed=color_seed.value,
        )

    target_texture = build_target_texture(layout)
    _rgb = np.repeat(target_texture[:, :, None], 3, axis=2)
    mo.vstack(
        [
            mo.md(
                f"**{len(layout.quads)} markers** · {layout.width_px}×{layout.height_px} px · "
                f"side {layout.side_px} px · spacing {layout.spacing_px} px"
            ),
            mo.image(
                (_rgb * 255).astype(np.uint8),
                width="100%",
                style={"image-rendering": "pixelated"},
            ),
        ]
    )
    return (layout,)


@app.cell
def _(Path, mo):
    sample_count = mo.ui.slider(16, 200, value=40, step=4, label="Total scenes")
    output_width = mo.ui.slider(320, 1280, value=640, step=64, label="Training width")
    output_height = mo.ui.slider(240, 960, value=480, step=48, label="Training height")
    natural_fraction = mo.ui.slider(0.0, 0.6, value=0.35, step=0.05, label="Natural negatives")
    natural_dir = mo.ui.text(
        value=str(Path(__file__).parent / "natural_images"),
        label="Natural image folder (optional)",
        full_width=True,
    )
    generate_button = mo.ui.run_button(label="Generate heatmaps")
    mo.vstack(
        [
            mo.md("## Dataset"),
            mo.md(
                "Target scenes use a small Gaussian around each square center for training; exact projected centers are retained for evaluation. "
                "The area mask is the union of the projected hex cells, not the full target rectangle. "
                "Square colors are randomized independently for each scene. Natural scenes have all-zero targets. "
                "Drop ordinary photos into the folder to use them; otherwise a smooth procedural fallback is used. "
                "The U-Net uses wider channel counts to give the encoder more representational capacity."
            ),
            mo.hstack([sample_count, output_width, output_height, natural_fraction]),
            natural_dir,
            generate_button,
        ]
    )
    return (
        generate_button,
        natural_dir,
        natural_fraction,
        output_height,
        output_width,
        sample_count,
    )


@app.cell
def _(
    HeatmapDatasetConfig,
    generate_button,
    generate_heatmap_dataset,
    layout,
    natural_dir,
    natural_fraction,
    output_height,
    output_width,
    sample_count,
):
    dataset = None
    if generate_button.value:
        dataset = generate_heatmap_dataset(
            HeatmapDatasetConfig(
                output_width=output_width.value,
                output_height=output_height.value,
                samples=sample_count.value,
                natural_fraction=natural_fraction.value,
                natural_dir=natural_dir.value or None,
                seed=7,
            ),
            layout,
        )
    return (dataset,)


@app.cell
def _(dataset, mo):
    upper = max(0, len(dataset) - 1) if dataset is not None else 0
    preview_index = mo.ui.slider(0, upper, value=0, step=1, label="Preview scene")
    preview_index
    return (preview_index,)


@app.cell(hide_code=True)
def _(dataset, mo, np, plt, preview_index):
    if dataset is None:
        dataset_preview = mo.md("Generate a dataset to preview it.")
    else:
        _idx = int(preview_index.value)
        _image = dataset.images[_idx, :, :, 0]
        _heat = dataset.heatmaps[_idx, :, :, 0]
        _mask = dataset.masks[_idx, :, :, 0]
        _junctions = dataset.junctions[_idx, :, :, 0]
        _heat_rgb = (plt.cm.inferno(np.clip(_heat, 0, 1))[..., :3] * 255).astype(np.uint8)
        _junction_rgb = (plt.cm.inferno(np.clip(_junctions, 0, 1))[..., :3] * 255).astype(np.uint8)
        dataset_preview = mo.vstack(
            [
                mo.md(
                    f"**{len(dataset)} scenes** · {sum(1 for m in dataset.metadata if m.get('natural'))} natural · "
                    f"{int(dataset.heatmaps.sum())} positive pixels · "
                    f"target size {dataset.config.output_width}×{dataset.config.output_height}"
                ),
                mo.md("### Input"),
                mo.image(_image, vmin=0, vmax=1, width="100%", style={"image-rendering": "pixelated"}),
                mo.md("### Small-Gaussian square-center target (exact centers retained for evaluation)"),
                mo.image(_heat_rgb, width="100%", style={"image-rendering": "pixelated"}),
                mo.hstack([mo.md("**0.0**"), mo.md("**1.0**")], justify="space-between"),
                mo.md("### Hexagon-contour target mask"),
                mo.image(
                    (_mask * 255).astype(np.uint8),
                    width="100%",
                    style={"image-rendering": "pixelated"},
                ),
                mo.md("### Small-Gaussian junction target"),
                mo.image(_junction_rgb, width="100%", style={"image-rendering": "pixelated"}),
            ]
        )
    dataset_preview
    return


@app.cell
def _(Path, mo):
    checkpoint_path = Path(__file__).parent / "blob_lab_data" / "unet_checkpoint.pt"
    filters = mo.ui.slider(4, 24, value=8, step=1, label="Conv filters")
    reset_button = mo.ui.run_button(label="Reset model checkpoint")
    mo.vstack([mo.md("## Model"), filters, reset_button])
    return checkpoint_path, filters, reset_button


@app.cell(hide_code=True)
def _(
    TinyUNet,
    checkpoint_path,
    filters,
    mo,
    reset_button,
    torch_unet_parameter_count,
):
    if reset_button.value and checkpoint_path.exists():
        checkpoint_path.unlink()
    _probe = TinyUNet(int(filters.value))
    _params = torch_unet_parameter_count(_probe)
    architecture = mo.md(
        "### Architecture\n\n"
        "```text\n"
        "input 4 channels: gray, |grad|, local variance, DoH\n"
        "encoder: f -> 2f -> 4f -> 8f bottleneck\n"
        "         1/1   1/2   1/4   1/8\n"
        "decoder: 8f -> 4f -> 2f -> f\n"
        "heads: center (full), junction (full)\n"
        "area: full-resolution decoder head\n"
        "no dilation; context comes from pooling\n"
        "```"
    )
    reset_status = mo.md(
        f"**Parameters:** {_params:,} · checkpoint: `{checkpoint_path}` · "
        + ("exists" if checkpoint_path.exists() else "empty")
    )
    mo.vstack([architecture, reset_status])
    return (reset_status,)


@app.cell
def _(mo):
    epochs = mo.ui.slider(5, 100, value=30, step=1, label="Epochs")
    batch_size = mo.ui.slider(1, 8, value=4, step=1, label="Batch size")
    train_button = mo.ui.run_button(label="Train / continue tiny U-Net")
    mo.vstack([mo.md("## Training"), mo.hstack([epochs, batch_size]), train_button])
    return batch_size, epochs, train_button


@app.cell(hide_code=True)
def _(
    TrainedUNetTorch,
    UNetTorchConfig,
    batch_size,
    checkpoint_path,
    dataset,
    epochs,
    filters,
    load_unet_checkpoint,
    mo,
    plt,
    reset_status,
    resolve_device,
    save_unet_checkpoint,
    torch_unet_parameter_count,
    train_button,
    train_unet_torch,
):
    model = None
    _ = reset_status  # resetting the checkpoint must invalidate this cell's cached model
    training_output = mo.md("Train the U-Net after generating a dataset.")
    initial_model = load_unet_checkpoint(checkpoint_path, filters.value, input_channels=4) if dataset is not None else None
    if initial_model is not None and not train_button.value:
        model = TrainedUNetTorch(
            initial_model,
            {},
            UNetTorchConfig(filters=filters.value),
            resolve_device(),
        )
        training_output = mo.md("Loaded the existing checkpoint. Press Train to continue it.")
    if train_button.value and dataset is not None:
        train, validation, test = dataset.split((0.7, 0.15, 0.15))
        resumed = initial_model is not None
        with mo.status.progress_bar(
            total=epochs.value,
            title="Training tiny U-Net (MPS)",
            subtitle="Starting…",
        ) as progress:
            def _on_epoch(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
                progress.update(
                    increment=1,
                    subtitle=f"Epoch {epoch}/{total} · train {train_loss:.4f} · val {val_loss:.4f}",
                )

            model = train_unet_torch(
                train.images,
                train.heatmaps,
                validation.images,
                validation.heatmaps,
                UNetTorchConfig(epochs=epochs.value, filters=filters.value, batch_size=batch_size.value, seed=3),
                on_epoch=_on_epoch,
                initial_model=initial_model,
                train_features=train.features,
                validation_features=validation.features,
                train_masks=train.masks,
                validation_masks=validation.masks,
                train_junctions=train.junctions,
                validation_junctions=validation.junctions,
            )
        save_unet_checkpoint(checkpoint_path, model.model, filters.value)
        _history = model.history
        _fig, _axis = plt.subplots(figsize=(5, 3))
        _axis.plot(_history["train_loss"], label="train")
        _axis.plot(_history["val_loss"], label="validation")
        _axis.set_xlabel("epoch")
        _axis.set_ylabel("weighted BCE")
        _axis.legend()
        training_output = mo.vstack(
            [
                mo.md(
                    f"**{torch_unet_parameter_count(model.model)} parameters** · "
                    f"{model.device} · {len(train)} train / {len(validation)} val / {len(test)} test scenes · "
                    + ("resumed from checkpoint" if resumed else "fresh model")
                ),
                _fig,
            ]
        )
    training_output
    return (model,)


@app.cell
def _(mo):
    peak_threshold = mo.ui.slider(0.05, 0.9, value=0.3, step=0.01, label="Peak threshold")
    nms_radius = mo.ui.slider(1, 5, value=2, step=1, label="NMS radius")
    evaluate_button = mo.ui.run_button(label="Evaluate peaks")
    mo.vstack([mo.md("## Peak detector"), mo.hstack([peak_threshold, nms_radius]), evaluate_button])
    return evaluate_button, nms_radius, peak_threshold


@app.cell(hide_code=True)
def _(dataset, evaluate_button, mo, model):
    test_bundle = None
    if evaluate_button.value and dataset is not None and model is not None:
        _train, _validation, _test = dataset.split((0.7, 0.15, 0.15))
        _predicted = model.score(_test.features)
        test_bundle = (_test, _predicted)
        prediction_output = mo.md(
            f"Cached predictions for **{len(_test)} test scenes** on `{model.device}`. "
            "Change the Test scene slider without rerunning the model."
        )
    else:
        prediction_output = mo.md("Train the model, then press **Evaluate peaks**.")
    prediction_output
    return (test_bundle,)


@app.cell(hide_code=True)
def _(dataset, mo):
    if dataset is None:
        _upper = 0
    else:
        _n = len(dataset)
        _n_train = int(_n * 0.7)
        _n_val = int(_n * 0.15)
        _upper = max(0, _n - _n_train - _n_val - 1)
    test_index = mo.ui.slider(0, _upper, value=0, step=1, label="Test scene")
    test_index
    return (test_index,)


@app.cell(hide_code=True)
def _(evaluate_heatmaps, mo, nms_radius, peak_threshold, test_bundle):
    test_metrics = None
    if test_bundle is not None:
        _test, _predicted = test_bundle
        test_metrics = evaluate_heatmaps(
            _predicted,
            _test.metadata,
            threshold=peak_threshold.value,
            nms_radius=nms_radius.value,
            tolerance=2.0,
        )
    else:
        _ = mo.md("No cached predictions yet.")
    return (test_metrics,)


@app.cell(hide_code=True)
def _(model, test_bundle, test_index):
    selected_aux = None
    if test_bundle is not None:
        _test, _predicted = test_bundle
        _idx = min(int(test_index.value), len(_predicted) - 1)
        _, _area, _junction = model.score_heads(_test.features[[_idx]])
        selected_aux = (_area[0], _junction[0])
    return (selected_aux,)


@app.cell(hide_code=True)
def _(
    detect_peaks,
    mo,
    nms_radius,
    np,
    overlay_peaks,
    peak_threshold,
    plt,
    selected_aux,
    test_bundle,
    test_index,
    test_metrics,
):
    evaluation_output = mo.md("Train and evaluate the model first.")
    if test_bundle is not None and test_metrics is not None:
        _test, _predicted = test_bundle
        _idx = min(int(test_index.value), len(_predicted) - 1)
        _area_selected, _junction_selected = selected_aux if selected_aux is not None else (None, None)
        _heat = np.clip(_predicted[_idx], 0, 1)
        _peaks = detect_peaks(
            _heat,
            threshold=peak_threshold.value,
            nms_radius=nms_radius.value,
        )
        _overlay = (overlay_peaks(_test.images[_idx], _peaks) * 255).astype(np.uint8)
        _heat_rgb = (plt.cm.inferno(np.clip(_heat, 0, 1))[..., :3] * 255).astype(np.uint8)
        _area_rgb = (plt.cm.inferno(np.clip(_area_selected, 0, 1))[..., :3] * 255).astype(np.uint8)
        _junction_rgb = (plt.cm.inferno(np.clip(_junction_selected, 0, 1))[..., :3] * 255).astype(np.uint8)
        _colorbar = (plt.cm.inferno(np.linspace(0, 1, 256))[None, :, :3] * 255).astype(np.uint8)
        _colorbar = np.repeat(_colorbar, 32, axis=0)
        evaluation_output = mo.vstack(
            [
                mo.md(
                    f"**Cached test set** · peaks {test_metrics.predicted_count} · "
                    f"truth {test_metrics.true_count} · precision {test_metrics.precision:.3f} · "
                    f"recall {test_metrics.recall:.3f} · F1 {test_metrics.f1:.3f} · "
                    f"mean error {test_metrics.mean_error_px:.2f}px"
                ),
                mo.md(
                    f"### Scene {_idx} · min {_heat.min():.4f} · mean {_heat.mean():.4f} · "
                    f"max {_heat.max():.4f}"
                ),
                mo.image(_heat_rgb, width="100%", style={"image-rendering": "pixelated"}),
                mo.image(
                    _colorbar,
                    width="100%",
                    style={"image-rendering": "pixelated"},
                ),
                mo.hstack([mo.md("**0.0**"), mo.md("inferno colormap"), mo.md("**1.0**")], justify="space-between"),
                mo.md("### Predicted hexagon-contour mask"),
                mo.image(_area_rgb, width="100%", style={"image-rendering": "pixelated"}),
                mo.md("### Predicted junction heatmap"),
                mo.image(_junction_rgb, width="100%", style={"image-rendering": "pixelated"}),
                mo.md(f"### Detected peaks ({len(_peaks)})"),
                mo.image(_overlay, width="100%", style={"image-rendering": "pixelated"}),
            ]
        )
    evaluation_output
    return


if __name__ == "__main__":
    app.run()
