"""Fast smoke test for the full-target heatmap detector harness."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from blob_lab import (
    HeatmapDatasetConfig,
    UNetTorchConfig,
    build_hex_target_layout,
    build_target_texture,
    detect_peaks,
    evaluate_heatmaps,
    generate_heatmap_dataset,
    torch_unet_parameter_count,
    train_unet_torch,
)
from blob_lab.camera import CameraPose
from blob_lab.render import camera_homography, render_target


def main() -> None:
    layout = build_hex_target_layout(
        width_px=320,
        height_px=240,
        columns=7,
        points_per_edge=4,
        marker_scale=0.45,
        color_seed=1,
    )
    print("layout", len(layout.quads), layout.side_px, layout.spacing_px)
    texture = jnp.asarray(build_target_texture(layout))
    pose = CameraPose(0, 0, 0, 3.2, 2.3)
    h = camera_homography(
        yaw_deg=pose.yaw_deg,
        pitch_deg=pose.pitch_deg,
        roll_deg=pose.roll_deg,
        distance=pose.distance,
        focal=pose.focal,
        tx=pose.tx,
        ty=pose.ty,
        target_size=(layout.width_px, layout.height_px),
        output_size=(layout.width_px, layout.height_px),
    )
    image = render_target(h, texture, output_size=(layout.width_px, layout.height_px))
    gradient = jax.grad(
        lambda matrix: render_target(matrix, texture, output_size=(layout.width_px, layout.height_px)).sum()
    )(h)
    print(
        "render",
        image.shape,
        float(image.min()),
        float(image.max()),
        "grad_finite",
        bool(np.isfinite(np.asarray(gradient)).all()),
    )

    dataset = generate_heatmap_dataset(
        HeatmapDatasetConfig(
            output_width=128,
            output_height=96,
            samples=8,
            natural_fraction=0.25,
            seed=4,
        ),
        layout,
    )
    print(
        "dataset",
        dataset.images.shape,
        dataset.heatmaps.shape,
        "positive_pixels",
        float(dataset.heatmaps.sum()),
        "natural",
        sum(1 for meta in dataset.metadata if meta.get("natural")),
    )

    model = train_unet_torch(
        dataset.images,
        dataset.heatmaps,
        dataset.images,
        dataset.heatmaps,
        UNetTorchConfig(epochs=1, filters=3, batch_size=1, seed=1),
        train_features=dataset.features,
        validation_features=dataset.features,
        train_masks=dataset.masks,
        validation_masks=dataset.masks,
        train_junctions=dataset.junctions,
        validation_junctions=dataset.junctions,
    )
    print("unet", torch_unet_parameter_count(model.model), "device", model.device)
    heatmaps, areas, junctions = model.score_heads(dataset.features)
    peaks = detect_peaks(heatmaps[0], threshold=0.05, nms_radius=1)
    print("prediction", heatmaps.shape, "area", areas.shape, "junctions", junctions.shape, "peaks", len(peaks))
    print("metrics", evaluate_heatmaps(heatmaps, dataset.metadata, threshold=0.05, nms_radius=1))


if __name__ == "__main__":
    main()
