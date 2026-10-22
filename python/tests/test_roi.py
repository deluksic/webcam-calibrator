import unittest

import numpy as np
import torch

from blob_lab.roi_dataset import RoiDatasetConfig, generate_roi_dataset
from blob_lab.roi_net import (
    RoiNetConfig,
    TinyRoiNet,
    largest_roi_polygon,
    mask_loss,
    parameter_count,
    roi_mask_from_probability,
)
from blob_lab.target_layout import build_hex_target_layout, build_target_texture


class RoiDatasetTest(unittest.TestCase):
    def test_browser_padding_is_preserved_in_texture(self):
        layout = build_hex_target_layout(
            width_px=640,
            height_px=480,
            columns=8,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
            padding_fraction=0.1,
        )
        self.assertEqual(layout.padding_x_px, 64)
        self.assertEqual(layout.padding_y_px, 48)
        texture = build_target_texture(layout)
        self.assertEqual(texture.shape, (480, 640))
        self.assertTrue(np.allclose(texture[:48], 0.5))
        self.assertTrue(np.allclose(texture[-48:], 0.5))
        self.assertTrue(np.allclose(texture[:, :64], 0.5))
        self.assertTrue(np.allclose(texture[:, -64:], 0.5))

    def test_positive_composite_contains_centers(self):
        layout = build_hex_target_layout(
            width_px=320,
            height_px=240,
            columns=4,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
            padding_fraction=0.1,
        )
        dataset = generate_roi_dataset(
            RoiDatasetConfig(
                output_width=160,
                output_height=120,
                samples=1,
                natural_fraction=0.0,
                natural_dir=None,
                target_width_fraction=(0.65, 0.65),
                mask_mode="hull",
                mask_margin=0.03,
                randomize_colors=False,
                seed=3,
            ),
            layout,
        )
        self.assertEqual(dataset.images.shape, (1, 120, 160, 1))
        self.assertEqual(dataset.masks.shape, (1, 120, 160, 1))
        self.assertEqual(float(dataset.presence[0]), 1.0)
        self.assertGreater(float(dataset.masks.sum()), 0.0)
        meta = dataset.metadata[0]
        self.assertGreater(len(meta["points"]), 0)
        contained = 0
        for x, y in meta["points"]:
            ix = int(np.clip(round(x), 0, 159))
            iy = int(np.clip(round(y), 0, 119))
            contained += int(dataset.masks[0, iy, ix, 0] > 0.5)
        self.assertGreater(contained / len(meta["points"]), 0.95)

    def test_horizontal_flip_mirrors_mask_and_points(self):
        layout = build_hex_target_layout(
            width_px=320,
            height_px=240,
            columns=4,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
        )
        dataset = generate_roi_dataset(
            RoiDatasetConfig(
                output_width=160,
                output_height=120,
                samples=1,
                natural_fraction=0.0,
                natural_dir=None,
                target_width_fraction=(0.65, 0.65),
                target_gray_range=(0.4, 0.4),
                randomize_colors=False,
                seed=6,
            ),
            layout,
        )
        flipped = dataset.flipped(horizontal=True)
        self.assertTrue(
            np.array_equal(flipped.masks, np.flip(dataset.masks, axis=2))
        )
        self.assertAlmostEqual(
            float(flipped.metadata[0]["board_gray"]),
            float(dataset.metadata[0]["board_gray"]),
        )
        original_point = dataset.metadata[0]["points"][0]
        flipped_point = flipped.metadata[0]["points"][0]
        self.assertAlmostEqual(flipped_point[0], 159.0 - original_point[0], places=4)
        self.assertAlmostEqual(flipped_point[1], original_point[1], places=4)
        self.assertEqual(len(dataset.augmented(horizontal=True)), 2)

    def test_closeup_blur_metadata(self):
        layout = build_hex_target_layout(
            width_px=320,
            height_px=240,
            columns=4,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
        )
        dataset = generate_roi_dataset(
            RoiDatasetConfig(
                output_width=160,
                output_height=120,
                samples=1,
                natural_fraction=0.0,
                natural_dir=None,
                closeup_fraction=1.0,
                closeup_width_fraction=(0.8, 0.8),
                heavy_blur_fraction=1.0,
                heavy_blur_sigma=(1.5, 1.5),
                frame_blur_fraction=1.0,
                frame_blur_sigma=(1.0, 1.0),
                canvas_aspects=(4 / 3, 16 / 9),
                randomize_colors=False,
                seed=8,
            ),
            layout,
        )
        meta = dataset.metadata[0]
        self.assertTrue(meta["closeup"])
        self.assertTrue(meta["heavy_blur"])
        self.assertGreaterEqual(meta["blur_scale"], 0.25)
        self.assertLessEqual(meta["blur_scale"], 0.5)
        self.assertLess(meta["blur_sigma"], 0.8)
        self.assertLess(meta["frame_blur_sigma"], 0.8)
        self.assertIn(round(meta["canvas_aspect"], 3), (1.333, 1.778))

    def test_cropped_target_is_visible_and_metadata_marked(self):
        layout = build_hex_target_layout(
            width_px=320,
            height_px=240,
            columns=4,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
        )
        dataset = generate_roi_dataset(
            RoiDatasetConfig(
                output_width=160,
                output_height=120,
                samples=3,
                natural_fraction=0.0,
                natural_dir=None,
                cropped_fraction=1.0,
                cropped_width_fraction=(0.8, 0.8),
                cropped_min_visible_fraction=0.02,
                randomize_colors=False,
                seed=10,
            ),
            layout,
        )
        self.assertTrue(all(meta["cropped"] for meta in dataset.metadata))
        self.assertTrue(any(meta["visible_fraction"] < 0.999 for meta in dataset.metadata))
        self.assertGreater(float(dataset.masks.sum()), 0.0)

    def test_negative_composite_has_empty_mask(self):
        layout = build_hex_target_layout(
            width_px=320,
            height_px=240,
            columns=4,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
        )
        dataset = generate_roi_dataset(
            RoiDatasetConfig(
                output_width=160,
                output_height=120,
                samples=2,
                natural_fraction=1.0,
                natural_dir=None,
                randomize_colors=False,
                seed=4,
            ),
            layout,
        )
        self.assertTrue(np.all(dataset.presence == 0.0))
        self.assertEqual(float(dataset.masks.sum()), 0.0)


class RoiNetTest(unittest.TestCase):
    def test_forward_shapes_and_parameters(self):
        config = RoiNetConfig(
            channels=(4, 8, 16, 32),
            input_width=96,
            input_height=72,
            output_stride=2,
        )
        model = TinyRoiNet(config)
        self.assertLess(parameter_count(model), 100_000)
        logits, presence = model(torch.randn(2, 1, 72, 96))
        self.assertEqual(tuple(logits.shape), (2, 1, 36, 48))
        self.assertEqual(tuple(presence.shape), (2, 1))

    def test_false_negative_weight_increases_loss(self):
        logits = torch.full((1, 1, 8, 8), -4.0)
        targets = torch.ones((1, 1, 8, 8))
        plain = mask_loss(logits, targets, false_negative_weight=1.0)
        weighted = mask_loss(logits, targets, false_negative_weight=5.0)
        self.assertGreater(float(weighted), float(plain))

    def test_largest_roi_polygon(self):
        mask = np.zeros((60, 80), dtype=np.float32)
        mask[15:45, 20:60] = 1.0
        polygon = largest_roi_polygon(mask)
        self.assertEqual(polygon.shape[1], 2)
        self.assertGreater(polygon.shape[0], 3)
        self.assertGreater(float(polygon[:, 0].min()), 15.0)
        self.assertLess(float(polygon[:, 0].max()), 65.0)

    def test_full_res_mask_from_probability(self):
        probability = np.zeros((36, 48), dtype=np.float32)
        probability[10:26, 14:34] = 1.0
        mask = roi_mask_from_probability(
            probability, 160, 120, threshold=0.5, margin_fraction=0.0
        )
        self.assertEqual(mask.shape, (120, 160))
        self.assertGreater(float(mask.sum()), 0.0)
        self.assertEqual(float(mask[60, 80]), 1.0)


if __name__ == "__main__":
    unittest.main()
