import unittest

import numpy as np

from blob_lab.augment import (
    CameraSharpenConfig,
    SharpenParams,
    apply_camera_sharpen,
    sample_augmentation_rng,
    sample_sharpen_params,
)
from blob_lab.camera import (
    CameraPose,
    CameraRanges,
    PoseAugmentConfig,
    local_foreshortening,
    pose_tilt_deg,
    random_pose,
    tilt_bin,
)
from blob_lab.heatmap_dataset import HeatmapDatasetConfig, generate_heatmap_dataset
from blob_lab.roi_dataset import RoiDatasetConfig, generate_roi_dataset
from blob_lab.target_layout import build_hex_target_layout


def _step_image() -> np.ndarray:
    image = np.zeros((32, 32, 1), dtype=np.float32)
    image[:, :16] = 0.2
    image[:, 16:] = 0.8
    return image


class CameraSharpenTest(unittest.TestCase):
    def test_probability_zero_is_noop(self):
        config = CameraSharpenConfig()
        params = sample_sharpen_params(config, sample_augmentation_rng(1, 0), probability=0.0)
        self.assertIsNone(params)
        image = _step_image()
        output = apply_camera_sharpen(image, None)
        np.testing.assert_allclose(output, image)

    def test_step_edge_gains_halo(self):
        params = SharpenParams(
            mode="usm",
            amount=1.5,
            sigma=1.0,
            threshold=0.0,
            space="linear",
            overshoot_limit=None,
        )
        output = apply_camera_sharpen(_step_image(), params)
        self.assertGreater(float(output[16, 17, 0]), 0.8)
        self.assertLess(float(output[16, 14, 0]), 0.2)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_multiscale_display_space_and_overshoot(self):
        params = SharpenParams(
            mode="multiscale",
            amount=1.0,
            sigma=0.8,
            threshold=0.005,
            space="linear",
            secondary_sigma=1.8,
            secondary_weight=0.4,
            overshoot_limit=0.05,
        )
        output = apply_camera_sharpen(_step_image(), params)
        self.assertEqual(output.shape, (32, 32, 1))
        self.assertTrue(np.isfinite(output).all())
        delta = output - _step_image()
        self.assertLessEqual(float(np.abs(delta).max()), 0.05 + 1e-6)

        display = SharpenParams(
            mode="multiscale",
            amount=1.0,
            sigma=0.8,
            threshold=0.005,
            space="display",
            secondary_sigma=1.8,
            secondary_weight=0.4,
            overshoot_limit=None,
        )
        display_output = apply_camera_sharpen(_step_image(), display)
        self.assertEqual(display_output.shape, (32, 32, 1))
        self.assertTrue(np.isfinite(display_output).all())
        self.assertGreaterEqual(float(display_output.min()), 0.0)
        self.assertLessEqual(float(display_output.max()), 1.0)

    def test_sampling_is_deterministic(self):
        config = CameraSharpenConfig()
        first = sample_sharpen_params(config, sample_augmentation_rng(3, 4), probability=1.0)
        second = sample_sharpen_params(config, sample_augmentation_rng(3, 4), probability=1.0)
        self.assertEqual(first, second)
        self.assertIsNotNone(first)


class PoseAugmentTest(unittest.TestCase):
    def test_pose_tilt_is_roll_independent(self):
        base = CameraPose(50.0, 20.0, 0.0, 3.0, 2.2)
        rolled = CameraPose(50.0, 20.0, 40.0, 3.0, 2.2)
        self.assertAlmostEqual(pose_tilt_deg(base), pose_tilt_deg(rolled), places=6)
        self.assertAlmostEqual(pose_tilt_deg(CameraPose(50.0, 0.0, 0.0, 3.0, 2.2)), 50.0, places=5)

    def test_oblique_branch_meets_min_tilt(self):
        config = PoseAugmentConfig(
            oblique_fraction=1.0,
            oblique_yaw_deg=(35.0, 65.0),
            oblique_pitch_deg=(20.0, 45.0),
            min_oblique_tilt_deg=40.0,
            max_attempts=16,
        )
        rng = np.random.default_rng(11)
        poses = [random_pose(rng, CameraRanges(), config) for _ in range(200)]
        oblique = [pose for pose in poses if pose.pose_branch == "oblique"]
        self.assertGreater(len(oblique), 190)
        for pose in oblique:
            self.assertGreaterEqual(pose_tilt_deg(pose), 40.0 - 1e-6)

    def test_zero_oblique_fraction_uses_uniform(self):
        rng = np.random.default_rng(5)
        poses = [
            random_pose(rng, CameraRanges(), PoseAugmentConfig(oblique_fraction=0.0))
            for _ in range(50)
        ]
        self.assertTrue(all(pose.pose_branch == "uniform" for pose in poses))

    def test_local_foreshortening(self):
        identity = np.eye(3, dtype=np.float64)
        ratio, tilt = local_foreshortening(identity, (0.0, 0.0))
        self.assertAlmostEqual(ratio, 1.0, places=6)
        self.assertAlmostEqual(tilt, 0.0, places=6)
        compressed = np.asarray([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ratio, tilt = local_foreshortening(compressed, (0.0, 0.0))
        self.assertAlmostEqual(ratio, 0.5, places=6)
        self.assertAlmostEqual(tilt, 60.0, places=5)

    def test_tilt_bin_boundaries(self):
        self.assertEqual(tilt_bin(0.0), "frontal")
        self.assertEqual(tilt_bin(24.9), "frontal")
        self.assertEqual(tilt_bin(25.0), "moderate")
        self.assertEqual(tilt_bin(40.0), "oblique")
        self.assertEqual(tilt_bin(55.0), "steep")
        self.assertEqual(tilt_bin(None), "unknown")


class DatasetAugmentationTest(unittest.TestCase):
    def _layout(self):
        return build_hex_target_layout(
            width_px=320,
            height_px=240,
            columns=4,
            points_per_edge=4,
        )

    def test_heatmap_sharpening_changes_image_not_target(self):
        clean_config = HeatmapDatasetConfig(
            output_width=160,
            output_height=120,
            samples=1,
            natural_fraction=0.0,
            blur_sigma=(0.8, 0.8),
            noise_sigma=(0.0, 0.0),
            randomize_colors=False,
            min_visible_marker_fraction=0.1,
            seed=9,
            pose_augment=PoseAugmentConfig(oblique_fraction=0.0),
            sharpen=CameraSharpenConfig(probability=0.0),
        )
        sharp_config = HeatmapDatasetConfig(
            output_width=160,
            output_height=120,
            samples=1,
            natural_fraction=0.0,
            blur_sigma=(0.8, 0.8),
            noise_sigma=(0.0, 0.0),
            randomize_colors=False,
            min_visible_marker_fraction=0.1,
            seed=9,
            pose_augment=PoseAugmentConfig(oblique_fraction=0.0),
            sharpen=CameraSharpenConfig(
                probability=1.0,
                amount=(1.5, 1.5),
                sigma=(1.0, 1.0),
                threshold=(0.0, 0.0),
                multiscale_probability=0.0,
                display_space_probability=0.0,
                overshoot_clip_probability=0.0,
            ),
        )
        clean = generate_heatmap_dataset(clean_config, self._layout(), count=1)
        sharp = generate_heatmap_dataset(sharp_config, self._layout(), count=1)
        np.testing.assert_array_equal(clean.heatmaps, sharp.heatmaps)
        np.testing.assert_array_equal(clean.masks, sharp.masks)
        self.assertFalse(np.allclose(clean.images, sharp.images))
        clean_meta = clean.metadata[0]
        sharp_meta = sharp.metadata[0]
        self.assertIsNone(clean_meta["sharpen"])
        self.assertIsNotNone(sharp_meta["sharpen"])
        self.assertEqual(len(sharp_meta["points"]), len(sharp_meta["point_targets"]))
        self.assertEqual(len(sharp_meta["points"]), len(sharp_meta["point_colors"]))
        self.assertIsNotNone(sharp_meta["H"])
        self.assertIsNotNone(sharp_meta["tilt_deg"])

    def test_roi_sharpening_changes_image_not_mask(self):
        base = dict(
            output_width=160,
            output_height=120,
            samples=1,
            natural_fraction=0.0,
            target_width_fraction=(0.6, 0.6),
            mask_mode="hull",
            target_gray_range=(0.5, 0.5),
            randomize_colors=False,
            seed=4,
            pose_augment=PoseAugmentConfig(oblique_fraction=0.0),
        )
        clean = generate_roi_dataset(
            RoiDatasetConfig(
                **base,
                sharpen=CameraSharpenConfig(probability=0.0),
            ),
            self._layout(),
            count=1,
        )
        sharp = generate_roi_dataset(
            RoiDatasetConfig(
                **base,
                sharpen=CameraSharpenConfig(
                    probability=1.0,
                    amount=(1.5, 1.5),
                    sigma=(1.0, 1.0),
                    threshold=(0.0, 0.0),
                    multiscale_probability=0.0,
                    display_space_probability=0.0,
                    overshoot_clip_probability=0.0,
                ),
            ),
            self._layout(),
            count=1,
        )
        np.testing.assert_array_equal(clean.masks, sharp.masks)
        self.assertFalse(np.allclose(clean.images, sharp.images))
        self.assertIsNone(clean.metadata[0]["sharpen"])
        self.assertIsNotNone(sharp.metadata[0]["sharpen"])


if __name__ == "__main__":
    unittest.main()
