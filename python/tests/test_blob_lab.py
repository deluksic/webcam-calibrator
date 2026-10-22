import unittest

from blob_lab.target_layout import build_hex_target_layout, random_binary_colors


class TargetLayoutTest(unittest.TestCase):
    def test_default_layout_matches_export_values(self):
        layout = build_hex_target_layout(
            width_px=640,
            height_px=480,
            columns=8,
            points_per_edge=4,
            marker_scale=0.45,
            color_seed=1,
        )
        self.assertEqual(len(layout.quads), 476)
        self.assertEqual(layout.side_px, 4)
        self.assertEqual(layout.spacing_px, 10)
        first = layout.quads[0]
        self.assertEqual(
            (first.design_id, first.color, first.left, first.top, first.right, first.bottom),
            (0, 1, 88, 78, 92, 82),
        )

    def test_colors_match_mulberry32(self):
        self.assertEqual(random_binary_colors(8, 1), (1, 0, 1, 1, 1, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()
