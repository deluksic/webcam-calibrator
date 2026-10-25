import unittest

import numpy as np

from blob_lab.local_peak_fit import (
    LocalPeakCandidate,
    _design_matrix,
    _solve_cubic_vertex,
    curvature_maps,
    detect_local_peaks,
    estimate_spacing,
    fit_peak,
    fit_peaks,
    smooth_luma,
    support_radii,
    support_radii_for_candidates,
)


def gaussian_image(
    cx: float,
    cy: float,
    *,
    sigma_x: float = 1.2,
    sigma_y: float = 1.2,
    angle: float = 0.0,
    sign: float = 1.0,
    size: int = 64,
    amplitude: float = 0.9,
) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    x = xx + 0.5 - cx
    y = yy + 0.5 - cy
    c = np.cos(angle)
    s = np.sin(angle)
    u = c * x + s * y
    v = -s * x + c * y
    image = sign * amplitude * np.exp(
        -(u * u / (2.0 * sigma_x * sigma_x) + v * v / (2.0 * sigma_y * sigma_y))
    )
    return image.astype(np.float32)


class CubicSolverTest(unittest.TestCase):
    def test_exact_cubic_vertex(self):
        coefficients = np.asarray(
            [0.0, 0.0, 0.0, -2.0, 0.1, -1.5, 0.2, -0.05, 0.03, -0.1],
            dtype=np.float64,
        )
        u0, v0 = 0.3, -0.2
        # Choose the linear terms so the gradient vanishes at (u0, v0).
        coefficients[1] = -(
            2.0 * coefficients[3] * u0
            + coefficients[4] * v0
            + 3.0 * coefficients[6] * u0**2
            + 2.0 * coefficients[7] * u0 * v0
            + coefficients[8] * v0**2
        )
        coefficients[2] = -(
            coefficients[4] * u0
            + 2.0 * coefficients[5] * v0
            + coefficients[7] * u0**2
            + 2.0 * coefficients[8] * u0 * v0
            + 3.0 * coefficients[9] * v0**2
        )
        grid = np.linspace(-1.0, 1.0, 5)
        uu, vv = np.meshgrid(grid, grid)
        design = _design_matrix(uu.ravel(), vv.ravel())
        values = design @ coefficients
        recovered = np.linalg.solve(design.T @ design, design.T @ values)
        np.testing.assert_allclose(recovered, coefficients, atol=1e-10)
        solved = _solve_cubic_vertex(recovered)
        self.assertIsNotNone(solved)
        assert solved is not None
        self.assertAlmostEqual(float(solved[0]), u0, places=5)
        self.assertAlmostEqual(float(solved[1]), v0, places=5)

    def test_saddle_has_no_maximum(self):
        coefficients = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(_solve_cubic_vertex(coefficients))


class LocalPeakFitTest(unittest.TestCase):
    def test_symmetric_gaussian_subpixel_center(self):
        truth = (30.37, 40.62)
        image = gaussian_image(*truth)
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidates = detect_local_peaks(smoothed, min_prominence=0.02)
        self.assertGreaterEqual(len(candidates), 1)
        peak = fit_peak(
            smoothed,
            candidates[0],
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="adaptive_circle_concave",
        )
        self.assertTrue(peak.fit_ok)
        error = float(np.hypot(peak.x - truth[0], peak.y - truth[1]))
        self.assertLess(error, 0.15)

    def test_black_marker_uses_negative_response(self):
        truth = (30.37, 40.62)
        image = gaussian_image(*truth, sign=-1.0)
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidates = detect_local_peaks(smoothed, min_prominence=0.02)
        self.assertEqual(candidates[0].polarity, 0)
        peak = fit_peak(
            smoothed,
            candidates[0],
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="adaptive_circle_concave",
        )
        self.assertTrue(peak.fit_ok)
        self.assertLess(float(np.hypot(peak.x - truth[0], peak.y - truth[1])), 0.2)

    def test_anisotropic_peak_uses_ellipse(self):
        truth = (30.0, 40.0)
        image = gaussian_image(
            *truth,
            sigma_x=1.0,
            sigma_y=2.5,
            angle=float(np.deg2rad(30.0)),
        )
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidates = detect_local_peaks(smoothed, min_prominence=0.02)
        compact = fit_peak(
            smoothed,
            candidates[0],
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="compact",
        )
        ellipse = fit_peak(
            smoothed,
            candidates[0],
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="adaptive_ellipse_concave",
        )
        self.assertEqual(ellipse.support_kind, "ellipse")
        self.assertTrue(ellipse.fit_ok)
        ellipse_error = float(np.hypot(ellipse.x - truth[0], ellipse.y - truth[1]))
        compact_error = float(np.hypot(compact.x - truth[0], compact.y - truth[1]))
        self.assertLess(ellipse_error, 0.1)
        self.assertLess(ellipse_error, compact_error)

    def test_flat_surface_falls_back(self):
        image = np.zeros((48, 48), dtype=np.float32)
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidate = LocalPeakCandidate(
            x=24,
            y=24,
            polarity=1,
            peak=0.0,
            valley=0.0,
            prominence=0.0,
        )
        peak = fit_peak(
            smoothed,
            candidate,
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="adaptive_circle_concave",
        )
        self.assertFalse(peak.fit_ok)
        self.assertTrue(peak.fallback)
        self.assertAlmostEqual(peak.x, 24.5)
        self.assertAlmostEqual(peak.y, 24.5)

    def test_plateau_collapses_to_one_candidate(self):
        image = np.zeros((48, 48), dtype=np.float32)
        image[20:25, 20:25] = 0.8
        smoothed = smooth_luma(image, 0.8)
        candidates = detect_local_peaks(smoothed, min_prominence=0.2)
        self.assertEqual(len(candidates), 1)

    def test_border_candidate_does_not_crash(self):
        image = gaussian_image(1.0, 1.0)
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidate = LocalPeakCandidate(
            x=0,
            y=0,
            polarity=1,
            peak=float(smoothed[0, 0]),
            valley=0.0,
            prominence=float(smoothed[0, 0]),
        )
        peak = fit_peak(
            smoothed,
            candidate,
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="adaptive_ellipse_concave",
        )
        self.assertIsNotNone(peak)
        self.assertTrue(np.isfinite(peak.x) and np.isfinite(peak.y))

    def test_connected_curvature_region(self):
        truth = (30.0, 40.0)
        image = gaussian_image(
            *truth,
            sigma_x=1.0,
            sigma_y=2.5,
            angle=float(np.deg2rad(30.0)),
        )
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidates = detect_local_peaks(smoothed, min_prominence=0.02)
        peak = fit_peak(
            smoothed,
            candidates[0],
            curvature,
            compact_radius=2.0,
            max_radius=3.0,
            mode="curvature_region",
        )
        self.assertEqual(peak.support_kind, "region")
        self.assertTrue(peak.used_curvature_mask)
        self.assertTrue(peak.fit_ok)
        self.assertLess(float(np.hypot(peak.x - truth[0], peak.y - truth[1])), 0.1)

    def test_local_support_radii_follow_density(self):
        dense = [
            LocalPeakCandidate(x=x, y=y, polarity=1, peak=1.0, valley=0.0, prominence=1.0)
            for y in range(0, 40, 4)
            for x in range(0, 40, 4)
        ]
        sparse = dense + [
            LocalPeakCandidate(x=200, y=200, polarity=1, peak=1.0, valley=0.0, prominence=1.0)
        ]
        radii = support_radii_for_candidates(sparse, fallback_spacing=10.0)
        sparse_radius = radii[-1][1]
        dense_radius = radii[0][1]
        self.assertGreater(sparse_radius, dense_radius)
        for compact, maximum in radii:
            self.assertLessEqual(compact, maximum)
            self.assertGreaterEqual(compact, 2.0)

    def test_all_modes_run_on_a_peak(self):
        image = gaussian_image(30.37, 40.62)
        smoothed = smooth_luma(image, 0.8)
        curvature = curvature_maps(smoothed)
        candidates = detect_local_peaks(smoothed, min_prominence=0.02)
        for mode in (
            "compact",
            "full",
            "adaptive_circle",
            "adaptive_circle_concave",
            "adaptive_ellipse_concave",
            "curvature_region",
        ):
            peak = fit_peak(
                smoothed,
                candidates[0],
                curvature,
                compact_radius=2.0,
                max_radius=3.0,
                mode=mode,
            )
            self.assertTrue(np.isfinite(peak.x))
            self.assertTrue(np.isfinite(peak.y))
            self.assertIn("support_radius", peak.descriptor_dict())

    def test_spacing_estimate(self):
        candidates = [
            LocalPeakCandidate(x=x, y=y, polarity=1, peak=1.0, valley=0.0, prominence=1.0)
            for y in range(0, 50, 10)
            for x in range(0, 50, 10)
        ]
        self.assertAlmostEqual(estimate_spacing(candidates), 10.0, places=5)

    def test_support_radii_scale_with_spacing(self):
        compact, maximum = support_radii(10.0)
        self.assertAlmostEqual(maximum, 3.0)
        self.assertAlmostEqual(compact, 2.0)
        compact, maximum = support_radii(20.0)
        self.assertAlmostEqual(maximum, 6.0)
        self.assertAlmostEqual(compact, 3.6)

    def test_low_prominence_noise_is_rejected(self):
        rng = np.random.default_rng(3)
        image = 0.5 + rng.normal(0.0, 0.005, size=(64, 64)).astype(np.float32)
        smoothed = smooth_luma(image, 0.8)
        candidates = detect_local_peaks(smoothed, min_prominence=0.02)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
