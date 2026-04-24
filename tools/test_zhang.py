#!/usr/bin/env python3
"""
Debug Zhang's calibration with proper homography normalization.
"""

import numpy as np
import cv2

def create_synthetic_calibration():
    """Create synthetic calibration data with known ground truth."""
    fx, fy = 1000.0, 1050.0
    cx, cy = 640.0, 360.0
    skew = 0.0

    K_gt = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    w, h = 1280, 720

    # Create synthetic layout (grid of points)
    grid_rows, grid_cols = 6, 8
    spacing = 0.05

    layout_points = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            layout_points.append([c * spacing, r * spacing, 0])

    layout_points = np.array(layout_points, dtype=np.float64)

    # Create synthetic views
    views = []
    np.random.seed(42)

    for i in range(10):
        rx = np.random.uniform(-0.3, 0.3)
        ry = np.random.uniform(-0.3, 0.3)
        rz = np.random.uniform(-0.1, 0.1)

        tz = np.random.uniform(1.0, 2.0)
        tx = np.random.uniform(-0.3, 0.3)
        ty = np.random.uniform(-0.2, 0.2)

        rvec = np.array([rx, ry, rz])
        R, _ = cv2.Rodrigues(rvec)
        t = np.array([tx, ty, tz])

        projected = []
        for p in layout_points:
            Xc = R @ p + t
            x = Xc[0] / Xc[2]
            y = Xc[1] / Xc[2]
            u = fx * x + skew * y + cx
            v = fy * y + cy
            projected.append([u, v])

        projected = np.array(projected, dtype=np.float64)
        noise = np.random.normal(0, 0.5, projected.shape)
        projected_noisy = projected + noise

        views.append({
            'object_points': layout_points.copy(),
            'image_points': projected_noisy,
            'R': R,
            't': t
        })

    return K_gt, w, h, views


def normalize_homography(H):
    """
    Normalize homography so that ||h1||^2 + ||h2||^2 = 2.

    This is what OpenCV's calibrateCamera does internally.
    """
    # h1 and h2 are columns 0 and 1
    h1 = H[:, 0]
    h2 = H[:, 1]

    norm_sq = np.dot(h1, h1) + np.dot(h2, h2)

    if norm_sq < 1e-10:
        return H

    # Scale factor
    s = np.sqrt(2.0 / norm_sq)

    return H * s


def compute_v_ij(h_i, h_j):
    """Compute v_ij vector from homography columns h_i and h_j."""
    return np.array([
        h_i[0] * h_j[0],
        h_i[0] * h_j[1] + h_i[1] * h_j[0],
        h_i[1] * h_j[1],
        h_i[2] * h_j[0] + h_i[0] * h_j[2],
        h_i[2] * h_j[1] + h_i[1] * h_j[2],
        h_i[2] * h_j[2]
    ])


def zhang_calibration(homographies, normalize=True):
    """Zhang's calibration."""
    # Build V matrix
    V = []
    for H in homographies:
        if normalize:
            H = normalize_homography(H)

        h1 = H[:, 0]
        h2 = H[:, 1]

        v12 = compute_v_ij(h1, h2)
        v11_minus_v22 = compute_v_ij(h1, h1) - compute_v_ij(h2, h2)

        V.append(v12)
        V.append(v11_minus_v22)

    V = np.array(V)

    # SVD
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1, :]

    # Build B matrix
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    eigenvalues = np.linalg.eigvalsh(B)

    # Ensure positive definite
    if eigenvalues[0] < 0:
        b = -b
        B = -B
        eigenvalues = -eigenvalues

    # Extract K
    B11, B12, B22, B13, B23, B33 = b

    denom = B11 * B22 - B12 * B12
    cy = (B12 * B13 - B11 * B23) / denom
    λ = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(λ / B11)
    fy = np.sqrt(λ * B11 / denom)
    cx = (-B13 * fx * fx) / λ

    return fx, fy, cx, cy, eigenvalues


def main():
    print("=" * 70)
    print("ZHANG CALIBRATION WITH NORMALIZATION")
    print("=" * 70)

    K_gt, w, h, views = create_synthetic_calibration()

    print(f"\nGround truth K:")
    print(f"  fx={K_gt[0,0]:.2f}, fy={K_gt[1,1]:.2f}, cx={K_gt[0,2]:.2f}, cy={K_gt[1,2]:.2f}")

    # Compute homographies
    homographies = []
    for view in views:
        obj_pts = view['object_points'][:, :2]
        img_pts = view['image_points']
        H, _ = cv2.findHomography(obj_pts, img_pts, method=0)
        homographies.append(H)

    print(f"\nComputed {len(homographies)} homographies")

    # Test without normalization
    print("\n" + "-" * 70)
    print("WITHOUT homography normalization:")
    fx, fy, cx, cy, ev = zhang_calibration(homographies, normalize=False)
    print(f"  K: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  B eigenvalues: {ev}")

    # Test with normalization
    print("\n" + "-" * 70)
    print("WITH homography normalization:")
    fx, fy, cx, cy, ev = zhang_calibration(homographies, normalize=True)
    print(f"  K: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  B eigenvalues: {ev}")

    # OpenCV calibrateCamera
    print("\n" + "-" * 70)
    print("OpenCV calibrateCamera:")

    obj_pts_list = [v['object_points'] for v in views]
    img_pts_list = [v['image_points'] for v in views]

    rms, K_cv, dist, rvecs, tvecs = cv2.calibrateCamera(
        [o.astype(np.float32) for o in obj_pts_list],
        [i.astype(np.float32) for i in img_pts_list],
        (w, h),
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_USE_QR
    )

    print(f"  RMS={rms:.4f}")
    print(f"  K: fx={K_cv[0,0]:.2f}, fy={K_cv[1,1]:.2f}, cx={K_cv[0,2]:.2f}, cy={K_cv[1,2]:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<25} {'fx':<12} {'fy':<12} {'cx':<12} {'cy':<12}")
    print("-" * 70)
    print(f"{'Ground Truth':<25} {K_gt[0,0]:<12.2f} {K_gt[1,1]:<12.2f} {K_gt[0,2]:<12.2f} {K_gt[1,2]:<12.2f}")

    fx_n, fy_n, cx_n, cy_n, _ = zhang_calibration(homographies, normalize=False)
    print(f"{'Zhang (no norm)':<25} {fx_n:<12.2f} {fy_n:<12.2f} {cx_n:<12.2f} {cy_n:<12.2f}")

    fx_y, fy_y, cx_y, cy_y, _ = zhang_calibration(homographies, normalize=True)
    print(f"{'Zhang (normalized)':<25} {fx_y:<12.2f} {fy_y:<12.2f} {cx_y:<12.2f} {cy_y:<12.2f}")

    print(f"{'OpenCV QR':<25} {K_cv[0,0]:<12.2f} {K_cv[1,1]:<12.2f} {K_cv[0,2]:<12.2f} {K_cv[1,2]:<12.2f}")


if __name__ == '__main__':
    main()
