#!/usr/bin/env python3
"""
Debug script to compare TypeScript and OpenCV calibration algorithms step-by-step.
Tests if point normalization is the key difference.
"""

import json
import sys
import numpy as np
import cv2
from pathlib import Path

def load_calibration_data(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_v12(H):
    """Compute v12 vector from homography H."""
    h1 = H[:, 0]
    h2 = H[:, 1]

    h1x, h1y, h1z = h1[0], h1[1], h1[2]
    h2x, h2y, h2z = h2[0], h2[1], h2[2]

    v12 = np.array([
        h1x * h2x,
        h1x * h2y + h2x * h1y,
        h1y * h2y,
        h1x * h2z + h2x * h1z,
        h1y * h2z + h2y * h1z,
        h1z * h2z
    ])
    return v12


def compute_v11_minus_v22(H):
    """Compute v11 - v22 vector from homography H."""
    h1 = H[:, 0]
    h2 = H[:, 1]

    h1x, h1y, h1z = h1[0], h1[1], h1[2]
    h2x, h2y, h2z = h2[0], h2[1], h2[2]

    v11 = np.array([
        h1x * h1x,
        h1x * h1y,
        h1y * h1y,
        h1x * h1z,
        h1y * h1z,
        h1z * h1z
    ])

    v22 = np.array([
        h2x * h2x,
        h2x * h2y,
        h2y * h2y,
        h2x * h2z,
        h2y * h2z,
        h2z * h2z
    ])

    return v11 - v22


def extract_K_from_B(B):
    """Extract K matrix from B vector using Zhang's formula."""
    B11, B12, B22, B13, B23, B33 = B

    denom = B11 * B22 - B12 * B12
    cy = (B12 * B13 - B11 * B23) / denom
    λ = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(λ / B11)
    fy = np.sqrt(λ * B11 / denom)
    cx = (-B13 * fx * fx) / λ

    return fx, fy, cx, cy


def normalize_points_2d(points):
    """Normalize 2D points: mean 0, RMS = sqrt(2)."""
    pts = np.array(points, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    avg_dist = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    scale = np.sqrt(2) / (avg_dist if avg_dist > 1e-10 else 1)
    normalized = centered * scale
    return normalized, centroid, scale


def zhang_calibration_python(homographies, image_size, normalize=True):
    """Zhang's calibration with optional normalization."""
    num_views = len(homographies)
    print(f"\n=== Zhang Calibration ({'Normalized' if normalize else 'Raw'}) with {num_views} homographies ===")

    # Build V matrix
    V = []
    for H in homographies:
        v12 = compute_v12(H)
        v11_minus_v22 = compute_v11_minus_v22(H)
        V.append(v12)
        V.append(v11_minus_v22)

    V = np.array(V)
    print(f"V matrix shape: {V.shape}")
    print(f"V matrix condition number: {np.linalg.cond(V):.2e}")

    # Solve V * b = 0 using SVD
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1, :]

    # Normalize b so B is positive definite
    B11, B12, B22, B13, B23, B33 = b
    B_matrix = np.array([
        [B11, B12, B13],
        [B12, B22, B23],
        [B13, B23, B33]
    ])
    eigenvalues = np.linalg.eigvalsh(B_matrix)

    # Ensure positive definite
    if eigenvalues[0] < 0 or B33 < 0:
        b = -b
        B11, B12, B22, B13, B23, B33 = b

    print(f"Singular values: {S}")
    print(f"B eigenvalues: {eigenvalues}")

    # Extract K from B
    fx, fy, cx, cy = extract_K_from_B(b)
    print(f"K: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    return fx, fy, cx, cy


def compute_homographies_normalized_dlt(src_points_list, dst_points_list):
    """Compute homographies using normalized DLT."""
    homographies = []

    for src_pts, dst_pts in zip(src_points_list, dst_points_list):
        src_pts = np.array(src_pts, dtype=np.float64)
        dst_pts = np.array(dst_pts, dtype=np.float64)

        # Normalize source points
        src_norm, src_mean, src_scale = normalize_points_2d(src_pts)

        # Normalize destination points
        dst_norm, dst_mean, dst_scale = normalize_points_2d(dst_pts)

        # Build A matrix for normalized points
        n = len(src_norm)
        A = np.zeros((2*n, 9))
        for i in range(n):
            x, y = src_norm[i]
            xp, yp = dst_norm[i]
            A[2*i] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp]
            A[2*i+1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp]

        # SVD
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :]

        # Denormalize
        T1 = np.array([
            [src_scale, 0, -src_scale * src_mean[0]],
            [0, src_scale, -src_scale * src_mean[1]],
            [0, 0, 1]
        ])
        T2_inv = np.array([
            [1/dst_scale, 0, dst_mean[0]],
            [0, 1/dst_scale, dst_mean[1]],
            [0, 0, 1]
        ])

        H_norm = h.reshape(3, 3)
        H = T2_inv @ H_norm @ T1

        # Normalize so H[2,2] = 1
        H = H / H[2, 2]

        homographies.append(H)

    return homographies


def compute_homographies_standard_dlt(src_points_list, dst_points_list):
    """Compute homographies using standard DLT (no normalization)."""
    homographies = []

    for src_pts, dst_pts in zip(src_points_list, dst_points_list):
        src_pts = np.array(src_pts, dtype=np.float64)
        dst_pts = np.array(dst_pts, dtype=np.float64)

        # Build A matrix
        n = len(src_pts)
        A = np.zeros((2*n, 9))
        for i in range(n):
            x, y = src_pts[i]
            xp, yp = dst_pts[i]
            A[2*i] = [x, y, 1, 0, 0, 0, -x*xp, -y*xp, -xp]
            A[2*i+1] = [0, 0, 0, x, y, 1, -x*yp, -y*yp, -yp]

        # SVD
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :]

        H = h.reshape(3, 3)
        # Normalize so H[2,2] = 1
        H = H / H[2, 2]

        homographies.append(H)

    return homographies


def compute_homographies_opencv(src_points_list, dst_points_list):
    """Compute homographies using OpenCV findHomography."""
    homographies = []

    for src_pts, dst_pts in zip(src_points_list, dst_points_list):
        H, mask = cv2.findHomography(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32),
            method=0
        )
        homographies.append(H)

    return homographies


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 debug_algorithm.py <calibration_export.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_calibration_data(json_path)

    print("=" * 70)
    print("ALGORITHM COMPARISON: TypeScript vs OpenCV")
    print("=" * 70)

    image_size = (int(data['resolution']['width']), int(data['resolution']['height']))
    print(f"\nImage size: {image_size}")

    # Load TypeScript calibration result
    ts_result = data.get('tsCalibResult')
    if not ts_result:
        print("ERROR: No TypeScript calibration result found in export")
        sys.exit(1)

    print(f"\nTypeScript K from export:")
    print(f"  fx = {ts_result['K']['fx']:.4f}")
    print(f"  fy = {ts_result['K']['fy']:.4f}")
    print(f"  cx = {ts_result['K']['cx']:.4f}")
    print(f"  cy = {ts_result['K']['cy']:.4f}")
    print(f"  RMS = {ts_result['rmsPx']:.4f}")

    # Build point_map
    point_map = {}
    for tag in data['layout']:
        tag_id = tag['tagId']
        for corner_idx, (x, y) in enumerate(tag['corners']):
            point_id = tag_id * 10000 + corner_idx
            point_map[point_id] = ([x, y], tag_id, corner_idx)

    # Scale
    all_x = []
    all_y = []
    for tag in data['layout']:
        for corner in tag['corners']:
            all_x.append(corner[0])
            all_y.append(corner[1])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width_units = max_x - min_x
    height_units = max_y - min_y

    img_width, img_height = image_size
    scale = max(img_width / width_units, img_height / height_units) * 0.8

    # Collect point correspondences for each frame
    src_points_list = []
    dst_points_list = []

    for frame in data['calibrationFrames']:
        src_pts = []
        dst_pts = []

        for fp in frame['framePoints']:
            point_id = fp['pointId']
            img_x, img_y = fp['imagePoint']

            if point_id in point_map:
                world_pos, _, _ = point_map[point_id]
                obj_x = (world_pos[0] - min_x) * scale
                obj_y = (world_pos[1] - min_y) * scale
                src_pts.append([float(obj_x), float(obj_y)])
                dst_pts.append([float(img_x), float(img_y)])

        if len(src_pts) >= 4:
            src_points_list.append(src_pts)
            dst_points_list.append(dst_pts)

    print(f"\nCollected {len(src_points_list)} frames with point correspondences")

    # Test different homography computation methods
    print("\n" + "=" * 70)
    print("HOMOGRAPHY COMPUTATION COMPARISON")
    print("=" * 70)

    # OpenCV findHomography
    H_opencv = compute_homographies_opencv(src_points_list, dst_points_list)
    print(f"\nOpenCV findHomography - First H:")
    print(H_opencv[0])

    # Standard DLT
    H_std = compute_homographies_standard_dlt(src_points_list, dst_points_list)
    print(f"\nStandard DLT - First H:")
    print(H_std[0])

    # Normalized DLT
    H_norm = compute_homographies_normalized_dlt(src_points_list, dst_points_list)
    print(f"\nNormalized DLT - First H:")
    print(H_norm[0])

    # Compare homographies
    print("\n--- Homography differences ---")
    print(f"OpenCV vs Standard DLT: max diff = {np.max(np.abs(H_opencv[0] - H_std[0])):.2e}")
    print(f"OpenCV vs Normalized DLT: max diff = {np.max(np.abs(H_opencv[0] - H_norm[0])):.2e}")

    # Run Zhang with different homography sources
    print("\n" + "=" * 70)
    print("ZHANG CALIBRATION WITH DIFFERENT HOMOGRAPHIES")
    print("=" * 70)

    results = {}

    # OpenCV homographies
    fx, fy, cx, cy = zhang_calibration_python(H_opencv, image_size)
    results['OpenCV H'] = (fx, fy, cx, cy)

    # Standard DLT homographies
    fx, fy, cx, cy = zhang_calibration_python(H_std, image_size)
    results['Standard DLT H'] = (fx, fy, cx, cy)

    # Normalized DLT homographies
    fx, fy, cx, cy = zhang_calibration_python(H_norm, image_size)
    results['Normalized DLT H'] = (fx, fy, cx, cy)

    # OpenCV calibrateCamera
    print("\n" + "=" * 70)
    print("OPENCV CALIBRATECAMERA")
    print("=" * 70)

    frame_obj_points = []
    frame_img_points = []

    for src_pts, dst_pts in zip(src_points_list, dst_points_list):
        obj_pts = np.array([[p[0], p[1], 0.0] for p in src_pts], dtype=np.float32)
        img_pts = np.array(dst_pts, dtype=np.float32)
        frame_obj_points.append(obj_pts)
        frame_img_points.append(img_pts)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        frame_obj_points,
        frame_img_points,
        image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_USE_QR
    )

    print(f"RMS: {rms:.4f}")
    print(f"K: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    results['calibrateCamera'] = (K[0,0], K[1,1], K[0,2], K[1,2])

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Method':<20} {'fx':<12} {'fy':<12} {'cx':<12} {'cy':<12}")
    print("-" * 70)
    for name, (fx, fy, cx, cy) in results.items():
        print(f"{name:<20} {fx:<12.2f} {fy:<12.2f} {cx:<12.2f} {cy:<12.2f}")


if __name__ == '__main__':
    main()
