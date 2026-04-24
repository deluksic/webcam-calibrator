#!/usr/bin/env python3
"""
Compare homography computation between Python and TypeScript.
This helps isolate where the 68% RMS discrepancy comes from.
"""

import json
import numpy as np


def solve_homography_dlt(points_plane, points_image):
    """
    8-point DLT algorithm for homography H.
    planes: N×2 array of (X,Y) world coordinates
    images: N×2 array of (u,v) image coordinates
    """
    N = len(points_plane)
    if N < 4:
        return None

    # Build A matrix (2N×9)
    A = []

    for i in range(N):
        X, Y = points_plane[i]
        u, v = points_image[i]

        # Row for u = (h0*X + h1*Y + h2) / (h6*X + h7*Y + h8)
        A.append([-X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
        # Row for v = (h3*X + h4*Y + h5) / (h6*X + h7*Y + h8)
        A.append([-X, -Y, -1, 0, 0, 0, X*v, Y*v, v])

    A = np.array(A, dtype=np.float64)

    # SVD to find null space (last column of V)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # Last row of Vt

    # Normalize: H[8] = 1
    if abs(h[8]) < 1e-12:
        return None

    h = h / h[8]
    return h.reshape(3, 3)


def homography_residual(h, points_plane, points_image):
    """Compute reprojection error for homography."""
    h = h.reshape(3, 3)
    h_inv = np.linalg.inv(h)

    errors = []
    for i in range(len(points_plane)):
        X, Y = points_plane[i]
        u, v = points_image[i]

        # Transform world point to image
        denom = h_inv[0, 2] * X + h_inv[1, 2] * Y + 1
        u_proj = (h_inv[0, 0] * X + h_inv[1, 0] * Y + h_inv[2, 0]) / denom
        v_proj = (h_inv[0, 1] * X + h_inv[1, 1] * Y + h_inv[2, 1]) / denom

        errors.append(np.sqrt((u - u_proj) ** 2 + (v - v_proj) ** 2))

    return np.array(errors)


def main():
    json_path = '/calibration_export_1777037266230.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("=" * 70)
    print("HOMOGRAPHY COMPARISON TEST")
    print("=" * 70)

    # Get calibration frames
    frames = data['calibrationFrames']
    print(f"\nTotal frames: {len(frames)}")

    # Analyze each frame individually
    for i, frame in enumerate(frames[:3]):  # Test first 3 frames
        print(f"\n{'=' * 70}")
        print(f"Frame {i}: {frame['frameId']}")
        print(f"{'=' * 70}")

        # Parse pointId → (world_x, world_y)
        point_map = {}
        for tag in data['layout']:
            tag_id = tag['tagId']
            for corner_idx, (x, y) in enumerate(tag['corners']):
                point_id = tag_id * 10000 + corner_idx
                point_map[point_id] = [x, y]

        # Extract correspondences for this frame
        plane_pts = []
        img_pts = []

        for fp in frame['framePoints']:
            point_id = fp['pointId']
            img_x, img_y = fp['imagePoint']

            if point_id in point_map:
                world_x, world_y = point_map[point_id]
                plane_pts.append([world_x, world_y])
                img_pts.append([img_x, img_y])

        if len(plane_pts) < 4:
            print(f"  Skip: only {len(plane_pts)} points")
            continue

        plane_pts = np.array(plane_pts)
        img_pts = np.array(img_pts)

        print(f"  Points: {len(plane_pts)}")

        # Check if points are consistent (planar)
        # Compute spread in world coordinates
        x_spread = np.max(plane_pts[:, 0]) - np.min(plane_pts[:, 0])
        y_spread = np.max(plane_pts[:, 1]) - np.min(plane_pts[:, 1])
        print(f"  World coordinate spread: X={x_spread:.2f}, Y={y_spread:.2f}")

        # Compute homography
        h = solve_homography_dlt(plane_pts, img_pts)
        if h is None:
            print("  ERROR: Could not compute homography")
            print("    This may indicate inconsistent planar projection")
            continue

        h_norm = h.reshape(3, 3)
        print(f"\n  Homography (row-major, H[8]=1):")
        for row in h_norm:
            print(f"    {row}")

        # Compute RMS from this homography
        errors = homography_residual(h_norm, plane_pts, img_pts)
        rms = np.sqrt(np.mean(errors ** 2))
        print(f"\n  RMS from homography: {rms:.3f} px")
        print(f"  RMS stats: min={np.min(errors):.3f}, median={np.median(errors):.3f}, max={np.max(errors):.3f}")

        # Show some sample residuals
        print(f"\n  Sample residuals:")
        for j in range(min(5, len(errors))):
            print(f"    Point {j}: {errors[j]:.3f} px")

    # Compare homographies directly if possible
    print(f"\n{'=' * 70}")
    print("PARALLAX CHECK")
    print(f"{'=' * 70}")

    # Extract first two frames and check if they represent the same plane
    if len(frames) >= 2:
        frame1 = frames[0]
        frame2 = frames[1]

        # Build point maps for both frames
        point_map1 = {}
        for tag in data['layout']:
            tag_id = tag['tagId']
            for corner_idx, (x, y) in enumerate(tag['corners']):
                point_id = tag_id * 10000 + corner_idx
                point_map1[point_id] = [x, y]

        # Extract point pairs that appear in both frames
        points_in_both = []

        fp1_ids = {fp['pointId'] for fp in frame1['framePoints']}
        fp2_ids = {fp['pointId'] for fp in frame2['framePoints']}

        common_ids = sorted(fp1_ids & fp2_ids)

        print(f"\n  Common points across frames 0 and 1: {len(common_ids)}")

        for pid in common_ids[:10]:  # Check first 10
            if pid in point_map1:
                world_pt = point_map1[pid]
                img_pt1 = next(fp['imagePoint'] for fp in frame1['framePoints'] if fp['pointId'] == pid)
                img_pt2 = next(fp['imagePoint'] for fp in frame2['framePoints'] if fp['pointId'] == pid)

                print(f"    Point {pid}: {world_pt} → ({img_pt1}, {img_pt2})")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()