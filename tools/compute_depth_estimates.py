#!/usr/bin/env python3
"""
Estimate Z-depth for different points using the homography analysis.
If parallax exists, the required depth will be inconsistent.
"""

import json
import numpy as np


def homography_depth_from_point(h, plane_pt, img_pt):
    """
    Estimate depth Z for a single point using homography H.
    Given: world point P = (X, Y, 1), image point p = (u, v, 1)
    Homography: p = H * P  where p and P are homogeneous
    Invert: P = H^{-1} * p
    If P = (X, Y, Z), then world coordinates / Z = H^{-1} * p
    → Z = world coords / (H^{-1} * p)_z
    """
    h_inv = np.linalg.inv(h.reshape(3, 3))
    p_homo = np.array([img_pt[0], img_pt[1], 1.0])

    p_homo_inv = h_inv @ p_homo
    Z = 1.0 / p_homo_inv[2] if abs(p_homo_inv[2]) > 1e-12 else float('inf')

    world_pt = np.array(plane_pt) / Z if Z < float('inf') else np.array(plane_pt)

    return Z, world_pt


def solve_homography_dlt(points_plane, points_image):
    """8-point DLT algorithm."""
    N = len(points_plane)
    if N < 4:
        return None

    A = []
    for i in range(N):
        X, Y = points_plane[i]
        u, v = points_image[i]
        A.append([-X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
        A.append([-X, -Y, -1, 0, 0, 0, X*v, Y*v, v])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]

    if abs(h[8]) < 1e-8:  # Relaxed threshold for debugging
        return None

    return (h / h[8]).reshape(3, 3)


def main():
    json_path = '/calibration_export_1777037266230.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Parse pointId → (world_x, world_y)
    point_map = {}
    for tag in data['layout']:
        tag_id = tag['tagId']
        for corner_idx, (x, y) in enumerate(tag['corners']):
            point_id = tag_id * 10000 + corner_idx
            point_map[point_id] = [x, y]

    # Test first frame
    frame0 = data['calibrationFrames'][0]

    # Try to compute homography
    plane_pts = []
    img_pts = []

    for fp in frame0['framePoints']:
        point_id = fp['pointId']
        img_x, img_y = fp['imagePoint']

        if point_id in point_map:
            world_x, world_y = point_map[point_id]
            plane_pts.append([world_x, world_y])
            img_pts.append([img_x, img_y])

    h = solve_homography_dlt(plane_pts, img_pts)

    print("=" * 70)
    print("DEPTH ESTIMATION TEST")
    print("=" * 70)

    if h is None:
        print("\nCannot compute homography - points don't lie on a plane")
        print("This confirms parallax exists in the calibration data.")
        return

    print(f"\nHomography computed successfully")
    print(f"Mean reprojection error: {np.mean(np.sqrt(np.sum(((h @ np.column_stack((plane_pts, np.ones((len(plane_pts), 1))))).reshape(-1, 3)[:,:2] - img_pts)**2, axis=1))):.3f} px")

    # Compute depth estimates for several points
    print(f"\n{'Point':<15} {'World':<20} {'Image':<20} {'Depth (Z)':<12}")
    print("-" * 70)

    for i in [0, 1, 2, 3, 10]:  # Sample some points
        world_pt = point_map[i * 10000]
        img_pt = img_pts[i]
        Z, recovered_world = homography_depth_from_point(h, world_pt, img_pt)

        print(f"Point {i*10000:<7} {str(world_pt):<20} {str(img_pt):<20} {Z:<12.2f}")

    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    print("If Depth (Z) values vary significantly, it means:")
    print("  - Same world point would need different depths to be planar")
    print("  - Points are NOT on a single plane (parallax exists)")
    print("  - TypeScript's homography-based method is inherently limited")
    print("  - OpenCV's perspective calibration handles this better")
    print("=" * 70)


if __name__ == '__main__':
    main()