#!/usr/bin/env python3
"""Debug why homography computation fails."""

import json
import numpy as np


def solve_homography_dlt_debug(points_plane, points_image):
    """8-point DLT with debugging."""
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

    print(f"  A shape: {A.shape}")
    print(f"  A norm: {np.linalg.norm(A)}")
    print(f"  Condition number: {np.linalg.cond(A)}")

    # Check rank
    print(f"  Rank: {np.linalg.matrix_rank(A)}")

    # Check if all zeros in any column
    col_sums = np.sum(np.abs(A), axis=0)
    print(f"  Column sum norms: {col_sums}")

    # SVD
    _, s, Vt = np.linalg.svd(A)
    print(f"  Singular values: {s}")
    print(f"  Ratio s[0]/s[-1]: {s[0]/s[-1] if len(s) > 0 else 'N/A'}")

    # Check null space
    null_vec = Vt[-1, :]
    print(f"  Null vector norm: {np.linalg.norm(null_vec)}")
    print(f"  h[8]: {null_vec[8]}")

    # Check if h[8] is near zero
    if abs(null_vec[8]) < 1e-12:
        print(f"  ⚠ h[8] is near zero - cannot normalize")
        return None

    h = null_vec / null_vec[8]
    return h.reshape(3, 3)


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

    # Test on frame 0
    frame = data['calibrationFrames'][0]

    plane_pts = []
    img_pts = []

    for fp in frame['framePoints']:
        point_id = fp['pointId']
        img_x, img_y = fp['imagePoint']

        if point_id in point_map:
            world_x, world_y = point_map[point_id]
            plane_pts.append([world_x, world_y])
            img_pts.append([img_x, img_y])

    plane_pts = np.array(plane_pts)
    img_pts = np.array(img_pts)

    print(f"Frame 0:")
    print(f"  Number of points: {len(plane_pts)}")
    print(f"  World coordinates: min X={plane_pts[:, 0].min():.2f}, max X={plane_pts[:, 0].max():.2f}")
    print(f"               min Y={plane_pts[:, 1].min():.2f}, max Y={plane_pts[:, 1].max():.2f}")
    print(f"  Image coordinates: min u={img_pts[:, 0].min():.1f}, max u={img_pts[:, 0].max():.1f}")
    print(f"               min v={img_pts[:, 1].min():.1f}, max v={img_pts[:, 1].max():.1f}\n")

    print(f"Attempting DLT...")
    h = solve_homography_dlt_debug(plane_pts, img_pts)

    if h is not None:
        print(f"\n✓ Homography computed successfully:")
        print(f"  H[0,0]={h[0,0]:.6f}, H[0,1]={h[0,1]:.6f}, H[0,2]={h[0,2]:.6f}")
        print(f"  H[1,0]={h[1,0]:.6f}, H[1,1]={h[1,1]:.6f}, H[1,2]={h[1,2]:.6f}")
        print(f"  H[2,0]={h[2,0]:.6f}, H[2,1]={h[2,1]:.6f}, H[2,2]={h[2,2]:.6f}")


if __name__ == '__main__':
    main()