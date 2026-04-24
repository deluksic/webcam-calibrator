#!/usr/bin/env python3
"""
Compute individual homographies for each frame and analyze errors.
This shows whether the homography solver itself is working correctly.
"""

import json
import numpy as np


def solve_homography_dlt(points_plane, points_image):
    """8-point DLT algorithm for homography H."""
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
    print("INDIVIDUAL HOMOGRAPHY ANALYSIS")
    print("=" * 70)

    # Parse pointId → (world_x, world_y)
    point_map = {}
    for tag in data['layout']:
        tag_id = tag['tagId']
        for corner_idx, (x, y) in enumerate(tag['corners']):
            point_id = tag_id * 10000 + corner_idx
            point_map[point_id] = [x, y]

    frame_rms = []

    for i, frame in enumerate(data['calibrationFrames']):
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
            print(f"Frame {i}: Skip ({len(plane_pts)} points)")
            continue

        plane_pts = np.array(plane_pts)
        img_pts = np.array(img_pts)

        # Compute homography
        h = solve_homography_dlt(plane_pts, img_pts)

        if h is None:
            print(f"Frame {i}: FAILED to compute homography")
            continue

        # Compute RMS from this homography
        errors = homography_residual(h, plane_pts, img_pts)
        rms = np.sqrt(np.mean(errors ** 2))

        frame_rms.append({
            'frame': i,
            'frameId': frame['frameId'],
            'num_points': len(plane_pts),
            'rms': rms,
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
        })

    print(f"\n{'=' * 70}")
    print(f"Successfully computed {len(frame_rms)}/{len(data['calibrationFrames'])} homographies")
    print(f"{'=' * 70}\n")

    if frame_rms:
        print(f"{'Frame':<8} {'ID':<20} {'Points':<8} {'RMS':<12} {'Min':<8} {'Median':<10} {'Max':<8}")
        print("-" * 74)
        for r in frame_rms:
            print(f"{r['frame']:<8} {r['frameId']:<20} {r['num_points']:<8} "
                  f"{r['rms']:<12.3f} {r['min_error']:<8.3f} {r['median_error']:<10.3f} {r['max_error']:<8.3f}")

        # Summary
        rms_values = [r['rms'] for r in frame_rms]
        print(f"\n{'=' * 70}")
        print(f"RMS Statistics:")
        print(f"  Mean: {np.mean(rms_values):.3f} px")
        print(f"  Median: {np.median(rms_values):.3f} px")
        print(f"  Std dev: {np.std(rms_values):.3f} px")
        print(f"  Min: {np.min(rms_values):.3f} px")
        print(f"  Max: {np.max(rms_values):.3f} px")
        print(f"{'=' * 70}")

        # Check if homographies are consistent
        print(f"\n{'=' * 70}")
        print(f"PARALLAX ANALYSIS")
        print(f"{'=' * 70}")

        # Extract observations for first point
        first_point_id = next(iter(point_map.keys()))
        first_obs = []

        for i, frame in enumerate(data['calibrationFrames']):
            for fp in frame['framePoints']:
                if fp['pointId'] == first_point_id:
                    first_obs.append({
                        'frame': i,
                        'frameId': frame['frameId'],
                        'imgPoint': fp['imagePoint'],
                    })
                    break

        if len(first_obs) >= 2:
            print(f"\nFirst point (ID={first_point_id}) observed from {len(first_obs)} frames:")
            print(f"  World coordinates: {point_map[first_point_id]}")
            print(f"\n  Image positions:")
            for obs in first_obs[:5]:
                print(f"    Frame {obs['frame']} ({obs['frameId']}): ({obs['imgPoint'][0]:.1f}, {obs['imgPoint'][1]:.1f})")

            # Calculate spread
            img_spreads = []
            for j in range(1, len(first_obs)):
                p1 = np.array(first_obs[j-1]['imgPoint'])
                p2 = np.array(first_obs[j]['imgPoint'])
                dist = np.linalg.norm(p1 - p2)
                img_spreads.append(dist)

            print(f"\n  Camera movement (first 4 observations):")
            for j, spread in enumerate(img_spreads[:4]):
                print(f"    Movement {j+1}: {spread:.1f} px")

            avg_movement = np.mean(img_spreads)
            print(f"\n  Average movement: {avg_movement:.1f} px per frame")
            print(f"\n  ⚠ Large image movement suggests points are NOT on a plane")
            print(f"     → Homography-based calibration will be inherently limited")

    print(f"\n{'=' * 70}")


if __name__ == '__main__':
    main()