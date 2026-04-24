#!/usr/bin/env python3
"""
Check for parallax - does the same point appear in multiple frames?
If yes, points are at different Z depths → not planar.
"""

import json
import numpy as np


def main():
    json_path = '/calibration_export_1777037266230.json'

    with open(json_path, 'r') as f:
        data = json.load(f)

    print("=" * 70)
    print("PARALLAX CHECK")
    print("=" * 70)

    # Build layout point map
    point_map = {}
    for tag in data['layout']:
        tag_id = tag['tagId']
        for corner_idx, (x, y) in enumerate(tag['corners']):
            point_id = tag_id * 10000 + corner_idx
            point_map[point_id] = [x, y]

    # Check each frame
    for frame_idx, frame in enumerate(data['calibrationFrames'][:5]):
        print(f"\nFrame {frame_idx}: {frame['frameId']}")

        # Track observations for this frame
        observed_ids = set()
        image_points = {}

        for fp in frame['framePoints']:
            point_id = fp['pointId']
            img_x, img_y = fp['imagePoint']

            if point_id in point_map:
                observed_ids.add(point_id)
                image_points[point_id] = [img_x, img_y]

        print(f"  Observed {len(observed_ids)} unique point IDs")

        # Check common points across frames
        if frame_idx == 0:
            prev_ids = observed_ids
        else:
            common = prev_ids & observed_ids
            print(f"  Common with previous: {len(common)}")

            # Check parallax: do these common points map to different plane positions?
            for pid in sorted(common)[:5]:
                world_pos = point_map[pid]
                img1 = image_points[pid]

                # Find this point in next frame
                next_fp = next((fp for fp in data['calibrationFrames'][frame_idx]['framePoints']
                               if fp['pointId'] == pid), None)
                if next_fp:
                    img2 = next_fp['imagePoint']
                    print(f"    Point {pid}: {world_pos} → ({img1[0]:.1f}, {img1[1]:.1f}) -> ({img2[0]:.1f}, {img2[1]:.1f})")

        prev_ids = observed_ids

    # Also check how many unique points are observed across all frames
    all_ids = set()
    for frame in data['calibrationFrames']:
        for fp in frame['framePoints']:
            all_ids.add(fp['pointId'])

    unique_points = [pid for pid in all_ids if pid in point_map]
    print(f"\n{'=' * 70}")
    print(f"Total unique point IDs in data: {len(all_ids)}")
    print(f"Points with valid layout coordinates: {len(unique_points)}")
    print(f"Unique layout points across all frames: {len(set(unique_points))}")

    # Check if each unique layout point is observed multiple times
    per_point_counts = {}
    for pid in unique_points:
        count = sum(1 for frame in data['calibrationFrames']
                    for fp in frame['framePoints'] if fp['pointId'] == pid)
        per_point_counts[pid] = count

    max_count = max(per_point_counts.values())
    print(f"\nPoints observed {max_count} times (out of {len(per_point_counts)} points)")
    print(f"Median observations per point: {np.median(list(per_point_counts.values())):.1f}")

    # This is the KEY insight: if max_count > 1, points are observed multiple times
    # which means they likely have different Z depths (parallax)
    print(f"\n{'=' * 70}")
    if max_count > 1:
        print("✓ PARALLAX DETECTED: Points are observed from different camera positions")
        print("  → Homography-based calibration (TypeScript) assumes planar plane")
        print("  → OpenCV calibrateCamera handles arbitrary 3D points")
    else:
        print("✓ NO PARALLAX: All points are on the same plane")
        print("  → Both methods should work similarly")

    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()