#!/usr/bin/env python3
"""
Calibration analysis tool comparing TypeScript vs OpenCV calibration approaches.

This script loads calibration data exported from CalibrationView and performs:
1. Layout analysis (regularity, spacing uniformity)
2. Initial RMS using TS camera matrix
3. OpenCV linear calibration (pinhole model)
4. OpenCV distorted calibration (radial + tangential distortion)
5. Comprehensive comparison report
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


def load_calibration_data(json_path: str) -> Dict:
    """Load calibration export JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_layout(layout: List[Dict]) -> Dict:
    """
    Analyze layout regularity:
    - Extract corners from each tag
    - Check if points form a regular grid
    - Spacing uniformity (variance)
    - Missing points
    """
    if not layout:
        return {'grid': 'insufficient data', 'rows': 0, 'cols': 0}

    # Extract all corner positions
    x_coords = []
    y_coords = []
    for tag in layout:
        for corner in tag['corners']:
            x_coords.append(corner[0])
            y_coords.append(corner[1])

    # Sort unique positions by tagId and cornerId (tagId*10000 + cornerId)
    sorted_layout = sorted(layout, key=lambda p: p['tagId'])

    # Find grid dimensions by analyzing gaps
    unique_x = sorted(set(x_coords))
    unique_y = sorted(set(y_coords))

    # Estimate grid size using gap analysis
    x_gaps = np.diff(unique_x)
    y_gaps = np.diff(unique_y)

    # Most common gap is likely the grid spacing
    unique_gaps_x, counts_x = np.unique(x_gaps[x_gaps > 0], return_counts=True)
    unique_gaps_y, counts_y = np.unique(y_gaps[y_gaps > 0], return_counts=True)

    avg_gap_x = float(np.mean(x_gaps[x_gaps > 0])) if len(x_gaps) > 0 else 0
    avg_gap_y = float(np.mean(y_gaps[y_gaps > 0])) if len(y_gaps) > 0 else 0

    # Calculate spacing variance (uniformity)
    spacing_variance_x = float(np.var(x_gaps[x_gaps > 0])) if len(x_gaps) > 0 else 0
    spacing_variance_y = float(np.var(y_gaps[y_gaps > 0])) if len(y_gaps) > 0 else 0

    # Check if most gaps are equal (uniform spacing)
    uniform_spacing_x = float(counts_x[0] / len(x_gaps[x_gaps > 0])) if len(unique_gaps_x) > 0 else 0
    uniform_spacing_y = float(counts_y[0] / len(y_gaps[y_gaps > 0])) if len(unique_gaps_y) > 0 else 0

    return {
        'num_points': len(layout),
        'x_range': (min(x_coords), max(x_coords)),
        'y_range': (min(y_coords), max(y_coords)),
        'avg_gap_x': avg_gap_x,
        'avg_gap_y': avg_gap_y,
        'spacing_variance_x': spacing_variance_x,
        'spacing_variance_y': spacing_variance_y,
        'uniform_spacing_x': uniform_spacing_x,
        'uniform_spacing_y': uniform_spacing_y,
        'is_regular_grid': uniform_spacing_x > 0.85 and uniform_spacing_y > 0.85,
    }


def compute_initial_rms(k: Dict, calibration_frames: List[Dict], resolution: Tuple[int, int], point_map: dict,
                         layout: List[Dict]) -> float:
    """
    Compute RMS using TS camera matrix on observations.
    This shows best achievable with TS approach.

    Note: Layout coordinates are in world space (anchor = unit square, others via homography).
    No rescaling needed - just project directly.
    """
    if not calibration_frames or not layout:
        return 0.0

    fx = k['fx']
    fy = k['fy']
    cx = k['cx']
    cy = k['cy']

    total_error = 0.0
    total_points = 0

    for frame in calibration_frames:
        image_points = np.array([fp['imagePoint'] for fp in frame['framePoints']], dtype=np.float32)
        point_ids = [fp['pointId'] for fp in frame['framePoints']]

        # Calculate expected image coordinates using K matrix
        plane_points = []
        for point_id in point_ids:
            if point_id in point_map:
                plane_points.append(point_map[point_id][0])

        if len(plane_points) < 4:
            continue

        plane_points = np.array(plane_points, dtype=np.float32)

        # Project plane points using K matrix directly
        projected = []
        for pp in plane_points:
            u = fx * pp[0] + cx
            v = fy * pp[1] + cy
            projected.append([u, v])

        projected = np.array(projected, dtype=np.float32)

        # Compute RMS
        errors = np.sqrt(np.sum((projected - image_points) ** 2, axis=1))
        total_error += np.sum(errors)
        total_points += len(errors)

    if total_points == 0:
        return 0.0

    return float(total_error / total_points)


def opencv_linear_calibration(layout: List[Dict], calibration_frames: List[Dict],
                                resolution: Tuple[int, int]) -> Dict:
    """
    cv2.calibrateCamera with distCoeffs=None (pinhole model).
    """
    if not layout or not calibration_frames:
        return {'error': 'Insufficient data'}

    # Build pointId -> worldPos mapping
    point_map = {}
    for tag in layout:
        tag_id = tag['tagId']
        for corner_idx, (x, y) in enumerate(tag['corners']):
            point_id = tag_id * 10000 + corner_idx
            point_map[point_id] = ([x, y], tag_id, corner_idx)

    # Analyze layout to determine scaling
    all_x = []
    all_y = []
    for tag in layout:
        for corner in tag['corners']:
            all_x.append(corner[0])
            all_y.append(corner[1])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width_units = max_x - min_x
    height_units = max_y - min_y

    # Determine image scaling
    img_width, img_height = resolution

    # Compute scale factor based on layout coverage
    # Use the larger dimension for scaling to fit everything nicely
    scale = max(img_width / width_units, img_height / height_units) * 0.8  # 80% of image

    # Calculate min values to center the layout in the image
    layout_min_x, layout_min_y = min_x, min_y

    # Collect frames with >= 8 points
    frame_obj_points = []
    frame_img_points = []

    for frame in calibration_frames:
        obj_pts = []
        img_pts = []

        for fp in frame['framePoints']:
            point_id = fp['pointId']
            img_x, img_y = fp['imagePoint']

            if point_id in point_map:
                world_pos, _, _ = point_map[point_id]
                # Scale and center the layout
                obj_x = (world_pos[0] - layout_min_x) * scale
                obj_y = (world_pos[1] - layout_min_y) * scale
                obj_pts.append([float(obj_x), float(obj_y), 0.0])
                img_pts.append([float(img_x), float(img_y)])

        if len(obj_pts) >= 8:
            frame_obj_points.append(np.array(obj_pts, dtype=np.float32))
            frame_img_points.append(np.array(img_pts, dtype=np.float32))

    if len(frame_obj_points) < 3:
        return {'error': f'Only {len(frame_obj_points)} frames have enough points (need >= 3)'}

    h, w = resolution

    # Linear calibration (no distortion)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        frame_obj_points,
        frame_img_points,
        (w, h),
        cameraMatrix=None,
        distCoeffs=None,
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return {
        'type': 'linear',
        'rms': float(rms),
        'K': {'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy)},
        'distortion': [0.0] * 5,
        'num_frames': len(frame_obj_points),
        'point_map': point_map,
    }


def opencv_distorted_calibration(layout: List[Dict], calibration_frames: List[Dict],
                                   resolution: Tuple[int, int], point_map: dict) -> Dict:
    """
    cv2.calibrateCamera with distCoeffs=[0,0,0,0,0] (allows distortion estimation).
    """
    # This is the same data as linear calibration, just allowing distortion
    # to see if it improves RMS

    # Analyze layout to determine scaling
    all_x = []
    all_y = []
    for tag in layout:
        for corner in tag['corners']:
            all_x.append(corner[0])
            all_y.append(corner[1])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width_units = max_x - min_x
    height_units = max_y - min_y

    # Determine image scaling
    img_width, img_height = resolution
    scale = max(img_width / width_units, img_height / height_units) * 0.8

    # Collect frames with >= 8 points
    frame_obj_points = []
    frame_img_points = []

    for frame in calibration_frames:
        obj_pts = []
        img_pts = []

        for fp in frame['framePoints']:
            point_id = fp['pointId']
            img_x, img_y = fp['imagePoint']

            if point_id in point_map:
                world_pos, _, _ = point_map[point_id]
                # Scale and center the layout
                obj_x = (world_pos[0] - min_x) * scale
                obj_y = (world_pos[1] - min_y) * scale
                obj_pts.append([float(obj_x), float(obj_y), 0.0])
                img_pts.append([float(img_x), float(img_y)])

        if len(obj_pts) >= 8:
            frame_obj_points.append(np.array(obj_pts, dtype=np.float32))
            frame_img_points.append(np.array(img_pts, dtype=np.float32))

    if len(frame_obj_points) < 3:
        return {'error': f'Only {len(frame_obj_points)} frames have >= 8 points'}

    h, w = resolution

    # Distorted calibration (allows distortion estimation)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        frame_obj_points,
        frame_img_points,
        (w, h),
        cameraMatrix=np.zeros((3, 3), dtype=np.float32),
        distCoeffs=np.zeros(5, dtype=np.float32),
    )

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    return {
        'type': 'distorted',
        'rms': float(rms),
        'K': {'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy)},
        'distortion': [float(d) for d in dist],
        'num_frames': len(frame_obj_points),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_calibration.py <calibration_export.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_calibration_data(json_path)

    print("=" * 60)
    print("CALIBRATION ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nExport time: {data['exportTime']}")
    print(f"Resolution: {data['resolution']['width']}x{data['resolution']['height']}")

    # Layout analysis
    print("\n" + "-" * 60)
    print("LAYOUT ANALYSIS")
    print("-" * 60)
    layout_analysis = analyze_layout(data['layout'])

    print(f"Number of points: {layout_analysis['num_points']}")
    print(f"X range: {layout_analysis['x_range'][0]:.2f} to {layout_analysis['x_range'][1]:.2f}")
    print(f"Y range: {layout_analysis['y_range'][0]:.2f} to {layout_analysis['y_range'][1]:.2f}")
    print(f"Average gap X: {layout_analysis['avg_gap_x']:.4f}")
    print(f"Average gap Y: {layout_analysis['avg_gap_y']:.4f}")
    print(f"Spacing variance X: {layout_analysis['spacing_variance_x']:.6f}")
    print(f"Spacing variance Y: {layout_analysis['spacing_variance_y']:.6f}")
    print(f"Uniform spacing X: {layout_analysis['uniform_spacing_x'] * 100:.1f}%")
    print(f"Uniform spacing Y: {layout_analysis['uniform_spacing_y'] * 100:.1f}%")
    print(f"Is regular grid: {layout_analysis['is_regular_grid']}")

    # Data statistics
    num_frames = len(data['calibrationFrames'])
    total_points = sum(len(f['framePoints']) for f in data['calibrationFrames'])
    print(f"\nCalibration frames: {num_frames}")
    print(f"Total observed points: {total_points}")
    print(f"Points per frame (avg): {total_points / num_frames:.1f}")

    # TS calibration result
    ts_result = data.get('tsCalibResult')
    print("\n" + "-" * 60)
    print("TYPESCRIPT CALIBRATION RESULT")
    print("-" * 60)
    if ts_result:
        print(f"RMS: {ts_result['rmsPx']:.3f} px")
        print(f"K matrix: {ts_result['K']['fx']:.1f} / {ts_result['K']['fy']:.1f} px")
        print(f"  cx: {ts_result['K']['cx']:.1f}, cy: {ts_result['K']['cy']:.1f}")
        print(f"  FOV (x): {2 * np.arctan(data['resolution']['width'] / (2 * ts_result['K']['fx'])) * 180 / np.pi:.1f}°")
        print(f"Homographies: {len(ts_result['homographies'])}")
        rms_values = [h['rms'] for h in ts_result['homographies']]
        print(f"  RMS min: {min(rms_values):.3f}, median: {np.median(rms_values):.3f}, max: {max(rms_values):.3f}")
    else:
        print("No TypeScript calibration result available")

    # Initial RMS estimate using TS K matrix
    print("\n" + "-" * 60)
    print("INITIAL RMS (TS Camera Matrix on Observations)")
    print("-" * 60)
    if ts_result:
        # Build point_map for initial RMS calculation
        point_map = {}
        for tag in data['layout']:
            tag_id = tag['tagId']
            for corner_idx, (x, y) in enumerate(tag['corners']):
                point_id = tag_id * 10000 + corner_idx
                point_map[point_id] = ([x, y], tag_id, corner_idx)
        initial_rms = compute_initial_rms(ts_result['K'], data['calibrationFrames'],
                                   (int(data['resolution']['width']), int(data['resolution']['height'])),
                                   point_map, data['layout'])
        print(f"RMS: {initial_rms:.3f} px")
        print(f"Interpretation: Best achievable with TS layout detection + calibration")
    else:
        print("No camera matrix available to compute initial RMS")

    # OpenCV linear calibration
    print("\n" + "-" * 60)
    print("OPENCV LINEAR CALIBRATION (Pinhole Model)")
    print("-" * 60)
    linear_result = opencv_linear_calibration(data['layout'], data['calibrationFrames'],
                                               (data['resolution']['width'], data['resolution']['height']))

    if 'error' in linear_result:
        print(f"Error: {linear_result['error']}")
    else:
        print(f"RMS: {linear_result['rms']:.3f} px")
        print(f"K matrix: {linear_result['K']['fx']:.1f} / {linear_result['K']['fy']:.1f} px")
        print(f"  cx: {linear_result['K']['cx']:.1f}, cy: {linear_result['K']['cy']:.1f}")
        print(f"FOV (x): {2 * np.arctan(data['resolution']['width'] / (2 * linear_result['K']['fx'])) * 180 / np.pi:.1f}°")
        print(f"Frames used: {linear_result['num_frames']}")

    # OpenCV distorted calibration
    print("\n" + "-" * 60)
    print("OPENCV DISTORTED CALIBRATION (Radial + Tangential)")
    print("-" * 60)
    distorted_result = opencv_distorted_calibration(data['layout'], data['calibrationFrames'],
                                                     (data['resolution']['width'], data['resolution']['height']),
                                                     linear_result.get('point_map', {}))

    if 'error' in distorted_result:
        print(f"Error: {distorted_result['error']}")
    else:
        print(f"RMS: {distorted_result['rms']:.3f} px")
        print(f"K matrix: {distorted_result['K']['fx']:.1f} / {distorted_result['K']['fy']:.1f} px")
        print(f"  cx: {distorted_result['K']['cx']:.1f}, cy: {distorted_result['K']['cy']:.1f}")
        print(f"FOV (x): {2 * np.arctan(data['resolution']['width'] / (2 * distorted_result['K']['fx'])) * 180 / np.pi:.1f}°")
        print(f"Distortion coefficients: {[f'{d:.6f}' for d in distorted_result['distortion']]}")
        print(f"Frames used: {distorted_result['num_frames']}")

    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if ts_result and 'error' not in linear_result:
        print(f"\nTS RMS:      {ts_result['rmsPx']:.3f} px")
        print(f"Linear CV2:  {linear_result['rms']:.3f} px")
        diff = abs(ts_result['rmsPx'] - linear_result['rms'])
        pct = (diff / ts_result['rmsPx']) * 100 if ts_result['rmsPx'] > 0 else 0
        print(f"Difference:  {diff:.3f} px ({pct:.1f}%)")
        print(f"TS is closer by: {abs(ts_result['rmsPx'] - linear_result['rms']):.3f} px")

    if 'error' not in linear_result and 'error' not in distorted_result:
        print(f"\nLinear CV2:  {linear_result['rms']:.3f} px")
        print(f"Distorted CV2: {distorted_result['rms']:.3f} px")
        diff = abs(linear_result['rms'] - distorted_result['rms'])
        pct = (diff / linear_result['rms']) * 100 if linear_result['rms'] > 0 else 0
        print(f"Difference:  {diff:.3f} px ({pct:.1f}%)")
        print(f"Distortion helps by: {abs(distorted_result['rms'] - linear_result['rms']):.3f} px")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if ts_result and 'error' not in linear_result:
        ts_linear_diff = abs(ts_result['rmsPx'] - linear_result['rms'])

        if ts_linear_diff < 0.5:
            print("\n✓ TS and OpenCV achieve nearly identical results.")
            print("  The TS implementation is likely optimal for this dataset.")
        elif ts_linear_diff < 1.0:
            print("\n✓ TS and OpenCV results are very close.")
            print("  Minor differences likely due to algorithm variations, not bugs.")
        else:
            print("\n⚠ TS results differ significantly from OpenCV.")
            print("  Check implementation for bugs or limitations.")
            if ts_result['rmsPx'] > linear_result['rms']:
                print("  TS RMS is higher than OpenCV - implementation may be losing precision.")
            else:
                print("  TS RMS is lower than OpenCV - implementation may have lucky calibration.")

    if 'error' not in linear_result and 'error' not in distorted_result:
        dist_linear_diff = abs(linear_result['rms'] - distorted_result['rms'])

        if dist_linear_diff < 0.3:
            print("\n✓ Linear and distorted models produce nearly identical results.")
            print("  Distortion coefficients are very small - your camera has minimal distortion.")
        elif dist_linear_diff < 1.0:
            print("\n⚠ Distortion model shows some improvement.")
            print("  Adding distortion correction could improve calibration by ~1px.")
        else:
            print(f"\n⚠ Significant improvement with distortion model!")
            print(f"  Distortion reduces RMS from {linear_result['rms']:.3f} to {distorted_result['rms']:.3f} px ({dist_linear_diff:.3f} px, {pct:.1f}%).")

    print("\n")


if __name__ == '__main__':
    main()