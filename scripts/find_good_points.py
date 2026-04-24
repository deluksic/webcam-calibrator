#!/usr/bin/env python3
"""Find test points that work well with Hartley normalization"""

import numpy as np

def hartley2(pts):
    n = len(pts)
    if n == 0:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    r2 = np.sum((pts - np.array([cx, cy]))**2, axis=1)
    r = np.sqrt((r2 / n) + 1e-18)
    s = np.sqrt(2) / r

    return np.array([
        [s, 0, -s * cx],
        [0, s, -s * cy],
        [0, 0, 1]
    ])

# Test identity H
H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Try different test point sets
test_sets = [
    ("unit square corners (0,0) etc", [[0, 0], [1, 0], [0, 1], [1, 1]]),
    ("shifted square (1,1) etc", [[1, 1], [2, 1], [1, 2], [2, 2]]),
    ("rect (1,0) etc", [[1, 0], [2, 0], [1, 1], [2, 1]]),
    ("random", [[0.5, 0.3], [1.2, 0.8], [0.9, 1.5], [2.1, 0.7]]),
]

for name, points in test_sets:
    plane_pts = np.array(points)
    img_pts = (H @ np.vstack([plane_pts.T, np.ones((1, len(points)))]))[:2, :].T

    tP = hartley2(plane_pts)
    tI = hartley2(img_pts)

    Xn = tP @ np.vstack([plane_pts.T, np.ones((1, 4))])
    Un = tI @ np.vstack([img_pts.T, np.ones((1, 4))])

    print(f"\n{name}:")
    print(f"  Plane pts: {plane_pts.tolist()}")
    print(f"  Image pts: {img_pts.tolist()}")
    print(f"  tP: {tP}")
    print(f"  tI: {tI}")
    print(f"  Xn:\n{Xn}")
    print(f"  Un:\n{Un}")

    # Check for zero rows in normalized coordinates
    zero_rows = np.where((np.abs(Xn[0, :]) < 1e-10) & (np.abs(Xn[1, :]) < 1e-10))
    print(f"  Zero rows in Xn: {zero_rows[0].tolist()}")
    if len(zero_rows[0]) > 0:
        print(f"    -> SKIP: Zero rows cause degenerate constraints")
    else:
        print(f"    -> OK")