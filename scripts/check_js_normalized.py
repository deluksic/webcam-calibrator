#!/usr/bin/env python3
"""Check normalization for unit square"""

import numpy as np

# Unit square points
plane_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Simple hartley2
def hartley2(pts):
    n = len(pts)
    if n == 0:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    r2 = 0.0
    for i in range(n):
        dx = pts[i, 0] - cx
        dy = pts[i, 1] - cy
        r2 += dx*dx + dy*dy
    r = float(np.sqrt((r2 / n) + 1e-18))
    s = float(np.sqrt(2.0) / r)

    return np.array([
        [s, 0.0, -s * cx],
        [0.0, s, -s * cy],
        [0.0, 0.0, 1.0]
    ])

tP = hartley2(plane_pts)
print(f"tP:")
for row in tP:
    print(f"  {row}")

Xn = tP @ np.vstack([plane_pts.T, np.ones((1, 4))])
print(f"\nXn (3x4):")
print(Xn)

print(f"\nXn[0, :]: {Xn[0, :]}")
print(f"Xn[1, :]: {Xn[1, :]}")

# Check which points have Xn[0] and Xn[1] both non-zero
for i in range(4):
    x_val = Xn[0, i]
    y_val = Xn[1, i]
    print(f"Point {i}: ({plane_pts[i, 0]}, {plane_pts[i, 1]}) -> Xn=[{x_val:.6f}, {y_val:.6f}, 1.000000]")