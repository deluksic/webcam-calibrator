#!/usr/bin/env python3
"""Explicit constraint matrix construction to debug"""

import numpy as np

# True homography
H_true = np.array([
    [2, 0, 10],
    [0, 1.5, 5],
    [0, 0, 1]
])

# Points - avoiding origin
plane_pts = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
img_pts = (H_true @ np.vstack([plane_pts.T, np.ones((1, 4))]))[:2, :].T

print("Input points:")
for i, (px, py) in enumerate(plane_pts):
    u, v = img_pts[i]
    print(f"  {i}: plane=({px},{py}), image=({u},{v})")
print()

# Hartley normalization for plane
tP = np.array([
    [2.0, 0, -1.0],
    [0, 2.0, -1.0],
    [0, 0, 1]
])

# Hartley normalization for image
tI = np.array([
    [1.131370849898476, 0, -12.445079348883237],
    [0, 1.131370849898476, -6.505382386916237],
    [0, 0, 1]
])

# Normalize plane points
Xn = tP @ np.vstack([plane_pts.T, np.ones((1, 4))])
print("Normalized plane points Xn (3x4):")
print(Xn)
print()

Xn_x = Xn[0, :]
Xn_y = Xn[1, :]

# Normalize image points
Un = tI @ np.vstack([img_pts.T, np.ones((1, 4))])
print("Normalized image points Un (3x4):")
print(Un)
print()

Un_u = Un[0, :]
Un_v = Un[1, :]

print("Normalized plane Xn_x:", Xn_x)
print("Normalized plane Xn_y:", Xn_y)
print("Normalized image Un_u:", Un_u)
print("Normalized image Un_v:", Un_v)
print()

# Build constraint matrix directly
A = []
for i in range(len(plane_pts)):
    xn = Xn_x[i]
    yn = Xn_y[i]
    un = Un_u[i]
    vn = Un_v[i]

    # Row 0: [Xn, Yn, 1, 0, 0, 0, -Xn*un, -Yn*un, -un]
    row0 = [xn, yn, 1, 0, 0, 0, -xn * un, -yn * un, -un]
    # Row 1: [0, 0, 0, Xn, Yn, 1, -Xn*vn, -Yn*vn, -vn]
    row1 = [0, 0, 0, xn, yn, 1, -xn * vn, -yn * vn, -vn]
    A.extend([row0, row1])

print("Constraint matrix A:")
for i, row in enumerate(A):
    print(f"  {i}: {row}")

# SVD to find null vector
U, s, Vt = np.linalg.svd(np.array(A))
h = Vt[-1, :] / Vt[-1, 8]  # Normalize to h[8] = 1
print(f"\nNull vector h (h[8] = 1):")
print(h)

# Test if null vector works
residual = 0
for i in range(len(A)):
    row = np.array(A[i])
    dot = row @ h
    residual += dot * dot
print(f"Residual ||A*h||: {np.sqrt(residual)}")

# Compute H from null vector
hN = h.reshape(3, 3)
tInvI = np.linalg.inv(tI)
H_computed = tInvI @ hN @ tP

print("\nH computed:")
print(H_computed)
print("\nH true:")
print(H_true)