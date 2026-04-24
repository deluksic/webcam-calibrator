#!/usr/bin/env python3
"""Verify constraint matrix construction"""

import numpy as np

# True homography
H_true = np.array([[2, 0, 10], [0, 1.5, 5], [0, 0, 1]])

# Points
plane_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
img_pts = (H_true @ np.vstack([plane_pts.T, np.ones((1, 4))]))[:2, :].T

# Hartley normalization
tP = np.array([[2.0, 0, -1.0], [0, 2.0, -1.0], [0, 0, 1]])
tI = np.array([[1.131370849898476, 0, -12.445079348883237], [0, 1.131370849898476, -6.505382386916237], [0, 0, 1]])

Xn = tP @ np.vstack([plane_pts.T, np.ones((1, 4))])
Un = tI @ np.vstack([img_pts.T, np.ones((1, 4))])

print("Xn (3x4):")
print(Xn)
print("\nUn (3x4):")
print(Un)

# Build constraint matrix A as Python does
A = []
for i in range(4):
    xn = Xn[0, i]
    yn = Xn[1, i]
    un = Un[0, i]
    vn = Un[1, i]

    row0 = [xn, yn, 1, 0, 0, 0, -xn * un, -yn * un, -un]
    row1 = [0, 0, 0, xn, yn, 1, -xn * vn, -yn * vn, -vn]
    A.extend([row0, row1])

print("\nConstraint matrix A (8x9):")
for i, row in enumerate(A):
    print(f"  {i}: {row}")

# Find null vector via SVD
U, s, Vt = np.linalg.svd(np.array(A))
h = Vt[-1, :]
h /= h[-1]  # Normalize

print(f"\nh (from Python SVD): {h}")

# Now verify: A @ h = 0 should hold
residual = 0
for i in range(8):
    row = np.array(A[i])
    dot = row @ h
    residual += dot * dot
print(f"Residual ||A*h||: {np.sqrt(residual)}")

# Now compute hN
hN = h.reshape(3, 3)
tInvI = np.linalg.inv(tI)

# Denormalize: H = tInvI @ hN @ tP
H_computed = tInvI @ hN @ tP

print(f"\nH computed (denormalized):")
print(H_computed)
print("\nH true:")
print(H_true)