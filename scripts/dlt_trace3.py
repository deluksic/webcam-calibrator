#!/usr/bin/env python3
"""Debug trace to verify transformation direction"""

import numpy as np

# True homography
H_true = np.array([
    [2, 0, 10],
    [0, 1.5, 5],
    [0, 0, 1]
])

# Points on unit square
plane_pts = np.array([[0.5, 0.5], [1.5, 0.5]])
img_pts = (H_true @ np.vstack([plane_pts.T, np.ones((1, 2))]))[:2, :].T

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

tInvI = np.linalg.inv(tI)

# Normalize plane points: Xn = tP * [x, y, 1]^T
Xn = tP @ np.vstack([plane_pts.T, np.ones((1, 2))])
print("Xn (3x2):")
print(Xn)
print()

# Normalize image points: Un = tI * [u, v, 1]^T
Un = tI @ np.vstack([img_pts.T, np.ones((1, 2))])
print("Un (3x2):")
print(Un)
print()

# The DLT finds hN such that Un = hN * Xn
# So hN satisfies: Un * Xn^T = hN * Xn * Xn^T = S
# And we want hN = Un * Xn^T * (Xn * Xn^T)^-1
# Let me check: For normalized coordinates, the constraint is:
#   Xn[0]*hN[0] + Xn[1]*hN[1] + hN[2] = Un[0]*wN
#   Xn[0]*hN[3] + Xn[1]*hN[4] + hN[5] = Un[1]*wN
#   Xn[0]*hN[6] + Xn[1]*hN[7] + hN[8] = wN

# Combine to get A * hN = 0 where:
#   [Xn[0], Xn[1], 1, 0, 0, 0, -Xn[0]*Un[0], -Xn[1]*Un[0], -Un[0]]
#   [0, 0, 0, Xn[0], Xn[1], 1, -Xn[0]*Un[1], -Xn[1]*Un[1], -Un[1]]

A = []
for i in range(len(plane_pts)):
    xn = Xn[0, i]
    yn = Xn[1, i]
    un = Un[0, i]
    vn = Un[1, i]

    row0 = [xn, yn, 1, 0, 0, 0, -xn * un, -yn * un, -un]
    row1 = [0, 0, 0, xn, yn, 1, -xn * vn, -yn * vn, -vn]
    A.extend([row0, row1])

print("A (4x9):")
for i, row in enumerate(A):
    print(f"  {i}: {row}")
print()

# SVD to find null vector
U, s, Vt = np.linalg.svd(np.array(A))
hN = Vt[-1, :]
hN /= hN[-1]  # Normalize to hN[8] = 1

print("hN (normalized to hN[8] = 1):")
print(hN)
print()

# Verify: Un = hN * Xn should hold (approximately)
for i in range(len(plane_pts)):
    xn = Xn[0, i]
    yn = Xn[1, i]
    un = Un[0, i]
    vn = Un[1, i]

    hN_3x3 = hN.reshape(3, 3)
    computed = hN_3x3 @ np.array([xn, yn, 1])
    print(f"Point {i}:")
    print(f"  Xn = {xn}, {yn}")
    print(f"  Un = {un}, {vn}")
    print(f"  hN * Xn = {computed[0]:.10f}, {computed[1]:.10f}")
    print(f"  Error = {abs(un - computed[0]):.10e}, {abs(vn - computed[1]):.10e}")
    print()

# Denormalize: We have H_true = tI^-1 @ hN_true @ tP^-1
# So hN_true = tI @ H_true @ tP

hN_true = tI @ H_true @ np.linalg.inv(tP)
print("hN_true (should match hN):")
print(hN_true)
print()

# Verify the forward direction
print("Verification:")
print("H_true @ [x, y, 1]:")
for i in range(len(plane_pts)):
    xn = Xn[0, i]
    yn = Xn[1, i]
    computed = H_true @ np.array([xn, yn, 1])
    print(f"  Xn = {xn}, {yn} -> {computed[0]:.10f}, {computed[1]:.10f}")

print("\ntI @ computed:")
for i in range(len(plane_pts)):
    un = Un[0, i]
    vn = Un[1, i]
    computed = tI @ np.array([un, vn, 1])
    print(f"  Un = {un}, {vn} -> {computed[0]:.10f}, {computed[1]:.10f}")

print("\nhN_true @ Xn:")
for i in range(len(plane_pts)):
    xn = Xn[0, i]
    yn = Xn[1, i]
    computed = hN_true @ np.array([xn, yn, 1])
    print(f"  Xn = {xn}, {yn} -> {computed[0]:.10f}, {computed[1]:.10f}")