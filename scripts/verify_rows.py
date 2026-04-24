#!/usr/bin/env python3

import numpy as np

# True homography
H_true = np.array([[2, 0, 10], [0, 1.5, 5], [0, 0, 1]])

# Points
plane_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
img_pts = (H_true @ np.vstack([plane_pts.T, np.ones((1, 4))]))[:2, :].T

# Hartley normalization
tP = np.array([[2.0, 0, -1.0], [0, 2.0, -1.0], [0, 0, 1]])
tI = np.array([[1.131370849898476, 0, -12.445079348883237],
               [0, 1.131370849898476, -6.505382386916237],
               [0, 0, 1]])

Xn = tP @ np.vstack([plane_pts.T, np.ones((1, 4))])
Un = tI @ np.vstack([img_pts.T, np.ones((1, 4))])

# Python constraint matrix
A = []
for i in range(4):
    xn = Xn[0, i]
    yn = Xn[1, i]
    un = Un[0, i]
    vn = Un[1, i]

    row0 = [xn, yn, 1, 0, 0, 0, -xn * un, -yn * un, -un]
    row1 = [0, 0, 0, xn, yn, 1, -xn * vn, -yn * vn, -vn]
    A.extend([row0, row1])

print("Python rows:")
for i, row in enumerate(A):
    print(f"  {i}: {row}")

# Verify with SVD
U, s, Vt = np.linalg.svd(np.array(A))
h = Vt[-1, :]
print(f"\nh (from Python SVD): {h}")
print(f"h[8] = {h[8]:.10f}")

hN = h.reshape(3, 3)
tInvI = np.linalg.inv(tI)

# Compute hN @ tP
hN_tP = hN @ tP
print(f"\nhN @ tP:")
print(hN_tP)

# Compute H = tInvI @ hN @ tP
H = tInvI @ hN_tP
print(f"\nH = tInvI @ hN @ tP:")
print(H)

# Normalize
H_norm = H / H[2, 2]
print(f"\nH normalized (h22=1):")
print(H_norm)
