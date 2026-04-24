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

print("Xn[0]:", Xn[:, 0])
print("Un[0]:", Un[:, 0])
print("Xn[0]*Un[0] =", Xn[0, 0] * Un[0, 0])

# Python constraint (with - signs)
row0_py = [Xn[0, 0], Xn[1, 0], 1, 0, 0, 0, -Xn[0, 0] * Un[0, 0], -Xn[1, 0] * Un[0, 0], -Un[0, 0]]
row1_py = [0, 0, 0, Xn[0, 0], Xn[1, 0], 1, -Xn[0, 0] * Un[1, 0], -Xn[1, 0] * Un[1, 0], -Un[1, 0]]

print("\nPython row0:", row0_py)
print("Python row1:", row1_py)

# JS constraint (with - signs)
row0_js = [-1.0, -1.0, 1, 0, 0, 0, -(-1.0)*(-1.131370849898476), -(-1.0)*(-1.131370849898476), -(-1.131370849898476)]
row1_js = [0, 0, 0, -1.0, -1.0, 1, -(-1.0)*(-0.848528137423857), -(-1.0)*(-0.848528137423857), -(-0.848528137423857)]

print("\nJS row0:", row0_js)
print("JS row1:", row1_js)

print("\nrow0_js is row0_py * -1:", np.allclose(row0_js, np.array(row0_py) * -1))
print("row1_js is row1_py * -1:", np.allclose(row1_js, np.array(row1_py) * -1))

# The null vector from JS will be -h_py, which is fine
# But the problem is the denormalization expects hN, not -hN
