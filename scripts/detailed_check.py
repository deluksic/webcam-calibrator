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

# Point 0
xn0, yn0, wn0 = Xn[0, 0], Xn[1, 0], Xn[2, 0]
un0, vn0, wn0_img = Un[0, 0], Un[1, 0], Un[2, 0]

print(f"Point 0:")
print(f"  Xn = [{xn0}, {yn0}, {wn0}]")
print(f"  Un = [{un0}, {vn0}, {wn0_img}]")

# Python constraint row0
row0 = [xn0, yn0, 1, 0, 0, 0, -xn0*un0, -yn0*un0, -un0]
row1 = [0, 0, 0, xn0, yn0, 1, -xn0*vn0, -yn0*vn0, -vn0]

print(f"\nPython row0: {row0}")
print(f"Python row1: {row1}")

# What about h * row?
h_py = np.array([0.631973, 0, 0, 0, 0.499000, 0, 0, 0, 0.57735])
dot = h_py @ np.array(row0)
print(f"\nDot product h @ row0 = {dot}")

# With signs flipped
h_py_neg = -h_py
dot_neg = h_py_neg @ np.array(row0)
print(f"Dot product (-h) @ row0 = {dot_neg}")
