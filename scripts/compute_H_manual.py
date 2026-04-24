#!/usr/bin/env python3

import numpy as np

# True homography
H_true = np.array([[2, 0, 10], [0, 1.5, 5], [0, 0, 1]])

# Points
plane_pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
img_pts = (H_true @ np.vstack([plane_pts.T, np.ones((1, 4))]))[:2, :].T

print("H_true:", H_true)
print("plane_pts:", plane_pts)
print("img_pts:", img_pts)

# Hartley normalization
tP = np.array([[2.0, 0, -1.0], [0, 2.0, -1.0], [0, 0, 1]])
tI = np.array([[1.131370849898476, 0, -12.445079348883237],
               [0, 1.131370849898476, -6.505382386916237],
               [0, 0, 1]])

print("\ntP:", tP)
print("tI:", tI)

tInvI = np.linalg.inv(tI)
print("tInvI:", tInvI)

hN = np.array([[1.131370849898476, 0, 0],
               [0, 0.848528137423857, 0],
               [0, 0, 1]])
print("hN:", hN)

# Compute hN @ tP
print("\nhN @ tP:")
print(np.dot(hN, tP))

# Compute tInvI @ (hN @ tP)
print("\ntInvI @ (hN @ tP):")
print(np.dot(tInvI, np.dot(hN, tP)))

# Normalize by last element
H_manual = np.dot(tInvI, np.dot(hN, tP)) / np.dot(tInvI, np.dot(hN, tP))[2, 2]
print("\nH_manual (normalized):")
print(H_manual)

print("\nH_true:")
print(H_true)

print("\nDifference:")
print(H_manual - H_true)
