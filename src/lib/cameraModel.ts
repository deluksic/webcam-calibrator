/** Pinhole intrinsics (skew fixed at 0 for v1 solver). */
export interface CameraIntrinsics {
  fx: number
  fy: number
  cx: number
  cy: number
}

/**
 * OpenCV `CALIB_RATIONAL_MODEL` order (8 coeffs):
 * k1, k2, p1, p2, k3, k4, k5, k6
 */
export type RationalDistortion8 = readonly [
  k1: number,
  k2: number,
  p1: number,
  p2: number,
  k3: number,
  k4: number,
  k5: number,
  k6: number,
]

export function zeroRationalDistortion8(): RationalDistortion8 {
  return [0, 0, 0, 0, 0, 0, 0, 0]
}
