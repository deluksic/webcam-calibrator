/**
 * Project 3D points to 2D image coordinates.
 *
 * Reference: opencv/modules/calib3d/src/calibration.cpp (projectPoints)
 *
 * Pipeline:
 * 1. Apply extrinsics: Xc = R*X + t
 * 2. Normalize: x = Xc[0]/Xc[2], y = Xc[1]/Xc[2]
 * 3. Apply distortion: x', y' = distort(x, y)
 * 4. Project: u = fx*x' + cx, v = fy*y' + cy
 */

import type { Vec3 } from './vec3'
import { Mat3 } from './mat3'
import type { Intrinsics } from './zhangCalibration'
import type { DistortionCoeffs } from './distortion'
import { distortPoint } from './distortion'
import { rodriguesToMatrix } from './rodrigues'

/** 2D image point */
export interface ImagePoint {
  u: number
  v: number
}

/**
 * Project a single 3D world point to 2D image coordinates.
 *
 * @param worldPt 3D world point [X, Y, Z]
 * @param rvec Rotation vector (Rodrigues)
 * @param tvec Translation vector
 * @param intrinsics Camera intrinsic matrix K
 * @param distortion Distortion coefficients
 * @returns Image coordinates [u, v]
 */
export function projectPoint(
  worldPt: Vec3,
  rvec: Vec3,
  tvec: Vec3,
  intrinsics: Intrinsics,
  distortion?: DistortionCoeffs
): ImagePoint {
  const R = rodriguesToMatrix(rvec)

  // Apply extrinsics: Xc = R * worldPt + tvec
  const Xc = Mat3.mulVec(R, worldPt)
  const XcX = Xc.x + tvec.x
  const XcY = Xc.y + tvec.y
  const XcZ = Xc.z + tvec.z

  // Normalize: x = Xc/Z, y = Yc/Z
  const invZ = 1 / XcZ
  let x = XcX * invZ
  let y = XcY * invZ

  // Apply distortion
  if (distortion) {
    const distorted = distortPoint({ x, y }, distortion)
    x = distorted.x
    y = distorted.y
  }

  // Project to image: u = fx*x + cx, v = fy*y + cy
  const u = intrinsics.fx * x + intrinsics.skew * y + intrinsics.cx
  const v = intrinsics.fy * y + intrinsics.cy

  return { u, v }
}

/**
 * Project multiple 3D points to 2D image coordinates.
 *
 * @param worldPts Array of 3D world points
 * @param rvec Rotation vector
 * @param tvec Translation vector
 * @param intrinsics Camera intrinsics
 * @param distortion Optional distortion coefficients
 * @returns Array of image points
 */
export function projectPoints(
  worldPts: readonly Vec3[],
  rvec: Vec3,
  tvec: Vec3,
  intrinsics: Intrinsics,
  distortion?: DistortionCoeffs
): ImagePoint[] {
  const R = rodriguesToMatrix(rvec)
  const { fx, fy, cx, cy, skew } = intrinsics

  return worldPts.map((pt) => {
    // Extrinsics
    const Xc = Mat3.mulVec(R, pt)
    const XcX = Xc.x + tvec.x
    const XcY = Xc.y + tvec.y
    const XcZ = Xc.z + tvec.z

    // Normalize
    const invZ = 1 / XcZ
    let x = XcX * invZ
    let y = XcY * invZ

    // Distortion
    if (distortion) {
      const d = distortPoint({ x, y }, distortion)
      x = d.x
      y = d.y
    }

    // Project
    return {
      u: fx * x + skew * y + cx,
      v: fy * y + cy,
    }
  })
}

/**
 * Compute reprojection error for a single point.
 */
export function reprojectionError(
  worldPt: Vec3,
  imagePt: ImagePoint,
  rvec: Vec3,
  tvec: Vec3,
  intrinsics: Intrinsics,
  distortion?: DistortionCoeffs
): number {
  const projected = projectPoint(worldPt, rvec, tvec, intrinsics, distortion)
  const dx = projected.u - imagePt.u
  const dy = projected.v - imagePt.v
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Compute total RMS reprojection error.
 */
export function reprojectionRms(
  observations: Array<{
    worldPt: Vec3
    imagePt: ImagePoint
    rvec: Vec3
    tvec: Vec3
  }>,
  intrinsics: Intrinsics,
  distortion?: DistortionCoeffs
): number {
  let sumSq = 0
  for (const obs of observations) {
    const err = reprojectionError(
      obs.worldPt,
      obs.imagePt,
      obs.rvec,
      obs.tvec,
      intrinsics,
      distortion
    )
    sumSq += err * err
  }
  return Math.sqrt(sumSq / observations.length)
}
