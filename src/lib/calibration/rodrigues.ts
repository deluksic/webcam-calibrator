/**
 * Rodrigues rotation representation conversions.
 * Reference: opencv/modules/calib3d/src/rodrigues.cpp
 */

import type { Vec3 } from './vec3'
import type { Mat3 } from './mat3'

/**
 * Convert rotation vector (Rodrigues) to rotation matrix.
 *
 * rvec: 3-element rotation vector (axis * angle)
 * Returns: 3x3 rotation matrix
 */
export function rodriguesToMatrix(rvec: Vec3): Mat3 {
  const { x, y, z } = rvec
  const theta = Math.sqrt(x * x + y * y + z * z)

  if (theta < 1e-10) {
    // Near-zero rotation -> identity
    return [1, 0, 0, 0, 1, 0, 0, 0, 1] as Mat3
  }

  // Unit axis
  const ux = x / theta
  const uy = y / theta
  const uz = z / theta

  const c = Math.cos(theta)
  const s = Math.sin(theta)

  // R = I*cos(theta) + (1-cos(theta))*u*u^T + [u]_x*sin(theta)
  // Where [u]_x is the skew-symmetric cross-product matrix
  const mc = 1 - c

  const R: number[] = new Array(9)
  R[0] = c + ux * ux * mc
  R[1] = ux * uy * mc - uz * s
  R[2] = ux * uz * mc + uy * s
  R[3] = uy * ux * mc + uz * s
  R[4] = c + uy * uy * mc
  R[5] = uy * uz * mc - ux * s
  R[6] = uz * ux * mc - uy * s
  R[7] = uz * uy * mc + ux * s
  R[8] = c + uz * uz * mc

  return R as unknown as Mat3
}

/**
 * Convert rotation matrix to rotation vector (Rodrigues).
 *
 * R: 3x3 rotation matrix
 * Returns: 3-element rotation vector (axis * angle)
 */
export function matrixToRodrigues(R: Mat3): Vec3 {
  const trace = R[0]! + R[4]! + R[8]!
  const theta = Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2)))

  if (Math.abs(theta) < 1e-10) {
    // Near-zero rotation -> zero vector
    return { x: 0, y: 0, z: 0 }
  }

  // Axis from antisymmetric part
  const rx = (R[7]! - R[5]!) / (2 * Math.sin(theta))
  const ry = (R[2]! - R[6]!) / (2 * Math.sin(theta))
  const rz = (R[3]! - R[1]!) / (2 * Math.sin(theta))

  return {
    x: theta * rx,
    y: theta * ry,
    z: theta * rz,
  }
}
