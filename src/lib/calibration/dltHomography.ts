/**
 * Direct Linear Transform (DLT) for homography estimation.
 * Computes 3x3 homography matrix from 2D point correspondences.
 *
 * OpenCV reference: opencv/modules/calib3d/src/usac/degeneracy.cpp
 */

import { svdNullVector } from './svd'
import type { Mat3 } from './mat3'

export interface Vec2 {
  x: number
  y: number
}

/**
 * Compute homography H such that: p' = s * H * p
 *
 * @param srcPoints Source points (e.g., object plane coordinates)
 * @param dstPoints Destination points (e.g., image coordinates)
 * @returns Homography matrix as flat array [h00, h01, h02, h10, h11, h12, h20, h21, h22]
 *          i.e., column-major: H = [[h00,h10,h20],[h01,h11,h21],[h02,h12,h22]]
 */
export function computeHomography(
  srcPoints: readonly Vec2[],
  dstPoints: readonly Vec2[]
): Mat3 {
  const n = srcPoints.length
  if (n < 4) {
    throw new Error(`Need at least 4 points, got ${n}`)
  }
  if (n !== dstPoints.length) {
    throw new Error(`Point count mismatch: ${n} vs ${dstPoints.length}`)
  }

  // Build 2n × 9 matrix A
  // For each correspondence (x_i, y_i) ↔ (x'_i, y'_i):
  //   [-x', x, y, 1, 0, 0, 0, 0, 0]
  //   [0, 0, 0, -y', x, y, 1, 0, 0]
  // But typically: [x, y, 1, 0, 0, 0, -x'*x, -x'*y, -x']
  //                [0, 0, 0, x, y, 1, -y'*x, -y'*y, -y']
  const numRows = n * 2
  const numCols = 9
  const a: number[] = new Array(numRows * numCols)

  for (let i = 0; i < n; i++) {
    const sx = srcPoints[i]!.x
    const sy = srcPoints[i]!.y
    const dx = dstPoints[i]!.x
    const dy = dstPoints[i]!.y

    // Row 2i: [-dx, sx, sy, 1, 0, 0, 0, 0, 0]
    // Row 2i+1: [0, 0, 0, -dy, sx, sy, 1, 0, 0]
    const r0 = i * numCols * 2
    const r1 = r0 + numCols

    a[r0 + 0] = sx
    a[r0 + 1] = sy
    a[r0 + 2] = 1
    a[r0 + 3] = 0
    a[r0 + 4] = 0
    a[r0 + 5] = 0
    a[r0 + 6] = -dx * sx
    a[r0 + 7] = -dx * sy
    a[r0 + 8] = -dx

    a[r1 + 0] = 0
    a[r1 + 1] = 0
    a[r1 + 2] = 0
    a[r1 + 3] = sx
    a[r1 + 4] = sy
    a[r1 + 5] = 1
    a[r1 + 6] = -dy * sx
    a[r1 + 7] = -dy * sy
    a[r1 + 8] = -dy
  }

  // Compute null vector of A (last column of V from SVD)
  const nullVec = svdNullVector(a, numRows, numCols)
  if (!nullVec) {
    throw new Error('Homography computation failed: no null vector')
  }

  // Convert 9-element null vector to 3x3 matrix
  // nullVec from SVD is in row-major order: [h00, h01, h02, h10, h11, h12, h20, h21, h22]
  return [
    nullVec[0]!,  // h00
    nullVec[1]!,  // h01
    nullVec[2]!,  // h02
    nullVec[3]!,  // h10
    nullVec[4]!,  // h11
    nullVec[5]!,  // h12
    nullVec[6]!,  // h20
    nullVec[7]!,  // h21
    nullVec[8]!,  // h22
  ] as Mat3
}

/**
 * Apply homography to a 2D point.
 * H maps: [x, y, 1]^T → [x', y', w'] → (x'/w', y'/w')
 */
export function applyHomography(H: Mat3, p: Vec2): Vec2 {
  const w = H[6]! * p.x + H[7]! * p.y + H[8]!
  const wx = H[0]! * p.x + H[1]! * p.y + H[2]!
  const wy = H[3]! * p.x + H[4]! * p.y + H[5]!
  return { x: wx / w, y: wy / w }
}

// ============================================================================
// NORMALIZED DLT (for accurate calibration)
// ============================================================================

export interface Normalization {
  /** Mean to subtract from points */
  mean: Vec2
  /** Scale to normalize distance */
  scale: number
}

/**
 * Compute similarity transform that normalizes points to:
 * - Zero mean
 * - Average distance from origin = √2
 */
export function normalizePoints(points: readonly Vec2[]): { normalized: Vec2[], T: Normalization } {
  // Compute centroid
  let cx = 0, cy = 0
  for (const p of points) {
    cx += p.x
    cy += p.y
  }
  cx /= points.length
  cy /= points.length

  // Compute average distance
  let avgDist = 0
  for (const p of points) {
    const dx = p.x - cx
    const dy = p.y - cy
    avgDist += Math.sqrt(dx * dx + dy * dy)
  }
  avgDist /= points.length

  // Scale factor: avg distance should be √2
  const scale = Math.sqrt(2) / (avgDist || 1)

  // Normalize: T * (p - mean)
  const normalized: Vec2[] = []
  for (const p of points) {
    normalized.push({
      x: (p.x - cx) * scale,
      y: (p.y - cy) * scale,
    })
  }

  return {
    normalized,
    T: { mean: { x: cx, y: cy }, scale },
  }
}

/**
 * Denormalize homography from normalized DLT.
 * If H_norm = T2 * H * T1^(-1), then H = T2^(-1) * H_norm * T1
 */
export function denormalizeHomography(H_norm: Mat3, Tsrc: Normalization, Tdst: Normalization): Mat3 {
  // T = [[s, 0, -s*mx], [0, s, -s*my], [0, 0, 1]]
  const Ts = [
    Tsrc.scale, 0, -Tsrc.scale * Tsrc.mean.x,
    0, Tsrc.scale, -Tsrc.scale * Tsrc.mean.y,
    0, 0, 1,
  ] as Mat3

  const Tdinverse = [
    1/Tdst.scale, 0, Tdst.mean.x,
    0, 1/Tdst.scale, Tdst.mean.y,
    0, 0, 1,
  ] as Mat3

  // H = T2^(-1) * H_norm * T1
  return mat3Mul(mat3Mul(Tdinverse, H_norm), Ts)
}

function mat3Mul(A: Mat3, B: Mat3): Mat3 {
  const result: number[] = new Array(9)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let sum = 0
      for (let k = 0; k < 3; k++) {
        sum += A[r * 3 + k]! * B[k * 3 + c]!
      }
      result[r * 3 + c] = sum
    }
  }
  return result as unknown as Mat3
}

/**
 * Compute homography with normalized DLT for better numerical stability.
 * This is the recommended approach for camera calibration.
 */
export function computeHomographyNormalized(
  srcPoints: readonly Vec2[],
  dstPoints: readonly Vec2[]
): Mat3 {
  // Normalize both point sets
  const { normalized: srcNorm, T: Tsrc } = normalizePoints(srcPoints)
  const { normalized: dstNorm, T: Tdst } = normalizePoints(dstPoints)

  // Compute homography in normalized space
  const H_norm = computeHomography(srcNorm, dstNorm)

  // Denormalize to original space
  return denormalizeHomography(H_norm, Tsrc, Tdst)
}