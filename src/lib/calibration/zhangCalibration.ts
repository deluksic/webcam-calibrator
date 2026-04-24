/**
 * Zhang's Camera Calibration Method (Closed-Form)
 *
 * Reference: Z. Zhang, "A flexible new technique for camera calibration"
 * IEEE TPAMI, 22(11):1330-1334, 2000
 *
 * OpenCV Reference: opencv/modules/calib3d/src/calibration.cpp
 */

import { Mat3 } from './mat3'
import { Vec3, Vec6 } from './vec3'
import { svdNullVector } from './svd'

// Homography: 2D plane to image plane mapping
export interface Homography {
  /** 3x3 homography matrix (row-major) */
  H: Mat3
}

/**
 * Intrinsic camera parameters (3x3 calibration matrix)
 * K = [fx,  s, cx]
 *     [ 0, fy, cy]
 *     [ 0,  0,  1]
 */
export interface Intrinsics {
  fx: number
  fy: number
  cx: number
  cy: number
  /** Skew parameter (typically ~0) */
  skew: number
}

/**
 * Extrinsic parameters for each view
 */
export interface Extrinsics {
  /** Rotation vector (Rodrigues format, 3 elements) */
  rvec: Vec3
  /** Translation vector (3 elements) */
  tvec: Vec3
}

/**
 * Camera calibration result
 */
export interface CalibrationResult {
  intrinsics: Intrinsics
  extrinsics: Extrinsics[]
  /** Reprojection RMS error */
  rmsError: number
}

// ============================================================================
// VECTOR FORMULAS FROM HOMOGRAPHY
// ============================================================================

/**
 * v12 vector from homography columns h1, h2.
 *
 * Used in constraint: v12^T * b = 0
 *
 * Vector formula:
 * v12 = [h1x*h2x, h1x*h2y + h2x*h1y, h1y*h2y,
 *        h1x*h2z + h2x*h1z, h1y*h2z + h2y*h1z, h1z*h2z]
 *
 * b = [B11, B12, B22, B13, B23, B33]^T
 */
export function computeV12(H: Mat3): Vec6 {
  // Extract column 1 (h1) and column 2 (h2)
  const h1 = column(H, 1)
  const h2 = column(H, 2)

  return [
    h1.x * h2.x,
    h1.x * h2.y + h2.x * h1.y,
    h1.y * h2.y,
    h1.x * h2.z + h2.x * h1.z,
    h1.y * h2.z + h2.y * h1.z,
    h1.z * h2.z,
  ] as Vec6
}

/**
 * v11 - v22 vector from homography.
 *
 * Used in constraint: (v11 - v22)^T * b = 0
 *
 * Vector formula:
 * v11 = [h1x², h1x*h1y, h1y², h1x*h1z, h1y*h1z, h1z²]
 * v22 = [h2x², h2x*h2y, h2y², h2x*h2z, h2y*h2z, h2z²]
 *
 * v11 - v22 = [h1x²-h2x², h1x*h1y-h2x*h2y, h1y²-h2y²,
 *              h1x*h1z-h2x*h2z, h1y*h1z-h2y*h2z, h1z²-h2z²]
 */
export function computeV11MinusV22(H: Mat3): Vec6 {
  const h1 = column(H, 1)
  const h2 = column(H, 2)

  return [
    h1.x * h1.x - h2.x * h2.x,
    h1.x * h1.y - h2.x * h2.y,
    h1.y * h1.y - h2.y * h2.y,
    h1.x * h1.z - h2.x * h2.z,
    h1.y * h1.z - h2.y * h2.z,
    h1.z * h1.z - h2.z * h2.z,
  ] as Vec6
}

/**
 * Extract column from 3x3 matrix as Vec3.
 * H[row*3 + col]
 *
 * col 0: indices [0, 3, 6] -> [H00, H10, H20]
 * col 1: indices [1, 4, 7] -> [H01, H11, H21]
 * col 2: indices [2, 5, 8] -> [H02, H12, H22]
 */
function column(H: Mat3, col: number): Vec3 {
  return {
    x: H[col]!,
    y: H[3 + col]!,
    z: H[6 + col]!,
  }
}

// ============================================================================
// SOLVE FOR B MATRIX (Absolute Conic Image)
// ============================================================================

/**
 * Build V matrix from homographies.
 * Each homography contributes 2 rows: [v12^T, (v11-v22)^T]
 */
export function buildVMatrix(homographies: readonly Homography[]): number[] {
  const numViews = homographies.length
  const numRows = numViews * 2
  const numCols = 6
  const V: number[] = new Array(numRows * numCols)

  for (let i = 0; i < numViews; i++) {
    const H = homographies[i]!.H
    const v12 = computeV12(H)
    const v11MinusV22 = computeV11MinusV22(H)

    // Row 2i: v12
    for (let j = 0; j < 6; j++) {
      V[i * 2 * 6 + j] = v12[j]!
    }

    // Row 2i+1: v11 - v22
    for (let j = 0; j < 6; j++) {
      V[(i * 2 + 1) * 6 + j] = v11MinusV22[j]!
    }
  }

  return V
}

/**
 * Solve for B matrix from V*b=0.
 * B = λ*K^(-T) * K^(-1) is the image of the absolute conic.
 *
 * Returns 6-element vector: [B11, B12, B22, B13, B23, B33]
 */
export function solveForB(V: number[], numViews: number): Vec6 | undefined {
  const nullVec = svdNullVector(V, numViews * 2, 6)
  if (!nullVec) {
    return undefined
  }
  return nullVec as Vec6
}

// ============================================================================
// EXTRACT K FROM B
// ============================================================================

/**
 * Extract camera intrinsic matrix K from B.
 *
 * Formulas (from Zhang 2000):
 * cy = (B12*B13 - B11*B23) / (B11*B22 - B12²)
 * λ = B33 - (B13² + cy*(B12*B13 - B11*B23)) / B11
 * fx = √(λ / B11)
 * fy = √(λ * B11 / (B11*B22 - B12²))
 * cx = (-B13 * fx²) / λ
 *
 * Returns K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
 */
export function extractKFromB(B: Vec6): Intrinsics {
  const B11 = B[0]!
  const B12 = B[1]!
  const B22 = B[2]!
  const B13 = B[3]!
  const B23 = B[4]!
  const B33 = B[5]!

  // cy = (B12*B13 - B11*B23) / (B11*B22 - B12²)
  const denom = B11 * B22 - B12 * B12
  const cy = (B12 * B13 - B11 * B23) / denom

  // λ = B33 - (B13² + cy*(B12*B13 - B11*B23)) / B11
  const λ = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11

  // fx = √(λ / B11)
  const fx = Math.sqrt(λ / B11)

  // fy = √(λ * B11 / (B11*B22 - B12²))
  const fy = Math.sqrt(λ * B11 / denom)

  // cx = (-B13 * fx²) / λ
  const cx = (-B13 * fx * fx) / λ

  // Skew s ≈ 0 for typical cameras
  const skew = 0

  return { fx, fy, cx, cy, skew }
}

// ============================================================================
// EXTRACT EXTRINSICS FROM HOMOGRAPHY
// ============================================================================

/**
 * Compute extrinsics (R, t) from homography using K.
 *
 * Steps:
 * 1. Compute A = K^(-1) * H
 * 2. λ = 2 / (||a0||² + ||a1||²) where a0, a1 are columns of A
 * 3. r0 = λ * a0, r1 = λ * a1
 * 4. r2 = r0 × r1 (cross product to enforce orthonormality)
 * 5. t = λ * a2
 */
export function computeExtrinsicsFromHomography(H: Mat3, K: Intrinsics): Extrinsics {
  // Compute K^(-1)
  const Kinv = invertK(K)

  // A = K^(-1) * H
  const A = mat3Mul(Kinv, H)

  // Extract columns of A
  const a0 = column(A, 0)
  const a1 = column(A, 1)
  const a2 = column(A, 2)

  // λ = 2 / (||a0||² + ||a1||²)
  const normSq = a0.x * a0.x + a0.y * a0.y + a0.z * a0.z +
                 a1.x * a1.x + a1.y * a1.y + a1.z * a1.z
  const λ = 2 / normSq

  // Rotation columns
  const r0 = vec3Scale(a0, λ)
  const r1 = vec3Scale(a1, λ)
  const r2 = vec3Cross(r0, r1)

  // Translation
  const t = vec3Scale(a2, λ)

  // Convert rotation matrix to Rodrigues vector
  const rvec = matrixToRodrigues([r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z] as Mat3)

  return { rvec, tvec: t }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Invert the intrinsic matrix K.
 */
function invertK(K: Intrinsics): Mat3 {
  const { fx, fy, cx, cy, skew } = K

  // K^(-1) = [[1/fx, -s/(fx*fy), (s*cy - cx*fy)/(fx*fy)],
  //           [0,    1/fy,      -cy/fy],
  //           [0,    0,         1]]

  const a = 1 / fx
  const b = -skew / (fx * fy)
  const c = (skew * cy - cx * fy) / (fx * fy)
  const d = 0
  const e = 1 / fy
  const f = -cy / fy
  const g = 0
  const h = 0
  const i = 1

  return [a, b, c, d, e, f, g, h, i] as Mat3
}

/** Matrix multiplication: A * B */
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

/** Scale a 3D vector */
function vec3Scale(v: Vec3, s: number): Vec3 {
  return { x: v.x * s, y: v.y * s, z: v.z * s }
}

/** Cross product of two 3D vectors */
function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  }
}

/**
 * Convert 3x3 rotation matrix to Rodrigues vector.
 * R = I + sin(θ)/θ * (K) + (1-cos(θ))/θ² * K²
 *
 * Returns 3-element vector: [rx, ry, rz]
 */
function matrixToRodrigues(R: Mat3): Vec3 {
  // Extract rotation axis and angle from matrix
  // θ = arccos((trace(R) - 1) / 2)
  const trace = R[0]! + R[4]! + R[8]!
  const θ = Math.acos((trace - 1) / 2)

  if (Math.abs(θ) < 1e-10) {
    // Near-zero rotation -> zero vector
    return { x: 0, y: 0, z: 0 }
  }

  // Axis from antisymmetric part of R
  const rx = (R[7]! - R[5]!) / (2 * Math.sin(θ))
  const ry = (R[2]! - R[6]!) / (2 * Math.sin(θ))
  const rz = (R[3]! - R[1]!) / (2 * Math.sin(θ))

  // Rodrigues vector = θ * axis
  return { x: θ * rx, y: θ * ry, z: θ * rz }
}

// ============================================================================
// MAIN CALIBRATION FUNCTION
// ============================================================================

/**
 * Zhang's closed-form camera calibration.
 *
 * @param homographies Array of homographies from planar calibration grids
 * @returns Intrinsic parameters K and per-view extrinsics
 */
export function zhangCalibration(homographies: readonly Homography[]): CalibrationResult {
  if (homographies.length < 3) {
    throw new Error('Need at least 3 homographies for calibration')
  }

  // Step 1: Build V matrix
  const V = buildVMatrix(homographies)

  // Step 2: Solve for B
  const B = solveForB(V, homographies.length)
  if (!B) {
    throw new Error('Failed to solve for B matrix')
  }

  // Step 3: Extract K from B
  const intrinsics = extractKFromB(B)

  // Step 4: Compute extrinsics for each view
  const extrinsics: Extrinsics[] = []
  for (const hg of homographies) {
    extrinsics.push(computeExtrinsicsFromHomography(hg.H, intrinsics))
  }

  return {
    intrinsics,
    extrinsics,
    rmsError: 0, // Will be computed after reprojection
  }
}