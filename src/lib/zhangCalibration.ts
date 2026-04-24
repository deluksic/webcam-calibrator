// Zhang: intrinsics from planar homographies; extrinsics M = K^{-1} H.

import type { CameraIntrinsics } from './cameraModel'
import { solveHomogeneousNullVector, symPowInvSqrt } from './calibration/jacobiEigenSym'
import type { Mat3 } from './geometry'

const { abs, sqrt } = Math

export type Mat3R = [number, number, number, number, number, number, number, number, number]
export type Vec3 = { x: number; y: number; z: number }

/** Columns of H (row-major storage): c0 = (h0,h3,h6), c1, c2. */
function hCol0(h: Mat3): [number, number, number] {
  return [h[0]!, h[3]!, h[6]!]
}
function hCol1(h: Mat3): [number, number, number] {
  return [h[1]!, h[4]!, h[7]!]
}

/**
 * b = (B11, B12, B22, B13, B23, B33) in row-major upper triangle; B = K^{-T}K^{-1} symmetric 3x3.
 * Two rows: h1' B h2 = 0, h1' B h1 - h2' B h2 = 0
 */
/** 6-tuple (B11, B12, B22, B13, B23, B33) for h_i^T B h_j. */
function quadFormCoeffs(a: [number, number, number], b: [number, number, number]): number[] {
  const [a0, a1, a2] = a
  const [b0, b1, b2] = b
  return [
    a0 * b0,
    a0 * b1 + a1 * b0,
    a1 * b1,
    a0 * b2 + a2 * b0,
    a1 * b2 + a2 * b1,
    a2 * b2,
  ]
}

/**
 * 6 elements (B11, B12, B22, B13, B23, B33) to rows for V: each row 6-coeff, second row for equality constraint.
 */
export function zhangVRowsH(h: Mat3): { row0: number[]; row1: number[] } {
  const c0 = hCol0(h) as [number, number, number]
  const c1 = hCol1(h) as [number, number, number]
  const r0 = quadFormCoeffs(c0, c1)
  const v11 = quadFormCoeffs(c0, c0)
  const v22 = quadFormCoeffs(c1, c1)
  const r1 = v11.map((x, i) => x - v22[i]!)
  return { row0: r0, row1: r1 }
}

/**
 * V has 2 rows per homography; 6 columns. b minimizing ||V b||, then recover K.
 */
function solveBFromV(vRows: number[][]): [number, number, number, number, number, number] | undefined {
  const m = vRows.length
  if (m < 6) {
    return undefined
  }
  const a = new Float64Array(m * 6)
  for (let i = 0; i < m; i++) {
    const r = vRows[i]!
    for (let j = 0; j < 6; j++) {
      a[i * 6 + j] = r[j]!
    }
  }
  const bVec = solveHomogeneousNullVector(a, m, 6)
  return [bVec[0]!, bVec[1]!, bVec[2]!, bVec[3]!, bVec[4]!, bVec[5]!] as [number, number, number, number, number, number]
}

function intrinsicsFromB(
  bIn: [number, number, number, number, number, number],
): CameraIntrinsics | undefined {
  // B and -B are both in the conic's null space; B from K is PSD so B11 > 0.
  const s = bIn[0]! < 0 ? -1 : 1
  const b = bIn.map((x) => s * x) as [number, number, number, number, number, number]
  const [B11, B12, B22, B13, B23, B33] = b
  const d = B11 * B22 - B12 * B12
  if (abs(d) < 1e-18) {
    return undefined
  }
  const cy = (B12 * B13 - B11 * B23) / d
  const b13term = B12 * B13 - B11 * B23
  const l = B33 - (B13 * B13 + cy * b13term) / B11
  if (l <= 0 || B11 === 0) {
    return undefined
  }
  const fx2 = l / B11
  if (fx2 <= 0) {
    return undefined
  }
  const fx = sqrt(fx2)
  const d2 = B11 * B22 - B12 * B12
  if (d2 <= 0) {
    return undefined
  }
  const fy2 = (l * B11) / d2
  if (fy2 <= 0) {
    return undefined
  }
  const fy = sqrt(fy2)
  const cx = (-B13 * fx * fx) / l
  if (![fx, fy, cx, cy].every((x) => Number.isFinite(x))) {
    return undefined
  }
  return { fx, fy, cx, cy }
}

export function solveIntrinsicsFromHomographies(hs: Mat3[]): CameraIntrinsics | undefined {
  if (hs.length < 3) {
    return undefined
  }
  const vRows: number[][] = []
  for (const h of hs) {
    const { row0, row1 } = zhangVRowsH(h)
    vRows.push(row0, row1)
  }
  const b = solveBFromV(vRows)
  if (!b) {
    return undefined
  }
  return intrinsicsFromB(b)
}

export function kInverse(k: CameraIntrinsics): Mat3R {
  const { fx, fy, cx, cy } = k
  if (abs(fx) < 1e-12 || abs(fy) < 1e-12) {
    return [0, 0, 0, 0, 0, 0, 0, 0, 0]
  }
  return [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1] as const
}

function matMul3(a: Mat3R, b: Mat3): Mat3R {
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!
  const a10 = a[3]!,
    a11 = a[4]!,
    a12 = a[5]!
  const a20 = a[6]!,
    a21 = a[7]!,
    a22 = a[8]!
  const b00 = b[0]!,
    b01 = b[1]!,
    b02 = b[2]!
  const b10 = b[3]!,
    b11 = b[4]!,
    b12 = b[5]!
  const b20 = b[6]!,
    b21 = b[7]!,
    b22 = b[8]!
  return [
    a00 * b00 + a01 * b10 + a02 * b20,
    a00 * b01 + a01 * b11 + a02 * b21,
    a00 * b02 + a01 * b12 + a02 * b22,
    a10 * b00 + a11 * b10 + a12 * b20,
    a10 * b01 + a11 * b11 + a12 * b21,
    a10 * b02 + a11 * b12 + a12 * b22,
    a20 * b00 + a21 * b10 + a22 * b20,
    a20 * b01 + a21 * b11 + a22 * b21,
    a20 * b02 + a21 * b12 + a22 * b22,
  ]
}

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!]
}

function len3(v: [number, number, number]): number {
  return sqrt(v[0]! * v[0]! + v[1]! * v[1]! + v[2]! * v[2]!)
}

/**
 * M = K^{-1} H; R,t from first two columns; orthonormalize R.
 */
export function extrinsicsFromHomography(h: Mat3, k: CameraIntrinsics): { R: Mat3R; t: Vec3 } | undefined {
  const kInv = kInverse(k)
  if (abs(kInv[0]!) < 1e-15) {
    return undefined
  }
  const m = matMul3(kInv, h)
  const m0: [number, number, number] = [m[0]!, m[3]!, m[6]!]
  const m1: [number, number, number] = [m[1]!, m[4]!, m[7]!]
  const m2: [number, number, number] = [m[2]!, m[5]!, m[8]!]
  const l0 = len3(m0)
  const l1 = len3(m1)
  if (l0 < 1e-15 || l1 < 1e-15) {
    return undefined
  }
  const la = 2 / (l0 + l1)
  const r1: [number, number, number] = [la * m0[0]!, la * m0[1]!, la * m0[2]!]
  const r2: [number, number, number] = [la * m1[0]!, la * m1[1]!, la * m1[2]!]
  const r3v = cross(r1, r2)
  const t: Vec3 = { x: la * m2[0]!, y: la * m2[1]!, z: la * m2[2]! }
  // Columns c0,c1,c2: row major [c0x,c1x,c2x, c0y,c1y,c2y, c0z,c1z,c2z]
  const R: Mat3R = [r1[0]!, r2[0]!, r3v[0]!, r1[1]!, r2[1]!, r3v[1]!, r1[2]!, r2[2]!, r3v[2]!]

  const c0: [number, number, number] = [R[0]!, R[3]!, R[6]!]
  const c1b: [number, number, number] = [R[1]!, R[4]!, R[7]!]
  const c2b: [number, number, number] = [R[2]!, R[5]!, R[8]!]
  // Gram: G[i,j] = c_i·c_j, stored row-major: G_00 = c0·c0, G_01 = c0·c1, etc.
  const g: Float64Array = new Float64Array(9)
  const dot3 = (u: [number, number, number], v2: [number, number, number]) =>
    u[0]! * v2[0]! + u[1]! * v2[1]! + u[2]! * v2[2]!
  g[0] = dot3(c0, c0)
  g[1] = dot3(c0, c1b)
  g[2] = dot3(c0, c2b)
  g[3] = g[1]
  g[4] = dot3(c1b, c1b)
  g[5] = dot3(c1b, c2b)
  g[6] = g[2]
  g[7] = g[5]
  g[8] = dot3(c2b, c2b)

  const sInv = symPowInvSqrt(g, 3)
  const sMat = sInv
  if (sMat[0] === 0 && sMat[4] === 0) {
    return { R, t } // fall through if degenerate
  }
  const rOrtho = matMul3(R, sMat as unknown as Mat3R)
  const det = rOrtho[0]! * (rOrtho[4]! * rOrtho[8]! - rOrtho[5]! * rOrtho[7]!) - rOrtho[1]! * (rOrtho[3]! * rOrtho[8]! - rOrtho[5]! * rOrtho[6]!) + rOrtho[2]! * (rOrtho[3]! * rOrtho[7]! - rOrtho[4]! * rOrtho[6]!)
  let rr = rOrtho
  if (det < 0) {
    rr = [rOrtho[0]!, rOrtho[1]!, -rOrtho[2]!, rOrtho[3]!, rOrtho[4]!, -rOrtho[5]!, rOrtho[6]!, rOrtho[7]!, -rOrtho[8]!] as Mat3R
  }
  return { R: rr, t }
}
