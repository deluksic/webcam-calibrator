/**
 * 3x3 matrix operations using ml-matrix.
 *
 * Storage: row-major flat array [r0c0, r0c1, r0c2, r1c0, ...]
 * Access: index = row * 3 + col
 */

import { Matrix, SingularValueDecomposition, inverse } from 'ml-matrix'
import type { Vec3 } from './vec3'

export type Mat3 = readonly [
  number, number, number, // row 0
  number, number, number, // row 1
  number, number, number,  // row 2
]

export type Mat3Mut = [number, number, number, number, number, number, number, number, number]

/** Index helper */
const idx = (r: number, c: number) => r * 3 + c

/** Convert flat array to Matrix */
function toMatrix(a: readonly number[]): Matrix {
  return Matrix.from1DArray(3, 3, [...a])
}

/** Convert Matrix to flat array */
function toArray(m: Matrix): Mat3 {
  const result = new Array<number>(9)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      result[r * 3 + c] = m.get(r, c)
    }
  }
  return result as unknown as Mat3
}

export const Mat3 = {
  /** Identity matrix */
  identity(): Mat3 {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  },

  /** Zero matrix */
  zero(): Mat3 {
    return [0, 0, 0, 0, 0, 0, 0, 0, 0]
  },

  /** Create from row-major array */
  of(a: readonly number[]): Mat3 {
    return [a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!, a[6]!, a[7]!, a[8]!]
  },

  /** Create from row vectors */
  fromRows(r0: Vec3, r1: Vec3, r2: Vec3): Mat3 {
    return [r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z]
  },

  /** Create from column vectors */
  fromCols(c0: Vec3, c1: Vec3, c2: Vec3): Mat3 {
    return [c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z]
  },

  /** Create diagonal matrix from vector */
  diag(v: Vec3): Mat3 {
    return [v.x, 0, 0, 0, v.y, 0, 0, 0, v.z]
  },

  /** Get element at row, col */
  get(a: readonly number[], row: number, col: number): number {
    return a[idx(row, col)]!
  },

  /** Set element at row, col (returns new array) */
  set(a: number[], row: number, col: number, value: number): Mat3 {
    const out = [...a] as Mat3Mut
    out[idx(row, col)] = value
    return out
  },

  /** Get row as Vec3 */
  row(a: readonly number[], r: number): Vec3 {
    return { x: a[idx(r, 0)]!, y: a[idx(r, 1)]!, z: a[idx(r, 2)]! }
  },

  /** Get column as Vec3 */
  col(a: readonly number[], c: number): Vec3 {
    return { x: a[idx(0, c)]!, y: a[idx(1, c)]!, z: a[idx(2, c)]! }
  },

  /** Matrix addition: A + B */
  add(a: readonly number[], b: readonly number[]): Mat3 {
    return [
      a[0]! + b[0]!, a[1]! + b[1]!, a[2]! + b[2]!,
      a[3]! + b[3]!, a[4]! + b[4]!, a[5]! + b[5]!,
      a[6]! + b[6]!, a[7]! + b[7]!, a[8]! + b[8]!,
    ]
  },

  /** Matrix subtraction: A - B */
  sub(a: readonly number[], b: readonly number[]): Mat3 {
    return [
      a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!,
      a[3]! - b[3]!, a[4]! - b[4]!, a[5]! - b[5]!,
      a[6]! - b[6]!, a[7]! - b[7]!, a[8]! - b[8]!,
    ]
  },

  /** Scalar multiplication */
  scale(a: readonly number[], s: number): Mat3 {
    return [
      a[0]! * s, a[1]! * s, a[2]! * s,
      a[3]! * s, a[4]! * s, a[5]! * s,
      a[6]! * s, a[7]! * s, a[8]! * s,
    ]
  },

  /** Matrix multiplication: A * B */
  mul(a: readonly number[], b: readonly number[]): Mat3 {
    return [
      a[0]! * b[0]! + a[1]! * b[3]! + a[2]! * b[6]!,
      a[0]! * b[1]! + a[1]! * b[4]! + a[2]! * b[7]!,
      a[0]! * b[2]! + a[1]! * b[5]! + a[2]! * b[8]!,
      a[3]! * b[0]! + a[4]! * b[3]! + a[5]! * b[6]!,
      a[3]! * b[1]! + a[4]! * b[4]! + a[5]! * b[7]!,
      a[3]! * b[2]! + a[4]! * b[5]! + a[5]! * b[8]!,
      a[6]! * b[0]! + a[7]! * b[3]! + a[8]! * b[6]!,
      a[6]! * b[1]! + a[7]! * b[4]! + a[8]! * b[7]!,
      a[6]! * b[2]! + a[7]! * b[5]! + a[8]! * b[8]!,
    ]
  },

  /** Matrix-vector multiplication: A * v */
  mulVec(a: readonly number[], v: Vec3): Vec3 {
    return {
      x: a[0]! * v.x + a[1]! * v.y + a[2]! * v.z,
      y: a[3]! * v.x + a[4]! * v.y + a[5]! * v.z,
      z: a[6]! * v.x + a[7]! * v.y + a[8]! * v.z,
    }
  },

  /** Transpose */
  transpose(a: readonly number[]): Mat3 {
    return [a[0]!, a[3]!, a[6]!, a[1]!, a[4]!, a[7]!, a[2]!, a[5]!, a[8]!]
  },

  /** Determinant */
  det(a: readonly number[]): number {
    return (
      a[0]! * (a[4]! * a[8]! - a[5]! * a[7]!) -
      a[1]! * (a[3]! * a[8]! - a[5]! * a[6]!) +
      a[2]! * (a[3]! * a[7]! - a[4]! * a[6]!)
    )
  },

  /** Inverse (returns undefined if singular) */
  inverse(a: readonly number[]): Mat3 | undefined {
    const M = toMatrix(a)
    try {
      const inv = inverse(M)
      if (!inv) return undefined
      return toArray(inv)
    } catch {
      return undefined
    }
  },

  /** Trace (sum of diagonal) */
  trace(a: readonly number[]): number {
    return a[0]! + a[4]! + a[8]!
  },

  /** Check if matrix is symmetric */
  isSymmetric(a: readonly number[], tol = 1e-12): boolean {
    return (
      Math.abs(a[1]! - a[3]!) <= tol &&
      Math.abs(a[2]! - a[6]!) <= tol &&
      Math.abs(a[5]! - a[7]!) <= tol
    )
  },

  /** Check if matrix is orthogonal (A^T * A = I) */
  isOrthogonal(a: readonly number[], tol = 1e-10): boolean {
    const AtA = Mat3.mul(Mat3.transpose(a), a)
    const I = Mat3.identity()
    for (let i = 0; i < 9; i++) {
      if (Math.abs(AtA[i]! - I[i]!) > tol) return false
    }
    return true
  },

  /** Check if rotation matrix (orthogonal + det = 1) */
  isRotation(a: readonly number[], tol = 1e-10): boolean {
    return Mat3.isOrthogonal(a, tol) && Math.abs(Mat3.det(a) - 1) <= tol
  },

  /** Matrix equals with tolerance */
  equals(a: readonly number[], b: readonly number[], tol = 1e-12): boolean {
    for (let i = 0; i < 9; i++) {
      if (Math.abs(a[i]! - b[i]!) > tol) return false
    }
    return true
  },

  /** SVD - returns { singularValues, U, Vt } */
  svd(a: readonly number[]): {
    s: [number, number, number]
    U: Mat3
    Vt: Mat3
  } {
    const M = toMatrix(a)
    const svd = new SingularValueDecomposition(M, {
      computeLeftSingularVectors: true,
      computeRightSingularVectors: true,
    })
    return {
      s: svd.diagonal as [number, number, number],
      U: toArray(svd.leftSingularVectors),
      Vt: toArray(svd.rightSingularVectors.transpose()),
    }
  },

  /** Eigen decomposition for symmetric matrices */
  eigenSym(a: readonly number[]): { values: Vec3; vectors: [Vec3, Vec3, Vec3] } {
    const M = toMatrix(a)
    // Use SVD for symmetric case: eigenvalues from singular values
    const { s, Vt } = Mat3.svd(a)
    return {
      values: { x: s[0]!, y: s[1]!, z: s[2]! },
      vectors: [
        { x: Vt[0]!, y: Vt[3]!, z: Vt[6]! },
        { x: Vt[1]!, y: Vt[4]!, z: Vt[7]! },
        { x: Vt[2]!, y: Vt[5]!, z: Vt[8]! },
      ],
    }
  },

  /** Convert to string for debugging */
  toString(a: readonly number[], precision = 4): string {
    const f = (n: number) => n.toFixed(precision).padStart(precision + 3)
    return [
      `[${f(a[0]!)}, ${f(a[1]!)}, ${f(a[2]!)}]`,
      `[${f(a[3]!)}, ${f(a[4]!)}, ${f(a[5]!)}]`,
      `[${f(a[6]!)}, ${f(a[7]!)}, ${f(a[8]!)}]`,
    ].join('\n')
  },
}
