/**
 * SVD decomposition using ml-matrix.
 * A = U * S * V^T where S is diagonal singular values.
 */

import { Matrix, SingularValueDecomposition } from 'ml-matrix'
import type { Mat3 } from './mat3'

export interface SVDResult {
  /** Left singular vectors (U), columns are orthonormal vectors */
  U: Mat3
  /** Singular values (S), diagonal as plain object */
  s: { x: number; y: number; z: number }
  /** Right singular vectors transpose (V^T), rows are orthonormal vectors */
  Vt: Mat3
}

/** Compute SVD of 3x3 matrix. Returns U, singular values, V^T. */
export function svd3x3(a: readonly number[]): SVDResult {
  const M = Matrix.from1DArray(3, 3, [...a])
  const result = new SingularValueDecomposition(M, {
    computeLeftSingularVectors: true,
    computeRightSingularVectors: true,
  })

  // Extract singular values as plain object
  const diag = result.diagonal as unknown as number[]
  const s = { x: diag[0]!, y: diag[1]!, z: diag[2]! }

  // U: left singular vectors (3x3 matrix)
  const Uarr = result.leftSingularVectors.to1DArray()
  const U: Mat3 = [Uarr[0]!, Uarr[1]!, Uarr[2]!, Uarr[3]!, Uarr[4]!, Uarr[5]!, Uarr[6]!, Uarr[7]!, Uarr[8]!]

  // V^T: right singular vectors transposed (3x3 matrix)
  const VtArr = result.rightSingularVectors.transpose().to1DArray()
  const Vt: Mat3 = [VtArr[0]!, VtArr[1]!, VtArr[2]!, VtArr[3]!, VtArr[4]!, VtArr[5]!, VtArr[6]!, VtArr[7]!, VtArr[8]!]

  return { U, s, Vt }
}

/** Compute SVD with m×n matrix, returns null vector (smallest right singular vector).
 *  Returns array of n elements, normalized so last element = 1.
 */
export function svdNullVector(a: readonly number[], m: number, n: number): number[] | undefined {
  const M = Matrix.from1DArray(m, n, [...a])
  const result = new SingularValueDecomposition(M, {
    autoTranspose: m >= n,
    computeLeftSingularVectors: false,
    computeRightSingularVectors: true,
  })

  const V = result.rightSingularVectors
  const vCols = V.columns
  const minDim = Math.min(m, n)

  // Get null vector from last column of V
  let col: number
  if (vCols === minDim) {
    col = minDim - 1
  } else {
    col = n - 1
  }

  // Normalize by last element
  const last = V.get(n - 1, col)!
  if (Math.abs(last) < 1e-12) {
    return undefined
  }

  // Return null vector normalized so last element = 1
  const nullVec: number[] = new Array(n)
  for (let i = 0; i < n; i++) {
    nullVec[i] = V.get(i, col)! / last
  }
  return nullVec
}
