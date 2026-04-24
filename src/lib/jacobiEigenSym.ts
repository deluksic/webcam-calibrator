/**
 * Smallest subspace of homogeneous linear systems + SPD inverse-sqrt, via ml-matrix.
 * Matrices stored row-major: index i*n + j = A[i][j].
 */

import { EigenvalueDecomposition, Matrix, SingularValueDecomposition } from 'ml-matrix'

const { sqrt } = Math

/**
 * Smallest n×n symmetric eigendecomposition: eigenvalues sorted ascending, columns of `vectors` match.
 */
export function jacobiEigenSym(a0: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const m = Matrix.from1DArray(n, n, a0)
  const evd = new EigenvalueDecomposition(m, { assumeSymmetric: true })
  const rawVals = evd.realEigenvalues
  const V = evd.eigenvectorMatrix
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => rawVals[i]! - rawVals[j]!)
  const values = new Float64Array(n)
  const vectors = new Float64Array(n * n)
  for (let j = 0; j < n; j++) {
    const oj = order[j]!
    values[j] = rawVals[oj]!
    for (let i = 0; i < n; i++) {
      vectors[i * n + j] = V.get(i, oj)
    }
  }
  return { values, vectors }
}

/**
 * A is m x n, row-major. Returns unit x minimizing ||A x|| (right singular vector for smallest SV).
 */
export function solveHomogeneousNullVector(a: Float64Array, m: number, n: number): Float64Array {
  const M = Matrix.from1DArray(m, n, a)
  const svd = new SingularValueDecomposition(M, {
    autoTranspose: m >= n,
    computeLeftSingularVectors: false,
    computeRightSingularVectors: true,
  })

  const s = svd.diagonal
  const V = svd.rightSingularVectors

  const minDim = Math.min(m, n)
  const vCols = V.columns

  // Determine the null space index (largest nullity)
  // The smallest singular value(s) correspond to the null space
  // When m < n, there are n - m null vectors
  // When m >= n, there's 1 null vector
  let nullIndex = n - 1
  if (m < n) {
    nullIndex = n - 1 - (n - m)
  }

  const v0 = new Float64Array(n)
  const Vdata = V.data

  // ml-matrix behavior with autoTranspose:
  // - When autoTranspose=false with m < n: V is n x n, null vector is LAST COLUMN
  // - When autoTranspose=true with m < n: V is n x minDim, null vector is LAST COLUMN

  // Get the null vector from the last column of V
  let nullVec: Float64Array
  if (vCols === minDim && V.rows === n) {
    // autoTranspose=true: V is n x minDim, extract from last column
    nullVec = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      nullVec[i] = Vdata[i]![minDim - 1]!
    }
  } else {
    // Default: extract from last column
    nullVec = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      nullVec[i] = Vdata[i]![n - 1]!
    }
  }

  for (let i = 0; i < n; i++) {
    v0[i] = nullVec[i]!
  }

  // Normalize by last element (h22) if it's not zero
  const h22 = v0[n - 1]!
  if (Math.abs(h22) > 1e-12) {
    for (let i = 0; i < n; i++) {
      v0[i]! /= h22
    }
  }
  return v0
}

/** S SPD, row-major. Returns S^{-1/2} (3×3 typical). */
export function symPowInvSqrt(sIn: Float64Array, n: number): Float64Array {
  const { values, vectors } = jacobiEigenSym(sIn, n)
  const out = new Float64Array(n * n)
  for (let k = 0; k < n; k++) {
    const l = values[k]!
    if (l <= 0) {
      return new Float64Array(n * n)
    }
    const invS = 1 / sqrt(l)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        out[i * n + j]! += vectors[i * n + k]! * invS * vectors[j * n + k]!
      }
    }
  }
  return out
}
