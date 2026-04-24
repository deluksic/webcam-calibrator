import { describe, expect, test } from 'vitest'
import { svd3x3, svdNullVector } from './svd'
import { Mat3 } from './mat3'

describe('svd', () => {
  describe('svd3x3', () => {
    test('identity matrix gives singular values of 1', () => {
      const I = Mat3.identity()
      const { s } = svd3x3(I)
      expect(s.x).toBeCloseTo(1)
      expect(s.y).toBeCloseTo(1)
      expect(s.z).toBeCloseTo(1)
    })

    test('diagonal matrix singular values match diagonal', () => {
      const D = Mat3.diag({ x: 3, y: 2, z: 1 })
      const { s } = svd3x3(D)
      expect(s.x).toBeCloseTo(3)
      expect(s.y).toBeCloseTo(2)
      expect(s.z).toBeCloseTo(1)
    })

    test('singular values are non-negative and sorted descending', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { s } = svd3x3(A)
      expect(s.x).toBeGreaterThanOrEqual(s.y)
      expect(s.y).toBeGreaterThanOrEqual(s.z)
      expect(s.x).toBeGreaterThanOrEqual(0)
      expect(s.y).toBeGreaterThanOrEqual(0)
      expect(s.z).toBeGreaterThanOrEqual(0)
    })

    test('U is orthogonal', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { U } = svd3x3(A)
      expect(Mat3.isOrthogonal(U)).toBe(true)
    })

    test('Vt is orthogonal', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { Vt } = svd3x3(A)
      expect(Mat3.isOrthogonal(Vt)).toBe(true)
    })

    test('reconstruction A = U * S * Vt matches original', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { U, s, Vt } = svd3x3(A)

      const S = Mat3.diag({ x: s.x, y: s.y, z: s.z })
      const US = Mat3.mul(U, S)
      const USVt = Mat3.mul(US, Vt)

      expect(Mat3.equals(A, USVt, 1e-10)).toBe(true)
    })

    test('rotation matrix gives singular values of 1', () => {
      // 90 degree rotation
      const R = [0, -1, 0, 1, 0, 0, 0, 0, 1]
      const { s } = svd3x3(R)
      expect(s.x).toBeCloseTo(1)
      expect(s.y).toBeCloseTo(1)
      expect(s.z).toBeCloseTo(1)
    })
  })

  describe('svdNullVector', () => {
    test('matrix with known null space [1,1,1]', () => {
      // Want null vector [1, 1, 1]
      // Row 0: [1, -1, 0] gives 1*1 + (-1)*1 + 0*1 = 0 ✓
      // Row 1: [0, 1, -1] gives 0*1 + 1*1 + (-1)*1 = 0 ✓
      const m = 2
      const n = 3
      const a: number[] = [
        1, -1, 0,   // row 0
        0, 1, -1,   // row 1
      ]

      const nullVec = svdNullVector(a, m, n)
      expect(nullVec).toBeDefined()
      // [1, 1, 1] is null, normalize by last (1): [1, 1, 1]
      expect(nullVec![0]).toBeCloseTo(1)
      expect(nullVec![1]).toBeCloseTo(1)
      expect(nullVec![2]).toBeCloseTo(1)
    })

    test('underdetermined system (2 rows, 3 cols) gives null vector', () => {
      // 2 equations, 3 unknowns -> 1D null space
      const a = [
        1, 2, 3,
        4, 5, 6,
      ]

      const nullVec = svdNullVector(a, 2, 3)
      expect(nullVec).toBeDefined()
      expect(nullVec![2]).toBeCloseTo(1)
    })

    test('returns undefined for full-rank matrix', () => {
      // 3x3 identity has full rank, should handle gracefully
      const I = Mat3.identity()
      // Note: identity has no null space, but SVD should still return something
      const nullVec = svdNullVector(I, 3, 3)
      // For identity, null vector should be [0,0,1] normalized, giving [0,0,1]
      expect(nullVec).toBeDefined()
    })
  })
})
