import { describe, expect, test } from 'vitest'
import { Mat3, type Mat3 as Mat3Type } from './mat3'
import { Vec3 } from './vec3'

/** Helper to create Mat3 */
const m = (a: number[]): Mat3Type => a as unknown as Mat3Type

describe('Mat3', () => {
  describe('construction', () => {
    test('identity has correct values', () => {
      const I = Mat3.identity()
      expect(I[0]!).toBe(1)
      expect(I[4]!).toBe(1)
      expect(I[8]!).toBe(1)
      expect(I[1]!).toBe(0)
      expect(I[2]!).toBe(0)
      expect(I[3]!).toBe(0)
      expect(I[5]!).toBe(0)
      expect(I[6]!).toBe(0)
      expect(I[7]!).toBe(0)
    })

    test('zero matrix all zeros', () => {
      const Z = Mat3.zero()
      expect(Z.every((v) => v === 0)).toBe(true)
    })

    test('diag creates diagonal matrix', () => {
      const D = Mat3.diag(Vec3.of(1, 2, 3))
      expect(D).toEqual([1, 0, 0, 0, 2, 0, 0, 0, 3])
    })

    test('of clones array', () => {
      const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const M = Mat3.of(arr)
      expect([...M]).toEqual(arr)
    })

    test('fromRows creates matrix from vectors', () => {
      const M = Mat3.fromRows(
        Vec3.of(1, 2, 3),
        Vec3.of(4, 5, 6),
        Vec3.of(7, 8, 9)
      )
      expect(M).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9])
    })

    test('fromCols creates matrix from column vectors', () => {
      const M = Mat3.fromCols(
        Vec3.of(1, 4, 7),
        Vec3.of(2, 5, 8),
        Vec3.of(3, 6, 9)
      )
      expect(M).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9])
    })
  })

  describe('element access', () => {
    const M = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    test('get retrieves correct elements', () => {
      expect(Mat3.get(M, 0, 0)).toBe(1)
      expect(Mat3.get(M, 0, 1)).toBe(2)
      expect(Mat3.get(M, 0, 2)).toBe(3)
      expect(Mat3.get(M, 1, 0)).toBe(4)
      expect(Mat3.get(M, 1, 1)).toBe(5)
      expect(Mat3.get(M, 1, 2)).toBe(6)
      expect(Mat3.get(M, 2, 0)).toBe(7)
      expect(Mat3.get(M, 2, 1)).toBe(8)
      expect(Mat3.get(M, 2, 2)).toBe(9)
    })

    test('set creates new matrix', () => {
      const M2 = Mat3.set([...M], 1, 1, 99)
      expect(M2[4]!).toBe(99)
      expect(M[4]!).toBe(5) // original unchanged
    })

    test('row extracts row as Vec3', () => {
      expect(Mat3.row(M, 0)).toEqual(Vec3.of(1, 2, 3))
      expect(Mat3.row(M, 1)).toEqual(Vec3.of(4, 5, 6))
      expect(Mat3.row(M, 2)).toEqual(Vec3.of(7, 8, 9))
    })

    test('col extracts column as Vec3', () => {
      expect(Mat3.col(M, 0)).toEqual(Vec3.of(1, 4, 7))
      expect(Mat3.col(M, 1)).toEqual(Vec3.of(2, 5, 8))
      expect(Mat3.col(M, 2)).toEqual(Vec3.of(3, 6, 9))
    })
  })

  describe('arithmetic', () => {
    test('add combines matrices', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      const R = Mat3.add(A, B)
      expect(R).toEqual([10, 10, 10, 10, 10, 10, 10, 10, 10])
    })

    test('sub subtracts matrices', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      const R = Mat3.sub(A, B)
      expect(R).toEqual([-8, -6, -4, -2, 0, 2, 4, 6, 8])
    })

    test('scale multiplies all elements', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const R = Mat3.scale(A, 2.5)
      expect(R).toEqual([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5])
    })

    test('mul computes matrix product', () => {
      // A = [[1, 2], [3, 4], [5, 6]] in row-major 3x2
      // But we're 3x3, so test with simple matrices
      const A = [1, 0, 0, 0, 2, 0, 0, 0, 3] // diag(1,2,3)
      const B = [4, 0, 0, 0, 5, 0, 0, 0, 6] // diag(4,5,6)
      const R = Mat3.mul(A, B)
      expect(R).toEqual([4, 0, 0, 0, 10, 0, 0, 0, 18]) // diag(4,10,18)
    })

    test('mul with non-diagonal matrices', () => {
      // [[1, 2, 3], [4, 5, 6], [7, 8, 9]] * [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const I = Mat3.identity()
      const R = Mat3.mul(A, I)
      expect(R).toEqual(A) // A * I = A
    })

    test('mul is NOT commutative', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      const AB = Mat3.mul(A, B)
      const BA = Mat3.mul(B, A)
      expect(AB).not.toEqual(BA)
    })

    test('mulVec computes matrix-vector product', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const v = Vec3.of(1, 0, 0)
      const r = Mat3.mulVec(A, v)
      expect(r).toEqual(Vec3.of(1, 4, 7)) // first column
    })

    test('transpose swaps rows and columns', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const T = Mat3.transpose(A)
      expect(T).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9])
    })

    test('transpose of transpose is original', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.transpose(Mat3.transpose(A))).toEqual([...A])
    })
  })

  describe('determinant', () => {
    test('identity determinant is 1', () => {
      expect(Mat3.det(Mat3.identity())).toBe(1)
    })

    test('zero matrix determinant is 0', () => {
      expect(Mat3.det(Mat3.zero())).toBe(0)
    })

    test('det of diag(1,2,3) = 6', () => {
      expect(Mat3.det(Mat3.diag(Vec3.of(1, 2, 3)))).toBe(6)
    })

    test('det of scalar multiple scales by k³', () => {
      const I = Mat3.identity()
      const k = 2.5
      const kI = Mat3.scale(I, k)
      expect(Mat3.det(kI)).toBeCloseTo(k ** 3)
    })

    test('det(A*B) = det(A) * det(B)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      const detA = Mat3.det(A)
      const detB = Mat3.det(B)
      const detAB = Mat3.det(Mat3.mul(A, B))
      expect(detAB).toBeCloseTo(detA * detB)
    })

    test('det(A^T) = det(A)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.det(Mat3.transpose(A))).toBeCloseTo(Mat3.det(A))
    })

    test('det preserves sign on swap rows', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10] // non-singular
      const detA = Mat3.det(A)
      // Swap row 0 and row 1: row0<->row1
      const swap01 = [0, 1, 0, 1, 0, 0, 0, 0, 1] as Mat3Type
      const R = Mat3.mul(swap01, A)
      expect(Mat3.det(R)).toBeCloseTo(-detA)
    })
  })

  describe('inverse', () => {
    test('inverse of identity is identity', () => {
      const I = Mat3.identity()
      const Iinv = Mat3.inverse(I)
      expect(Iinv).toEqual(I)
    })

    test('inverse of diag(1,2,3) is diag(1, 0.5, 0.333...)', () => {
      const D = Mat3.diag(Vec3.of(1, 2, 3))
      const Dinv = Mat3.inverse(D)
      expect(Dinv).toBeDefined()
      expect(Dinv![0]!).toBeCloseTo(1)
      expect(Dinv![4]!).toBeCloseTo(0.5)
      expect(Dinv![8]!).toBeCloseTo(1 / 3)
    })

    test('A * A^(-1) = I', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10] // non-singular
      const Ainv = Mat3.inverse(A)
      expect(Ainv).toBeDefined()
      const I = Mat3.mul(A, Ainv!)
      expect(Mat3.equals(I, Mat3.identity())).toBe(true)
    })

    test('inverse of singular matrix is undefined', () => {
      const Z = Mat3.zero()
      expect(Mat3.inverse(Z)).toBeUndefined()
    })

    test('inverse of det=0 matrix is undefined', () => {
      // Determinant 0 -> singular
      const singular = [1, 2, 3, 2, 4, 6, 1, 2, 3]
      expect(Mat3.inverse(singular)).toBeUndefined()
    })

    test('(A^-1)^-1 = A', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const Ainv = Mat3.inverse(A)!
      const AinvInv = Mat3.inverse(Ainv)
      expect(Mat3.equals(A, AinvInv!)).toBe(true)
    })

    test('(AB)^-1 = B^-1 * A^-1', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 5] // non-singular (det != 0)
      const ABinv = Mat3.inverse(Mat3.mul(A, B))!
      const BinvAinv = Mat3.mul(Mat3.inverse(B)!, Mat3.inverse(A)!)
      expect(Mat3.equals(ABinv, BinvAinv)).toBe(true)
    })
  })

  describe('trace', () => {
    test('trace of identity is 3', () => {
      expect(Mat3.trace(Mat3.identity())).toBe(3)
    })

    test('trace of diag(a,b,c) = a+b+c', () => {
      const D = Mat3.diag(Vec3.of(1, 2, 3))
      expect(Mat3.trace(D)).toBe(6)
    })

    test('trace(A^T) = trace(A)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.trace(Mat3.transpose(A))).toBeCloseTo(Mat3.trace(A))
    })

    test('trace(A+B) = trace(A) + trace(B)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      expect(Mat3.trace(Mat3.add(A, B))).toBeCloseTo(Mat3.trace(A) + Mat3.trace(B))
    })
  })

  describe('orthogonality', () => {
    test('identity is orthogonal', () => {
      expect(Mat3.isOrthogonal(Mat3.identity())).toBe(true)
    })

    test('rotation by 90 degrees is orthogonal', () => {
      // 90 degree rotation in XY plane
      const R = [0, -1, 0, 1, 0, 0, 0, 0, 1]
      expect(Mat3.isOrthogonal(R)).toBe(true)
    })

    test('non-orthogonal matrix fails', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.isOrthogonal(A)).toBe(false)
    })

    test('reflection has det=-1', () => {
      // Reflection across X axis
      const R = [1, 0, 0, 0, -1, 0, 0, 0, 1]
      expect(Mat3.isOrthogonal(R)).toBe(true)
      expect(Mat3.det(R)).toBe(-1)
    })
  })

  describe('rotation matrix', () => {
    test('identity is rotation', () => {
      expect(Mat3.isRotation(Mat3.identity())).toBe(true)
    })

    test('rotation by 90 degrees is rotation', () => {
      const R = [0, -1, 0, 1, 0, 0, 0, 0, 1]
      expect(Mat3.isRotation(R)).toBe(true)
    })

    test('reflection is NOT rotation', () => {
      const R = [1, 0, 0, 0, -1, 0, 0, 0, 1]
      expect(Mat3.isRotation(R)).toBe(false)
    })

    test('non-orthogonal matrix is NOT rotation', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.isRotation(A)).toBe(false)
    })

    test('determinant of rotation is 1', () => {
      // 45 degree rotation
      const c = Math.cos(Math.PI / 4)
      const s = Math.sin(Math.PI / 4)
      const R = [c, -s, 0, s, c, 0, 0, 0, 1]
      expect(Mat3.det(R)).toBeCloseTo(1)
    })
  })

  describe('symmetry', () => {
    test('identity is symmetric', () => {
      expect(Mat3.isSymmetric(Mat3.identity())).toBe(true)
    })

    test('diagonal matrix is symmetric', () => {
      const D = Mat3.diag(Vec3.of(1, 2, 3))
      expect(Mat3.isSymmetric(D)).toBe(true)
    })

    test('non-symmetric matrix fails', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.isSymmetric(A)).toBe(false)
    })

    test('A = A^T for symmetric matrix', () => {
      // Symmetric matrix
      const S = [1, 2, 3, 2, 5, 6, 3, 6, 9]
      expect(Mat3.transpose(S)).toEqual(S)
    })
  })

  describe('SVD', () => {
    test('SVD on identity gives singular values of 1', () => {
      const { s } = Mat3.svd(Mat3.identity())
      expect(s[0]).toBeCloseTo(1)
      expect(s[1]).toBeCloseTo(1)
      expect(s[2]).toBeCloseTo(1)
    })

    test('SVD gives orthonormal U and V', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { U, Vt } = Mat3.svd(A)
      expect(Mat3.isOrthogonal(U)).toBe(true)
      expect(Mat3.isOrthogonal(Vt)).toBe(true)
    })

    test('SVD: A = U * S * V^T', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { s, U, Vt } = Mat3.svd(A)

      // Create diagonal matrix of singular values
      const S = Mat3.diag(Vec3.of(s[0], s[1], s[2]))

      // Compute U * S * V^T
      const US = Mat3.mul(U, S)
      const USVt = Mat3.mul(US, Vt)

      // Should reconstruct A
      expect(Mat3.equals(A, USVt, 1e-10)).toBe(true)
    })

    test('Singular values are non-negative and sorted descending', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 10]
      const { s } = Mat3.svd(A)
      expect(s[0]!).toBeGreaterThanOrEqual(s[1]!)
      expect(s[1]!).toBeGreaterThanOrEqual(s[2]!)
      expect(s.every((v) => v >= 0)).toBe(true)
    })
  })

  describe('equals', () => {
    test('identical matrices are equal', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.equals(A, A)).toBe(true)
    })

    test('different matrices are not equal', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      expect(Mat3.equals(A, B)).toBe(false)
    })

    test('approximately equal with default tolerance', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = A.map((v) => v + 1e-13)
      expect(Mat3.equals(A, B as unknown as Mat3Type)).toBe(true)
    })

    test('custom tolerance works', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = A.map((v) => v + 0.001)
      expect(Mat3.equals(A, B as unknown as Mat3Type, 0.01)).toBe(true)
      expect(Mat3.equals(A, B as unknown as Mat3Type, 1e-6)).toBe(false)
    })
  })

  describe('matrix identities', () => {
    test('A + B = B + A (commutative)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      expect(Mat3.add(A, B)).toEqual(Mat3.add(B, A))
    })

    test('A * (B + C) = A*B + A*C (distributive)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const C = [0, 1, 0, 1, 0, 1, 1, 1, 0]
      const lhs = Mat3.mul(A, Mat3.add(B, C))
      const rhs = Mat3.add(Mat3.mul(A, B), Mat3.mul(A, C))
      expect(Mat3.equals(lhs, rhs)).toBe(true)
    })

    test('(A*B)*C = A*(B*C) (associative)', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const B = [9, 8, 7, 6, 5, 4, 3, 2, 1]
      const C = Mat3.identity()
      expect(Mat3.equals(
        Mat3.mul(Mat3.mul(A, B), C),
        Mat3.mul(A, Mat3.mul(B, C))
      )).toBe(true)
    })

    test('A * I = A', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.mul(A, Mat3.identity())).toEqual([...A])
    })

    test('I * A = A', () => {
      const A = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      expect(Mat3.mul(Mat3.identity(), A)).toEqual([...A])
    })

    test('I^T = I', () => {
      expect(Mat3.transpose(Mat3.identity())).toEqual(Mat3.identity())
    })
  })

  describe('rotation-specific tests', () => {
    test('rotation matrix preserves vector length', () => {
      // 45 degree rotation
      const c = Math.cos(Math.PI / 4)
      const s = Math.sin(Math.PI / 4)
      const R = [c, -s, 0, s, c, 0, 0, 0, 1]

      const v = Vec3.of(3, 4, 0)
      const v2 = Mat3.mulVec(R, v)
      expect(Vec3.length(v2)).toBeCloseTo(Vec3.length(v))
    })

    test('composition of rotations is rotation', () => {
      // Rotation by 30 degrees
      const angle = Math.PI / 6
      const c = Math.cos(angle)
      const s = Math.sin(angle)
      const R = [c, -s, 0, s, c, 0, 0, 0, 1]

      // R * R should also be a rotation
      const R2 = Mat3.mul(R, R)
      expect(Mat3.isRotation(R2)).toBe(true)
    })

    test('rotation inverse equals transpose', () => {
      const c = Math.cos(Math.PI / 4)
      const s = Math.sin(Math.PI / 4)
      const R = [c, -s, 0, s, c, 0, 0, 0, 1]

      const Rinv = Mat3.inverse(R)
      const Rt = Mat3.transpose(R)
      expect(Mat3.equals(Rinv!, Rt)).toBe(true)
    })
  })

  describe('toString', () => {
    test('formats matrix nicely', () => {
      const M = [1, 2, 3, 4, 5, 6, 7, 8, 9]
      const s = Mat3.toString(M, 2)
      expect(s).toContain('[')
      expect(s).toContain('1.00')
      expect(s).toContain('9.00')
    })
  })
})
