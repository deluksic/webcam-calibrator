/**
 * Tests for Rodrigues rotation conversions
 */
import { describe, it, expect } from 'vitest'
import { rodriguesToMatrix, matrixToRodrigues } from './rodrigues'
import { Mat3 } from './mat3'

describe('rodriguesToMatrix', () => {
  it('should return identity for zero rotation', () => {
    const R = rodriguesToMatrix({ x: 0, y: 0, z: 0 })
    expect(Mat3.isRotation(R)).toBe(true)
    expect(R).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1])
  })

  it('should create proper rotation matrix for 90 degree around Z', () => {
    // Rodrigues vector for 90 degrees around Z axis: (0, 0, PI/2)
    const theta = Math.PI / 2
    const rvec = { x: 0, y: 0, z: theta }
    const R = rodriguesToMatrix(rvec)

    // Verify rotation property
    expect(Mat3.isRotation(R, 1e-10)).toBe(true)

    // Rotate point (1, 0, 0) - should get (0, 1, 0)
    const Xc = Mat3.mulVec(R, { x: 1, y: 0, z: 0 })
    expect(Xc.x).toBeCloseTo(0, 3)
    expect(Xc.y).toBeCloseTo(1, 3)
    expect(Xc.z).toBeCloseTo(0, 3)
  })

  it('should create proper rotation matrix for 90 degree around X', () => {
    const theta = Math.PI / 2
    const rvec = { x: theta, y: 0, z: 0 }
    const R = rodriguesToMatrix(rvec)

    expect(Mat3.isRotation(R, 1e-10)).toBe(true)

    // Rotate (0, 1, 0) - should get (0, 0, 1)
    const Xc = Mat3.mulVec(R, { x: 0, y: 1, z: 0 })
    expect(Xc.x).toBeCloseTo(0, 3)
    expect(Xc.y).toBeCloseTo(0, 3)
    expect(Xc.z).toBeCloseTo(1, 3)
  })

  it('should round-trip through matrixToRodrigues', () => {
    const original = { x: 0.3, y: -0.5, z: 0.7 }
    const R = rodriguesToMatrix(original)
    const recovered = matrixToRodrigues(R)
    expect(recovered.x).toBeCloseTo(original.x, 5)
    expect(recovered.y).toBeCloseTo(original.y, 5)
    expect(recovered.z).toBeCloseTo(original.z, 5)
  })
})

describe('Mat3.mulVec', () => {
  it('should multiply row-major matrix with vector correctly', () => {
    // Row-major matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    const M = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    const v = { x: 1, y: 2, z: 3 }

    const result = Mat3.mulVec(M, v)
    // Expected: [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3] = [14, 32, 50]
    expect(result.x).toBeCloseTo(14, 10)
    expect(result.y).toBeCloseTo(32, 10)
    expect(result.z).toBeCloseTo(50, 10)
  })

  it('should handle identity matrix', () => {
    const I = Mat3.identity()
    const v = { x: 1, y: 2, z: 3 }
    const result = Mat3.mulVec(I, v)
    expect(result.x).toBeCloseTo(1, 10)
    expect(result.y).toBeCloseTo(2, 10)
    expect(result.z).toBeCloseTo(3, 10)
  })
})
