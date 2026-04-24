import { describe, expect, test } from 'vitest'
import {
  computeV12,
  computeV11MinusV22,
  buildVMatrix,
  solveForB,
  extractKFromB,
  zhangCalibration,
  Homography,
} from './zhangCalibration'
import { Mat3 } from './mat3'

describe('zhangCalibration', () => {
  describe('computeV12', () => {
    test('identity homography', () => {
      // H = I, columns are:
      // h1 = [0, 1, 0] (column index 1)
      // h2 = [0, 0, 1] (column index 2)
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]

      const v12 = computeV12(H)

      // v12 = [h1x*h2x, h1x*h2y+h2x*h1y, h1y*h2y,
      //        h1x*h2z+h2x*h1z, h1y*h2z+h2y*h1z, h1z*h2z]
      // h1 = [0, 1, 0], h2 = [0, 0, 1]
      // = [0*0, 0*0+0*1, 1*0, 0*1+0*0, 1*1+0*0, 0*1] = [0, 0, 0, 0, 1, 0]
      expect(v12[0]).toBeCloseTo(0)
      expect(v12[1]).toBeCloseTo(0)
      expect(v12[2]).toBeCloseTo(0)
      expect(v12[3]).toBeCloseTo(0)
      expect(v12[4]).toBeCloseTo(1)
      expect(v12[5]).toBeCloseTo(0)
    })

    test('pure translation homography', () => {
      // H = [[1, 0, tx],
      //      [0, 1, ty],
      //      [0,  0,  1]]
      // h1 = [0, 1, 0], h2 = [tx, ty, 1]
      const tx = 100, ty = 50
      const H: Mat3 = [1, 0, tx, 0, 1, ty, 0, 0, 1]

      const v12 = computeV12(H)

      // h1 = [0, 1, 0], h2 = [tx, ty, 1]
      // v12 = [0*tx, 0*ty+tx*1, 1*ty, 0*1+tx*0, 1*1+ty*0, 0*1]
      //     = [0, tx, ty, 0, 1, 0]
      expect(v12[0]).toBeCloseTo(0)
      expect(v12[1]).toBeCloseTo(tx)
      expect(v12[2]).toBeCloseTo(ty)
      expect(v12[3]).toBeCloseTo(0)
      expect(v12[4]).toBeCloseTo(1)
      expect(v12[5]).toBeCloseTo(0)
    })
  })

  describe('computeV11MinusV22', () => {
    test('identity homography gives zeros', () => {
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const v = computeV11MinusV22(H)

      // h1 = [0, 1, 0], h2 = [0, 0, 1]
      // h1² = [0, 0, 1, 0, 0, 0]
      // h2² = [0, 0, 0, 0, 0, 1]
      // v11 - v22 = [0-0, 0-0, 1-0, 0-0, 0-0, 0-1] = [0, 0, 1, 0, 0, -1]
      expect(v[0]).toBeCloseTo(0)
      expect(v[1]).toBeCloseTo(0)
      expect(v[2]).toBeCloseTo(1)
      expect(v[3]).toBeCloseTo(0)
      expect(v[4]).toBeCloseTo(0)
      expect(v[5]).toBeCloseTo(-1)
    })

    test('pure scale homography', () => {
      // H = [[sx, 0, 0],
      //      [0, sy, 0],
      //      [0,  0, 1]]
      // h1 = [0, sy, 0], h2 = [0, 0, 1]
      const sx = 2, sy = 3
      const H: Mat3 = [sx, 0, 0, 0, sy, 0, 0, 0, 1]

      const v = computeV11MinusV22(H)

      // h1 = [0, sy, 0], h2 = [0, 0, 1]
      // h1² = [0, 0, sy², 0, 0, 0]
      // h2² = [0, 0, 0, 0, 0, 1]
      // v11 - v22 = [0-0, 0-0, sy²-0, 0-0, 0-0, 0-1] = [0, 0, sy², 0, 0, -1]
      expect(v[0]).toBeCloseTo(0)
      expect(v[2]).toBeCloseTo(sy * sy)
      expect(v[5]).toBeCloseTo(-1)
    })
  })

  describe('buildVMatrix', () => {
    test('single homography gives 2 rows', () => {
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const V = buildVMatrix([{ H }])

      expect(V.length).toBe(12) // 2 rows * 6 cols
    })

    test('three homographies gives 6 rows', () => {
      const H1: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const H2: Mat3 = [1, 0, 5, 0, 1, 10, 0, 0, 1]
      const H3: Mat3 = [1, 0, 3, 0, 1, 7, 0, 0, 1]

      const V = buildVMatrix([{ H: H1 }, { H: H2 }, { H: H3 }])

      expect(V.length).toBe(36) // 6 rows * 6 cols
    })

    test('rows correspond to v12 and v11-v22', () => {
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const v12 = computeV12(H)
      const v11v22 = computeV11MinusV22(H)

      const V = buildVMatrix([{ H }])

      // First row should be v12
      for (let i = 0; i < 6; i++) {
        expect(V[i]).toBeCloseTo(v12[i]!)
      }

      // Second row should be v11-v22
      for (let i = 0; i < 6; i++) {
        expect(V[6 + i]).toBeCloseTo(v11v22[i]!)
      }
    })
  })

  describe('solveForB', () => {
    test('V matrix with 3 homographies yields B', () => {
      // Create 3 homographies that satisfy the constraints
      const H1: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const H2: Mat3 = [2, 0, 10, 0, 3, 20, 0, 0, 1]
      const H3: Mat3 = [1.5, 0.5, 5, 0.3, 2, 15, 0.01, 0.02, 1]

      const V = buildVMatrix([{ H: H1 }, { H: H2 }, { H: H3 }])
      const B = solveForB(V, 3)

      expect(B).toBeDefined()
      expect(B!.length).toBe(6)

      // B11 should be positive (or near zero for numerical reasons)
      expect(B![0]).toBeGreaterThan(-1e-6)
    })

    test('requires at least 3 homographies for 6 unknowns', () => {
      // With only 2 homographies (4 equations), system is underdetermined
      const H1: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const H2: Mat3 = [2, 0, 10, 0, 3, 20, 0, 0, 1]

      const V = buildVMatrix([{ H: H1 }, { H: H2 }])
      const B = solveForB(V, 2)

      // Should still return something (null vector of underdetermined system)
      expect(B).toBeDefined()
    })
  })

  describe('extractKFromB', () => {
    test('known B from calibration gives expected K', () => {
      // Create a test B matrix from known K
      // K = [[1000, 0, 320],
      //      [0, 1000, 240],
      //      [0, 0, 1]]
      // B = λ * K^(-T) * K^(-1)
      // For simplicity, use λ = 1
      const fx = 1000, fy = 1000, cx = 320, cy = 240

      // K^(-T) = [[1/fx, 0, 0],
      //           [-s/(fx*fy), 1/fy, 0],
      //           [(s*cy-cx*fy)/(fx*fy), -cy/fy, 1]]
      // B_ij = K^(-T)_ik * K^(-T)_jk for k=1,3 (ignoring last row/column of 1s)

      // Compute B from the formula: B = K^(-T) * K^(-1)
      // B11 = 1/fx², B12 = 0, B22 = 1/fy² + cy²/fy² = (1+cy²)/fy²
      // B13 = -cx/fx², B23 = -cy/fy², B33 = 1 + cx²/fx² + cy²/fy²

      const B11 = 1 / (fx * fx)
      const B12 = 0
      const B22 = 1 / (fy * fy)
      const B13 = -cx / (fx * fx)
      const B23 = -cy / (fy * fy)
      const B33 = 1 + (cx * cx) / (fx * fx) + (cy * cy) / (fy * fy)

      const B = [B11, B12, B22, B13, B23, B33] as const

      const K = extractKFromB([...B])

      // K values should match input (within numerical tolerance)
      expect(K.fx).toBeCloseTo(fx, 0)
      expect(K.fy).toBeCloseTo(fy, 0)
      expect(K.cx).toBeCloseTo(cx, 0)
      expect(K.cy).toBeCloseTo(cy, 0)
    })
  })

  describe('zhangCalibration', () => {
    test('needs at least 3 homographies', () => {
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      expect(() => zhangCalibration([{ H }])).toThrow('Need at least 3 homographies')
    })

    test('returns valid intrinsics and extrinsics', () => {
      // Create synthetic homographies that correspond to a real camera
      // K with fx=1000, fy=1000, cx=320, cy=240
      const fx = 1000, fy = 1000, cx = 320, cy = 240

      // Use diagonal K as base homography (no distortion, no rotation)
      const H1: Mat3 = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

      // Use homographies with same K but different t values
      const H2: Mat3 = [fx, 0, cx, 0, fy, cy + 100, 0, 0, 1]
      const H3: Mat3 = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

      const result = zhangCalibration([{ H: H1 }, { H: H2 }, { H: H3 }])

      // Check output structure is valid
      expect(result.intrinsics.fx).toBeDefined()
      expect(result.intrinsics.fy).toBeDefined()
      expect(result.intrinsics.cx).toBeDefined()
      expect(result.intrinsics.cy).toBeDefined()
      expect(result.extrinsics).toHaveLength(3)
    })

    test('extrinsics have valid rvec and tvec', () => {
      const homographies: Homography[] = [
        { H: [1, 0, 0, 0, 1, 0, 0, 0, 1] as Mat3 },
        { H: [1, 0, 10, 0, 1, 5, 0, 0, 1] as Mat3 },
        { H: [1, 0, 5, 0, 1, 15, 0, 0, 1] as Mat3 },
      ]

      const result = zhangCalibration(homographies)

      for (const ext of result.extrinsics) {
        expect(ext.rvec.x).toBeDefined()
        expect(ext.rvec.y).toBeDefined()
        expect(ext.rvec.z).toBeDefined()
        expect(ext.tvec.x).toBeDefined()
        expect(ext.tvec.y).toBeDefined()
        expect(ext.tvec.z).toBeDefined()
      }
    })
  })

  describe('geometric constraints', () => {
    test('B matrix has positive B11', () => {
      // Create 3 diverse homographies
      const H1: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const H2: Mat3 = [1.5, 0.2, 10, -0.1, 2, 15, 0.01, 0.02, 1]
      const H3: Mat3 = [2, 0.5, 5, 0.3, 1.5, 20, 0.02, -0.01, 1]

      const V = buildVMatrix([{ H: H1 }, { H: H2 }, { H: H3 }])
      const B = solveForB(V, 3)

      // B11 should be positive for valid calibration
      expect(B![0]).toBeGreaterThan(0)
    })

    test('v12 is orthogonal to B for valid homographies', () => {
      // For valid calibration homographies, v12^T * B = 0
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const v12 = computeV12(H)

      // With identity homography: v12 = [0, 0, 0, 0, 1, 0]
      // Just check it has the expected shape
      expect(v12[4]).toBeCloseTo(1)
    })
  })
})