/**
 * Tests for Brown-Conrady distortion model
 */
import { describe, it, expect } from 'vitest'
import { distortPoint, undistortPoint } from './distortion'
import type { DistortionCoeffs } from './distortion'

describe('distortPoint', () => {
  it('should return input point for zero distortion', () => {
    const dist: DistortionCoeffs = { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
    const result = distortPoint({ x: 0.5, y: 0.3 }, dist)
    expect(result.x).toBeCloseTo(0.5, 10)
    expect(result.y).toBeCloseTo(0.3, 10)
  })

  it('should apply radial distortion', () => {
    const dist: DistortionCoeffs = { k1: 0.1, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
    const result = distortPoint({ x: 0.5, y: 0.3 }, dist)

    // Radial distortion should push points outward
    const r2 = 0.5 * 0.5 + 0.3 * 0.3  // 0.25 + 0.09 = 0.34
    const radial = 1 + 0.1 * r2  // ~1.034
    expect(result.x).toBeCloseTo(0.5 * radial, 5)
    expect(result.y).toBeCloseTo(0.3 * radial, 5)
  })

  it('should apply tangential distortion', () => {
    const dist: DistortionCoeffs = { k1: 0, k2: 0, p1: 0.1, p2: 0.05, k3: 0, k4: 0, k5: 0, k6: 0 }
    const result = distortPoint({ x: 0.5, y: 0.3 }, dist)

    // Tangential distortion formulas
    // dx = (1 + k1*r^2 + k2*r^4)*x + 2*p1*x*y + p2*(r^2 + 2*x^2)
    // dy = (1 + k1*r^2 + k2*r^4)*y + p1*(r^2 + 2*y^2) + 2*p2*x*y
    // With k1=k2=0: dx = 2*p1*x*y + p2*(r^2 + 2*x^2), dy = p1*(r^2 + 2*y^2) + 2*p2*x*y
    const r2 = 0.34
    const dx = 2 * 0.1 * 0.5 * 0.3 + 0.05 * (r2 + 2 * 0.25)  // 0.03 + 0.042 = 0.072
    const dy = 0.1 * (r2 + 2 * 0.09) + 2 * 0.05 * 0.5 * 0.3  // 0.052 + 0.03 = 0.082
    expect(result.x).toBeCloseTo(0.5 + dx, 5)
    expect(result.y).toBeCloseTo(0.3 + dy, 5)
  })
})

describe('undistortPoint', () => {
  it('should invert distortion approximately for small distortion', () => {
    const dist: DistortionCoeffs = { k1: 0.05, k2: 0.01, p1: 0.001, p2: 0.001, k3: 0, k4: 0, k5: 0, k6: 0 }
    const original = { x: 0.3, y: 0.4 }
    const distorted = distortPoint(original, dist)
    const recovered = undistortPoint(distorted, dist)

    expect(recovered.x).toBeCloseTo(original.x, 3)
    expect(recovered.y).toBeCloseTo(original.y, 3)
  })

  it('should invert distortion for moderate distortion', () => {
    const dist: DistortionCoeffs = { k1: 0.2, k2: 0.05, p1: 0.005, p2: 0.005, k3: 0, k4: 0, k5: 0, k6: 0 }
    const original = { x: 0.5, y: 0.5 }
    const distorted = distortPoint(original, dist)
    const recovered = undistortPoint(distorted, dist)

    expect(recovered.x).toBeCloseTo(original.x, 2)
    expect(recovered.y).toBeCloseTo(original.y, 2)
  })

  it('should handle points near origin', () => {
    const dist: DistortionCoeffs = { k1: 0.1, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
    const original = { x: 0.01, y: 0.01 }
    const distorted = distortPoint(original, dist)
    const recovered = undistortPoint(distorted, dist)

    expect(recovered.x).toBeCloseTo(original.x, 5)
    expect(recovered.y).toBeCloseTo(original.y, 5)
  })
})

describe('round-trip distortion', () => {
  it('should preserve points through distort-undistort cycle', () => {
    const dist: DistortionCoeffs = { k1: 0.1, k2: 0.02, p1: 0.005, p2: 0.002, k3: 0, k4: 0, k5: 0, k6: 0 }

    const testPoints = [
      { x: 0.1, y: 0.2 },
      { x: 0.5, y: 0.5 },
      { x: -0.3, y: 0.4 },
      { x: 0.0, y: 0.0 },
    ]

    for (const pt of testPoints) {
      const distorted = distortPoint(pt, dist)
      const recovered = undistortPoint(distorted, dist)
      expect(recovered.x).toBeCloseTo(pt.x, 3)
      expect(recovered.y).toBeCloseTo(pt.y, 3)
    }
  })
})
