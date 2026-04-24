import { describe, expect, test } from 'vitest'
import { computeHomography, applyHomography, Vec2 } from './dltHomography'

describe('dltHomography', () => {
  describe('computeHomography', () => {
    test('identity homography from identity mapping', () => {
      // Four corners of unit square
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
      ]
      const dst: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
      ]

      const H = computeHomography(src, dst)

      // Verify identity mapping
      for (let i = 0; i < src.length; i++) {
        const mapped = applyHomography(H, src[i]!)
        expect(mapped.x).toBeCloseTo(dst[i]!.x)
        expect(mapped.y).toBeCloseTo(dst[i]!.y)
      }
    })

    test('translation only', () => {
      const tx = 50
      const ty = 30
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 1 },
      ]
      const dst: Vec2[] = src.map(p => ({ x: p.x + tx, y: p.y + ty }))

      const H = computeHomography(src, dst)

      for (let i = 0; i < src.length; i++) {
        const mapped = applyHomography(H, src[i]!)
        expect(mapped.x).toBeCloseTo(dst[i]!.x, 5)
        expect(mapped.y).toBeCloseTo(dst[i]!.y, 5)
      }
    })

    test('scale transformation', () => {
      const sx = 2
      const sy = 3
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 1 },
      ]
      const dst: Vec2[] = src.map(p => ({ x: p.x * sx, y: p.y * sy }))

      const H = computeHomography(src, dst)

      for (let i = 0; i < src.length; i++) {
        const mapped = applyHomography(H, src[i]!)
        expect(mapped.x).toBeCloseTo(dst[i]!.x)
        expect(mapped.y).toBeCloseTo(dst[i]!.y)
      }
    })

    test('rotation 90 degrees', () => {
      // (x, y) -> (-y, x)
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
      ]
      const dst: Vec2[] = [
        { x: 0, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 1 },
        { x: -1, y: 0 },
      ]

      const H = computeHomography(src, dst)

      for (let i = 0; i < src.length; i++) {
        const mapped = applyHomography(H, src[i]!)
        expect(mapped.x).toBeCloseTo(dst[i]!.x, 4)
        expect(mapped.y).toBeCloseTo(dst[i]!.y, 4)
      }
    })

    test('homography is projective invariant', () => {
      // Test that applying homography to scaled homogeneous coords gives same result
      const H = [2, 0, 10, 0, 3, 20, 0.1, 0.2, 1] as const
      const src: Vec2[] = [
        { x: 1, y: 2 },
        { x: 3, y: 4 },
      ]

      // Apply H to points
      const results = src.map(p => applyHomography(H as any, p))

      // Verify projective nature: scaling H should give same result
      const Hscaled = [4, 0, 20, 0, 6, 40, 0.2, 0.4, 2] as const
      const resultsScaled = src.map(p => applyHomography(Hscaled as any, p))

      for (let i = 0; i < src.length; i++) {
        expect(resultsScaled[i]!.x).toBeCloseTo(results[i]!.x)
        expect(resultsScaled[i]!.y).toBeCloseTo(results[i]!.y)
      }
    })

    test('needs at least 4 points', () => {
      expect(() => computeHomography([{ x: 0, y: 0 }], [{ x: 0, y: 0 }])).toThrow('Need at least 4 points')
    })

    test('requires matching point counts', () => {
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 1 },
      ]
      const dst: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
      ]
      expect(() => computeHomography(src, dst)).toThrow('Point count mismatch')
    })
  })

  describe('applyHomography', () => {
    test('identity homography', () => {
      const I = [1, 0, 0, 0, 1, 0, 0, 0, 1] as const
      const p = { x: 5, y: 7 }
      const result = applyHomography(I as any, p)
      expect(result.x).toBe(5)
      expect(result.y).toBe(7)
    })

    test('pure translation via homogeneous coordinates', () => {
      const tx = 100
      const ty = 50
      const H = [1, 0, tx, 0, 1, ty, 0, 0, 1] as const
      const p = { x: 10, y: 20 }
      const result = applyHomography(H as any, p)
      expect(result.x).toBeCloseTo(110)
      expect(result.y).toBeCloseTo(70)
    })

    test('affine scale', () => {
      const sx = 2
      const sy = 0.5
      const H = [sx, 0, 0, 0, sy, 0, 0, 0, 1] as const
      const p = { x: 3, y: 4 }
      const result = applyHomography(H as any, p)
      expect(result.x).toBeCloseTo(6)
      expect(result.y).toBeCloseTo(2)
    })

    test('projective distortion', () => {
      // H with perspective effect
      const H = [1, 0, 0, 0, 1, 0, 0.01, 0, 1] as const
      const p = { x: 10, y: 5 }
      const result = applyHomography(H as any, p)
      // w = 0.01*10 + 0*5 + 1 = 1.1
      // x' = 10 / 1.1 ≈ 9.09
      // y' = 5 / 1.1 ≈ 4.55
      expect(result.x).toBeCloseTo(9.09, 2)
      expect(result.y).toBeCloseTo(4.55, 2)
    })
  })

  describe('geometric consistency', () => {
    test('collinearity preserved under homography', () => {
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
        { x: 2, y: 2 },
        { x: 3, y: 3 },
      ]
      // Non-trivial homography
      const H = [2, 1, 5, 3, 2, 7, 0.1, 0.2, 1] as const

      const dst = src.map(p => applyHomography(H as any, p))

      // All mapped points should lie on a line (within tolerance)
      // Pick first two points as reference line
      const A = dst[0]!
      const B = dst[1]!
      const ABx = B.x - A.x
      const ABy = B.y - A.y
      const AB_len = Math.sqrt(ABx * ABx + ABy * ABy)

      for (let i = 2; i < dst.length; i++) {
        const P = dst[i]!
        const APx = P.x - A.x
        const APy = P.y - A.y
        // Cross product magnitude / base = distance from line
        const cross = Math.abs(APx * ABy - APy * ABx)
        const dist = cross / AB_len
        expect(dist).toBeLessThan(1e-6)
      }
    })

    test('cross ratio invariance (projective)', () => {
      // Four collinear points
      const src: Vec2[] = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 2, y: 0 },
        { x: 3, y: 0 },
      ]

      // Apply random homography
      const H = [2, 0, 1, 0, 2, 2, 0.1, 0.05, 1] as const
      const dst = src.map(p => applyHomography(H as any, p))

      // Cross ratio of (P0, P1; P2, P3) should be preserved
      // For points at t=0,1,2,3 on a line: cross ratio = (2-0)/(3-0) / (2-1)/(3-1) = 2/3 / 1/2 = 4/3
      // In homogeneous: (t2-t0)/(t3-t0) / ((t2-t1)/(t3-t1))

      // Actually compute cross ratio from projected points
      function crossRatio(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2): number {
        // Use x-coordinate (they're collinear in x after projection)
        const t0 = p0.x, t1 = p1.x, t2 = p2.x, t3 = p3.x
        return ((t2 - t0) / (t3 - t0)) / ((t2 - t1) / (t3 - t1))
      }

      const cr_src = crossRatio(src[0]!, src[1]!, src[2]!, src[3]!)
      const cr_dst = crossRatio(dst[0]!, dst[1]!, dst[2]!, dst[3]!)

      expect(cr_dst).toBeCloseTo(cr_src, 5)
    })
  })
})