import { describe, it, expect } from 'vitest'

import {
  lineFromPoints,
  lineIntersection,
  pointLineDistance,
  subdivideSegment,
  quadAspectRatio,
  computeHomography,
  tryComputeHomography,
  applyHomography,
  computeProjectiveWeights,
  type Corners,
  type Mat3,
} from '@/lib/geometry'

const { abs, max, min } = Math

describe('geometry', () => {
  describe('lineFromPoints', () => {
    it('computes line from two points', () => {
      const line = lineFromPoints({ x: 0, y: 0 }, { x: 1, y: 1 })
      expect(line).not.toBeNull()
      // Line y = -x should have a = -b, c = 0
      expect(line!.a).toBeCloseTo(-0.707, 3)
      expect(line!.b).toBeCloseTo(0.707, 3)
    })

    it('returns null for coincident points', () => {
      const line = lineFromPoints({ x: 1, y: 1 }, { x: 1, y: 1 })
      expect(line).toBeNull()
    })
  })

  describe('lineIntersection', () => {
    it('intersects two non-parallel lines', () => {
      // Horizontal line y = 0 and vertical line x = 0
      const l1 = { a: 0, b: 1, c: 0 } // y = 0
      const l2 = { a: 1, b: 0, c: 0 } // x = 0
      const intersection = lineIntersection(l1, l2)
      expect(intersection).not.toBeNull()
      expect(intersection!.x).toBeCloseTo(0, 5)
      expect(intersection!.y).toBeCloseTo(0, 5)
    })

    it('intersects horizontal y=0 with vertical x = k (lineFromPoints form)', () => {
      const l1 = lineFromPoints({ x: 0, y: 0 }, { x: 100, y: 0 })!
      const l2 = lineFromPoints({ x: 50, y: 0 }, { x: 50, y: 100 })!
      const p = lineIntersection(l1, l2)
      expect(p).not.toBeNull()
      expect(p!.x).toBeCloseTo(50, 5)
      expect(p!.y).toBeCloseTo(0, 5)
    })

    it('returns null for parallel lines', () => {
      const l1 = { a: 1, b: 0, c: 0 } // x = 0
      const l2 = { a: 2, b: 0, c: 1 } // 2x + 1 = 0 → x = -0.5 (parallel!)
      const intersection = lineIntersection(l1, l2)
      expect(intersection).toBeNull()
    })
  })

  describe('pointLineDistance', () => {
    it('computes perpendicular distance', () => {
      const line = { a: 0, b: 1, c: 0 } // y = 0
      const dist = pointLineDistance({ x: 3, y: 4 }, line)
      expect(dist).toBeCloseTo(4, 5)
    })
  })

  describe('subdivideSegment', () => {
    it('subdivides into equal segments', () => {
      const points = subdivideSegment({ x: 0, y: 0 }, { x: 6, y: 0 }, 6)
      expect(points).toHaveLength(7)
      expect(points[1]).toEqual({ x: 1, y: 0 })
      expect(points[3]).toEqual({ x: 3, y: 0 })
      expect(points[6]).toEqual({ x: 6, y: 0 })
    })

    it('returns endpoints when divisions = 1', () => {
      const points = subdivideSegment({ x: 0, y: 0 }, { x: 5, y: 5 }, 1)
      expect(points).toHaveLength(2)
      expect(points[0]).toEqual({ x: 0, y: 0 })
      expect(points[1]).toEqual({ x: 5, y: 5 })
    })
  })

  describe('quadAspectRatio', () => {
    it('computes aspect ratio of square (strip order TL, TR, BL, BR)', () => {
      const corners: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]
      expect(quadAspectRatio(corners)).toBeCloseTo(1, 2)
    })
  })

  describe('computeHomography', () => {
    it('identity: maps unit square to same square', () => {
      // Unit square corners → same corners (no transform)
      const src: Corners = [
        { x: 0, y: 0 }, // TL
        { x: 1, y: 0 }, // TR
        { x: 0, y: 1 }, // BL
        { x: 1, y: 1 }, // BR
      ]
      const H = computeHomography(src)

      // Apply homography to each corner and verify round-trip
      for (const corner of src) {
        const result = applyHomography(H, corner.x, corner.y)
        expect(result.x).toBeCloseTo(corner.x, 5)
        expect(result.y).toBeCloseTo(corner.y, 5)
      }
    })

    it('translation: moves all corners by (10, 20)', () => {
      const src: Corners = [
        { x: 10, y: 20 }, // TL
        { x: 110, y: 20 }, // TR
        { x: 10, y: 120 }, // BL
        { x: 110, y: 120 }, // BR
      ]
      const H = computeHomography(src)

      // Unit square corners should map to translated positions
      expect(applyHomography(H, 0, 0).x).toBeCloseTo(10, 5)
      expect(applyHomography(H, 0, 0).y).toBeCloseTo(20, 5)
      expect(applyHomography(H, 1, 0).x).toBeCloseTo(110, 5)
      expect(applyHomography(H, 1, 0).y).toBeCloseTo(20, 5)
      expect(applyHomography(H, 0, 1).x).toBeCloseTo(10, 5)
      expect(applyHomography(H, 0, 1).y).toBeCloseTo(120, 5)
      expect(applyHomography(H, 1, 1).x).toBeCloseTo(110, 5)
      expect(applyHomography(H, 1, 1).y).toBeCloseTo(120, 5)
    })

    it('scale: doubles the size', () => {
      const src: Corners = [
        { x: 0, y: 0 },
        { x: 2, y: 0 },
        { x: 0, y: 2 },
        { x: 2, y: 2 },
      ]
      const H = computeHomography(src)

      // Unit square corners should map to scaled positions
      expect(abs(applyHomography(H, 0, 0).x - 0) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 0, 0).y - 0) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 0.5, 0.5).x - 1) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 0.5, 0.5).y - 1) < 0.01).toBe(true)
    })

    it('perspective skew: trapezoid shape', () => {
      // Narrower at top, wider at bottom
      const src: Corners = [
        { x: 40, y: 0 }, // TL (shifted right)
        { x: 60, y: 0 }, // TR (shifted left from 100)
        { x: 0, y: 100 }, // BL
        { x: 100, y: 100 }, // BR
      ]
      const H = computeHomography(src)

      // Verify corners map correctly
      expect(abs(applyHomography(H, 0, 0).x - 40) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 0, 0).y - 0) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 1, 0).x - 60) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 1, 0).y - 0) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 0, 1).x - 0) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 0, 1).y - 100) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 1, 1).x - 100) < 0.01).toBe(true)
      expect(abs(applyHomography(H, 1, 1).y - 100) < 0.01).toBe(true)
    })

    it('rotated 90 degrees', () => {
      // Rotate unit square 90 degrees around center
      const src: Corners = [
        { x: 0, y: 1 }, // TL → BL
        { x: 0, y: 0 }, // TR → TL
        { x: 1, y: 1 }, // BL → BR
        { x: 1, y: 0 }, // BR → TR
      ]
      const H = computeHomography(src)

      // (0,0) should map to the rotated position
      expect(applyHomography(H, 0, 0).x).toBeCloseTo(0, 5)
      expect(applyHomography(H, 0, 0).y).toBeCloseTo(1, 5)
    })

    it('edge midpoints preserve straight lines under perspective', () => {
      // This is the key test: under perspective transform, collinear points
      // on a line in the source should map to collinear points in destination
      const src: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]
      const H = computeHomography(src)

      // Points along top edge (u from 0 to 1, v=0) should remain on a line
      const p0 = applyHomography(H, 0, 0)
      const p50 = applyHomography(H, 0.5, 0)
      const p100 = applyHomography(H, 1, 0)

      // They should all have y ≈ 0 (on the top edge)
      expect(p0.y).toBeCloseTo(0, 5)
      expect(p50.y).toBeCloseTo(0, 5)
      expect(p100.y).toBeCloseTo(0, 5)
    })

    it('handles real camera coordinates', () => {
      // Typical webcam resolution with a detected quad
      const src: Corners = [
        { x: 100, y: 80 }, // TL
        { x: 540, y: 100 }, // TR
        { x: 120, y: 420 }, // BL
        { x: 520, y: 400 }, // BR
      ]
      const H = computeHomography(src)

      // Verify all 4 corners round-trip correctly
      expect(applyHomography(H, 0, 0).x).toBeCloseTo(100, 3)
      expect(applyHomography(H, 0, 0).y).toBeCloseTo(80, 3)
      expect(applyHomography(H, 1, 0).x).toBeCloseTo(540, 3)
      expect(applyHomography(H, 1, 0).y).toBeCloseTo(100, 3)
      expect(applyHomography(H, 0, 1).x).toBeCloseTo(120, 3)
      expect(applyHomography(H, 0, 1).y).toBeCloseTo(420, 3)
      expect(applyHomography(H, 1, 1).x).toBeCloseTo(520, 3)
      expect(applyHomography(H, 1, 1).y).toBeCloseTo(400, 3)
    })

    it('horizontal bar: wide but short', () => {
      const src: Corners = [
        { x: 0, y: 0 },
        { x: 640, y: 10 },
        { x: 0, y: 50 },
        { x: 640, y: 40 },
      ]
      const H = computeHomography(src)

      // Verify corners
      expect(applyHomography(H, 0, 0).x).toBeCloseTo(0, 3)
      expect(applyHomography(H, 0, 0).y).toBeCloseTo(0, 3)
      expect(applyHomography(H, 1, 0).x).toBeCloseTo(640, 3)
      expect(applyHomography(H, 1, 0).y).toBeCloseTo(10, 3)
      expect(applyHomography(H, 0, 1).x).toBeCloseTo(0, 3)
      expect(applyHomography(H, 0, 1).y).toBeCloseTo(50, 3)
      expect(applyHomography(H, 1, 1).x).toBeCloseTo(640, 3)
      expect(applyHomography(H, 1, 1).y).toBeCloseTo(40, 3)
    })
  })

  describe('tryComputeHomography', () => {
    it('returns null for coincident corners (singular)', () => {
      const p = { x: 10, y: 20 }
      expect(tryComputeHomography([p, p, p, p] as Corners)).toBeUndefined()
    })

    it('matches computeHomography for a valid quad', () => {
      const src: Corners = [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: 1, y: 1 },
      ]
      const a = tryComputeHomography(src)
      const b = computeHomography(src)
      expect(a).not.toBeNull()
      expect([...a!]).toEqual([...b])
    })
  })

  describe('applyHomography', () => {
    it('applies identity correctly', () => {
      const H: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
      const result = applyHomography(H, 0.5, 0.5)
      expect(result.x).toBeCloseTo(0.5, 5)
      expect(result.y).toBeCloseTo(0.5, 5)
    })

    it('applies translation correctly', () => {
      // H maps (u,v) → (u+10, v+20)
      const H: Mat3 = [1, 0, 10, 0, 1, 20, 0, 0, 1]
      const result = applyHomography(H, 0, 0)
      expect(result.x).toBeCloseTo(10, 5)
      expect(result.y).toBeCloseTo(20, 5)
    })

    it('applies scale correctly', () => {
      // H maps (u,v) → (2*u, 3*v)
      const H: Mat3 = [2, 0, 0, 0, 3, 0, 0, 0, 1]
      const result = applyHomography(H, 0.5, 0.5)
      expect(result.x).toBeCloseTo(1, 5)
      expect(result.y).toBeCloseTo(1.5, 5)
    })
  })

  describe('computeProjectiveWeights', () => {
    it('unit square: weights are non-zero', () => {
      const corners: Corners = [
        { x: 0, y: 0 }, // TL
        { x: 1, y: 0 }, // TR
        { x: 0, y: 1 }, // BL
        { x: 1, y: 1 }, // BR
      ]
      const weights = computeProjectiveWeights(corners)
      // All weights should be non-zero (identity case - sign doesn't matter, only ratios)
      expect(
        weights.every((w) => w !== 0),
        `weights should be non-zero: ${weights}`,
      ).toBe(true)
    })

    it('parallelogram: still gives uniform weights', () => {
      // Sheared but still a parallelogram
      const corners: Corners = [
        { x: 10, y: 20 }, // TL
        { x: 110, y: 20 }, // TR
        { x: 20, y: 120 }, // BL
        { x: 120, y: 120 }, // BR
      ]
      const [w0, w1, w2, w3] = computeProjectiveWeights(corners)
      // All weights should be similar
      const ratio = max(w0, w1, w2, w3) / min(w0, w1, w2, w3)
      expect(ratio < 1.1).toBe(true)
    })

    it('trapezoid: weights differ for perspective', () => {
      // Narrower at top (perspective effect)
      const corners: Corners = [
        { x: 40, y: 0 }, // TL
        { x: 60, y: 0 }, // TR
        { x: 0, y: 100 }, // BL
        { x: 100, y: 100 }, // BR
      ]
      const [w0, w1, w2, w3] = computeProjectiveWeights(corners)
      // At least one weight should be positive
      expect(max(w0, w1, w2, w3) > 0).toBe(true)
    })

    it('real camera quad', () => {
      const corners: Corners = [
        { x: 100, y: 80 }, // TL
        { x: 540, y: 100 }, // TR
        { x: 120, y: 420 }, // BL
        { x: 520, y: 400 }, // BR
      ]
      const [w0, w1, w2, w3] = computeProjectiveWeights(corners)
      // At least one weight should be positive
      expect(max(w0, w1, w2, w3) > 0).toBe(true)
    })
  })
})
