import { describe, expect, it } from 'vitest'

import {
  affineCondition,
  affineDeterminant,
  applyAffine,
  fitAffine2D,
  invertAffine,
  isAffineUsable,
  type Affine2D,
} from '@/lib/pointIdentity/localAffine'

describe('localAffine', () => {
  it('fits and inverts a normal orientation-preserving affine', () => {
    const affine: Affine2D = { a: 2, b: 0.1, c: 10, d: -0.05, e: 1.8, f: -4 }
    expect(isAffineUsable(affine)).toBe(true)
    expect(affineDeterminant(affine)).toBeGreaterThan(0)
    const inverse = invertAffine(affine)
    expect(inverse).toBeDefined()
    const point = { x: 13, y: 7 }
    const roundTrip = applyAffine(inverse!, applyAffine(affine, point))
    expect(roundTrip.x).toBeCloseTo(point.x, 10)
    expect(roundTrip.y).toBeCloseTo(point.y, 10)
  })

  it('rejects mirrored transforms', () => {
    const mirrored: Affine2D = { a: 2, b: 0, c: 0, d: 0, e: -2, f: 0 }
    expect(affineDeterminant(mirrored)).toBeLessThan(0)
    expect(isAffineUsable(mirrored)).toBe(false)
  })

  it('rejects near-singular transforms that would splatter predictions', () => {
    const splatter: Affine2D = { a: 1e-4, b: 0, c: 0, d: 0, e: 1, f: 0 }
    expect(affineCondition(splatter)).toBeGreaterThan(50)
    expect(isAffineUsable(splatter)).toBe(false)
  })

  it('recovers an exact affine from non-collinear correspondences', () => {
    const affine: Affine2D = { a: 2, b: 0.1, c: 10, d: -0.05, e: 1.8, f: -4 }
    const sources = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 0, y: 10 },
      { x: 10, y: 10 },
      { x: 4, y: 7 },
    ]
    const fitted = fitAffine2D(
      sources.map((source) => ({
        source,
        target: applyAffine(affine, source),
      })),
    )
    expect(fitted).toBeDefined()
    expect(isAffineUsable(fitted!)).toBe(true)
    const probe = { x: 3, y: 9 }
    const expected = applyAffine(affine, probe)
    const actual = applyAffine(fitted!, probe)
    expect(actual.x).toBeCloseTo(expected.x, 1)
    expect(actual.y).toBeCloseTo(expected.y, 1)
  })
})
