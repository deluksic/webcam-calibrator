import { describe, expect, it } from 'vitest'

import { solveHomographyDLT } from '@/lib/dltHomography'
import { applyHomography, type Correspondence, type Mat3 } from '@/lib/geometry'

function hErr(h: Mat3, pairs: Correspondence[]): number {
  let s = 0
  for (const c of pairs) {
    const p = applyHomography(h, c.plane.x, c.plane.y)
    s += (p.x - c.image.x) ** 2 + (p.y - c.image.y) ** 2
  }
  return Math.sqrt(s / pairs.length)
}

describe('solveHomographyDLT', () => {
  it('recovers identity for plane==image (four corners of unit square)', () => {
    const I: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    const plane = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
      { x: 0.2, y: 0.8 },
    ]
    const pairs: Correspondence[] = plane.map((p) => ({ plane: p, image: applyHomography(I, p.x, p.y) }))
    const h = solveHomographyDLT(pairs)
    expect(h).toBeDefined()
    if (!h) {
      return
    }
    expect(hErr(h, pairs)).toBeLessThan(1e-6)
  })

  it('recovers a known affine+translate homography on the unit square (with Hartley path)', () => {
    // H: u = 2*X + 10, v = 1.5*Y + 5, w = 1  =>  row: [2,0,10,0,1.5,5,0,0,1]
    const Htrue: Mat3 = [2, 0, 10, 0, 1.5, 5, 0, 0, 1] as const
    const plane = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
      { x: 0.3, y: 0.7 },
      { x: 1.2, y: 0.2 },
    ]
    const pairs: Correspondence[] = plane.map((p) => ({
      plane: p,
      image: applyHomography(Htrue, p.x, p.y),
    }))
    const h = solveHomographyDLT(pairs)
    expect(h).toBeDefined()
    if (!h) {
      return
    }
    expect(hErr(h, pairs)).toBeLessThan(1e-2)
  })
})
