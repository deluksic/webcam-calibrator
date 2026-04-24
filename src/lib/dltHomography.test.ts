import { describe, it, expect } from 'vitest'

import { solveHomographyDLT } from '@/lib/dltHomography'
import { applyHomography, type Correspondence, type Mat3 } from '@/lib/geometry'

function hErr(h: Mat3, pairs: Correspondence[]): number {
  const apply = (x: number, y: number) => {
    const w = h[6]! * x + h[7]! * y + h[8]!
    return {
      x: (h[0]! * x + h[1]! * y + h[2]!) / w,
      y: (h[3]! * x + h[4]! * y + h[5]!) / w,
    }
  }
  let s = 0
  for (const c of pairs) {
    const p = apply(c.plane.x, c.plane.y)
    s += (p.x - c.image.x) ** 2 + (p.y - c.image.y) ** 2
  }
  return Math.sqrt(s / pairs.length)
}

describe('solveHomographyDLT', () => {
  it('recovers identity for plane==image (five corners of unit square)', () => {
    const I: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    const plane = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
      { x: 0.5, y: 0.5 },
    ]
    const pairs: Correspondence[] = plane.map((p) => ({ plane: p, image: applyHomography(I, p.x, p.y) }))

    const h = solveHomographyDLT(pairs)

    expect(h).toBeDefined()
    if (!h) {
      return
    }
    expect(hErr(h, pairs)).toBeLessThan(0.01)
  })
})