import { describe, expect, it } from 'vitest'

import { canonicalInnerCornersTagPlane } from '@/lib/canonicalTagGeometry'
import type { Point } from '@/lib/geometry'
import { buildTagGrid } from '@/lib/grid'

describe('canonicalInnerCornersTagPlane', () => {
  it('matches buildTagGrid on unit square with 49 points', () => {
    const unit: [Point, Point, Point, Point] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]
    const g = buildTagGrid(unit)
    const c = canonicalInnerCornersTagPlane()
    expect(c.length).toBe(49)
    for (let i = 0; i < 49; i++) {
      expect(c[i]!.x).toBeCloseTo(g.innerCorners[i]!.x, 10)
      expect(c[i]!.y).toBeCloseTo(g.innerCorners[i]!.y, 10)
      expect(c[i]!.z).toBe(0)
    }
  })
})
