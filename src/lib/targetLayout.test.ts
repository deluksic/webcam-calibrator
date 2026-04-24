import { describe, expect, it } from 'vitest'

import { learnLayoutFromFrame } from '@/lib/targetLayout'
import type { TagObservation } from '@/lib/calibrationTypes'
import type { Corners } from '@/lib/geometry'

const sq = (o: { x: number; y: number }): Corners => [
  { x: 100 + o.x, y: 100 + o.y },
  { x: 200 + o.x, y: 100 + o.y },
  { x: 100 + o.x, y: 200 + o.y },
  { x: 200 + o.x, y: 200 + o.y },
]

describe('learnLayoutFromFrame', () => {
  it('produces 2+ tags with anchor (lowest id) on unit square', () => {
    const tags: TagObservation[] = [
      { tagId: 2, rotation: 0, corners: sq({ x: 0, y: 0 }), score: 1 },
      { tagId: 5, rotation: 0, corners: sq({ x: 220, y: 0 }), score: 1 },
    ]
    const L = learnLayoutFromFrame(tags)
    expect(L).toBeDefined()
    if (!L) {
      return
    }
    expect(L.size).toBe(2)
    const a = L.get(2)!.map((p) => ({ x: p.x, y: p.y }))
    expect(a[0]).toEqual({ x: 0, y: 0 })
    expect(a[1]).toEqual({ x: 1, y: 0 })
    expect(a[2]).toEqual({ x: 0, y: 1 })
    expect(a[3]).toEqual({ x: 1, y: 1 })
    const b = L.get(5)!
    expect(b[0]!.x).toBeGreaterThan(1.5)
  })
})
