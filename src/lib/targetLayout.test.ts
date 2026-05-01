import { describe, expect, it } from 'vitest'

import type { TagObservation } from '@/lib/calibrationTypes'
import type { Corners } from '@/lib/geometry'
import { learnLayoutFromFrame } from '@/lib/targetLayout'

const sq = (o: { x: number; y: number }): Corners => [
  { x: 100 + o.x, y: 100 + o.y },
  { x: 200 + o.x, y: 100 + o.y },
  { x: 100 + o.x, y: 200 + o.y },
  { x: 200 + o.x, y: 200 + o.y },
]

function layoutCornerMeanXY(L: Map<number, Corners>): { x: number; y: number } {
  let sx = 0
  let sy = 0
  let n = 0
  for (const corners of L.values()) {
    for (const p of corners) {
      sx += p.x
      sy += p.y
      n++
    }
  }
  return n ? { x: sx / n, y: sy / n } : { x: 0, y: 0 }
}

describe('learnLayoutFromFrame', () => {
  it('produces 2+ tags with zero mean corner xy after anchor unit-square mapping', () => {
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
    const mean = layoutCornerMeanXY(L)
    expect(Math.abs(mean.x)).toBeLessThan(1e-9)
    expect(Math.abs(mean.y)).toBeLessThan(1e-9)
    const a = L.get(2)!
    const b = L.get(5)!
    expect(b[0]!.x - a[0]!.x).toBeGreaterThan(1.5)
  })
})
