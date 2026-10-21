import { describe, expect, it } from 'vitest'

import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import type { PatternPoint } from '@/lib/pointIdentity/types'

const points: PatternPoint[] = [
  { x: 0, y: 0, index: 10 },
  { x: 50, y: 0, index: 11 },
  { x: 100, y: 0, index: 12 },
  { x: 0, y: 100, index: 13 },
  { x: 100, y: 100, index: 14 },
]

describe('applyObservationMask', () => {
  it('keeps all points by default', () => {
    expect(applyObservationMask(points)).toEqual(points)
  })

  it('crops the right half while preserving original indices', () => {
    const kept = applyObservationMask(points, { ...DEFAULT_OBSERVATION_MASK, cropRight: 50 })
    expect(kept.map((point) => point.index)).toEqual([10, 11, 13])
  })

  it('drops all points at 100 percent', () => {
    expect(applyObservationMask(points, { ...DEFAULT_OBSERVATION_MASK, dropoutPercent: 100 })).toEqual([])
  })

  it('is reproducible from the dropout seed', () => {
    const first = applyObservationMask(points, { ...DEFAULT_OBSERVATION_MASK, dropoutPercent: 50, dropoutSeed: 7 })
    const second = applyObservationMask(points, { ...DEFAULT_OBSERVATION_MASK, dropoutPercent: 50, dropoutSeed: 7 })
    expect(first).toEqual(second)
  })
})
