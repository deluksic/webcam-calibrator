import { describe, expect, it } from 'vitest'

import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { createIdentityIndex } from '@/lib/pointIdentity/identity'
import {
  createColorIdentityIndex,
  matchObservedIdentity,
} from '@/lib/pointIdentity/match'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'

describe('matchObservedIdentity', () => {
  const pattern = generateHexPattern({ columns: 3, pointsPerEdge: 2 })
  const colors = randomBinaryColors(pattern.points.length, 1)
  const tuning = { pointsPerEdge: 2 }
  const designGraph = buildProximityGraph(pattern.points, tuning)
  const designIndex = createIdentityIndex(designGraph, colors, 3)
  const looseIndex = createColorIdentityIndex(designGraph, colors, 3)

  it('resolves an identical observed graph without unresolved points', () => {
    const observedGraph = buildProximityGraph(pattern.points, tuning)
    const exact = matchObservedIdentity(designIndex, observedGraph, colors, 3)
    const loose = matchObservedIdentity(designIndex, observedGraph, colors, 3, {
      looseByKey: looseIndex,
    })
    expect(exact.summary.total).toBe(pattern.points.length)
    expect(exact.summary.unresolved).toBe(0)
    expect(exact.summary.conflicts).toBe(0)
    expect(loose.summary.total).toBe(pattern.points.length)
    expect(loose.summary.unresolved).toBe(0)
    expect(loose.summary.conflicts).toBe(0)
  })

  it('returns an empty result for an empty observation', () => {
    const observedGraph = buildProximityGraph([], tuning)
    const result = matchObservedIdentity(designIndex, observedGraph, [], 3)
    expect(result.summary.total).toBe(0)
    expect(result.points).toHaveLength(0)
  })
})
