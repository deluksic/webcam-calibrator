import { describe, expect, it } from 'vitest'

import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import {
  computeFingerprint,
  createIdentityIndex,
  createIdentityIndexesByDepth,
  fingerprintKeyAtDepth,
} from '@/lib/pointIdentity/identity'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'

describe('identity index depths', () => {
  const pattern = generateHexPattern({ columns: 4, pointsPerEdge: 3 })
  const colors = randomBinaryColors(pattern.points.length, 7)
  const tuning = { pointsPerEdge: 3 }
  const graph = buildProximityGraph(pattern.points, tuning)

  it('prefix keys match a fingerprint computed at that depth', () => {
    for (let pointIndex = 0; pointIndex < graph.points.length; pointIndex += 1) {
      const full = computeFingerprint(graph, colors, pointIndex, 5)
      for (let depth = 1; depth <= 5; depth += 1) {
        const atDepth = computeFingerprint(graph, colors, pointIndex, depth)
        expect(fingerprintKeyAtDepth(full, depth)).toBe(atDepth.key)
      }
    }
  })

  it('builds the same indexes as one per-depth pass', () => {
    const indexes = createIdentityIndexesByDepth(graph, colors, 5)
    expect(indexes).toHaveLength(5)
    for (let depth = 1; depth <= 5; depth += 1) {
      const expected = createIdentityIndex(graph, colors, depth)
      const actual = indexes[depth - 1]!
      expect(actual.keyCount).toBe(expected.keyCount)
      expect(actual.uniqueKeys).toBe(expected.uniqueKeys)
      expect(actual.duplicateKeyCount).toBe(expected.duplicateKeyCount)
      expect(actual.collidingPointCount).toBe(expected.collidingPointCount)
      expect([...actual.byKey.entries()].sort()).toEqual([...expected.byKey.entries()].sort())
    }
  })

  it('returns no indexes for a non-positive depth', () => {
    expect(createIdentityIndexesByDepth(graph, colors, 0)).toEqual([])
  })
})
