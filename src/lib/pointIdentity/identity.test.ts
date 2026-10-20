import { describe, expect, it } from 'vitest'

import { chainEndpointFlagsForGraph } from '@/lib/pointIdentity/chainRuns'
import { canonicalRing, computeFingerprint, createIdentityIndex, evaluateIdentity } from '@/lib/pointIdentity/identity'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import type { GraphTuning, IdentityIndex, PatternPoint } from '@/lib/pointIdentity/types'

function points(coords: Array<readonly [number, number]>): PatternPoint[] {
  return coords.map(([x, y], index) => ({ x, y, index }))
}

const PATH_TUNING: GraphTuning = {
  pointsPerEdge: 5,
}

function pathPoints(): PatternPoint[] {
  return points([
    [0, 0],
    [10, 0],
    [20, 0],
    [30, 0],
    [40, 0],
  ])
}

describe('canonicalRing', () => {
  it('returns the smallest cyclic rotation', () => {
    expect(canonicalRing('1010')).toBe('0101')
    expect(canonicalRing('0110')).toBe('0011')
    expect(canonicalRing('0000')).toBe('0000')
    expect(canonicalRing('1')).toBe('1')
  })
})

describe('identity fingerprints', () => {
  it('assigns unique keys to an asymmetric color layout', () => {
    const graph = buildProximityGraph(pathPoints(), PATH_TUNING)
    const colors = [0, 0, 1, 0, 1] as const
    const index = createIdentityIndex(graph, colors, 2)
    expect(index.keyCount).toBe(5)
    expect(index.duplicateKeyCount).toBe(0)
    expect(index.collidingPointCount).toBe(0)
  })

  it('decodes an unchanged topology exactly', () => {
    const graph = buildProximityGraph(pathPoints(), PATH_TUNING)
    const colors = [0, 0, 1, 0, 1] as const
    const index = createIdentityIndex(graph, colors, 2)
    const evaluation = evaluateIdentity(graph, graph, index, colors, 2)
    expect(evaluation.summary.correct).toBe(5)
    expect(evaluation.summary.safeFail).toBe(0)
    expect(evaluation.summary.falseMatch).toBe(0)
    expect(evaluation.summary.edgeJaccard).toBe(1)
    expect(evaluation.summary.degreeChanged).toBe(0)
  })

  it('treats symmetric color layouts as ambiguous rather than dangerous', () => {
    const graph = buildProximityGraph(
      points([
        [0, 0],
        [10, 0],
        [20, 0],
      ]),
      { pointsPerEdge: 3 },
    )
    const colors = [0, 0, 0] as const
    const index = createIdentityIndex(graph, colors, 1)
    expect(index.duplicateKeyCount).toBe(1)
    expect(index.collidingPointCount).toBe(2)

    const evaluation = evaluateIdentity(graph, graph, index, colors, 1)
    expect(evaluation.summary.ambiguous).toBe(2)
    expect(evaluation.summary.falseMatch).toBe(0)
    expect(evaluation.summary.safeFail).toBe(2)
  })

  it('reports a wrong unique match as a false match', () => {
    const graph = buildProximityGraph(pathPoints(), PATH_TUNING)
    const colors = [0, 0, 1, 0, 1] as const
    const designIndex = createIdentityIndex(graph, colors, 2)
    const observedFingerprint = computeFingerprint(graph, colors, 0, 2)
    const poisoned: IdentityIndex = {
      ...designIndex,
      byKey: new Map([[observedFingerprint.key, [2]]]),
    }

    const evaluation = evaluateIdentity(graph, graph, poisoned, colors, 2)
    expect(evaluation.perPoint[0]!.outcome).toEqual({ kind: 'false-match', matchedIndex: 2 })
    expect(evaluation.summary.falseMatch).toBe(1)
    expect(evaluation.summary.safeFail).toBe(4)
  })

  it('marks chain endpoints and includes endpoint bits in the fingerprint', () => {
    const graph = buildProximityGraph(pathPoints(), PATH_TUNING)
    const colors = [0, 0, 1, 0, 1] as const
    expect([...chainEndpointFlagsForGraph(graph)]).toEqual([1, 0, 0, 0, 1])
    expect(computeFingerprint(graph, colors, 0, 1).endpoint).toBe(true)
    expect(computeFingerprint(graph, colors, 1, 1).endpoint).toBe(false)
  })

  it('includes ring sizes in the fingerprint key', () => {
    const graph = buildProximityGraph(pathPoints(), PATH_TUNING)
    const colors = [0, 0, 1, 0, 1] as const
    const fingerprint = computeFingerprint(graph, colors, 1, 2)
    expect(fingerprint.rings).toHaveLength(2)
    expect(fingerprint.rings[0]!.size).toBe(2)
    expect(fingerprint.rings[0]!.chains).toBe('11')
    expect(fingerprint.rings[1]!.size).toBe(1)
    expect(fingerprint.key).toContain('|2:')
    expect(fingerprint.key).toContain(':11')
  })
})
