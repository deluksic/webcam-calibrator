import { describe, expect, it } from 'vitest'

import { extractChainRuns } from '@/lib/pointIdentity/chainRuns'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import type { PatternPoint } from '@/lib/pointIdentity/types'

function points(coords: Array<readonly [number, number]>): PatternPoint[] {
  return coords.map(([x, y], index) => ({ x, y, index }))
}

describe('buildProximityGraph', () => {
  it('builds the graph from exact N-point chains', () => {
    const graph = buildProximityGraph(
      points([
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
      ]),
      { pointsPerEdge: 4 },
    )
    expect(graph.edges).toEqual([
      [0, 1],
      [1, 2],
      [2, 3],
    ])
    expect(graph.stats.chainEdges).toBe(3)
    expect(graph.stats.junctionEdges).toBe(0)
    expect(graph.edgeKinds).toEqual(['chain', 'chain', 'chain'])
  })

  it('builds the expected honeycomb graph', () => {
    const pattern = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
    const graph = buildProximityGraph(pattern.points, { pointsPerEdge: 4 })
    expect(graph.stats.chainEdges).toBe(357)
    expect(graph.stats.junctionEdges).toBe(212)
    expect(graph.stats.edges).toBe(569)

    const runs = extractChainRuns(graph)
    expect(runs).toHaveLength(119)
    expect(runs.every((run) => run.length === 4)).toBe(true)
  })

  it('extracts the same chain count for 4, 5 and 6 points per edge', () => {
    const expected = new Map([
      [4, { chain: 357, junction: 212, edges: 569 }],
      [5, { chain: 476, junction: 212, edges: 688 }],
      [6, { chain: 595, junction: 212, edges: 807 }],
    ])
    for (const [pointsPerEdge, counts] of expected) {
      const pattern = generateHexPattern({ columns: 8, pointsPerEdge })
      const graph = buildProximityGraph(pattern.points, { pointsPerEdge })
      expect(graph.stats.chainEdges).toBe(counts.chain)
      expect(graph.stats.junctionEdges).toBe(counts.junction)
      expect(graph.stats.edges).toBe(counts.edges)
      const runs = extractChainRuns(graph)
      expect(runs).toHaveLength(119)
      expect(runs.every((run) => run.length === pointsPerEdge)).toBe(true)
    }
  })

  it('is deterministic', () => {
    const pattern = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
    const first = buildProximityGraph(pattern.points, { pointsPerEdge: 4 })
    const second = buildProximityGraph(pattern.points, { pointsPerEdge: 4 })
    expect(first.edges).toEqual(second.edges)
    expect(first.stats).toEqual(second.stats)
  })
})
