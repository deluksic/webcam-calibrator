import { describe, expect, it } from 'vitest'

import { extractChainRuns, summarizeChainRuns } from '@/lib/pointIdentity/chainRuns'
import type { EdgeKind, GraphStats, PatternPoint, ProximityGraph } from '@/lib/pointIdentity/types'

function makeGraph(
  coords: ReadonlyArray<readonly [number, number]>,
  edges: ReadonlyArray<readonly [number, number]>,
  edgeKinds: ReadonlyArray<EdgeKind> = edges.map(() => 'chain'),
): ProximityGraph {
  const points: PatternPoint[] = coords.map(([x, y], index) => ({ x, y, index }))
  const neighbors: number[][] = Array.from({ length: points.length }, () => [])
  for (const [a, b] of edges) {
    neighbors[a]!.push(b)
    neighbors[b]!.push(a)
  }
  const rejections = { degree: 0, crossing: 0 }
  const stats: GraphStats = {
    inputPoints: points.length,
    candidates: edges.length,
    edges: edges.length,
    meanDegree: points.length === 0 ? 0 : (2 * edges.length) / points.length,
    isolated: neighbors.filter((list) => list.length === 0).length,
    components: 1,
    chainEdges: 0,
    junctionEdges: 0,
    rejections,
    localScaleMedian: 1,
  }
  return { points, neighbors, edges, edgeKinds, stats }
}

describe('extractChainRuns', () => {
  it('groups a straight path into one run', () => {
    const graph = makeGraph(
      [
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
      ],
      [
        [0, 1],
        [1, 2],
        [2, 3],
      ],
    )
    const runs = extractChainRuns(graph)
    expect(runs).toHaveLength(1)
    expect(runs[0]!.length).toBe(4)
    expect(runs[0]!.edgeIndices).toHaveLength(3)
  })

  it('splits at a turn', () => {
    const graph = makeGraph(
      [
        [0, 0],
        [10, 0],
        [20, 0],
        [20, 10],
        [20, 20],
      ],
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
      ],
    )
    const runs = extractChainRuns(graph)
    expect(runs.map((run) => run.length).sort((a, b) => a - b)).toEqual([3, 3])
    expect(runs.reduce((sum, run) => sum + run.edgeIndices.length, 0)).toBe(4)
  })

  it('ignores junction edges', () => {
    const graph = makeGraph(
      [
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
        [20, 10],
      ],
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [2, 4],
      ],
      ['chain', 'chain', 'chain', 'junction'],
    )
    const runs = extractChainRuns(graph)
    expect(runs).toHaveLength(1)
    expect(runs[0]!.length).toBe(4)
    expect(runs[0]!.edgeIndices).toEqual([0, 1, 2])
  })

  it('does not merge parallel disconnected runs', () => {
    const graph = makeGraph(
      [
        [0, 0],
        [10, 0],
        [0, 10],
        [10, 10],
      ],
      [
        [0, 1],
        [2, 3],
      ],
    )
    const runs = extractChainRuns(graph)
    expect(runs).toHaveLength(2)
    expect(runs.every((run) => run.length === 2)).toBe(true)
  })
})

describe('summarizeChainRuns', () => {
  it('counts short, exact and long runs against N', () => {
    const graph = makeGraph(
      [
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
        [40, 0],
        [50, 0],
      ],
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
      ],
    )
    const runs = extractChainRuns(graph)
    const summary = summarizeChainRuns(runs, 4, 3)
    expect(summary.observedChains).toBe(1)
    expect(summary.observedChainEdges).toBe(5)
    expect(summary.expectedChainEdges).toBe(9)
    expect(summary.short).toBe(0)
    expect(summary.exact).toBe(0)
    expect(summary.long).toBe(1)
    expect(summary.histogram).toEqual([{ length: 6, count: 1 }])
    expect(summary.coveredPoints).toBe(6)
  })

  it('uses the expected chain count for edge totals', () => {
    const graph = makeGraph(
      [
        [0, 0],
        [10, 0],
        [20, 0],
        [0, 20],
        [10, 20],
      ],
      [
        [0, 1],
        [1, 2],
        [3, 4],
      ],
    )
    const runs = extractChainRuns(graph)
    const summary = summarizeChainRuns(runs, 4, 2)
    expect(summary.observedChains).toBe(2)
    expect(summary.short).toBe(2)
    expect(summary.exact).toBe(0)
    expect(summary.long).toBe(0)
    expect(summary.observedChainEdges).toBe(3)
    expect(summary.expectedChainEdges).toBe(6)
    expect(summary.coveredPoints).toBe(5)
  })
})
