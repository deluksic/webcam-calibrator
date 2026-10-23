import { describe, expect, it } from 'vitest'

import { extractChainRuns } from '@/lib/pointIdentity/chainRuns'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { createRandom } from '@/lib/pointIdentity/random'
import type { PatternPoint } from '@/lib/pointIdentity/types'

function points(coords: Array<readonly [number, number]>): PatternPoint[] {
  return coords.map(([x, y], index) => ({ x, y, index }))
}

function distance(a: PatternPoint, b: PatternPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function edgeKey(first: number, second: number): string {
  return first < second ? `${first}:${second}` : `${second}:${first}`
}

const PATTERN = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
const TUNING = { pointsPerEdge: 4 }
const DESIGN = buildProximityGraph(PATTERN.points, TUNING)
const PROJECTED = projectPattern(PATTERN.points, { ...DEFAULT_VIEW, yawDeg: 0, pitchDeg: 0 })
const OBSERVED = applyObservationMask(PROJECTED, DEFAULT_OBSERVATION_MASK)
const DESIGN_EDGE_KEYS = new Set(
  DESIGN.edges.map(([a, b]) => edgeKey(DESIGN.points[a]!.index, DESIGN.points[b]!.index)),
)
const JITTER_SEED = (DEFAULT_OBSERVATION_MASK.dropoutSeed + 0x51ed270b) >>> 0

const projectedChainLengths = DESIGN.edges
  .map(([a, b], edgeIndex) => ({
    kind: DESIGN.edgeKinds[edgeIndex],
    length: distance(PROJECTED[DESIGN.points[a]!.index]!, PROJECTED[DESIGN.points[b]!.index]!),
  }))
  .filter((edge) => edge.kind === 'chain')
  .map((edge) => edge.length)
  .sort((first, second) => first - second)
const PROJECTED_CHAIN_SPACING = projectedChainLengths[projectedChainLengths.length >> 1]!

function jitteredGraph(percent: number): ReturnType<typeof buildProximityGraph> {
  const random = createRandom(JITTER_SEED)
  const spacing = DESIGN.stats.localScaleMedian
  const amplitude = spacing * (percent / 100)
  const observed = OBSERVED.map((point) => ({
    ...point,
    x: point.x + (random() * 2 - 1) * amplitude,
    y: point.y + (random() * 2 - 1) * amplitude,
  }))
  return buildProximityGraph(observed, TUNING)
}

function jitterSummary(graph: ReturnType<typeof buildProximityGraph>): {
  falseEdges: number
  longFalseEdges: number
  maxFalseLength: number
} {
  let falseEdges = 0
  let longFalseEdges = 0
  let maxFalseLength = 0
  for (const [a, b] of graph.edges) {
    const key = edgeKey(graph.points[a]!.index, graph.points[b]!.index)
    if (DESIGN_EDGE_KEYS.has(key)) {
      continue
    }
    falseEdges += 1
    const length = distance(graph.points[a]!, graph.points[b]!)
    maxFalseLength = Math.max(maxFalseLength, length)
    if (length > PROJECTED_CHAIN_SPACING * 2) {
      longFalseEdges += 1
    }
  }
  return { falseEdges, longFalseEdges, maxFalseLength }
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

  it('recovers a 3-point chain when one chain endpoint is dropped', () => {
    const pattern = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
    const full = buildProximityGraph(pattern.points, { pointsPerEdge: 4 })
    const firstRun = extractChainRuns(full)[0]!
    const chainDegree = new Map<number, number>()
    for (let edgeIndex = 0; edgeIndex < full.edges.length; edgeIndex += 1) {
      if (full.edgeKinds[edgeIndex] !== 'chain') {
        continue
      }
      const [a, b] = full.edges[edgeIndex]!
      chainDegree.set(a, (chainDegree.get(a) ?? 0) + 1)
      chainDegree.set(b, (chainDegree.get(b) ?? 0) + 1)
    }
    const endpoint = firstRun.pointIndices.find((pointIndex) => chainDegree.get(pointIndex) === 1)
    expect(endpoint).toBeDefined()

    const dropped = pattern.points.filter((point) => point.index !== endpoint)
    const graph = buildProximityGraph(dropped, { pointsPerEdge: 4 })
    const runs = extractChainRuns(graph)
    expect(runs.some((run) => run.length === 3)).toBe(true)
  })

  it('does not create long cross-chain edges at a left crop boundary', () => {
    const pattern = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
    const projected = projectPattern(pattern.points, { ...DEFAULT_VIEW, yawDeg: 0, pitchDeg: 0 })
    const observed = applyObservationMask(projected, {
      ...DEFAULT_OBSERVATION_MASK,
      cropLeft: 9,
    })
    const design = buildProximityGraph(pattern.points, { pointsPerEdge: 4 })
    const graph = buildProximityGraph(observed, { pointsPerEdge: 4 })
    const designEdges = new Set(
      design.edges.map(([a, b]) => {
        const first = design.points[a]!.index
        const second = design.points[b]!.index
        return first < second ? `${first}:${second}` : `${second}:${first}`
      }),
    )
    for (const [a, b] of graph.edges) {
      const first = graph.points[a]!.index
      const second = graph.points[b]!.index
      const key = first < second ? `${first}:${second}` : `${second}:${first}`
      expect(designEdges.has(key)).toBe(true)
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

describe('buildProximityGraph jitter regressions', () => {
  it('keeps the exact graph at 0% jitter', () => {
    const graph = jitteredGraph(0)
    expect(graph.stats.edges).toBe(569)
    expect(graph.stats.isolated).toBe(0)
    expect(jitterSummary(graph)).toEqual({ falseEdges: 0, longFalseEdges: 0, maxFalseLength: 0 })
  })

  it('keeps the exact graph at 1% jitter', () => {
    const graph = jitteredGraph(1)
    expect(graph.stats.edges).toBe(569)
    expect(graph.stats.isolated).toBe(0)
    expect(jitterSummary(graph).falseEdges).toBe(0)
  })

  it('recovers every junction at 2% jitter without false edges', () => {
    const graph = jitteredGraph(2)
    expect(graph.stats.edges).toBe(569)
    expect(graph.stats.chainEdges).toBe(357)
    expect(graph.stats.junctionEdges).toBe(212)
    expect(graph.stats.isolated).toBe(0)
    expect(jitterSummary(graph)).toEqual({ falseEdges: 0, longFalseEdges: 0, maxFalseLength: 0 })
  })

  it('degrades gracefully at 3% jitter without long false edges', () => {
    const graph = jitteredGraph(3)
    expect(graph.stats.edges).toBeGreaterThan(500)
    expect(graph.stats.isolated).toBeLessThan(20)
    expect(jitterSummary(graph).longFalseEdges).toBe(0)
  })

  it('remains fragmented but safe at 5% jitter', () => {
    const graph = jitteredGraph(5)
    expect(graph.stats.edges).toBeGreaterThan(300)
    expect(graph.stats.isolated).toBeGreaterThan(0)
    expect(jitterSummary(graph).longFalseEdges).toBe(0)
  })
})

describe('buildProximityGraph crop boundaries', () => {
  it('keeps crop left 2, 7 and 9 free of non-design edges', () => {
    for (const cropLeft of [2, 7, 9]) {
      const observed = applyObservationMask(PROJECTED, { ...DEFAULT_OBSERVATION_MASK, cropLeft })
      const graph = buildProximityGraph(observed, TUNING)
      for (const [a, b] of graph.edges) {
        expect(DESIGN_EDGE_KEYS.has(edgeKey(graph.points[a]!.index, graph.points[b]!.index))).toBe(true)
      }
    }
  })
})
