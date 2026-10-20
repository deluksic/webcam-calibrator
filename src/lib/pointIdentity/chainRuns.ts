import type { ProximityGraph } from '@/lib/pointIdentity/types'

/** Two graph edges count as a chain continuation when their turn is under ~18°. */
const COLLINEAR_DOT = -0.95

export type ChainRun = {
  /** Edge indices belonging to this straight run. */
  edgeIndices: number[]
  /** Unique point indices in the run. */
  pointIndices: number[]
  /** Number of points in the run. */
  length: number
}

export type ChainHistogramBin = {
  length: number
  count: number
}

export type ChainDiagnostics = {
  runs: ChainRun[]
  histogram: ChainHistogramBin[]
  expectedPointsPerEdge: number
  expectedChains: number
  expectedChainEdges: number
  observedChains: number
  observedChainEdges: number
  short: number
  exact: number
  long: number
  meanLength: number
  coveredPoints: number
}

function findRoot(parent: number[], index: number): number {
  let current = index
  while (parent[current] !== current) {
    parent[current] = parent[parent[current]!]!
    current = parent[current]!
  }
  return current
}

function union(parent: number[], first: number, second: number): void {
  const rootFirst = findRoot(parent, first)
  const rootSecond = findRoot(parent, second)
  if (rootFirst !== rootSecond) {
    parent[rootSecond] = rootFirst
  }
}

function otherEndpoint(edge: readonly [number, number], point: number): number {
  return edge[0] === point ? edge[1] : edge[0]
}

/**
 * Groups admitted edges into maximal collinear runs. Two edges are linked when
 * they meet at a point with a near-180° turn; junction turns and triangle
 * corners naturally end a run. Edge indices are unioned rather than points, so
 * a point shared by two runs (a corner) does not merge them.
 */
export function extractChainRuns(graph: ProximityGraph): ChainRun[] {
  const edges = graph.edges
  if (edges.length === 0) {
    return []
  }

  const chainEdgeIndices: number[] = []
  for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex += 1) {
    if (graph.edgeKinds[edgeIndex] === 'chain') {
      chainEdgeIndices.push(edgeIndex)
    }
  }
  if (chainEdgeIndices.length === 0) {
    return []
  }

  const incident: number[][] = Array.from({ length: graph.points.length }, () => [])
  for (const edgeIndex of chainEdgeIndices) {
    const edge = edges[edgeIndex]!
    incident[edge[0]]!.push(edgeIndex)
    incident[edge[1]]!.push(edgeIndex)
  }

  const parent = Array.from({ length: edges.length }, (_, index) => index)

  for (let pointIndex = 0; pointIndex < graph.points.length; pointIndex += 1) {
    const edgeIndices = incident[pointIndex]!
    if (edgeIndices.length < 2) {
      continue
    }
    const origin = graph.points[pointIndex]!
    let bestDot = COLLINEAR_DOT
    let bestFirst = -1
    let bestSecond = -1

    for (let firstIndex = 0; firstIndex < edgeIndices.length; firstIndex += 1) {
      const firstEdgeIndex = edgeIndices[firstIndex]!
      const firstPoint = graph.points[otherEndpoint(edges[firstEdgeIndex]!, pointIndex)]!
      const firstDx = firstPoint.x - origin.x
      const firstDy = firstPoint.y - origin.y
      const firstLength = Math.hypot(firstDx, firstDy)
      if (firstLength < 1e-9) {
        continue
      }
      const firstUnitX = firstDx / firstLength
      const firstUnitY = firstDy / firstLength

      for (let secondIndex = firstIndex + 1; secondIndex < edgeIndices.length; secondIndex += 1) {
        const secondEdgeIndex = edgeIndices[secondIndex]!
        const secondPoint = graph.points[otherEndpoint(edges[secondEdgeIndex]!, pointIndex)]!
        const secondDx = secondPoint.x - origin.x
        const secondDy = secondPoint.y - origin.y
        const secondLength = Math.hypot(secondDx, secondDy)
        if (secondLength < 1e-9) {
          continue
        }
        const dot = firstUnitX * (secondDx / secondLength) + firstUnitY * (secondDy / secondLength)
        if (dot < bestDot) {
          bestDot = dot
          bestFirst = firstEdgeIndex
          bestSecond = secondEdgeIndex
        }
      }
    }

    if (bestFirst >= 0 && bestSecond >= 0) {
      union(parent, bestFirst, bestSecond)
    }
  }

  const groups = new Map<number, number[]>()
  for (const edgeIndex of chainEdgeIndices) {
    const root = findRoot(parent, edgeIndex)
    const group = groups.get(root)
    if (group === undefined) {
      groups.set(root, [edgeIndex])
    } else {
      group.push(edgeIndex)
    }
  }

  const runs: ChainRun[] = []
  for (const edgeIndices of groups.values()) {
    const pointIndices = new Set<number>()
    for (const edgeIndex of edgeIndices) {
      const edge = edges[edgeIndex]!
      pointIndices.add(edge[0])
      pointIndices.add(edge[1])
    }
    const sortedPoints = [...pointIndices].sort((a, b) => a - b)
    runs.push({
      edgeIndices,
      pointIndices: sortedPoints,
      length: sortedPoints.length,
    })
  }

  runs.sort((first, second) => {
    if (first.length !== second.length) {
      return second.length - first.length
    }
    return (first.pointIndices[0] ?? 0) - (second.pointIndices[0] ?? 0)
  })

  return runs
}

export function chainIdsForGraph(graph: ProximityGraph): Int32Array {
  const ids = new Int32Array(graph.points.length)
  ids.fill(-1)
  const runs = extractChainRuns(graph)
  for (let chainIndex = 0; chainIndex < runs.length; chainIndex += 1) {
    for (const pointIndex of runs[chainIndex]!.pointIndices) {
      ids[pointIndex] = chainIndex
    }
  }
  return ids
}

export function chainEndpointFlagsForGraph(graph: ProximityGraph): Uint8Array {
  const degree = new Uint8Array(graph.points.length)
  for (let edgeIndex = 0; edgeIndex < graph.edges.length; edgeIndex += 1) {
    if (graph.edgeKinds[edgeIndex] !== 'chain') {
      continue
    }
    const edge = graph.edges[edgeIndex]!
    degree[edge[0]]! += 1
    degree[edge[1]]! += 1
  }
  const flags = new Uint8Array(graph.points.length)
  for (let index = 0; index < flags.length; index += 1) {
    flags[index] = degree[index] === 1 ? 1 : 0
  }
  return flags
}

export function summarizeChainRuns(
  runs: readonly ChainRun[],
  expectedPointsPerEdge: number,
  expectedChains: number,
): ChainDiagnostics {
  const histogramMap = new Map<number, number>()
  let observedChainEdges = 0
  let short = 0
  let exact = 0
  let long = 0
  let lengthSum = 0
  const coveredPoints = new Set<number>()

  for (const run of runs) {
    for (const pointIndex of run.pointIndices) {
      coveredPoints.add(pointIndex)
    }
    observedChainEdges += run.edgeIndices.length
    lengthSum += run.length
    histogramMap.set(run.length, (histogramMap.get(run.length) ?? 0) + 1)
    if (run.length < expectedPointsPerEdge) {
      short += 1
    } else if (run.length > expectedPointsPerEdge) {
      long += 1
    } else {
      exact += 1
    }
  }

  const histogram = [...histogramMap.entries()]
    .sort((first, second) => first[0] - second[0])
    .map(([length, count]) => ({ length, count }))

  return {
    runs: [...runs],
    histogram,
    expectedPointsPerEdge,
    expectedChains,
    expectedChainEdges: expectedChains * Math.max(0, expectedPointsPerEdge - 1),
    observedChains: runs.length,
    observedChainEdges,
    short,
    exact,
    long,
    meanLength: runs.length === 0 ? 0 : lengthSum / runs.length,
    coveredPoints: coveredPoints.size,
  }
}
