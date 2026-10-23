import { findChainCandidates, type ChainCandidate } from '@/lib/pointIdentity/chainCandidates'
import type {
  EdgeKind,
  GraphRejections,
  GraphStats,
  GraphTuning,
  PatternPoint,
  ProximityGraph,
  Vec2,
} from '@/lib/pointIdentity/types'

const NEIGHBOR_SAMPLE = 12
const LOCAL_SCALE_SAMPLE = 6
const MAX_FRAGMENT_GAP_FACTOR = 1.7

const NEAR_CHAIN_FACTOR = 2.5
/** Noisy chain lines rarely meet at an exact point; allow a quarter-spacing miss. */
const MAX_CONCURRENCE_RESIDUAL = 0.25
const MAX_CORNER_DISTANCE = 2
/** A corner edge longer than this bridges past its local cell at high noise. */
const MAX_JUNCTION_EDGE_FACTOR = 1.4
const MIN_LINE_SIN = 0.2
const LINE_TRIANGLE_MIN_QUALITY = 0.02
const MAX_DEGREE = 3

type ChainLine = {
  chainIndex: number
  point: Vec2
  direction: Vec2
  spacing: number
  first: number
  last: number
  maxCornerDistance: number
}

function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function unitVector(from: Vec2, to: Vec2): Vec2 | undefined {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const length = Math.hypot(dx, dy)
  if (length < 1e-9) {
    return undefined
  }
  return { x: dx / length, y: dy / length }
}

function dot(first: Vec2, second: Vec2): number {
  return first.x * second.x + first.y * second.y
}

function cross(first: Vec2, second: Vec2): number {
  return first.x * second.y - first.y * second.x
}

function lineIntersection(pointA: Vec2, directionA: Vec2, pointB: Vec2, directionB: Vec2): Vec2 | undefined {
  const denominator = cross(directionA, directionB)
  if (Math.abs(denominator) < 1e-9) {
    return undefined
  }
  const dx = pointB.x - pointA.x
  const dy = pointB.y - pointA.y
  const t = (dx * directionB.y - dy * directionB.x) / denominator
  return { x: pointA.x + directionA.x * t, y: pointA.y + directionA.y * t }
}

function distanceToLine(point: Vec2, linePoint: Vec2, direction: Vec2): number {
  return Math.abs(cross({ x: point.x - linePoint.x, y: point.y - linePoint.y }, direction))
}

function pairKey(first: number, second: number): string {
  return first < second ? `${first}:${second}` : `${second}:${first}`
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    return 0
  }
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  if (sorted.length % 2 === 1) {
    return sorted[middle]!
  }
  return (sorted[middle - 1]! + sorted[middle]!) / 2
}

function triangleCompactness(a: Vec2, b: Vec2, c: Vec2): number {
  const ab = distance(a, b)
  const ac = distance(a, c)
  const bc = distance(b, c)
  const longest = Math.max(ab, ac, bc)
  if (longest < 1e-9) {
    return 0
  }
  return Math.min(ab, ac, bc) / longest
}

function emptyGraph(points: readonly PatternPoint[]): ProximityGraph {
  const stats: GraphStats = {
    inputPoints: points.length,
    candidates: 0,
    edges: 0,
    meanDegree: 0,
    isolated: 0,
    components: 0,
    chainEdges: 0,
    junctionEdges: 0,
    rejections: { degree: 0, crossing: 0 },
    localScaleMedian: 0,
  }
  return {
    points,
    neighbors: points.map(() => []),
    edges: [],
    edgeKinds: [],
    stats,
  }
}

function orientation(a: Vec2, b: Vec2, c: Vec2): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

function boxesOverlap(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
  return (
    Math.max(a.x, b.x) >= Math.min(c.x, d.x) &&
    Math.max(c.x, d.x) >= Math.min(a.x, b.x) &&
    Math.max(a.y, b.y) >= Math.min(c.y, d.y) &&
    Math.max(c.y, d.y) >= Math.min(a.y, b.y)
  )
}

function properlyIntersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2): boolean {
  const first = orientation(c, d, a)
  const second = orientation(c, d, b)
  const third = orientation(a, b, c)
  const fourth = orientation(a, b, d)
  const epsilon = 1e-9
  return (
    ((first > epsilon && second < -epsilon) || (first < -epsilon && second > epsilon)) &&
    ((third > epsilon && fourth < -epsilon) || (third < -epsilon && fourth > epsilon))
  )
}

function crossesAccepted(
  points: readonly PatternPoint[],
  accepted: ReadonlyArray<readonly [number, number]>,
  a: number,
  b: number,
): boolean {
  const start = points[a]!
  const end = points[b]!
  for (const [u, v] of accepted) {
    if (u === a || u === b || v === a || v === b) {
      continue
    }
    const otherStart = points[u]!
    const otherEnd = points[v]!
    if (!boxesOverlap(start, end, otherStart, otherEnd)) {
      continue
    }
    if (properlyIntersect(start, end, otherStart, otherEnd)) {
      return true
    }
  }
  return false
}

function countComponents(adjacency: ReadonlyArray<readonly number[]>): number {
  const visited = new Uint8Array(adjacency.length)
  let components = 0
  for (let start = 0; start < adjacency.length; start += 1) {
    if (visited[start] === 1) {
      continue
    }
    components += 1
    const stack = [start]
    visited[start] = 1
    while (stack.length > 0) {
      const node = stack.pop()!
      for (const neighbor of adjacency[node]!) {
        if (visited[neighbor] === 0) {
          visited[neighbor] = 1
          stack.push(neighbor)
        }
      }
    }
  }
  return components
}

function computeNeighborhoods(points: readonly PatternPoint[]): {
  neighborLists: number[][]
  localScales: number[]
} {
  const count = points.length
  const neighborLists: number[][] = []
  const localScales: number[] = []
  const bestDistances = new Float64Array(NEIGHBOR_SAMPLE)
  const bestIndices = new Int32Array(NEIGHBOR_SAMPLE)

  for (let i = 0; i < count; i += 1) {
    const origin = points[i]!
    bestDistances.fill(Number.POSITIVE_INFINITY)
    bestIndices.fill(-1)
    for (let j = 0; j < count; j += 1) {
      if (j === i) {
        continue
      }
      const length = distance(origin, points[j]!)
      let position = NEIGHBOR_SAMPLE
      for (let slot = 0; slot < NEIGHBOR_SAMPLE; slot += 1) {
        const slotDistance = bestDistances[slot]!
        const tied = Math.abs(length - slotDistance) < 1e-6
        if (tied ? j < bestIndices[slot]! : length < slotDistance) {
          position = slot
          break
        }
      }
      if (position >= NEIGHBOR_SAMPLE) {
        continue
      }
      for (let slot = NEIGHBOR_SAMPLE - 1; slot > position; slot -= 1) {
        bestDistances[slot] = bestDistances[slot - 1]!
        bestIndices[slot] = bestIndices[slot - 1]!
      }
      bestDistances[position] = length
      bestIndices[position] = j
    }

    const neighborCount = Math.min(NEIGHBOR_SAMPLE, Math.max(0, count - 1))
    const neighbors: number[] = []
    for (let slot = 0; slot < neighborCount; slot += 1) {
      neighbors.push(bestIndices[slot]!)
    }
    neighborLists.push(neighbors)

    const scaleCount = Math.min(LOCAL_SCALE_SAMPLE, neighborCount)
    let localScale = 0
    if (scaleCount > 0) {
      const middle = scaleCount >> 1
      localScale =
        scaleCount % 2 === 1
          ? bestDistances[middle]!
          : (bestDistances[middle - 1]! + bestDistances[middle]!) / 2
    }
    localScales.push(Math.max(localScale, 1e-6))
  }
  return { neighborLists, localScales }
}

const MAX_COVERAGE_REPAIR_ROUNDS = 16
const COVERAGE_SCORE_EPSILON = 1e-9

function candidateKey(candidate: ChainCandidate): string {
  return [...candidate.pointIndices].sort((first, second) => first - second).join(':')
}

/**
 * Covers as many points as possible with non-overlapping chains. A globally
 * best-scoring window can block several correct chains, so selection first
 * respects coverage and only then prefers lower geometric scores. The greedy
 * seed is repaired by swapping one selected chain for every unselected chain
 * that becomes available.
 */
function selectChainsByCoverage(candidates: readonly ChainCandidate[], pointCount: number): ChainCandidate[] {
  if (candidates.length === 0) {
    return []
  }

  const order = candidates
    .map((candidate, index) => ({ candidate, index }))
    .sort((first, second) => {
      if (Math.abs(first.candidate.score - second.candidate.score) > COVERAGE_SCORE_EPSILON) {
        return first.candidate.score - second.candidate.score
      }
      return candidateKey(first.candidate).localeCompare(candidateKey(second.candidate))
    })
    .map((item) => item.index)

  const used = new Uint8Array(pointCount)
  const buildSelection = (excluded: number): { indices: number[]; coverage: number; score: number } => {
    used.fill(0)
    const indices: number[] = []
    let coverage = 0
    let score = 0
    for (const candidateIndex of order) {
      if (candidateIndex === excluded) {
        continue
      }
      const candidate = candidates[candidateIndex]!
      let available = true
      for (const pointIndex of candidate.pointIndices) {
        if (used[pointIndex] === 1) {
          available = false
          break
        }
      }
      if (!available) {
        continue
      }
      for (const pointIndex of candidate.pointIndices) {
        used[pointIndex] = 1
        coverage += 1
      }
      indices.push(candidateIndex)
      score += candidate.score
    }
    return { indices, coverage, score }
  }

  let selection = buildSelection(-1)
  let improved = true
  let round = 0
  while (improved && round < MAX_COVERAGE_REPAIR_ROUNDS) {
    improved = false
    round += 1
    for (const removed of selection.indices) {
      const trial = buildSelection(removed)
      const betterCoverage = trial.coverage > selection.coverage
      const betterTieBreak =
        trial.coverage === selection.coverage && trial.score < selection.score - COVERAGE_SCORE_EPSILON
      if (betterCoverage || betterTieBreak) {
        selection = trial
        improved = true
        break
      }
    }
  }

  return selection.indices
    .sort((first, second) => {
      if (Math.abs(candidates[first]!.score - candidates[second]!.score) > COVERAGE_SCORE_EPSILON) {
        return candidates[first]!.score - candidates[second]!.score
      }
      return candidateKey(candidates[first]!).localeCompare(candidateKey(candidates[second]!))
    })
    .map((index) => candidates[index]!)
}

/**
 * Build a deterministic proximity graph from detected point positions.
 *
 * Chains are extracted as N-point collinear runs with per-window PCA
 * validation. Selection covers as many points as possible with non-overlapping
 * chains before preferring lower geometric scores. Junction closure then
 * intersects chain lines at virtual cell corners: three-chain corners become
 * triangles, and two-chain boundary corners become single edges.
 */
export function buildProximityGraph(points: readonly PatternPoint[], tuning: GraphTuning): ProximityGraph {
  const count = points.length
  if (count === 0) {
    return emptyGraph(points)
  }

  const { neighborLists, localScales } = computeNeighborhoods(points)

  const chainCandidates = findChainCandidates(points, neighborLists, localScales, tuning.pointsPerEdge)
  const selected = selectChainsByCoverage(chainCandidates, count)

  const adjacency: number[][] = Array.from({ length: count }, () => [])
  const accepted: Array<readonly [number, number]> = []
  const edgeKinds: EdgeKind[] = []
  const acceptedKeys = new Set<string>()
  const rejections: GraphRejections = { degree: 0, crossing: 0 }
  let chainEdges = 0
  let junctionEdges = 0

  const addEdge = (a: number, b: number, kind: EdgeKind): void => {
    adjacency[a]!.push(b)
    adjacency[b]!.push(a)
    accepted.push(a < b ? [a, b] : [b, a])
    edgeKinds.push(kind)
    acceptedKeys.add(pairKey(a, b))
    if (kind === 'chain') {
      chainEdges += 1
    } else {
      junctionEdges += 1
    }
  }

  const admitChains = (chains: readonly ChainCandidate[]): void => {
    for (const chain of chains) {
      const pairs: Array<readonly [number, number]> = []
      let valid = true
      for (let index = 1; index < chain.pointIndices.length; index += 1) {
        const a = chain.pointIndices[index - 1]!
        const b = chain.pointIndices[index]!
        if (adjacency[a]!.length >= MAX_DEGREE || adjacency[b]!.length >= MAX_DEGREE) {
          rejections.degree += 1
          valid = false
          break
        }
        if (acceptedKeys.has(pairKey(a, b)) || crossesAccepted(points, accepted, a, b)) {
          rejections.crossing += 1
          valid = false
          break
        }
        pairs.push([a, b])
      }
      if (!valid) {
        continue
      }
      for (const [a, b] of pairs) {
        addEdge(a, b, 'chain')
      }
      admittedChains.push(chain)
    }
  }

  const admittedChains: ChainCandidate[] = []
  admitChains(selected)
  let candidateCount = chainCandidates.length

  // A dropped point can leave a 3-point side fragment with no complete
  // N-point window. Admit those fragments as chains, then let normal junction
  // closure reconnect them to the rest of the mesh.
  if (tuning.pointsPerEdge >= 4) {
    const remainingPositions: number[] = []
    for (let position = 0; position < count; position += 1) {
      if (adjacency[position]!.length === 0) {
        remainingPositions.push(position)
      }
    }
    if (remainingPositions.length >= 3) {
      const remainingPoints = remainingPositions.map((position) => points[position]!)
      const remainingNeighborhood = computeNeighborhoods(remainingPoints)
      const shortCandidates = findChainCandidates(
        remainingPoints,
        remainingNeighborhood.neighborLists,
        remainingNeighborhood.localScales,
        3,
      )
      candidateCount += shortCandidates.length
      const globalScale = median(localScales)
      const maxFragmentGap = globalScale * MAX_FRAGMENT_GAP_FACTOR
      const mappedCandidates: ChainCandidate[] = shortCandidates
        .filter((candidate) => {
          for (let index = 1; index < candidate.pointIndices.length; index += 1) {
            const first = remainingPoints[candidate.pointIndices[index - 1]!]!
            const second = remainingPoints[candidate.pointIndices[index]!]!
            if (distance(first, second) > maxFragmentGap) {
              return false
            }
          }
          return true
        })
        .map((candidate) => ({
          ...candidate,
          pointIndices: candidate.pointIndices.map((position) => remainingPositions[position]!),
        }))
      const shortUsed = new Set<number>()
      const shortSelected: ChainCandidate[] = []
      for (const candidate of mappedCandidates) {
        if (candidate.pointIndices.some((pointIndex) => shortUsed.has(pointIndex))) {
          continue
        }
        shortSelected.push(candidate)
        for (const pointIndex of candidate.pointIndices) {
          shortUsed.add(pointIndex)
        }
      }
      admitChains(shortSelected)
    }
  }

  const chainSpacings = admittedChains.map((chain) => {
    let total = 0
    let gaps = 0
    for (let index = 1; index < chain.pointIndices.length; index += 1) {
      total += distance(points[chain.pointIndices[index - 1]!]!, points[chain.pointIndices[index]!]!)
      gaps += 1
    }
    return gaps === 0 ? 0 : total / gaps
  })

  const chainLines: ChainLine[] = []
  for (let chainIndex = 0; chainIndex < admittedChains.length; chainIndex += 1) {
    const chain = admittedChains[chainIndex]!
    const first = chain.pointIndices[0]
    const last = chain.pointIndices[chain.pointIndices.length - 1]
    if (first === undefined || last === undefined || first === last) {
      continue
    }
    const firstPoint = points[first]!
    const lastPoint = points[last]!
    const direction = unitVector(firstPoint, lastPoint)
    const spacing = chainSpacings[chainIndex]!
    if (direction === undefined || spacing < 1e-9) {
      continue
    }
    chainLines.push({
      chainIndex,
      point: firstPoint,
      direction,
      spacing,
      first,
      last,
      maxCornerDistance: spacing * MAX_CORNER_DISTANCE,
    })
  }

  const nearbyChains = chainLines.map(() => new Set<number>())
  for (let firstIndex = 0; firstIndex < chainLines.length; firstIndex += 1) {
    const firstLine = chainLines[firstIndex]!
    const firstPoints = [points[firstLine.first]!, points[firstLine.last]!]
    for (let secondIndex = firstIndex + 1; secondIndex < chainLines.length; secondIndex += 1) {
      const secondLine = chainLines[secondIndex]!
      const secondPoints = [points[secondLine.first]!, points[secondLine.last]!]
      let nearest = Number.POSITIVE_INFINITY
      for (const firstPoint of firstPoints) {
        for (const secondPoint of secondPoints) {
          nearest = Math.min(nearest, distance(firstPoint, secondPoint))
        }
      }
      if (nearest <= Math.max(firstLine.spacing, secondLine.spacing) * NEAR_CHAIN_FACTOR) {
        nearbyChains[firstIndex]!.add(secondIndex)
        nearbyChains[secondIndex]!.add(firstIndex)
      }
    }
  }

  const chooseEndpoint = (corner: Vec2, line: ChainLine): { point: number; distance: number } | undefined => {
    const firstDistance = distance(corner, points[line.first]!)
    const lastDistance = distance(corner, points[line.last]!)
    const point = firstDistance <= lastDistance ? line.first : line.last
    const far = point === line.first ? line.last : line.first
    const cornerDistance = Math.min(firstDistance, lastDistance)
    if (cornerDistance > line.maxCornerDistance) {
      return undefined
    }
    const nearPoint = points[point]!
    const farPoint = points[far]!
    const outward = { x: nearPoint.x - farPoint.x, y: nearPoint.y - farPoint.y }
    const toCorner = { x: corner.x - nearPoint.x, y: corner.y - nearPoint.y }
    if (dot(toCorner, outward) < -1e-9) {
      return undefined
    }
    return { point, distance: cornerDistance }
  }

  const trianglesByChain = new Map<
    string,
    {
      a: number
      b: number
      c: number
      quality: number
      maxEdge: number
      residual: number
      cornerDistance: number
      chainIndices: [number, number, number]
    }
  >()

  for (let firstIndex = 0; firstIndex < chainLines.length; firstIndex += 1) {
    const firstLine = chainLines[firstIndex]!
    const firstNeighbors = nearbyChains[firstIndex]!
    for (const secondIndex of firstNeighbors) {
      if (secondIndex <= firstIndex) {
        continue
      }
      const secondLine = chainLines[secondIndex]!
      if (Math.abs(cross(firstLine.direction, secondLine.direction)) < MIN_LINE_SIN) {
        continue
      }
      const secondNeighbors = nearbyChains[secondIndex]!
      for (const thirdIndex of firstNeighbors) {
        if (thirdIndex <= secondIndex || !secondNeighbors.has(thirdIndex)) {
          continue
        }
        const thirdLine = chainLines[thirdIndex]!
        if (
          Math.abs(cross(firstLine.direction, thirdLine.direction)) < MIN_LINE_SIN ||
          Math.abs(cross(secondLine.direction, thirdLine.direction)) < MIN_LINE_SIN
        ) {
          continue
        }
        const corner = lineIntersection(firstLine.point, firstLine.direction, secondLine.point, secondLine.direction)
        if (corner === undefined) {
          continue
        }
        const averageSpacing = (firstLine.spacing + secondLine.spacing + thirdLine.spacing) / 3
        const residual = distanceToLine(corner, thirdLine.point, thirdLine.direction)
        if (residual > averageSpacing * MAX_CONCURRENCE_RESIDUAL) {
          continue
        }
        const firstEndpoint = chooseEndpoint(corner, firstLine)
        const secondEndpoint = chooseEndpoint(corner, secondLine)
        const thirdEndpoint = chooseEndpoint(corner, thirdLine)
        if (firstEndpoint === undefined || secondEndpoint === undefined || thirdEndpoint === undefined) {
          continue
        }
        const a = firstEndpoint.point
        const b = secondEndpoint.point
        const c = thirdEndpoint.point
        const quality = triangleCompactness(points[a]!, points[b]!, points[c]!)
        if (quality < LINE_TRIANGLE_MIN_QUALITY) {
          continue
        }
        const maxEdge = Math.max(
          distance(points[a]!, points[b]!),
          distance(points[a]!, points[c]!),
          distance(points[b]!, points[c]!),
        )
        const chainKey = [firstLine.chainIndex, secondLine.chainIndex, thirdLine.chainIndex]
          .sort((left, right) => left - right)
          .join(':')
        const cornerDistance = Math.max(firstEndpoint.distance, secondEndpoint.distance, thirdEndpoint.distance)
        const existing = trianglesByChain.get(chainKey)
        if (
          existing === undefined ||
          residual + cornerDistance * 0.01 < existing.residual + existing.cornerDistance * 0.01
        ) {
          trianglesByChain.set(chainKey, {
            a,
            b,
            c,
            quality,
            maxEdge,
            residual,
            cornerDistance,
            chainIndices: [firstLine.chainIndex, secondLine.chainIndex, thirdLine.chainIndex],
          })
        }
      }
    }
  }

  const triangles = [...trianglesByChain.values()].sort((first, second) => {
    const firstScore = first.residual + first.cornerDistance * 0.01
    const secondScore = second.residual + second.cornerDistance * 0.01
    if (firstScore !== secondScore) {
      return firstScore - secondScore
    }
    if (first.maxEdge !== second.maxEdge) {
      return first.maxEdge - second.maxEdge
    }
    return second.quality - first.quality
  })

  const threeChainPairKeys = new Set<string>()
  for (const triangle of triangles) {
    const [firstChain, secondChain, thirdChain] = triangle.chainIndices
    const pairs: Array<readonly [number, number]> = [
      [firstChain, secondChain],
      [firstChain, thirdChain],
      [secondChain, thirdChain],
    ]
    for (const [first, second] of pairs) {
      threeChainPairKeys.add(pairKey(first, second))
    }
  }

  for (const triangle of triangles) {
    const vertices = [triangle.a, triangle.b, triangle.c]
    if (vertices.some((vertex) => adjacency[vertex]!.length > MAX_DEGREE - 2)) {
      rejections.degree += 1
      continue
    }
    const pairs: Array<readonly [number, number]> = [
      [triangle.a, triangle.b],
      [triangle.a, triangle.c],
      [triangle.b, triangle.c],
    ]
    if (pairs.some(([a, b]) => acceptedKeys.has(pairKey(a, b)))) {
      continue
    }
    const [firstChainIndex, secondChainIndex, thirdChainIndex] = triangle.chainIndices
    const triangleSpacing =
      (chainSpacings[firstChainIndex]! + chainSpacings[secondChainIndex]! + chainSpacings[thirdChainIndex]!) / 3
    const maxJunctionLength = triangleSpacing * MAX_JUNCTION_EDGE_FACTOR
    if (pairs.some(([a, b]) => distance(points[a]!, points[b]!) > maxJunctionLength)) {
      continue
    }
    if (pairs.some(([a, b]) => crossesAccepted(points, accepted, a, b))) {
      rejections.crossing += 1
      continue
    }
    for (const [a, b] of pairs) {
      addEdge(a, b, 'junction')
    }
  }

  // Two-chain boundary corners run only after every three-chain triangle has
  // had a chance to claim its endpoints.
  for (let firstIndex = 0; firstIndex < chainLines.length; firstIndex += 1) {
    const firstLine = chainLines[firstIndex]!
    for (const secondIndex of nearbyChains[firstIndex]!) {
      if (secondIndex <= firstIndex) {
        continue
      }
      const secondLine = chainLines[secondIndex]!
      if (
        threeChainPairKeys.has(pairKey(firstLine.chainIndex, secondLine.chainIndex)) ||
        threeChainPairKeys.has(pairKey(secondLine.chainIndex, firstLine.chainIndex))
      ) {
        continue
      }
      if (Math.abs(cross(firstLine.direction, secondLine.direction)) < MIN_LINE_SIN) {
        continue
      }
      const corner = lineIntersection(firstLine.point, firstLine.direction, secondLine.point, secondLine.direction)
      if (corner === undefined) {
        continue
      }
      const averageSpacing = (firstLine.spacing + secondLine.spacing) / 2
      let hasThirdChain = false
      for (const otherLine of chainLines) {
        if (otherLine.chainIndex === firstLine.chainIndex || otherLine.chainIndex === secondLine.chainIndex) {
          continue
        }
        if (distanceToLine(corner, otherLine.point, otherLine.direction) > averageSpacing * MAX_CONCURRENCE_RESIDUAL) {
          continue
        }
        if (chooseEndpoint(corner, otherLine) !== undefined) {
          hasThirdChain = true
          break
        }
      }
      if (hasThirdChain) {
        continue
      }
      const firstEndpoint = chooseEndpoint(corner, firstLine)
      const secondEndpoint = chooseEndpoint(corner, secondLine)
      if (firstEndpoint === undefined || secondEndpoint === undefined) {
        continue
      }
      const a = firstEndpoint.point
      const b = secondEndpoint.point
      if (distance(points[a]!, points[b]!) > averageSpacing * MAX_JUNCTION_EDGE_FACTOR) {
        continue
      }
      if (adjacency[a]!.length >= MAX_DEGREE || adjacency[b]!.length >= MAX_DEGREE) {
        rejections.degree += 1
        continue
      }
      if (acceptedKeys.has(pairKey(a, b))) {
        continue
      }
      if (crossesAccepted(points, accepted, a, b)) {
        rejections.crossing += 1
        continue
      }
      addEdge(a, b, 'junction')
    }
  }

  const edges = accepted
  const stats: GraphStats = {
    inputPoints: count,
    candidates: candidateCount,
    edges: edges.length,
    meanDegree: count === 0 ? 0 : (2 * edges.length) / count,
    isolated: adjacency.filter((list) => list.length === 0).length,
    components: countComponents(adjacency),
    chainEdges,
    junctionEdges,
    rejections,
    localScaleMedian: median(localScales),
  }

  return {
    points,
    neighbors: adjacency,
    edges,
    edgeKinds,
    stats,
  }
}
