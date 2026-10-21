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

const NEAR_CHAIN_FACTOR = 2.5
const MAX_CONCURRENCE_RESIDUAL = 0.15
const MAX_CORNER_DISTANCE = 2
const MIN_LINE_SIN = 0.2
const LINE_TRIANGLE_MIN_QUALITY = 0.02
const MAX_DEGREE = 3
const REPAIR_REACH_FACTOR = 2.2
const REPAIR_MAX_PER_POINT = 2

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
    repairEdges: 0,
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
  const neighborLists: number[][] = []
  const localScales: number[] = []
  for (let i = 0; i < points.length; i += 1) {
    const origin = points[i]!
    const sorted: Array<{ index: number; length: number }> = []
    for (let j = 0; j < points.length; j += 1) {
      if (j !== i) {
        sorted.push({ index: j, length: distance(origin, points[j]!) })
      }
    }
    sorted.sort((first, second) =>
      Math.abs(first.length - second.length) < 1e-6 ? first.index - second.index : first.length - second.length,
    )
    neighborLists.push(sorted.slice(0, Math.min(NEIGHBOR_SAMPLE, sorted.length)).map((item) => item.index))
    localScales.push(
      Math.max(median(sorted.slice(0, Math.min(LOCAL_SCALE_SAMPLE, sorted.length)).map((item) => item.length)), 1e-6),
    )
  }
  return { neighborLists, localScales }
}

/**
 * Build a deterministic proximity graph from detected point positions.
 *
 * Chains are extracted as exact N-point collinear runs. Junction closure then
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
  const selected: ChainCandidate[] = []
  const usedPoints = new Set<number>()
  for (const candidate of chainCandidates) {
    if (candidate.pointIndices.some((pointIndex) => usedPoints.has(pointIndex))) {
      continue
    }
    selected.push(candidate)
    for (const pointIndex of candidate.pointIndices) {
      usedPoints.add(pointIndex)
    }
  }

  const adjacency: number[][] = Array.from({ length: count }, () => [])
  const accepted: Array<readonly [number, number]> = []
  const edgeKinds: EdgeKind[] = []
  const acceptedKeys = new Set<string>()
  const rejections: GraphRejections = { degree: 0, crossing: 0 }
  let chainEdges = 0
  let junctionEdges = 0
  let repairEdges = 0

  const addEdge = (a: number, b: number, kind: EdgeKind): void => {
    adjacency[a]!.push(b)
    adjacency[b]!.push(a)
    accepted.push(a < b ? [a, b] : [b, a])
    edgeKinds.push(kind)
    acceptedKeys.add(pairKey(a, b))
    if (kind === 'chain') {
      chainEdges += 1
    } else if (kind === 'junction') {
      junctionEdges += 1
    } else {
      repairEdges += 1
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
      const mappedCandidates: ChainCandidate[] = shortCandidates.map((candidate) => ({
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

  // A dropout or crop can leave completely disconnected fragments. Repair
  // only those isolated points, and cap each repaired fragment at one design
  // side so repair edges never extend an existing chain.
  const repairEligible = adjacency.map((list) => list.length === 0)
  const repairParent = Array.from({ length: count }, (_, index) => index)
  const repairSize = Array.from({ length: count }, () => 1)
  const findRepairRoot = (node: number): number => {
    let root = node
    while (repairParent[root] !== root) {
      root = repairParent[root]!
    }
    let current = node
    while (repairParent[current] !== current) {
      const next = repairParent[current]!
      repairParent[current] = root
      current = next
    }
    return root
  }
  const repairCandidates: Array<{ a: number; b: number; length: number }> = []
  for (let a = 0; a < count; a += 1) {
    if (!repairEligible[a]) {
      continue
    }
    for (let b = a + 1; b < count; b += 1) {
      if (!repairEligible[b]) {
        continue
      }
      const length = distance(points[a]!, points[b]!)
      if (length <= Math.min(localScales[a]!, localScales[b]!) * REPAIR_REACH_FACTOR) {
        repairCandidates.push({ a, b, length })
      }
    }
  }
  repairCandidates.sort((first, second) =>
    first.length !== second.length ? first.length - second.length : first.a - second.a || first.b - second.b,
  )
  const repairCount = new Uint8Array(count)
  for (const candidate of repairCandidates) {
    const { a, b } = candidate
    if (repairCount[a]! >= REPAIR_MAX_PER_POINT || repairCount[b]! >= REPAIR_MAX_PER_POINT) {
      continue
    }
    if (adjacency[a]!.length >= MAX_DEGREE || adjacency[b]!.length >= MAX_DEGREE) {
      continue
    }
    if (acceptedKeys.has(pairKey(a, b))) {
      continue
    }
    const rootA = findRepairRoot(a)
    const rootB = findRepairRoot(b)
    if (rootA === rootB || repairSize[rootA]! + repairSize[rootB]! > tuning.pointsPerEdge) {
      continue
    }
    if (crossesAccepted(points, accepted, a, b)) {
      rejections.crossing += 1
      continue
    }
    addEdge(a, b, 'repair')
    repairCount[a]! += 1
    repairCount[b]! += 1
    repairParent[rootB] = rootA
    repairSize[rootA]! += repairSize[rootB]!
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
    repairEdges,
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
