import { chainEndpointFlagsForGraph, chainIdsForGraph } from '@/lib/pointIdentity/chainRuns'
import type {
  BinaryColor,
  IdentityEvaluation,
  IdentityIndex,
  MatchOutcome,
  PointEvaluation,
  PointFingerprint,
  ProximityGraph,
  RingDescriptor,
} from '@/lib/pointIdentity/types'

function colorAt(colors: readonly BinaryColor[], index: number): BinaryColor {
  return colors[index] ?? 0
}

/**
 * Rotate a ring color string so that repeated/reversed traversal starts compare
 * equal. Only rotations are considered: a perspective view cannot mirror the
 * angular order of a neighborhood.
 */
function canonicalRingFeatures(
  colors: string,
  chainIds: readonly number[],
  endpoints: string,
  rootChain: number,
): { colors: string; partition: string; endpoints: string } {
  if (colors.length <= 1) {
    const label = rootChain >= 0 && chainIds[0] === rootChain ? 0 : 0
    return { colors, partition: String(label), endpoints }
  }

  let bestColors = ''
  let bestPartition = ''
  let bestEndpoints = ''
  let hasBest = false

  for (let shift = 0; shift < colors.length; shift += 1) {
    const candidateColors = colors.slice(shift) + colors.slice(0, shift)
    const candidateEndpoints = endpoints.slice(shift) + endpoints.slice(0, shift)
    const rotatedChains = [...chainIds.slice(shift), ...chainIds.slice(0, shift)]
    const labels = new Map<number, number>()
    let nextLabel = 1
    if (rootChain >= 0) {
      labels.set(rootChain, 0)
    } else {
      nextLabel = 0
    }
    const partitionLabels: number[] = []
    for (const chainId of rotatedChains) {
      let label = labels.get(chainId)
      if (label === undefined) {
        label = nextLabel
        nextLabel += 1
        labels.set(chainId, label)
      }
      partitionLabels.push(label)
    }
    const candidatePartition = partitionLabels.map((label) => label.toString(36)).join('')

    if (
      !hasBest ||
      candidateColors < bestColors ||
      (candidateColors === bestColors &&
        (candidatePartition < bestPartition ||
          (candidatePartition === bestPartition && candidateEndpoints < bestEndpoints)))
    ) {
      bestColors = candidateColors
      bestPartition = candidatePartition
      bestEndpoints = candidateEndpoints
      hasBest = true
    }
  }

  return { colors: bestColors, partition: bestPartition, endpoints: bestEndpoints }
}

export function canonicalRing(colors: string): string {
  if (colors.length <= 1) {
    return colors
  }
  let best = colors
  for (let shift = 1; shift < colors.length; shift += 1) {
    const candidate = colors.slice(shift) + colors.slice(0, shift)
    if (candidate < best) {
      best = candidate
    }
  }
  return best
}

/**
 * A fingerprint is the root color plus one canonical color ring per graph hop.
 * Ring sizes are part of the key, so a topology change cannot silently
 * masquerade as a color match.
 */
export function computeFingerprint(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  rootIndex: number,
  depth: number,
  chainIds: Int32Array = chainIdsForGraph(graph),
  endpointFlags: Uint8Array = chainEndpointFlagsForGraph(graph),
): PointFingerprint {
  const root = graph.points[rootIndex]
  if (root === undefined) {
    return { index: rootIndex, color: 0, endpoint: false, rings: [], key: '0' }
  }
  const rootChain = chainIds[rootIndex] ?? -1
  const rootEndpoint = (endpointFlags[rootIndex] ?? 0) === 1

  const seen = new Set<number>([rootIndex])
  let frontier: number[] = [rootIndex]
  const rings: RingDescriptor[] = []

  for (let level = 0; level < depth; level += 1) {
    const next = new Set<number>()
    for (const node of frontier) {
      for (const neighbor of graph.neighbors[node]!) {
        if (!seen.has(neighbor)) {
          next.add(neighbor)
        }
      }
    }
    if (next.size === 0) {
      break
    }

    const ordered = [...next].sort((a, b) => {
      const pointA = graph.points[a]!
      const pointB = graph.points[b]!
      const angleA = Math.atan2(pointA.y - root.y, pointA.x - root.x)
      const angleB = Math.atan2(pointB.y - root.y, pointB.x - root.x)
      if (angleA !== angleB) {
        return angleA - angleB
      }
      return a - b
    })

    const raw = ordered.map((node) => String(colorAt(colors, graph.points[node]!.index))).join('')
    const rawChains = ordered.map((node) => chainIds[node] ?? -1)
    const rawEndpoints = ordered.map((node) => String(endpointFlags[node] ?? 0)).join('')
    const canonical = canonicalRingFeatures(raw, rawChains, rawEndpoints, rootChain)
    rings.push({
      size: ordered.length,
      colors: canonical.colors,
      partition: canonical.partition,
      endpoints: canonical.endpoints,
    })
    for (const node of ordered) {
      seen.add(node)
    }
    frontier = ordered
  }

  const rootColor = colorAt(colors, root.index)
  let key = `${rootColor}:${rootEndpoint ? 1 : 0}`
  for (const ring of rings) {
    key += `|${ring.size}:${ring.colors}:${ring.partition}:${ring.endpoints}`
  }

  return { index: root.index, color: rootColor, endpoint: rootEndpoint, rings, key }
}

export function createIdentityIndex(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  depth: number,
): IdentityIndex {
  const fingerprints: PointFingerprint[] = []
  const byKey = new Map<string, number[]>()
  const chainIds = chainIdsForGraph(graph)
  const endpointFlags = chainEndpointFlagsForGraph(graph)

  for (let rootIndex = 0; rootIndex < graph.points.length; rootIndex += 1) {
    const fingerprint = computeFingerprint(graph, colors, rootIndex, depth, chainIds, endpointFlags)
    fingerprints.push(fingerprint)
    const existing = byKey.get(fingerprint.key)
    if (existing === undefined) {
      byKey.set(fingerprint.key, [fingerprint.index])
    } else {
      existing.push(fingerprint.index)
    }
  }

  let uniqueKeys = 0
  let duplicateKeyCount = 0
  let collidingPointCount = 0
  for (const indices of byKey.values()) {
    if (indices.length === 1) {
      uniqueKeys += 1
    } else {
      duplicateKeyCount += 1
      collidingPointCount += indices.length
    }
  }

  return {
    byKey,
    fingerprints,
    keyCount: byKey.size,
    uniqueKeys,
    duplicateKeyCount,
    collidingPointCount,
  }
}

function classify(fingerprint: PointFingerprint, designIndex: IdentityIndex): MatchOutcome {
  const candidates = designIndex.byKey.get(fingerprint.key)
  if (candidates === undefined) {
    return { kind: 'unresolved' }
  }
  if (candidates.length > 1) {
    return { kind: 'ambiguous', candidateIndices: [...candidates] }
  }
  const matchedIndex = candidates[0]!
  if (matchedIndex === fingerprint.index) {
    return { kind: 'correct', matchedIndex }
  }
  return { kind: 'false-match', matchedIndex }
}

function edgeSet(graph: ProximityGraph): Set<string> {
  const keys = new Set<string>()
  for (const [a, b] of graph.edges) {
    const first = graph.points[a]!.index
    const second = graph.points[b]!.index
    keys.add(first < second ? `${first}:${second}` : `${second}:${first}`)
  }
  return keys
}

function edgeJaccard(design: ProximityGraph, observed: ProximityGraph): number {
  const designEdges = edgeSet(design)
  const observedEdges = edgeSet(observed)
  let intersection = 0
  for (const key of designEdges) {
    if (observedEdges.has(key)) {
      intersection += 1
    }
  }
  const union = designEdges.size + observedEdges.size - intersection
  if (union === 0) {
    return 1
  }
  return intersection / union
}

function degreeChanges(design: ProximityGraph, observed: ProximityGraph): number {
  const designDegrees = new Map<number, number>()
  for (let position = 0; position < design.points.length; position += 1) {
    designDegrees.set(design.points[position]!.index, design.neighbors[position]!.length)
  }
  const observedDegrees = new Map<number, number>()
  for (let position = 0; position < observed.points.length; position += 1) {
    observedDegrees.set(observed.points[position]!.index, observed.neighbors[position]!.length)
  }
  let changed = 0
  for (const [id, degree] of designDegrees) {
    if (observedDegrees.get(id) !== degree) {
      changed += 1
    }
  }
  return changed
}

export function evaluateIdentity(
  designGraph: ProximityGraph,
  observedGraph: ProximityGraph,
  designIndex: IdentityIndex,
  colors: readonly BinaryColor[],
  depth: number,
): IdentityEvaluation {
  const observedChainIds = chainIdsForGraph(observedGraph)
  const observedEndpoints = chainEndpointFlagsForGraph(observedGraph)
  const observedFingerprints = observedGraph.points.map((_, rootIndex) =>
    computeFingerprint(observedGraph, colors, rootIndex, depth, observedChainIds, observedEndpoints),
  )

  const perPoint: PointEvaluation[] = observedFingerprints.map((fingerprint) => ({
    index: fingerprint.index,
    fingerprint,
    outcome: classify(fingerprint, designIndex),
  }))

  let correct = 0
  let falseMatch = 0
  let unresolved = 0
  let ambiguous = 0
  for (const point of perPoint) {
    switch (point.outcome.kind) {
      case 'correct':
        correct += 1
        break
      case 'false-match':
        falseMatch += 1
        break
      case 'ambiguous':
        ambiguous += 1
        break
      case 'unresolved':
        unresolved += 1
        break
    }
  }

  const total = perPoint.length
  const designUniquePoints = designIndex.fingerprints.length - designIndex.collidingPointCount

  return {
    summary: {
      total,
      correct,
      falseMatch,
      safeFail: unresolved + ambiguous,
      unresolved,
      ambiguous,
      accuracy: total === 0 ? 0 : correct / total,
      falseMatchRate: total === 0 ? 0 : falseMatch / total,
      designUniquePoints,
      designCollisions: designIndex.collidingPointCount,
      edgeJaccard: edgeJaccard(designGraph, observedGraph),
      degreeChanged: degreeChanges(designGraph, observedGraph),
    },
    perPoint,
    observedFingerprints,
  }
}
