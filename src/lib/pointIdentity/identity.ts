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
const BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz'

function canonicalRingFeatures(
  colors: string,
  chainIds: Int32Array,
  chainCount: number,
  endpoints: string,
  rootChain: number,
): { colors: string; partition: string; endpoints: string } {
  if (chainCount <= 1) {
    return { colors, partition: '0', endpoints }
  }

  let colorsUniform = true
  for (let index = 1; index < chainCount; index += 1) {
    if (colors[index] !== colors[0]) {
      colorsUniform = false
      break
    }
  }
  let endpointsUniform = true
  for (let index = 1; index < chainCount; index += 1) {
    if (endpoints[index] !== endpoints[0]) {
      endpointsUniform = false
      break
    }
  }

  let bestColors = ''
  let bestPartition = ''
  let bestEndpoints = ''
  let hasBest = false
  // Reused across shifts. No Map and no per-shift arrays.
  const distinctChains: number[] = []
  const distinctLabels: number[] = []

  for (let shift = 0; shift < chainCount; shift += 1) {
    const candidateColors = colorsUniform
      ? colors
      : colors.slice(shift) + colors.slice(0, shift)
    const candidateEndpoints = endpointsUniform
      ? endpoints
      : endpoints.slice(shift) + endpoints.slice(0, shift)

    distinctChains.length = 0
    distinctLabels.length = 0
    let nextLabel = rootChain >= 0 ? 1 : 0
    let candidatePartition = ''

    for (let index = 0; index < chainCount; index += 1) {
      const chainId = chainIds[(shift + index) % chainCount] ?? -1
      let label = -1
      if (rootChain >= 0 && chainId === rootChain) {
        label = 0
      } else {
        for (let known = 0; known < distinctChains.length; known += 1) {
          if (distinctChains[known] === chainId) {
            label = distinctLabels[known]!
            break
          }
        }
        if (label < 0) {
          label = nextLabel
          nextLabel += 1
          distinctChains.push(chainId)
          distinctLabels.push(label)
        }
      }
      candidatePartition += BASE36[label]
    }

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
  let rawChains = new Int32Array(0)

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

    if (rawChains.length < ordered.length) {
      rawChains = new Int32Array(ordered.length)
    }
    let raw = ''
    let rawEndpoints = ''
    for (let index = 0; index < ordered.length; index += 1) {
      const node = ordered[index]!
      raw += colorAt(colors, graph.points[node]!.index) === 1 ? '1' : '0'
      rawChains[index] = chainIds[node] ?? -1
      rawEndpoints += (endpointFlags[node] ?? 0) === 1 ? '1' : '0'
    }
    const canonical = canonicalRingFeatures(
      raw,
      rawChains,
      ordered.length,
      rawEndpoints,
      rootChain,
    )
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
  let key = fingerprintRootKey(rootColor, rootEndpoint)
  for (const ring of rings) {
    key += `|${fingerprintRingKey(ring)}`
  }

  return { index: root.index, color: rootColor, endpoint: rootEndpoint, rings, key }
}

export function fingerprintRootKey(color: BinaryColor, endpoint: boolean): string {
  return `${color}:${endpoint ? 1 : 0}`
}

export function fingerprintRingKey(ring: RingDescriptor): string {
  return `${ring.size}:${ring.colors}:${ring.partition}:${ring.endpoints}`
}

/**
 * Rebuild the lookup key for a fingerprint prefix. A full-depth fingerprint is
 * built once per point; its rings are already the exact BFS rings for every
 * shallower depth, so prefix keys avoid recomputing the graph traversal.
 *
 * Prefer the incremental builders below when walking every prefix depth: this
 * helper intentionally rebuilds from the root.
 */
export function fingerprintKeyAtDepth(fingerprint: PointFingerprint, depth: number): string {
  let key = fingerprintRootKey(fingerprint.color, fingerprint.endpoint)
  const limit = Math.max(0, Math.min(depth, fingerprint.rings.length))
  for (let index = 0; index < limit; index += 1) {
    key += `|${fingerprintRingKey(fingerprint.rings[index]!)}`
  }
  return key
}

export function computeFingerprints(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  depth: number,
): PointFingerprint[] {
  const chainIds = chainIdsForGraph(graph)
  const endpointFlags = chainEndpointFlagsForGraph(graph)
  const fingerprints: PointFingerprint[] = []
  for (let rootIndex = 0; rootIndex < graph.points.length; rootIndex += 1) {
    fingerprints.push(
      computeFingerprint(graph, colors, rootIndex, depth, chainIds, endpointFlags),
    )
  }
  return fingerprints
}

function identityIndexFromMap(
  byKey: Map<string, number[]>,
  fingerprints: PointFingerprint[],
): IdentityIndex {
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

export function createIdentityIndex(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  depth: number,
): IdentityIndex {
  const fingerprints = computeFingerprints(graph, colors, depth)
  const byKey = new Map<string, number[]>()
  for (const fingerprint of fingerprints) {
    const existing = byKey.get(fingerprint.key)
    if (existing === undefined) {
      byKey.set(fingerprint.key, [fingerprint.index])
    } else {
      existing.push(fingerprint.index)
    }
  }
  return identityIndexFromMap(byKey, fingerprints)
}

/**
 * Build one design index per depth from a single full-depth fingerprint pass.
 * `indexes[depth - 1]` is equivalent to `createIdentityIndex(graph, colors, depth)`.
 */
export function createIdentityIndexesByDepth(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  maxDepth: number,
): IdentityIndex[] {
  const depthCount = Math.max(0, Math.floor(maxDepth))
  if (depthCount === 0) {
    return []
  }
  const fingerprints = computeFingerprints(graph, colors, depthCount)
  const maps: Array<Map<string, number[]>> = []
  for (let depth = 0; depth < depthCount; depth += 1) {
    maps.push(new Map())
  }
  for (const fingerprint of fingerprints) {
    let key = fingerprintRootKey(fingerprint.color, fingerprint.endpoint)
    for (let depth = 0; depth < depthCount; depth += 1) {
      // Depth 1 is the root color plus the first ring, matching
      // fingerprintKeyAtDepth. Append first, then index this prefix.
      const ring = fingerprint.rings[depth]
      if (ring !== undefined) {
        key += `|${fingerprintRingKey(ring)}`
      }
      const byKey = maps[depth]!
      const existing = byKey.get(key)
      if (existing === undefined) {
        byKey.set(key, [fingerprint.index])
      } else {
        existing.push(fingerprint.index)
      }
    }
  }
  const indexes: IdentityIndex[] = []
  for (const map of maps) {
    indexes.push(identityIndexFromMap(map, fingerprints))
  }
  return indexes
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
  fingerprints?: readonly PointFingerprint[],
): IdentityEvaluation {
  const observedFingerprints =
    fingerprints !== undefined
      ? [...fingerprints]
      : (() => {
          const observedChainIds = chainIdsForGraph(observedGraph)
          const observedEndpoints = chainEndpointFlagsForGraph(observedGraph)
          return observedGraph.points.map((_, rootIndex) =>
            computeFingerprint(
              observedGraph,
              colors,
              rootIndex,
              depth,
              observedChainIds,
              observedEndpoints,
            ),
          )
        })()

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
