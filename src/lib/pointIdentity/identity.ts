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

const BASE36 = '0123456789abcdefghijklmnopqrstuvwxyz'

/**
 * Rotate a ring color string so that repeated/reversed traversal starts compare
 * equal. Only rotations are considered: a perspective view cannot mirror the
 * angular order of a neighborhood.
 */
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

/** Rings wider than a 32-bit word fall back to the element-wise path. */
const MAX_PACKED_RING = 32

type FingerprintScratch = {
  /** atan2 of each point, so the ring sort never recomputes it per comparison. */
  angle: Float64Array
  visited: Uint8Array
  ringColors: Uint8Array
  ringEndpoints: Uint8Array
  ringChains: Int32Array
  partition: Uint8Array
  bestPartition: Uint8Array
  distinctChains: Int32Array
  distinctLabels: Int32Array
}

function createFingerprintScratch(pointCount: number): FingerprintScratch {
  const points = Math.max(1, pointCount)
  const ring = Math.max(8, Math.min(points, 64))
  return {
    angle: new Float64Array(points),
    visited: new Uint8Array(points),
    ringColors: new Uint8Array(ring),
    ringEndpoints: new Uint8Array(ring),
    ringChains: new Int32Array(ring),
    partition: new Uint8Array(ring),
    bestPartition: new Uint8Array(ring),
    distinctChains: new Int32Array(ring),
    distinctLabels: new Int32Array(ring),
  }
}

function growRing(scratch: FingerprintScratch, size: number): void {
  if (scratch.ringColors.length >= size) {
    return
  }
  const capacity = Math.max(size, scratch.ringColors.length * 2)
  const colors = new Uint8Array(capacity)
  colors.set(scratch.ringColors)
  scratch.ringColors = colors
  const endpoints = new Uint8Array(capacity)
  endpoints.set(scratch.ringEndpoints)
  scratch.ringEndpoints = endpoints
  const chains = new Int32Array(capacity)
  chains.set(scratch.ringChains)
  scratch.ringChains = chains
  const partition = new Uint8Array(capacity)
  partition.set(scratch.partition)
  scratch.partition = partition
  const bestPartition = new Uint8Array(capacity)
  bestPartition.set(scratch.bestPartition)
  scratch.bestPartition = bestPartition
  const distinctChains = new Int32Array(capacity)
  distinctChains.set(scratch.distinctChains)
  scratch.distinctChains = distinctChains
  const distinctLabels = new Int32Array(capacity)
  distinctLabels.set(scratch.distinctLabels)
  scratch.distinctLabels = distinctLabels
}

function bitsToBinary(bits: number, count: number): string {
  return bits.toString(2).padStart(count, '0')
}

/**
 * Same algorithm without packing, for rings wider than one word. Compares
 * rotations element-wise: O(n²) per ring, which only matters for rings that
 * never occur in practice (the measured maximum is 18 on the default target).
 */
function canonicalRingWide(
  scratch: FingerprintScratch,
  count: number,
  rootChain: number,
): RingDescriptor {
  const ringColors = scratch.ringColors
  const ringEndpoints = scratch.ringEndpoints
  const ringChains = scratch.ringChains
  const partition = scratch.partition
  const bestPartition = scratch.bestPartition
  const distinctChains = scratch.distinctChains
  const distinctLabels = scratch.distinctLabels

  // Pass 1: the smallest colour rotation.
  let bestStart = 0
  for (let shift = 1; shift < count; shift += 1) {
    for (let index = 0; index < count; index += 1) {
      const candidate = ringColors[(shift + index) % count]!
      const current = ringColors[(bestStart + index) % count]!
      if (candidate !== current) {
        if (candidate < current) {
          bestStart = shift
        }
        break
      }
    }
  }

  // Pass 2: among the color ties, smallest partition, then smallest endpoints.
  let bestShift = bestStart
  let hasBest = false
  for (let shift = 0; shift < count; shift += 1) {
    let sameColor = true
    for (let index = 0; index < count; index += 1) {
      if (ringColors[(shift + index) % count] !== ringColors[(bestStart + index) % count]) {
        sameColor = false
        break
      }
    }
    if (!sameColor) {
      continue
    }

    let nextLabel = rootChain >= 0 ? 1 : 0
    let distinct = 0
    for (let index = 0; index < count; index += 1) {
      const chainId = ringChains[(shift + index) % count]!
      let label = -1
      if (rootChain >= 0 && chainId === rootChain) {
        label = 0
      } else {
        for (let known = 0; known < distinct; known += 1) {
          if (distinctChains[known] === chainId) {
            label = distinctLabels[known]!
            break
          }
        }
        if (label < 0) {
          label = nextLabel
          nextLabel += 1
          distinctChains[distinct] = chainId
          distinctLabels[distinct] = label
          distinct += 1
        }
      }
      partition[index] = label
    }

    let take = !hasBest
    if (!take) {
      let order = 0
      for (let index = 0; index < count; index += 1) {
        const candidate = partition[index]!
        const current = bestPartition[index]!
        if (candidate !== current) {
          order = candidate < current ? -1 : 1
          break
        }
      }
      if (order < 0) {
        take = true
      } else if (order === 0) {
        for (let index = 0; index < count; index += 1) {
          const candidate = ringEndpoints[(shift + index) % count]!
          const current = ringEndpoints[(bestShift + index) % count]!
          if (candidate !== current) {
            take = candidate < current
            break
          }
        }
      }
    }
    if (!take) {
      continue
    }
    hasBest = true
    bestShift = shift
    bestPartition.set(partition.subarray(0, count))
  }

  let colorText = ''
  for (let index = 0; index < count; index += 1) {
    colorText += ringColors[(bestStart + index) % count] === 1 ? '1' : '0'
  }
  let endpointText = ''
  for (let index = 0; index < count; index += 1) {
    endpointText += ringEndpoints[(bestShift + index) % count] === 1 ? '1' : '0'
  }
  let partitionText = ''
  for (let index = 0; index < count; index += 1) {
    partitionText += BASE36[bestPartition[index]!]
  }
  return {
    size: count,
    colors: colorText,
    partition: partitionText,
    endpoints: endpointText,
  }
}

/**
 * Canonical (rotation-invariant) form of one BFS ring.
 *
 * The previous implementation rebuilt three strings for every rotation and
 * compared them lexicographically. Here:
 *
 * - colors and endpoint flags are packed MSB-first into 32-bit words, so a
 *   rotation is one shift pair and the ordering is a single integer compare;
 * - only rotations that reach the minimal color word are considered at all
 *   (colors outrank the partition, so the rest can never win);
 * - chain labels are written into a scratch buffer and compared element-wise,
 *   which is the one part that cannot be packed — a ring of `n` points needs
 *   `n * log2(n)` bits of label data, and the first-appearance relabeling is
 *   inherently sequential.
 *
 * Only the winning rotation is materialised into strings, for the key and UI.
 */
function canonicalRingFromArrays(
  scratch: FingerprintScratch,
  count: number,
  rootChain: number,
): RingDescriptor {
  const ringColors = scratch.ringColors
  const ringEndpoints = scratch.ringEndpoints
  const ringChains = scratch.ringChains

  if (count <= 1) {
    return {
      size: count,
      colors: count === 0 ? '' : ringColors[0] === 1 ? '1' : '0',
      partition: '0',
      endpoints: count === 0 ? '' : ringEndpoints[0] === 1 ? '1' : '0',
    }
  }
  if (count > MAX_PACKED_RING) {
    return canonicalRingWide(scratch, count, rootChain)
  }

  let colorBits = 0
  let endpointBits = 0
  for (let index = 0; index < count; index += 1) {
    colorBits = (colorBits << 1) | (ringColors[index] === 1 ? 1 : 0)
    endpointBits = (endpointBits << 1) | (ringEndpoints[index] === 1 ? 1 : 0)
  }
  colorBits >>>= 0
  endpointBits >>>= 0
  const mask = count >= 32 ? 0xffffffff : (1 << count) - 1
  const rotate = (bits: number, shift: number): number =>
    shift === 0 ? bits : (((bits << shift) | (bits >>> (count - shift))) & mask) >>> 0

  // Pass 1: the smallest colour ring. Rotations above it cannot win.
  let bestColorBits = 0xffffffff
  for (let shift = 0; shift < count; shift += 1) {
    const value = rotate(colorBits, shift)
    if (value < bestColorBits) {
      bestColorBits = value
    }
  }

  const partition = scratch.partition
  const bestPartition = scratch.bestPartition
  const distinctChains = scratch.distinctChains
  const distinctLabels = scratch.distinctLabels

  // Pass 2: among the color ties, smallest partition, then smallest endpoints.
  let bestEndpoints = 0
  let hasBest = false
  for (let shift = 0; shift < count; shift += 1) {
    if (rotate(colorBits, shift) !== bestColorBits) {
      continue
    }

    let nextLabel = rootChain >= 0 ? 1 : 0
    let distinct = 0
    for (let index = 0; index < count; index += 1) {
      const chainId = ringChains[(shift + index) % count]!
      let label = -1
      if (rootChain >= 0 && chainId === rootChain) {
        label = 0
      } else {
        for (let known = 0; known < distinct; known += 1) {
          if (distinctChains[known] === chainId) {
            label = distinctLabels[known]!
            break
          }
        }
        if (label < 0) {
          label = nextLabel
          nextLabel += 1
          distinctChains[distinct] = chainId
          distinctLabels[distinct] = label
          distinct += 1
        }
      }
      partition[index] = label
    }

    const candidateEndpoints = rotate(endpointBits, shift)
    let take = !hasBest
    if (!take) {
      let order = 0
      for (let index = 0; index < count; index += 1) {
        const candidate = partition[index]!
        const current = bestPartition[index]!
        if (candidate !== current) {
          order = candidate < current ? -1 : 1
          break
        }
      }
      take = order < 0 || (order === 0 && candidateEndpoints < bestEndpoints)
    }
    if (!take) {
      continue
    }
    hasBest = true
    bestEndpoints = candidateEndpoints
    bestPartition.set(partition.subarray(0, count))
  }

  let partitionText = ''
  for (let index = 0; index < count; index += 1) {
    partitionText += BASE36[bestPartition[index]!]
  }
  return {
    size: count,
    colors: bitsToBinary(bestColorBits, count),
    partition: partitionText,
    endpoints: bitsToBinary(bestEndpoints, count),
  }
}

/**
 * Canonicalise one ring from angular-order arrays. Exposed for tests; the
 * pipeline uses the scratch variant to avoid per-ring allocation.
 */
export function canonicalRingDescriptor(
  ringColors: Uint8Array,
  ringEndpoints: Uint8Array,
  ringChains: Int32Array,
  count: number,
  rootChain: number,
): RingDescriptor {
  const scratch = createFingerprintScratch(Math.max(8, count))
  growRing(scratch, count)
  scratch.ringColors.set(ringColors.subarray(0, count))
  scratch.ringEndpoints.set(ringEndpoints.subarray(0, count))
  scratch.ringChains.set(ringChains.subarray(0, count))
  return canonicalRingFromArrays(scratch, count, rootChain)
}

function computeFingerprintInto(
  scratch: FingerprintScratch,
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  rootIndex: number,
  depth: number,
  chainIds: Int32Array,
  endpointFlags: Uint8Array,
): PointFingerprint {
  const root = graph.points[rootIndex]
  if (root === undefined) {
    return { index: rootIndex, color: 0, endpoint: false, rings: [], key: '0' }
  }
  const rootChain = chainIds[rootIndex] ?? -1
  const rootEndpoint = (endpointFlags[rootIndex] ?? 0) === 1

  const visited = scratch.visited
  const angle = scratch.angle
  visited.fill(0)
  visited[rootIndex] = 1
  let frontier: number[] = [rootIndex]
  const rings: RingDescriptor[] = []

  for (let level = 0; level < depth; level += 1) {
    const next: number[] = []
    for (const node of frontier) {
      for (const neighbor of graph.neighbors[node]!) {
        if (visited[neighbor] === 0) {
          visited[neighbor] = 1
          next.push(neighbor)
        }
      }
    }
    if (next.length === 0) {
      break
    }

    // One atan2 per ring point instead of one per sort comparison.
    for (const node of next) {
      const point = graph.points[node]!
      angle[node] = Math.atan2(point.y - root.y, point.x - root.x)
    }
    next.sort((a, b) => {
      const angleA = angle[a]!
      const angleB = angle[b]!
      if (angleA !== angleB) {
        return angleA - angleB
      }
      return a - b
    })

    growRing(scratch, next.length)
    const ringColors = scratch.ringColors
    const ringEndpoints = scratch.ringEndpoints
    const ringChains = scratch.ringChains
    for (let index = 0; index < next.length; index += 1) {
      const node = next[index]!
      ringColors[index] = colorAt(colors, graph.points[node]!.index)
      ringChains[index] = chainIds[node] ?? -1
      ringEndpoints[index] = (endpointFlags[node] ?? 0) === 1 ? 1 : 0
    }
    rings.push(canonicalRingFromArrays(scratch, next.length, rootChain))
    frontier = next
  }

  const rootColor = colorAt(colors, root.index)
  let key = fingerprintRootKey(rootColor, rootEndpoint)
  for (const ring of rings) {
    key += `|${fingerprintRingKey(ring)}`
  }

  return { index: root.index, color: rootColor, endpoint: rootEndpoint, rings, key }
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
  return computeFingerprintInto(
    createFingerprintScratch(graph.points.length),
    graph,
    colors,
    rootIndex,
    depth,
    chainIds,
    endpointFlags,
  )
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

/** Limits each fingerprint to its first `depth` rings.
 *
 *  A full-depth fingerprint already contains the exact BFS rings for every
 *  shallower depth, so a caller that needs both can traverse the graph once and
 *  derive the shallow keys here instead of running a second BFS pass. */
export function truncateFingerprints(
  fingerprints: readonly PointFingerprint[],
  depth: number,
): PointFingerprint[] {
  return fingerprints.map((fingerprint) => {
    const limit = Math.max(0, Math.min(depth, fingerprint.rings.length))
    if (limit === fingerprint.rings.length) {
      return fingerprint
    }
    const rings = fingerprint.rings.slice(0, limit)
    let key = fingerprintRootKey(fingerprint.color, fingerprint.endpoint)
    for (const ring of rings) {
      key += `|${fingerprintRingKey(ring)}`
    }
    return {
      index: fingerprint.index,
      color: fingerprint.color,
      endpoint: fingerprint.endpoint,
      rings,
      key,
    }
  })
}

export function computeFingerprints(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  depth: number,
): PointFingerprint[] {
  const chainIds = chainIdsForGraph(graph)
  const endpointFlags = chainEndpointFlagsForGraph(graph)
  // One scratch for the whole pass: the ring buffers are reused per root.
  const scratch = createFingerprintScratch(graph.points.length)
  const fingerprints: PointFingerprint[] = []
  for (let rootIndex = 0; rootIndex < graph.points.length; rootIndex += 1) {
    fingerprints.push(
      computeFingerprintInto(scratch, graph, colors, rootIndex, depth, chainIds, endpointFlags),
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
