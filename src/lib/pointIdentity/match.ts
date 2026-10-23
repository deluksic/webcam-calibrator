import { chainEndpointFlagsForGraph, chainIdsForGraph } from '@/lib/pointIdentity/chainRuns'
import { computeFingerprint } from '@/lib/pointIdentity/identity'
import type {
  BinaryColor,
  IdentityIndex,
  PointFingerprint,
  ProximityGraph,
} from '@/lib/pointIdentity/types'

export const IDENTITY_RESOLVED = 0
export const IDENTITY_AMBIGUOUS = 1
export const IDENTITY_UNRESOLVED = 2
export const IDENTITY_CONFLICT = 3
/** Identity assigned by geometric ICP, not by the local hash. */
export const IDENTITY_RECOVERED = 4
/** Virtual design point with no detection; never a measurement. */
export const IDENTITY_PREDICTED_MISSING = 5

export type IdentityStatus =
  | typeof IDENTITY_RESOLVED
  | typeof IDENTITY_AMBIGUOUS
  | typeof IDENTITY_UNRESOLVED
  | typeof IDENTITY_CONFLICT
  | typeof IDENTITY_RECOVERED
  | typeof IDENTITY_PREDICTED_MISSING

export type MatchedIdentityPoint = {
  x: number
  y: number
  color: BinaryColor
  designId: number | undefined
  status: IdentityStatus
  /** Position in the observed graph; useful when the observed index is not the design id. */
  observedPosition?: number
  source?: 'hash' | 'hash-loose' | 'geometry'
  proofPower?: number
  /** Design-space distance from the final ICP prediction. */
  designDistance?: number
  /** The ICP transform disagreed with a resolved hash identity. */
  geometryConflict?: boolean
  /** Number of overlapping local neighborhoods that voted for this assignment. */
  recoveryVotes?: number
}

export type IdentityMatchSummary = {
  total: number
  resolved: number
  ambiguous: number
  unresolved: number
  conflicts: number
}

export type ColorIdentityIndex = Map<string, number[]>

/** Loose key: root color plus ring sizes/colors; no chain partition/endpoint dependence. */
export function colorFingerprintKey(fingerprint: PointFingerprint): string {
  let key = String(fingerprint.color)
  for (const ring of fingerprint.rings) {
    key += `|${ring.size}:${ring.colors}`
  }
  return key
}

export function createColorIdentityIndex(
  graph: ProximityGraph,
  colors: readonly BinaryColor[],
  depth: number,
): ColorIdentityIndex {
  const chainIds = chainIdsForGraph(graph)
  const endpointFlags = chainEndpointFlagsForGraph(graph)
  const byKey: ColorIdentityIndex = new Map()
  for (let rootIndex = 0; rootIndex < graph.points.length; rootIndex += 1) {
    const fingerprint = computeFingerprint(graph, colors, rootIndex, depth, chainIds, endpointFlags)
    const key = colorFingerprintKey(fingerprint)
    const existing = byKey.get(key)
    if (existing === undefined) {
      byKey.set(key, [fingerprint.index])
    } else {
      existing.push(fingerprint.index)
    }
  }
  return byKey
}

export type IdentityMatchResult = {
  points: MatchedIdentityPoint[]
  fingerprints: PointFingerprint[]
  summary: IdentityMatchSummary
}

/**
 * Match observed graph fingerprints against the design index.
 *
 * Unlike `evaluateIdentity`, this does not compare the matched design index to
 * the observed array index; observed ids are assigned by fingerprint.
 */
export function matchObservedIdentity(
  designIndex: IdentityIndex,
  observedGraph: ProximityGraph,
  observedColors: readonly BinaryColor[],
  depth: number,
  options?: {
    looseByKey?: ColorIdentityIndex
    fingerprints?: readonly PointFingerprint[]
  },
): IdentityMatchResult {
  const fingerprints =
    options?.fingerprints !== undefined
      ? [...options.fingerprints]
      : (() => {
          const chainIds = chainIdsForGraph(observedGraph)
          const endpointFlags = chainEndpointFlagsForGraph(observedGraph)
          return observedGraph.points.map((_, rootIndex) =>
            computeFingerprint(
              observedGraph,
              observedColors,
              rootIndex,
              depth,
              chainIds,
              endpointFlags,
            ),
          )
        })()

  const candidateLists = fingerprints.map((fingerprint) =>
    options?.looseByKey !== undefined
      ? options.looseByKey.get(colorFingerprintKey(fingerprint))
      : designIndex.byKey.get(fingerprint.key),
  )
  const assigned: Array<number | undefined> = candidateLists.map((candidates) =>
    candidates !== undefined && candidates.length === 1 ? candidates[0] : undefined,
  )

  const usage = new Map<number, number[]>()
  for (let index = 0; index < assigned.length; index += 1) {
    const designId = assigned[index]
    if (designId === undefined) {
      continue
    }
    const users = usage.get(designId)
    if (users === undefined) {
      usage.set(designId, [index])
    } else {
      users.push(index)
    }
  }
  const conflicting = new Set<number>()
  for (const users of usage.values()) {
    if (users.length > 1) {
      for (const index of users) {
        conflicting.add(index)
      }
    }
  }

  let resolved = 0
  let ambiguous = 0
  let unresolved = 0
  let conflicts = 0
  const points: MatchedIdentityPoint[] = fingerprints.map((fingerprint, index) => {
    const candidates = candidateLists[index]
    const designId = assigned[index]
    let status: IdentityStatus
    if (conflicting.has(index)) {
      status = IDENTITY_CONFLICT
      conflicts += 1
    } else if (designId !== undefined) {
      status = IDENTITY_RESOLVED
      resolved += 1
    } else if (candidates !== undefined && candidates.length > 1) {
      status = IDENTITY_AMBIGUOUS
      ambiguous += 1
    } else {
      status = IDENTITY_UNRESOLVED
      unresolved += 1
    }
    const source = observedGraph.points[index]!
    return {
      x: source.x,
      y: source.y,
      color: observedColors[index] ?? 0,
      designId: status === IDENTITY_RESOLVED ? designId : undefined,
      status,
    }
  })

  return {
    points,
    fingerprints,
    summary: {
      total: points.length,
      resolved,
      ambiguous,
      unresolved,
      conflicts,
    },
  }
}
