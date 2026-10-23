import type { HexTargetModel } from '@/lib/hexTarget/target'
import { computeFingerprints } from '@/lib/pointIdentity/identity'
import {
  createLocalIcpOptions,
  MAX_FINGERPRINT_DEPTH,
  refineObservedIdentityWithLocalIcp,
} from '@/lib/pointIdentity/localIcpRefiner'
import {
  createColorIdentityIndex,
  IDENTITY_AMBIGUOUS,
  IDENTITY_CONFLICT,
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  IDENTITY_UNRESOLVED,
  matchObservedIdentity,
  type MatchedIdentityPoint,
} from '@/lib/pointIdentity/match'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import type { BinaryColor } from '@/lib/pointIdentity/types'

export type HexObservedPoint = {
  x: number
  y: number
  color: BinaryColor
}

export type HexTargetMatchSummary = {
  total: number
  resolved: number
  recovered: number
  ambiguous: number
  unresolved: number
  conflicts: number
}

export type HexTargetMatch = {
  points: MatchedIdentityPoint[]
  edges: ReadonlyArray<readonly [number, number]>
  summary: HexTargetMatchSummary
}

export type HexTargetMatchOptions = {
  recovery?: boolean
  depth?: number
}

function summarize(points: readonly MatchedIdentityPoint[]): HexTargetMatchSummary {
  const summary: HexTargetMatchSummary = {
    total: points.length,
    resolved: 0,
    recovered: 0,
    ambiguous: 0,
    unresolved: 0,
    conflicts: 0,
  }
  for (const point of points) {
    switch (point.status) {
      case IDENTITY_RESOLVED:
        summary.resolved += 1
        break
      case IDENTITY_RECOVERED:
        summary.recovered += 1
        break
      case IDENTITY_AMBIGUOUS:
        summary.ambiguous += 1
        break
      case IDENTITY_UNRESOLVED:
        summary.unresolved += 1
        break
      case IDENTITY_CONFLICT:
        summary.conflicts += 1
        break
    }
  }
  return summary
}

/**
 * Build the observed marker graph and resolve each marker against the fixed
 * design. Hash identities are tried first; ambiguous/unresolved markers can be
 * recovered with the local-ICP path used by Point Lab and ROI Lab.
 */
export function matchHexTargetPoints(
  design: HexTargetModel,
  observed: readonly HexObservedPoint[],
  options: HexTargetMatchOptions = {},
): HexTargetMatch {
  const depth = Math.max(
    1,
    Math.min(design.indexesByDepth.length, Math.round(options.depth ?? design.indexesByDepth.length)),
  )
  const observedPoints = observed.map((point, index) => ({
    x: point.x,
    y: point.y,
    index,
  }))
  const observedColors = observed.map((point) => point.color)
  const observedGraph = buildProximityGraph(observedPoints, {
    pointsPerEdge: design.pointsPerEdge,
  })
  const fingerprints = computeFingerprints(observedGraph, observedColors, depth)
  const identityIndex = design.indexesByDepth[depth - 1]!
  const recovery = options.recovery ?? true

  const baseline = matchObservedIdentity(identityIndex, observedGraph, observedColors, depth, {
    fingerprints,
    looseByKey: recovery ? undefined : createColorIdentityIndex(design.graph, design.colors, depth),
  })

  let points = baseline.points
  if (recovery) {
    const recoveryFingerprints =
      depth === MAX_FINGERPRINT_DEPTH
        ? fingerprints
        : computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
    points = refineObservedIdentityWithLocalIcp({
      designPattern: design.pattern,
      designColors: design.colors,
      designGraph: design.graph,
      designIndexesByDepth: design.indexesByDepth,
      observedGraph,
      observedColors,
      observedFingerprints: recoveryFingerprints,
      baseline,
      designCache: design.designCache,
      options: createLocalIcpOptions({
        identityDepth: depth,
        maxProofDepth: MAX_FINGERPRINT_DEPTH,
        minProofPower: 7,
        hopRadius: 8,
      }),
    }).points
  }

  return {
    points,
    edges: observedGraph.edges,
    summary: summarize(points),
  }
}
