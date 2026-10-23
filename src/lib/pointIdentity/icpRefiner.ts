/**
 * Legacy global-homography ICP. Point Lab and the stress harness now use the
 * hop-radius local matcher in `localIcpRefiner.ts`; this module is retained for
 * comparison and shared result types only.
 */
import { applyHomography, type Mat3 } from '@/lib/geometry'
import { collectAnchors, type Anchor } from '@/lib/pointIdentity/anchors'
import { createDesignPositionMap } from '@/lib/pointIdentity/designGrid'
import {
  runDesignSpaceIcp,
  type IcpPair,
  type IcpRejectionReason,
  type IcpResult,
} from '@/lib/pointIdentity/designSpaceIcp'
import { computeFingerprints, fingerprintKeyAtDepth } from '@/lib/pointIdentity/identity'
import {
  IDENTITY_AMBIGUOUS,
  IDENTITY_CONFLICT,
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  IDENTITY_UNRESOLVED,
  type IdentityMatchResult,
  type IdentityStatus,
  type MatchedIdentityPoint,
} from '@/lib/pointIdentity/match'
import type {
  BinaryColor,
  HexPattern,
  IdentityIndex,
  PatternPoint,
  ProximityGraph,
  Vec2,
} from '@/lib/pointIdentity/types'

export type FrameBounds = {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

export type IcpRefineOptions = {
  enabled: boolean
  identityDepth: number
  maxProofDepth: number
  minProofPower: number
  minAnchors: number
  minInitialInliers: number
  minInitialInlierRatio: number
  maxIcpIterations: number
  designGateFactor: number
  anchorGateFactor: number
  minPairCoverage: number
  maxDesignRmsFactor: number
  maxColorRejectRate: number
  maxConflictRate: number
  holdoutAnchorFraction: number
  minObservedSpacingPx: number
  allowProofFallback: boolean
  seed: number
  frameBounds?: FrameBounds
}

export const DEFAULT_ICP_REFINE_OPTIONS: IcpRefineOptions = {
  enabled: true,
  identityDepth: 6,
  maxProofDepth: 8,
  minProofPower: 4,
  minAnchors: 6,
  minInitialInliers: 6,
  minInitialInlierRatio: 0.5,
  maxIcpIterations: 10,
  designGateFactor: 0.4,
  anchorGateFactor: 0.2,
  minPairCoverage: 0.5,
  maxDesignRmsFactor: 0.35,
  maxColorRejectRate: 0.1,
  maxConflictRate: 0.05,
  holdoutAnchorFraction: 0.2,
  minObservedSpacingPx: 3,
  allowProofFallback: true,
  seed: 0x9e3779b9,
}

export function createIcpRefineOptions(overrides: Partial<IcpRefineOptions> = {}): IcpRefineOptions {
  return { ...DEFAULT_ICP_REFINE_OPTIONS, ...overrides }
}

export type PredictedMissingPoint = {
  designId: number
  predicted: Vec2
  color: BinaryColor
}

export type PredictedLatticePoint = {
  designId: number
  predicted: Vec2
  color: BinaryColor
}

export type IcpRefineRejectionReason =
  | IcpRejectionReason
  | 'disabled'
  | 'too-few-observed'
  | 'no-ambiguous-points'
  | 'local-icp-failed'

export type IcpRefineTransformSummary = {
  initial: Mat3
  final: Mat3
  designRms: number
  iterations: number
  fitAnchorMaxResidual: number
  validationAnchorMaxResidual?: number
}

export type IcpRefineMetrics = {
  observed: number
  anchors: number
  initialInliers: number
  pairs: number
  pairCoverage: number
  designSpacing: number
  designRms: number
  colorRejects: number
  recovered: number
  geometryConflicts: number
  predictedMissing: number
  strictResolved: number
  validationAnchors: number
  iterations: number
  usedProofPower: number
  /** Number of initialized local ICP neighborhoods (local refiner only). */
  neighborhoods?: number
}

export type IcpRefineResult = {
  accepted: boolean
  rejectionReason?: IcpRefineRejectionReason
  baseline: IdentityMatchResult
  points: MatchedIdentityPoint[]
  predictedMissing: PredictedMissingPoint[]
  predictedLattice: PredictedLatticePoint[]
  matches: IcpPair[]
  anchors: Anchor[]
  transform?: IcpRefineTransformSummary
  metrics: IcpRefineMetrics
  usedProofPower: number
}

export type IcpRefineInput = {
  designPattern: HexPattern
  designColors: readonly BinaryColor[]
  designGraph: ProximityGraph
  designIndexesByDepth: readonly IdentityIndex[]
  observedGraph: ProximityGraph
  observedColors: readonly BinaryColor[]
  baseline: IdentityMatchResult
  options: IcpRefineOptions
}

type StrictMatch = {
  status: IdentityStatus
  designId?: number
}

function strictMatches(
  observedFingerprints: ReturnType<typeof computeFingerprints>,
  designIndexesByDepth: readonly IdentityIndex[],
  identityDepth: number,
): StrictMatch[] {
  const selectedIndex = designIndexesByDepth[identityDepth - 1]
  const candidatesByObserved: Array<number[] | undefined> = observedFingerprints.map((fingerprint) => {
    if (selectedIndex === undefined) {
      return undefined
    }
    return selectedIndex.byKey.get(fingerprintKeyAtDepth(fingerprint, identityDepth))
  })

  const usage = new Map<number, number[]>()
  for (let observedPosition = 0; observedPosition < candidatesByObserved.length; observedPosition += 1) {
    const candidates = candidatesByObserved[observedPosition]
    if (candidates === undefined || candidates.length !== 1) {
      continue
    }
    const designId = candidates[0]!
    const users = usage.get(designId)
    if (users === undefined) {
      usage.set(designId, [observedPosition])
    } else {
      users.push(observedPosition)
    }
  }

  return candidatesByObserved.map((candidates) => {
    if (candidates === undefined) {
      return { status: IDENTITY_UNRESOLVED }
    }
    if (candidates.length > 1) {
      return { status: IDENTITY_AMBIGUOUS }
    }
    const designId = candidates[0]!
    const users = usage.get(designId) ?? []
    if (users.length > 1) {
      return { status: IDENTITY_CONFLICT }
    }
    return { status: IDENTITY_RESOLVED, designId }
  })
}

function baselinePointsWithPositions(
  baseline: IdentityMatchResult,
  proofPowers: readonly number[],
): MatchedIdentityPoint[] {
  return baseline.points.map((point, observedPosition) => ({
    ...point,
    observedPosition,
    proofPower: proofPowers[observedPosition],
    source: point.designId === undefined ? undefined : 'hash-loose',
  }))
}

function emptyMetrics(observed: number, designSpacing: number, usedProofPower: number): IcpRefineMetrics {
  return {
    observed,
    anchors: 0,
    initialInliers: 0,
    pairs: 0,
    pairCoverage: 0,
    designSpacing,
    designRms: Number.POSITIVE_INFINITY,
    colorRejects: 0,
    recovered: 0,
    geometryConflicts: 0,
    predictedMissing: 0,
    strictResolved: 0,
    validationAnchors: 0,
    iterations: 0,
    usedProofPower,
  }
}

function rejectionResult(
  input: IcpRefineInput,
  reason: IcpRefineRejectionReason,
  anchors: readonly Anchor[],
  usedProofPower: number,
  proofPowers: readonly number[],
  icp?: IcpResult,
): IcpRefineResult {
  const points = baselinePointsWithPositions(input.baseline, proofPowers)
  return {
    accepted: false,
    rejectionReason: reason,
    baseline: input.baseline,
    points,
    predictedMissing: [],
    predictedLattice: [],
    matches: [],
    anchors: [...anchors],
    transform:
      icp?.initialMatrix !== undefined && icp.matrix !== undefined
        ? {
            initial: icp.initialMatrix,
            final: icp.matrix,
            designRms: icp.designRms,
            iterations: icp.iterations.length,
            fitAnchorMaxResidual: icp.fitAnchorMaxResidual,
            validationAnchorMaxResidual: icp.validationAnchorMaxResidual,
          }
        : undefined,
    metrics: {
      ...emptyMetrics(input.observedGraph.points.length, icp?.designSpacing ?? 0, usedProofPower),
      anchors: anchors.length,
      initialInliers: icp?.fitAnchors.length ?? 0,
      pairs: icp?.pairs.length ?? 0,
      pairCoverage: icp?.pairCoverage ?? 0,
      designRms: icp?.designRms ?? Number.POSITIVE_INFINITY,
      colorRejects: icp?.colorRejects ?? 0,
      validationAnchors: icp?.validationAnchors.length ?? 0,
      iterations: icp?.iterations.length ?? 0,
    },
    usedProofPower,
  }
}

export function refineObservedIdentityWithIcp(input: IcpRefineInput): IcpRefineResult {
  const { observedGraph, observedColors, options } = input
  const observed = observedGraph.points.length
  if (!options.enabled) {
    return rejectionResult(input, 'disabled', [], options.minProofPower, [], undefined)
  }
  if (observed === 0) {
    return rejectionResult(input, 'too-few-observed', [], options.minProofPower, [], undefined)
  }

  const availableDepths = Math.max(1, input.designIndexesByDepth.length)
  const identityDepth = Math.min(Math.max(1, options.identityDepth), availableDepths)
  const maxProofDepth = Math.min(Math.max(identityDepth, options.maxProofDepth), availableDepths)

  // Fingerprints index colors by the graph point's stable `.index`, while the
  // caller supplies observed colors by array position. In Point Lab the
  // observed index is the design id; in the live path it is the array position.
  const observedColorByGraphIndex: BinaryColor[] = []
  for (let position = 0; position < observedGraph.points.length; position += 1) {
    const point = observedGraph.points[position]!
    observedColorByGraphIndex[point.index] = observedColors[position] ?? 0
  }
  const observedFingerprints = computeFingerprints(observedGraph, observedColorByGraphIndex, maxProofDepth)
  const designPointsByIndex = new Map<number, PatternPoint>()
  for (const point of input.designPattern.points) {
    designPointsByIndex.set(point.index, point)
  }

  const anchorCollection = collectAnchors(
    observedGraph,
    observedColors,
    observedFingerprints,
    input.designIndexesByDepth,
    {
      identityDepth,
      maxProofDepth,
      minProofPower: options.minProofPower,
      minAnchors: options.minAnchors,
      designPointsByIndex,
      allowProofFallback: options.allowProofFallback,
    },
  )
  const proofPowers = anchorCollection.proofPowers.map((proof) => proof.depth)

  if (anchorCollection.anchors.length < Math.max(6, options.minAnchors)) {
    return rejectionResult(
      input,
      'too-few-anchors',
      anchorCollection.anchors,
      anchorCollection.usedProofPower,
      proofPowers,
    )
  }

  const icp = runDesignSpaceIcp({
    designPoints: input.designPattern.points,
    designColors: input.designColors,
    observedPoints: observedGraph.points,
    observedColors,
    anchors: anchorCollection.anchors,
    options: {
      maxIterations: options.maxIcpIterations,
      designGateFactor: options.designGateFactor,
      anchorGateFactor: options.anchorGateFactor,
      minInitialInliers: options.minInitialInliers,
      minInitialInlierRatio: options.minInitialInlierRatio,
      minPairCoverage: options.minPairCoverage,
      maxDesignRmsFactor: options.maxDesignRmsFactor,
      maxColorRejectRate: options.maxColorRejectRate,
      holdoutAnchorFraction: options.holdoutAnchorFraction,
      minObservedSpacingPx: options.minObservedSpacingPx,
      seed: options.seed,
    },
  })

  if (!icp.accepted || icp.matrix === undefined || icp.initialMatrix === undefined) {
    return rejectionResult(
      input,
      icp.rejectionReason ?? 'icp-failed',
      anchorCollection.anchors,
      anchorCollection.usedProofPower,
      proofPowers,
      icp,
    )
  }

  const strict = strictMatches(observedFingerprints, input.designIndexesByDepth, identityDepth)
  const strictResolvedCount = strict.filter((match) => match.status === IDENTITY_RESOLVED).length
  const icpPairByObserved = new Map<number, IcpPair>()
  const icpPairByDesign = new Map<number, IcpPair>()
  for (const pair of icp.pairs) {
    icpPairByObserved.set(pair.observedPosition, pair)
    icpPairByDesign.set(pair.designId, pair)
  }

  const reservedDesignIds = new Set<number>()
  const conflictByObserved = new Set<number>()
  let geometryConflicts = 0
  for (let observedPosition = 0; observedPosition < strict.length; observedPosition += 1) {
    const match = strict[observedPosition]!
    if (match.status !== IDENTITY_RESOLVED || match.designId === undefined) {
      continue
    }
    reservedDesignIds.add(match.designId)
    const pair = icpPairByObserved.get(observedPosition)
    if (pair !== undefined && pair.designId !== match.designId) {
      conflictByObserved.add(observedPosition)
      geometryConflicts += 1
    } else if (pair === undefined && icpPairByDesign.has(match.designId)) {
      conflictByObserved.add(observedPosition)
      geometryConflicts += 1
    }
  }

  const maxConflicts = Math.max(2, Math.floor(strictResolvedCount * options.maxConflictRate))
  if (geometryConflicts > maxConflicts) {
    return rejectionResult(
      input,
      'too-many-conflicts',
      anchorCollection.anchors,
      anchorCollection.usedProofPower,
      proofPowers,
      icp,
    )
  }

  const points: MatchedIdentityPoint[] = input.baseline.points.map((base, observedPosition) => {
    const match = strict[observedPosition] ?? { status: IDENTITY_UNRESOLVED }
    const pair = icpPairByObserved.get(observedPosition)
    const proofPower = proofPowers[observedPosition]
    if (base.status === IDENTITY_CONFLICT) {
      return {
        ...base,
        observedPosition,
        proofPower,
        source: base.designId === undefined ? undefined : 'hash-loose',
        geometryConflict: pair !== undefined && pair.designId !== base.designId,
      }
    }
    if (match.status === IDENTITY_RESOLVED && match.designId !== undefined) {
      if (conflictByObserved.has(observedPosition)) {
        // The hash and the validated geometry disagree. Do not keep a possibly
        // false identity and do not replace it with an unverified one: abstain.
        return {
          ...base,
          designId: undefined,
          status: IDENTITY_CONFLICT,
          observedPosition,
          proofPower,
          source: 'hash',
          geometryConflict: true,
        }
      }
      return {
        ...base,
        designId: match.designId,
        status: IDENTITY_RESOLVED,
        observedPosition,
        proofPower,
        source: 'hash',
        geometryConflict: false,
      }
    }
    if (pair !== undefined) {
      return {
        ...base,
        designId: pair.designId,
        status: IDENTITY_RECOVERED,
        observedPosition,
        proofPower,
        source: 'geometry',
        designDistance: pair.designDistance,
        geometryConflict: false,
      }
    }
    return {
      ...base,
      observedPosition,
      proofPower,
      source: base.designId === undefined ? undefined : 'hash-loose',
    }
  })

  const recovered = points.filter((point) => point.status === IDENTITY_RECOVERED).length
  const pairedDesignIds = new Set(icp.pairs.map((pair) => pair.designId))
  const blockedDesignIds = new Set(icp.blockedDesignIds)
  const designPositionById = createDesignPositionMap(input.designPattern.points)
  const predictedMissing: PredictedMissingPoint[] = []
  const predictedLattice: PredictedLatticePoint[] = []

  for (let designPosition = 0; designPosition < input.designPattern.points.length; designPosition += 1) {
    const designPoint = input.designPattern.points[designPosition]!
    const predicted = applyHomography(icp.matrix, designPoint.x, designPoint.y)
    if (!Number.isFinite(predicted.x) || !Number.isFinite(predicted.y)) {
      continue
    }
    predictedLattice.push({
      designId: designPoint.index,
      predicted: { x: predicted.x, y: predicted.y },
      color: input.designColors[designPosition] ?? 0,
    })
  }

  for (const designPoint of input.designPattern.points) {
    const designId = designPoint.index
    if (pairedDesignIds.has(designId) || reservedDesignIds.has(designId) || blockedDesignIds.has(designId)) {
      continue
    }
    const designPosition = designPositionById.get(designId)
    if (designPosition === undefined) {
      continue
    }
    const neighborPositions = input.designGraph.neighbors[designPosition] ?? []
    let hasPairedNeighbor = false
    for (const neighborPosition of neighborPositions) {
      const neighbor = input.designPattern.points[neighborPosition]
      if (neighbor !== undefined && pairedDesignIds.has(neighbor.index)) {
        hasPairedNeighbor = true
        break
      }
    }
    if (!hasPairedNeighbor) {
      continue
    }
    const predicted = applyHomography(icp.matrix, designPoint.x, designPoint.y)
    if (!Number.isFinite(predicted.x) || !Number.isFinite(predicted.y)) {
      continue
    }
    if (options.frameBounds !== undefined) {
      const bounds = options.frameBounds
      if (
        predicted.x < bounds.minX ||
        predicted.x > bounds.maxX ||
        predicted.y < bounds.minY ||
        predicted.y > bounds.maxY
      ) {
        continue
      }
    }
    predictedMissing.push({
      designId,
      predicted: { x: predicted.x, y: predicted.y },
      color: input.designColors[designPosition] ?? 0,
    })
  }

  return {
    accepted: true,
    baseline: input.baseline,
    points,
    predictedMissing,
    predictedLattice,
    matches: icp.pairs,
    anchors: anchorCollection.anchors,
    transform: {
      initial: icp.initialMatrix,
      final: icp.matrix,
      designRms: icp.designRms,
      iterations: icp.iterations.length,
      fitAnchorMaxResidual: icp.fitAnchorMaxResidual,
      validationAnchorMaxResidual: icp.validationAnchorMaxResidual,
    },
    metrics: {
      observed,
      anchors: anchorCollection.anchors.length,
      initialInliers: icp.fitAnchors.length,
      pairs: icp.pairs.length,
      pairCoverage: icp.pairCoverage,
      designSpacing: icp.designSpacing,
      designRms: icp.designRms,
      colorRejects: icp.colorRejects,
      recovered,
      geometryConflicts,
      predictedMissing: predictedMissing.length,
      strictResolved: strictResolvedCount,
      validationAnchors: icp.validationAnchors.length,
      iterations: icp.iterations.length,
      usedProofPower: anchorCollection.usedProofPower,
    },
    usedProofPower: anchorCollection.usedProofPower,
  }
}
