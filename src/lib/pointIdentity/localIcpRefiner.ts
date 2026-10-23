import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import { applyHomography, type Mat3 } from '@/lib/geometry'
import { collectAnchors, type Anchor } from '@/lib/pointIdentity/anchors'
import type { IcpPair } from '@/lib/pointIdentity/designSpaceIcp'
import {
  computeDesignLatticeMetrics,
  createDesignGrid,
  createDesignPositionMap,
  type DesignGrid,
} from '@/lib/pointIdentity/designGrid'
import {
  fitHomography,
  homographyCentroid,
  isHomographyOrientationValid,
  robustHomography,
  type HomographyCorrespondence,
} from '@/lib/pointIdentity/homographyFit'
import { computeFingerprints } from '@/lib/pointIdentity/identity'
import type {
  FrameBounds,
  IcpRefineMetrics,
  IcpRefineResult,
  PredictedLatticePoint,
  PredictedMissingPoint,
} from '@/lib/pointIdentity/icpRefiner'
import {
  IDENTITY_AMBIGUOUS,
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  IDENTITY_UNRESOLVED,
  type IdentityMatchResult,
  type MatchedIdentityPoint,
} from '@/lib/pointIdentity/match'
import type {
  BinaryColor,
  HexPattern,
  IdentityIndex,
  PatternPoint,
  PointFingerprint,
  ProximityGraph,
  Vec2,
} from '@/lib/pointIdentity/types'

export type LocalIcpOptions = {
  enabled: boolean
  identityDepth: number
  maxProofDepth: number
  minProofPower: number
  minAnchors: number
  /** Graph-hop radius of the local observed support. */
  hopRadius: number
  /** Fewer than this many anchors in the local observed support means give up. */
  minLocalAnchors: number
  localGateFactor: number
  localAnchorGateFactor: number
  maxLocalIterations: number
  allowProofFallback: boolean
  frameBounds?: FrameBounds
}

/** Maximum fingerprint depth used by the local identity matcher (and its UI). */
export const MAX_FINGERPRINT_DEPTH = 12

export const DEFAULT_LOCAL_ICP_OPTIONS: LocalIcpOptions = {
  enabled: true,
  identityDepth: MAX_FINGERPRINT_DEPTH,
  maxProofDepth: MAX_FINGERPRINT_DEPTH,
  minProofPower: 7,
  minAnchors: 4,
  hopRadius: 8,
  minLocalAnchors: 4,
  localGateFactor: 0.35,
  localAnchorGateFactor: 0.5,
  maxLocalIterations: 3,
  // Min proof power is a hard floor: do not silently lower it when anchors are
  // scarce. Sparse frames should fail safely instead.
  allowProofFallback: false,
}

export function createLocalIcpOptions(overrides: Partial<LocalIcpOptions> = {}): LocalIcpOptions {
  return { ...DEFAULT_LOCAL_ICP_OPTIONS, ...overrides }
}

export type LocalIcpInput = {
  designPattern: HexPattern
  designColors: readonly BinaryColor[]
  designGraph: ProximityGraph
  designIndexesByDepth: readonly IdentityIndex[]
  observedGraph: ProximityGraph
  observedColors: readonly BinaryColor[]
  /** Already computed at maxProofDepth; avoids a duplicate fingerprint pass. */
  observedFingerprints?: readonly PointFingerprint[]
  baseline: IdentityMatchResult
  options: LocalIcpOptions
}

type LocalPair = {
  designPosition: number
  observedPosition: number
  distance: number
}

type LocalRegistration = {
  observedPosition: number
  designPosition: number
  designId: number
  distance: number
  observedToDesign: Mat3
  iterations: number
  anchorInliers: number
  colorRejects: number
}

type LocalChoice = {
  observedPosition: number
  designId: number
  distance: number
  observedToDesign: Mat3
}

function distance(first: Vec2, second: Vec2): number {
  return Math.hypot(first.x - second.x, first.y - second.y)
}

function pairKey(pairs: readonly LocalPair[]): string {
  return pairs
    .map((pair) => `${pair.observedPosition}:${pair.designPosition}`)
    .sort()
    .join('|')
}

/** BFS over either graph. Used for the local observed support. */
function collectHopGraph(
  graph: ProximityGraph,
  rootPosition: number,
  hopRadius: number,
): number[] {
  const visited = new Set<number>([rootPosition])
  let frontier = [rootPosition]
  for (let hop = 0; hop < hopRadius; hop += 1) {
    const next: number[] = []
    for (const node of frontier) {
      for (const neighbor of graph.neighbors[node] ?? []) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor)
          next.push(neighbor)
        }
      }
    }
    if (next.length === 0) {
      break
    }
    frontier = next
  }
  return [...visited]
}

function applyToDesign(matrix: Mat3, point: Vec2): Vec2 | undefined {
  const predicted = applyHomography(matrix, point.x, point.y)
  if (!Number.isFinite(predicted.x) || !Number.isFinite(predicted.y)) {
    return undefined
  }
  return { x: predicted.x, y: predicted.y }
}

function anchorCorrespondences(anchors: readonly Anchor[]): HomographyCorrespondence[] {
  return anchors.map((anchor) => ({
    design: anchor.design,
    observed: anchor.observed,
    weight: 1,
  }))
}

function fitFromAnchors(
  anchors: readonly Anchor[],
  designSpacing: number,
  gate: number,
  minInliers: number,
): { observedToDesign: Mat3; inliers: number } | undefined {
  const fit = robustHomography(anchorCorrespondences(anchors), {
    designSpacing,
    trimFactor: 2.5,
    trimFloor: Math.max(0.1, gate * 0.25),
    maxRounds: 6,
    minInliers,
    minInlierRatio: 0.5,
    seed: 0x9e3779b9,
  })
  if (fit === undefined) {
    return undefined
  }
  const centroid = homographyCentroid(anchors.map((anchor) => anchor.design))
  if (!isHomographyOrientationValid(fit.matrix, centroid)) {
    return undefined
  }
  const observedToDesign = invertMat3RowMajor(fit.matrix)
  if (observedToDesign === undefined) {
    return undefined
  }
  return { observedToDesign, inliers: fit.inlierIndices.length }
}

function matchLocalSupport(
  observedToDesign: Mat3,
  localObservedPositions: readonly number[],
  observedPoints: readonly Vec2[],
  observedColors: readonly BinaryColor[],
  designPoints: readonly PatternPoint[],
  designGrid: DesignGrid,
  gate: number,
): { pairs: LocalPair[]; colorRejects: number } {
  const candidates: LocalPair[] = []
  let colorRejects = 0
  for (const observedPosition of localObservedPositions) {
    const observedPoint = observedPoints[observedPosition]!
    const predicted = applyToDesign(observedToDesign, observedPoint)
    if (predicted === undefined) {
      colorRejects += 1
      continue
    }
    const designPosition = designGrid.nearest(
      predicted,
      gate,
      observedColors[observedPosition] ?? 0,
    )
    if (designPosition === undefined) {
      colorRejects += 1
      continue
    }
    const designPoint = designPoints[designPosition]!
    candidates.push({
      designPosition,
      observedPosition,
      distance: distance(predicted, designPoint),
    })
  }
  candidates.sort((first, second) => {
    if (first.distance !== second.distance) {
      return first.distance - second.distance
    }
    return first.observedPosition - second.observedPosition
  })
  const usedObserved = new Set<number>()
  const usedDesign = new Set<number>()
  const pairs: LocalPair[] = []
  for (const candidate of candidates) {
    if (usedObserved.has(candidate.observedPosition) || usedDesign.has(candidate.designPosition)) {
      continue
    }
    usedObserved.add(candidate.observedPosition)
    usedDesign.add(candidate.designPosition)
    pairs.push(candidate)
  }
  return { pairs, colorRejects }
}

function refitFromPairs(
  pairs: readonly LocalPair[],
  observedPoints: readonly Vec2[],
  designPoints: readonly PatternPoint[],
  anchors: readonly Anchor[],
  anchorGate: number,
): Mat3 | undefined {
  if (pairs.length < 4) {
    return undefined
  }
  const correspondences: HomographyCorrespondence[] = pairs.map((pair) => {
    const observedPoint = observedPoints[pair.observedPosition]!
    const designPoint = designPoints[pair.designPosition]!
    return {
      design: designPoint,
      observed: observedPoint,
      weight: 1,
    }
  })
  const fit = fitHomography(correspondences)
  if (fit === undefined) {
    return undefined
  }
  const centroid = homographyCentroid(pairs.map((pair) => designPoints[pair.designPosition]!))
  if (!isHomographyOrientationValid(fit, centroid)) {
    return undefined
  }
  const observedToDesign = invertMat3RowMajor(fit)
  if (observedToDesign === undefined) {
    return undefined
  }
  for (const anchor of anchors) {
    const predicted = applyToDesign(observedToDesign, anchor.observed)
    if (predicted === undefined || distance(predicted, anchor.design) > anchorGate) {
      return undefined
    }
  }
  return observedToDesign
}

function noRecoveryResult(
  input: LocalIcpInput,
  reason: IcpRefineResult['rejectionReason'],
  anchors: readonly Anchor[],
  proofPowers: readonly number[],
  usedProofPower: number,
  designSpacing: number,
): IcpRefineResult {
  const anchorByObservedPosition = new Map<number, Anchor>()
  for (const anchor of anchors) {
    anchorByObservedPosition.set(anchor.observedPosition, anchor)
  }
  const points: MatchedIdentityPoint[] = input.baseline.points.map((point, observedPosition) => {
    const anchor = anchorByObservedPosition.get(observedPosition)
    if (anchor !== undefined) {
      return {
        ...point,
        designId: anchor.designId,
        status: IDENTITY_RESOLVED,
        observedPosition,
        proofPower: proofPowers[observedPosition],
        source: 'hash',
      }
    }
    return {
      ...point,
      observedPosition,
      proofPower: proofPowers[observedPosition],
      source: point.designId === undefined ? undefined : 'hash-loose',
    }
  })
  const metrics: IcpRefineMetrics = {
    observed: input.observedGraph.points.length,
    anchors: anchors.length,
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
    neighborhoods: 0,
  }
  return {
    accepted: false,
    rejectionReason: reason,
    baseline: input.baseline,
    points,
    predictedMissing: [],
    predictedLattice: [],
    matches: [],
    anchors: [...anchors],
    metrics,
    usedProofPower,
  }
}

export function refineObservedIdentityWithLocalIcp(input: LocalIcpInput): IcpRefineResult {
  const { observedGraph, observedColors, options, designPattern, designColors, designGraph } = input
  const observed = observedGraph.points.length
  if (!options.enabled) {
    return noRecoveryResult(input, 'disabled', [], [], options.minProofPower, 0)
  }
  if (observed === 0) {
    return noRecoveryResult(input, 'too-few-observed', [], [], options.minProofPower, 0)
  }

  const availableDepths = Math.max(1, input.designIndexesByDepth.length)
  const identityDepth = Math.min(Math.max(1, options.identityDepth), availableDepths)
  const maxProofDepth = Math.min(Math.max(identityDepth, options.maxProofDepth), availableDepths)

  let observedFingerprints = input.observedFingerprints
  if (observedFingerprints === undefined) {
    const observedColorByGraphIndex: BinaryColor[] = []
    for (let position = 0; position < observedGraph.points.length; position += 1) {
      const point = observedGraph.points[position]!
      observedColorByGraphIndex[point.index] = observedColors[position] ?? 0
    }
    observedFingerprints = computeFingerprints(
      observedGraph,
      observedColorByGraphIndex,
      maxProofDepth,
    )
  }
  const designPointsByIndex = new Map<number, PatternPoint>()
  for (const point of designPattern.points) {
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
  const lattice = computeDesignLatticeMetrics(designPattern.points)
  const designSpacing = lattice.medianNearest

  if (designPattern.points.length < 4 || observed < 4 || !(designSpacing > 1e-9)) {
    return noRecoveryResult(
      input,
      'too-few-observed',
      anchorCollection.anchors,
      proofPowers,
      anchorCollection.usedProofPower,
      designSpacing,
    )
  }
  if (anchorCollection.anchors.length < Math.max(4, options.minAnchors)) {
    return noRecoveryResult(
      input,
      'too-few-anchors',
      anchorCollection.anchors,
      proofPowers,
      anchorCollection.usedProofPower,
      designSpacing,
    )
  }

  const anchors = anchorCollection.anchors
  const anchorByObservedPosition = new Map<number, Anchor>()
  const anchorDesignIds = new Set<number>()
  for (const anchor of anchors) {
    anchorByObservedPosition.set(anchor.observedPosition, anchor)
    anchorDesignIds.add(anchor.designId)
  }

  const ambiguousObservedPositions: number[] = []
  const reservedDesignIds = new Set<number>(anchorDesignIds)
  for (let position = 0; position < input.baseline.points.length; position += 1) {
    const point = input.baseline.points[position]!
    if (point.designId !== undefined) {
      reservedDesignIds.add(point.designId)
    }
    // Anchors are seeds, not refinement targets.
    if (anchorByObservedPosition.has(position)) {
      continue
    }
    if (point.status === IDENTITY_AMBIGUOUS || point.status === IDENTITY_UNRESOLVED) {
      ambiguousObservedPositions.push(position)
    }
  }
  if (ambiguousObservedPositions.length === 0) {
    return noRecoveryResult(
      input,
      'no-ambiguous-points',
      anchors,
      proofPowers,
      anchorCollection.usedProofPower,
      designSpacing,
    )
  }

  const gate = Math.max(0.1, lattice.minNearest * options.localGateFactor)
  const anchorGate = Math.max(0.1, designSpacing * options.localAnchorGateFactor)
  const designGrid = createDesignGrid(designPattern.points, designColors, gate)
  const designPositionById = createDesignPositionMap(designPattern.points)
  // Farthest-from-anchor first. A successful subgraph confirms every ambiguous
  // point it matched, so nearer points can be covered without their own run.
  const anchorDistance = new Int32Array(observed)
  anchorDistance.fill(-1)
  const distanceQueue: number[] = []
  for (const anchor of anchors) {
    const position = anchor.observedPosition
    if (position >= 0 && position < observed && anchorDistance[position] === -1) {
      anchorDistance[position] = 0
      distanceQueue.push(position)
    }
  }
  for (let head = 0; head < distanceQueue.length; head += 1) {
    const node = distanceQueue[head]!
    const nextDistance = anchorDistance[node]! + 1
    for (const neighbor of observedGraph.neighbors[node] ?? []) {
      if (anchorDistance[neighbor] === -1) {
        anchorDistance[neighbor] = nextDistance
        distanceQueue.push(neighbor)
      }
    }
  }
  const ambiguousSet = new Set(ambiguousObservedPositions)
  ambiguousObservedPositions.sort((first, second) => {
    const firstDistance = anchorDistance[first] ?? -1
    const secondDistance = anchorDistance[second] ?? -1
    if (firstDistance < 0 && secondDistance < 0) {
      return first - second
    }
    if (firstDistance < 0) {
      return 1
    }
    if (secondDistance < 0) {
      return -1
    }
    if (firstDistance !== secondDistance) {
      return secondDistance - firstDistance
    }
    return first - second
  })

  const registrations: LocalRegistration[] = []
  const choices: LocalChoice[] = []
  const choiceByObserved = new Map<number, LocalChoice>()
  const usedDesign = new Set<number>(reservedDesignIds)
  let anchorInliers = 0
  let colorRejects = 0
  let maxIterations = 0

  for (const observedPosition of ambiguousObservedPositions) {
    const nearestAnchorDistance = anchorDistance[observedPosition] ?? -1
    if (nearestAnchorDistance < 0 || nearestAnchorDistance > options.hopRadius) {
      continue
    }
    if (choiceByObserved.has(observedPosition)) {
      continue
    }
    const localObservedPositions = collectHopGraph(
      observedGraph,
      observedPosition,
      options.hopRadius,
    )
    const localAnchors: Anchor[] = []
    for (const position of localObservedPositions) {
      const anchor = anchorByObservedPosition.get(position)
      if (anchor !== undefined) {
        localAnchors.push(anchor)
      }
    }
    if (localAnchors.length < Math.max(4, options.minLocalAnchors)) {
      continue
    }
    const anchorFit = fitFromAnchors(
      localAnchors,
      designSpacing,
      gate,
      Math.max(4, options.minLocalAnchors),
    )
    if (anchorFit === undefined) {
      continue
    }

    let observedToDesign = anchorFit.observedToDesign
    let iterations = 0
    let lastKey = ''
    for (let iteration = 0; iteration < options.maxLocalIterations; iteration += 1) {
      const match = matchLocalSupport(
        observedToDesign,
        localObservedPositions,
        observedGraph.points,
        observedColors,
        designPattern.points,
        designGrid,
        gate,
      )
      colorRejects += match.colorRejects
      if (match.pairs.length < 4) {
        break
      }
      const next = refitFromPairs(
        match.pairs,
        observedGraph.points,
        designPattern.points,
        localAnchors,
        anchorGate,
      )
      if (next === undefined) {
        break
      }
      const key = pairKey(match.pairs)
      observedToDesign = next
      iterations += 1
      if (key === lastKey) {
        break
      }
      lastKey = key
    }

    const finalMatch = matchLocalSupport(
      observedToDesign,
      localObservedPositions,
      observedGraph.points,
      observedColors,
      designPattern.points,
      designGrid,
      gate,
    )
    colorRejects += finalMatch.colorRejects
    const rootPair = finalMatch.pairs.find(
      (pair) => pair.observedPosition === observedPosition,
    )
    if (rootPair === undefined) {
      continue
    }
    const rootDesignPoint = designPattern.points[rootPair.designPosition]!
    if (usedDesign.has(rootDesignPoint.index)) {
      continue
    }

    anchorInliers += anchorFit.inliers
    maxIterations = Math.max(maxIterations, iterations)
    registrations.push({
      observedPosition,
      designPosition: rootPair.designPosition,
      designId: rootDesignPoint.index,
      distance: rootPair.distance,
      observedToDesign,
      iterations,
      anchorInliers: anchorFit.inliers,
      colorRejects: finalMatch.colorRejects,
    })

    usedDesign.add(rootDesignPoint.index)
    choiceByObserved.set(observedPosition, {
      observedPosition,
      designId: rootDesignPoint.index,
      distance: rootPair.distance,
      observedToDesign,
    })
    choices.push(choiceByObserved.get(observedPosition)!)

    // The root matched, so confirm the rest of this local subgraph. Only
    // ambiguous, unconfirmed points may be claimed; hash-resolved points and
    // anchors remain reserved.
    for (const pair of finalMatch.pairs) {
      if (pair.observedPosition === observedPosition) {
        continue
      }
      if (choiceByObserved.has(pair.observedPosition)) {
        continue
      }
      if (anchorByObservedPosition.has(pair.observedPosition)) {
        continue
      }
      if (!ambiguousSet.has(pair.observedPosition)) {
        continue
      }
      const baselinePoint = input.baseline.points[pair.observedPosition]
      if (baselinePoint?.designId !== undefined) {
        continue
      }
      const designPoint = designPattern.points[pair.designPosition]!
      if (usedDesign.has(designPoint.index)) {
        continue
      }
      usedDesign.add(designPoint.index)
      const choice: LocalChoice = {
        observedPosition: pair.observedPosition,
        designId: designPoint.index,
        distance: pair.distance,
        observedToDesign,
      }
      choiceByObserved.set(pair.observedPosition, choice)
      choices.push(choice)
    }
  }

  if (choices.length === 0) {
    return noRecoveryResult(
      input,
      'local-icp-failed',
      anchors,
      proofPowers,
      anchorCollection.usedProofPower,
      designSpacing,
    )
  }

  const strictResolved = input.baseline.points.filter(
    (point) => point.designId !== undefined,
  ).length
  let geometryConflicts = 0
  const points: MatchedIdentityPoint[] = input.baseline.points.map((base, observedPosition) => {
    const choice = choiceByObserved.get(observedPosition)
    const proofPower = proofPowers[observedPosition]
    const anchor = anchorByObservedPosition.get(observedPosition)
    if (anchor !== undefined) {
      return {
        ...base,
        designId: anchor.designId,
        status: IDENTITY_RESOLVED,
        observedPosition,
        proofPower,
        source: 'hash',
      }
    }
    if (choice === undefined) {
      return {
        ...base,
        observedPosition,
        proofPower,
        source: base.designId === undefined ? undefined : 'hash-loose',
      }
    }
    const geometryConflict =
      base.designId !== undefined && base.designId !== choice.designId
    if (geometryConflict) {
      geometryConflicts += 1
    }
    return {
      ...base,
      designId: choice.designId,
      status: IDENTITY_RECOVERED,
      observedPosition,
      proofPower,
      source: 'geometry',
      designDistance: choice.distance,
      geometryConflict,
      recoveryVotes: 1,
    }
  })

  const predictedById = new Map<number, PredictedLatticePoint>()
  for (const registration of registrations) {
    const designToObserved = invertMat3RowMajor(registration.observedToDesign)
    if (designToObserved === undefined) {
      continue
    }
    const localDesignPositions = collectHopGraph(designGraph, registration.designPosition, 1)
    for (const designPosition of localDesignPositions) {
      const designPoint = designPattern.points[designPosition]!
      if (predictedById.has(designPoint.index)) {
        continue
      }
      const predicted = applyHomography(designToObserved, designPoint.x, designPoint.y)
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
      predictedById.set(designPoint.index, {
        designId: designPoint.index,
        predicted: { x: predicted.x, y: predicted.y },
        color: designColors[designPosition] ?? 0,
      })
    }
  }
  const predictedLattice = [...predictedById.values()]
  const assignedDesignIds = new Set(choices.map((choice) => choice.designId))
  const unavailableDesignIds = new Set<number>([...reservedDesignIds, ...assignedDesignIds])
  const predictedMissing: PredictedMissingPoint[] = []
  for (const prediction of predictedLattice) {
    if (unavailableDesignIds.has(prediction.designId)) {
      continue
    }
    const designPosition = designPositionById.get(prediction.designId)
    if (designPosition === undefined) {
      continue
    }
    const neighborPositions = designGraph.neighbors[designPosition] ?? []
    const supported = neighborPositions.some((neighborPosition) => {
      const neighbor = designPattern.points[neighborPosition]!
      return assignedDesignIds.has(neighbor.index)
    })
    if (!supported) {
      continue
    }
    predictedMissing.push(prediction)
  }

  const matches: IcpPair[] = choices.map((choice) => ({
    observedPosition: choice.observedPosition,
    designId: choice.designId,
    designDistance: choice.distance,
  }))
  const squared = matches.reduce(
    (sum, match) => sum + match.designDistance * match.designDistance,
    0,
  )
  const metrics: IcpRefineMetrics = {
    observed,
    anchors: anchors.length,
    initialInliers: anchorInliers,
    pairs: matches.length,
    pairCoverage: observed === 0 ? 0 : matches.length / observed,
    designSpacing,
    designRms: matches.length === 0 ? Number.POSITIVE_INFINITY : Math.sqrt(squared / matches.length),
    colorRejects,
    recovered: matches.length,
    geometryConflicts,
    predictedMissing: predictedMissing.length,
    strictResolved,
    validationAnchors: 0,
    iterations: maxIterations,
    usedProofPower: anchorCollection.usedProofPower,
    neighborhoods: registrations.length,
  }

  return {
    accepted: true,
    baseline: input.baseline,
    points,
    predictedMissing,
    predictedLattice,
    matches,
    anchors: [...anchors],
    metrics,
    usedProofPower: anchorCollection.usedProofPower,
  }
}
