/**
 * Legacy global design-space ICP. The active Point Lab path uses hop-radius
 * per-ambiguous-point local registration in `localIcpRefiner.ts`.
 */
import { invertMat3RowMajor } from '@/lib/geometry'
import { applyHomography, type Mat3 } from '@/lib/geometry'
import type { Anchor } from '@/lib/pointIdentity/anchors'
import { computeDesignLatticeMetrics, createDesignGrid, createDesignPositionMap } from '@/lib/pointIdentity/designGrid'
import {
  designResiduals,
  fitHomography,
  hasAnchorSpread,
  homographyCentroid,
  isHomographyOrientationValid,
  robustHomography,
  type HomographyCorrespondence,
} from '@/lib/pointIdentity/homographyFit'
import type { BinaryColor, PatternPoint, Vec2 } from '@/lib/pointIdentity/types'

/** Planar distance. `Math.hypot` is overflow-safe but markedly slower, and
 *  coordinates here are pixel-scale, so the plain form is safe and faster. */
function planarDistance(a: Vec2, b: Vec2): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.sqrt(dx * dx + dy * dy)
}

export type IcpPair = {
  observedPosition: number
  designId: number
  designDistance: number
}

export type IcpOptions = {
  maxIterations: number
  designGateFactor: number
  anchorGateFactor: number
  minInitialInliers: number
  minInitialInlierRatio: number
  minPairCoverage: number
  maxDesignRmsFactor: number
  maxColorRejectRate: number
  holdoutAnchorFraction: number
  minObservedSpacingPx: number
  seed: number
}

export type IcpIterationResult = {
  iteration: number
  pairCount: number
  designRms: number
  fitAnchorMaxResidual: number
}

export type IcpRejectionReason =
  | 'too-few-design-points'
  | 'degenerate-lattice'
  | 'too-few-anchors'
  | 'initial-fit-failed'
  | 'degenerate-anchors'
  | 'invalid-transform'
  | 'icp-failed'
  | 'anchor-drift'
  | 'validation-anchor-failed'
  | 'low-pair-coverage'
  | 'high-design-rms'
  | 'too-many-color-rejects'
  | 'too-many-conflicts'
  | 'too-small-target'

export type IcpResult = {
  accepted: boolean
  rejectionReason?: IcpRejectionReason
  matrix?: Mat3
  initialMatrix?: Mat3
  pairs: IcpPair[]
  iterations: IcpIterationResult[]
  designSpacing: number
  gate: number
  anchorGate: number
  fitAnchors: Anchor[]
  validationAnchors: Anchor[]
  fitAnchorMaxResidual: number
  validationAnchorMaxResidual?: number
  designRms: number
  colorRejects: number
  pairCoverage: number
  observedSpacingPx: number
  /** Design ids with an in-gate observed candidate, regardless of color. */
  observedCandidateDesignIds: number[]
  /** Design ids with an in-gate observed candidate but no accepted pair. */
  blockedDesignIds: number[]
}

export type IcpInput = {
  designPoints: readonly PatternPoint[]
  designColors: readonly BinaryColor[]
  observedPoints: readonly Vec2[]
  observedColors: readonly BinaryColor[]
  anchors: readonly Anchor[]
  options: IcpOptions
}

type MatchPass = {
  pairs: IcpPair[]
  colorRejects: number
  observedCandidateDesignIds: Set<number>
}

function splitAnchors(
  anchors: readonly Anchor[],
  holdoutFraction: number,
): { fitAnchors: Anchor[]; validationAnchors: Anchor[] } {
  if (anchors.length < 12 || holdoutFraction <= 0) {
    return { fitAnchors: [...anchors], validationAnchors: [] }
  }
  const every = Math.max(2, Math.round(1 / holdoutFraction))
  const fitAnchors: Anchor[] = []
  const validationAnchors: Anchor[] = []
  for (let index = 0; index < anchors.length; index += 1) {
    if (index % every === 0) {
      validationAnchors.push(anchors[index]!)
    } else {
      fitAnchors.push(anchors[index]!)
    }
  }
  if (fitAnchors.length < 6 || validationAnchors.length < 4) {
    return { fitAnchors: [...anchors], validationAnchors: [] }
  }
  return { fitAnchors, validationAnchors }
}

function anchorCorrespondence(anchor: Anchor): HomographyCorrespondence {
  return { design: anchor.design, observed: anchor.observed }
}

function medianNearestDistance(points: readonly Vec2[]): number {
  if (points.length < 2) {
    return 0
  }
  const nearest: number[] = []
  for (let first = 0; first < points.length; first += 1) {
    const origin = points[first]!
    let minimum = Number.POSITIVE_INFINITY
    for (let second = 0; second < points.length; second += 1) {
      if (first === second) {
        continue
      }
      const candidate = points[second]!
      minimum = Math.min(minimum, planarDistance(origin, candidate))
    }
    if (Number.isFinite(minimum)) {
      nearest.push(minimum)
    }
  }
  if (nearest.length === 0) {
    return 0
  }
  nearest.sort((first, second) => first - second)
  const middle = nearest.length >> 1
  return nearest.length % 2 === 1 ? nearest[middle]! : (nearest[middle - 1]! + nearest[middle]!) / 2
}

function designDistance(predicted: Vec2, target: Vec2): number {
  return planarDistance(predicted, target)
}

function matchPass(
  matrix: Mat3,
  designPoints: readonly PatternPoint[],
  designColors: readonly BinaryColor[],
  designPositionById: ReadonlyMap<number, number>,
  designGrid: ReturnType<typeof createDesignGrid>,
  observedPoints: readonly Vec2[],
  observedColors: readonly BinaryColor[],
  forcedAnchors: readonly Anchor[],
  gate: number,
): MatchPass | undefined {
  const inverse = invertMat3RowMajor(matrix)
  if (inverse === undefined) {
    return undefined
  }

  const observedInDesign = observedPoints.map((point) =>
    applyHomography(inverse, point.x, point.y),
  )
  const usedObserved = new Uint8Array(observedPoints.length)
  const usedDesign = new Uint8Array(designPoints.length)
  const pairs: IcpPair[] = []
  const observedCandidateDesignIds = new Set<number>()

  for (const anchor of forcedAnchors) {
    const observed = observedInDesign[anchor.observedPosition]
    const designPosition = designPositionById.get(anchor.designId)
    if (observed === undefined || designPosition === undefined) {
      return undefined
    }
    const distance = designDistance(observed, anchor.design)
    if (distance > gate) {
      return undefined
    }
    if (usedObserved[anchor.observedPosition] === 1 || usedDesign[designPosition] === 1) {
      return undefined
    }
    usedObserved[anchor.observedPosition] = 1
    usedDesign[designPosition] = 1
    observedCandidateDesignIds.add(anchor.designId)
    pairs.push({
      observedPosition: anchor.observedPosition,
      designId: anchor.designId,
      designDistance: distance,
    })
  }

  const candidates: Array<IcpPair & { colorMatch: boolean }> = []
  let colorRejects = 0
  for (let observedPosition = 0; observedPosition < observedPoints.length; observedPosition += 1) {
    if (usedObserved[observedPosition] === 1) {
      continue
    }
    const point = observedInDesign[observedPosition]!
    const nearby = designGrid.within(point, gate)
    if (nearby.length === 0) {
      continue
    }
    for (const position of nearby) {
      observedCandidateDesignIds.add(designPoints[position]!.index)
    }
    const observedColor = observedColors[observedPosition] ?? 0
    let bestPosition: number | undefined
    let bestDistance = Number.POSITIVE_INFINITY
    for (const position of nearby) {
      if ((designColors[position] ?? 0) !== observedColor) {
        continue
      }
      const distance = designDistance(point, designPoints[position]!)
      if (
        distance < bestDistance ||
        (distance === bestDistance &&
          bestPosition !== undefined &&
          designPoints[position]!.index < designPoints[bestPosition]!.index)
      ) {
        bestDistance = distance
        bestPosition = position
      }
    }
    if (bestPosition === undefined) {
      colorRejects += 1
      continue
    }
    candidates.push({
      observedPosition,
      designId: designPoints[bestPosition]!.index,
      designDistance: bestDistance,
      colorMatch: true,
    })
  }

  candidates.sort((first, second) => {
    if (first.designDistance !== second.designDistance) {
      return first.designDistance - second.designDistance
    }
    if (first.observedPosition !== second.observedPosition) {
      return first.observedPosition - second.observedPosition
    }
    return first.designId - second.designId
  })

  for (const candidate of candidates) {
    if (usedObserved[candidate.observedPosition] === 1) {
      continue
    }
    const designPosition = designPositionById.get(candidate.designId)
    if (designPosition === undefined || usedDesign[designPosition] === 1) {
      continue
    }
    usedObserved[candidate.observedPosition] = 1
    usedDesign[designPosition] = 1
    pairs.push({
      observedPosition: candidate.observedPosition,
      designId: candidate.designId,
      designDistance: candidate.designDistance,
    })
  }

  return { pairs, colorRejects, observedCandidateDesignIds }
}

function pairKey(pairs: readonly IcpPair[]): string {
  return pairs
    .map((pair) => `${pair.observedPosition}:${pair.designId}`)
    .sort()
    .join('|')
}

function pairResiduals(
  matrix: Mat3,
  pairs: readonly IcpPair[],
  designPoints: readonly PatternPoint[],
  designPositionById: ReadonlyMap<number, number>,
  observedPoints: readonly Vec2[],
): number[] {
  const correspondences: HomographyCorrespondence[] = []
  for (const pair of pairs) {
    const designPosition = designPositionById.get(pair.designId)
    const design = designPosition === undefined ? undefined : designPoints[designPosition]
    const observed = observedPoints[pair.observedPosition]
    if (design === undefined || observed === undefined) {
      continue
    }
    correspondences.push({ design, observed })
  }
  return designResiduals(matrix, correspondences)
}

function rms(values: readonly number[]): number {
  const finite = values.filter((value) => Number.isFinite(value))
  if (finite.length === 0) {
    return Number.POSITIVE_INFINITY
  }
  return Math.sqrt(finite.reduce((sum, value) => sum + value * value, 0) / finite.length)
}

export function runDesignSpaceIcp(input: IcpInput): IcpResult {
  const { designPoints, designColors, observedPoints, observedColors, anchors, options } = input
  const lattice = computeDesignLatticeMetrics(designPoints)
  const designSpacing = lattice.medianNearest
  const gateFactor = Math.min(0.49, Math.max(0.1, options.designGateFactor))
  const gate = Math.max(1e-6, lattice.minNearest * gateFactor)
  const anchorGate = Math.max(1e-6, designSpacing * options.anchorGateFactor)
  const emptyResult = (reason: IcpRejectionReason): IcpResult => ({
    accepted: false,
    rejectionReason: reason,
    pairs: [],
    iterations: [],
    designSpacing,
    gate,
    anchorGate,
    fitAnchors: [],
    validationAnchors: [],
    fitAnchorMaxResidual: Number.POSITIVE_INFINITY,
    designRms: Number.POSITIVE_INFINITY,
    colorRejects: 0,
    pairCoverage: 0,
    observedSpacingPx: medianNearestDistance(observedPoints),
    observedCandidateDesignIds: [],
    blockedDesignIds: [],
  })

  if (designPoints.length < 4 || observedPoints.length < 4) {
    return emptyResult('too-few-design-points')
  }
  if (!(designSpacing > 1e-9) || !(lattice.minNearest > 1e-9)) {
    return emptyResult('degenerate-lattice')
  }
  if (anchors.length < 6) {
    return emptyResult('too-few-anchors')
  }

  const observedSpacingPx = medianNearestDistance(observedPoints)
  if (observedSpacingPx > 0 && observedSpacingPx < options.minObservedSpacingPx) {
    return emptyResult('too-small-target')
  }

  const { fitAnchors, validationAnchors } = splitAnchors(anchors, options.holdoutAnchorFraction)
  if (fitAnchors.length < 6) {
    return emptyResult('too-few-anchors')
  }

  const initial = robustHomography(fitAnchors.map(anchorCorrespondence), {
    designSpacing,
    minInliers: options.minInitialInliers,
    minInlierRatio: options.minInitialInlierRatio,
    seed: options.seed,
    trimFloor: designSpacing * 0.1,
    inlierFactor: 0.3,
    ransacIterations: 256,
  })
  if (initial === undefined) {
    return emptyResult('initial-fit-failed')
  }

  const initialAnchors = initial.inlierIndices
    .map((index) => fitAnchors[index])
    .filter((anchor): anchor is Anchor => anchor !== undefined)
  if (initialAnchors.length < 6) {
    return emptyResult('initial-fit-failed')
  }
  if (!hasAnchorSpread(initialAnchors.map((anchor) => anchor.design), designSpacing)) {
    return emptyResult('degenerate-anchors')
  }
  const designCentroid = homographyCentroid(designPoints)
  if (!isHomographyOrientationValid(initial.matrix, designCentroid)) {
    return emptyResult('invalid-transform')
  }

  const designPositionById = createDesignPositionMap(designPoints)
  const designGrid = createDesignGrid(designPoints, designColors, gate)
  const fitAnchorCorrespondences = initialAnchors.map(anchorCorrespondence)
  const validationCorrespondences = validationAnchors.map(anchorCorrespondence)

  let matrix = initial.matrix
  let lastPairKey = ''
  const iterations: IcpIterationResult[] = []
  let finalPairs: IcpPair[] = []
  let finalColorRejects = 0
  let finalCandidateDesignIds = new Set<number>()

  for (let iteration = 0; iteration < options.maxIterations; iteration += 1) {
    const match = matchPass(
      matrix,
      designPoints,
      designColors,
      designPositionById,
      designGrid,
      observedPoints,
      observedColors,
      initialAnchors,
      gate,
    )
    if (match === undefined) {
      return emptyResult('anchor-drift')
    }
    if (match.pairs.length < 6) {
      return emptyResult('icp-failed')
    }

    const correspondences: HomographyCorrespondence[] = []
    const forcedObserved = new Set(initialAnchors.map((anchor) => anchor.observedPosition))
    const validationObserved = new Set(validationAnchors.map((anchor) => anchor.observedPosition))
    for (const pair of match.pairs) {
      if (validationObserved.has(pair.observedPosition)) {
        continue
      }
      const designPosition = designPositionById.get(pair.designId)
      const design = designPosition === undefined ? undefined : designPoints[designPosition]
      const observed = observedPoints[pair.observedPosition]
      if (design === undefined || observed === undefined) {
        continue
      }
      correspondences.push({
        design,
        observed,
        weight: forcedObserved.has(pair.observedPosition) ? 4 : 1,
      })
    }
    if (correspondences.length < 4) {
      return emptyResult('icp-failed')
    }

    const next = fitHomography(correspondences)
    if (next === undefined) {
      return emptyResult('icp-failed')
    }

    const fitAnchorResiduals = designResiduals(next, fitAnchorCorrespondences)
    const fitAnchorMaxResidual = fitAnchorResiduals.length === 0
      ? Number.POSITIVE_INFINITY
      : Math.max(...fitAnchorResiduals)
    if (!Number.isFinite(fitAnchorMaxResidual) || fitAnchorMaxResidual > anchorGate) {
      return emptyResult('anchor-drift')
    }

    const residuals = pairResiduals(next, match.pairs, designPoints, designPositionById, observedPoints)
    iterations.push({
      iteration,
      pairCount: match.pairs.length,
      designRms: rms(residuals),
      fitAnchorMaxResidual,
    })

    const nextKey = pairKey(match.pairs)
    matrix = next
    finalPairs = match.pairs
    finalColorRejects = match.colorRejects
    finalCandidateDesignIds = match.observedCandidateDesignIds
    if (nextKey === lastPairKey) {
      break
    }
    lastPairKey = nextKey
  }

  // Recompute the final matching against the final transform for consistent metrics.
  const finalMatch = matchPass(
    matrix,
    designPoints,
    designColors,
    designPositionById,
    designGrid,
    observedPoints,
    observedColors,
    initialAnchors,
    gate,
  )
  if (finalMatch === undefined) {
    return emptyResult('anchor-drift')
  }
  finalPairs = finalMatch.pairs
  finalColorRejects = finalMatch.colorRejects
  finalCandidateDesignIds = finalMatch.observedCandidateDesignIds

  const finalResiduals = pairResiduals(matrix, finalPairs, designPoints, designPositionById, observedPoints)
  const designRms = rms(finalResiduals)
  const finalFitAnchorResiduals = designResiduals(matrix, fitAnchorCorrespondences)
  const fitAnchorMaxResidual = finalFitAnchorResiduals.length === 0
    ? Number.POSITIVE_INFINITY
    : Math.max(...finalFitAnchorResiduals)

  const pairByObserved = new Map<number, IcpPair>()
  for (const pair of finalPairs) {
    pairByObserved.set(pair.observedPosition, pair)
  }
  for (const anchor of initialAnchors) {
    const pair = pairByObserved.get(anchor.observedPosition)
    if (pair === undefined || pair.designId !== anchor.designId) {
      return emptyResult('anchor-drift')
    }
  }

  let validationAnchorMaxResidual: number | undefined
  if (validationAnchors.length > 0) {
    const validationResiduals = designResiduals(matrix, validationCorrespondences)
    validationAnchorMaxResidual = validationResiduals.length === 0
      ? Number.POSITIVE_INFINITY
      : Math.max(...validationResiduals)
    // Proof-power anchors can still contain a small number of false matches
    // under heavy dropout. The holdout is a consensus check, not a per-point
    // veto: require most held-out anchors to validate while allowing a minority
    // to be hash failures.
    const requiredPasses = Math.max(2, Math.ceil(validationAnchors.length * 0.8))
    let residualPasses = 0
    for (const residual of validationResiduals) {
      if (Number.isFinite(residual) && residual <= anchorGate) {
        residualPasses += 1
      }
    }
    let pairPasses = 0
    for (const anchor of validationAnchors) {
      const pair = pairByObserved.get(anchor.observedPosition)
      if (pair !== undefined && pair.designId === anchor.designId) {
        pairPasses += 1
      }
    }
    if (residualPasses < requiredPasses || pairPasses < requiredPasses) {
      return emptyResult('validation-anchor-failed')
    }
  }

  const pairCoverage = observedPoints.length === 0 ? 0 : finalPairs.length / observedPoints.length
  if (pairCoverage < options.minPairCoverage) {
    return {
      ...emptyResult('low-pair-coverage'),
      matrix,
      initialMatrix: initial.matrix,
      pairs: finalPairs,
      iterations,
      fitAnchors: initialAnchors,
      validationAnchors,
      fitAnchorMaxResidual,
      validationAnchorMaxResidual,
      designRms,
      colorRejects: finalColorRejects,
      pairCoverage,
      observedSpacingPx,
      observedCandidateDesignIds: [...finalCandidateDesignIds].sort((first, second) => first - second),
    }
  }
  if (!Number.isFinite(designRms) || designRms > designSpacing * options.maxDesignRmsFactor) {
    return {
      ...emptyResult('high-design-rms'),
      matrix,
      initialMatrix: initial.matrix,
      pairs: finalPairs,
      iterations,
      fitAnchors: initialAnchors,
      validationAnchors,
      fitAnchorMaxResidual,
      validationAnchorMaxResidual,
      designRms,
      colorRejects: finalColorRejects,
      pairCoverage,
      observedSpacingPx,
      observedCandidateDesignIds: [...finalCandidateDesignIds].sort((first, second) => first - second),
    }
  }
  if (observedPoints.length > 0 && finalColorRejects / observedPoints.length > options.maxColorRejectRate) {
    return emptyResult('too-many-color-rejects')
  }
  if (!isHomographyOrientationValid(matrix, designCentroid)) {
    return emptyResult('invalid-transform')
  }

  const pairedDesignIds = new Set(finalPairs.map((pair) => pair.designId))
  const blockedDesignIds = [...finalCandidateDesignIds]
    .filter((designId) => !pairedDesignIds.has(designId))
    .sort((first, second) => first - second)

  return {
    accepted: true,
    matrix,
    initialMatrix: initial.matrix,
    pairs: finalPairs,
    iterations,
    designSpacing,
    gate,
    anchorGate,
    fitAnchors: initialAnchors,
    validationAnchors,
    fitAnchorMaxResidual,
    validationAnchorMaxResidual,
    designRms,
    colorRejects: finalColorRejects,
    pairCoverage,
    observedSpacingPx,
    observedCandidateDesignIds: [...finalCandidateDesignIds].sort((first, second) => first - second),
    blockedDesignIds,
  }
}
