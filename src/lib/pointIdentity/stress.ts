import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import {
  computeFingerprints,
  createIdentityIndex,
  createIdentityIndexesByDepth,
  evaluateIdentity,
} from '@/lib/pointIdentity/identity'
import {
  createLocalIcpOptions,
  refineObservedIdentityWithLocalIcp,
} from '@/lib/pointIdentity/localIcpRefiner'
import { IDENTITY_RECOVERED, matchObservedIdentity } from '@/lib/pointIdentity/match'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { createRandom, randomBinaryColors } from '@/lib/pointIdentity/random'

export type IdentityStressOptions = {
  runs: number
  depth: number
  pointsPerEdge: number
  columns?: number
  baseSeed?: number
  yawDeg?: number
  pitchDeg?: number
  cameraDistance?: number
  sphericalDistortion?: number
  dropoutPercent?: number
  dropoutSeed?: number
  cropLeft?: number
  cropRight?: number
  cropTop?: number
  cropBottom?: number
  jitterPercent?: number
  recovery?: boolean
  minProofPower?: number
  maxLocalIterations?: number
  hopRadius?: number
}

export type IdentityRecoveryStressReport = {
  acceptedRuns: number
  rejectedRuns: number
  meanRefinedAccuracy: number
  minRefinedAccuracy: number
  meanAcceptedRefinedAccuracy: number
  minAcceptedRefinedAccuracy: number
  refinedFalseMatches: number
  meanRecovered: number
  meanPredictedMissing: number
  predictedMissingFalse: number
  meanAnchors: number
  meanInliers: number
  meanPairs: number
  meanDesignRms: number
  meanConflicts: number
  rejectionReasons: Record<string, number>
}

export type IdentityStressReport = {
  runs: number
  depth: number
  pointsPerEdge: number
  view: {
    yawDeg: number
    pitchDeg: number
    cameraDistance: number
    sphericalDistortion: number
  }
  meanAccuracy: number
  minAccuracy: number
  maxAccuracy: number
  perfectRuns: number
  atLeast99Runs: number
  falseMatchRuns: number
  totalFalseMatches: number
  meanDesignCollisions: number
  maxDesignCollisions: number
  meanDetectedPoints: number
  recovery?: IdentityRecoveryStressReport
}

export function runIdentityStress(options: IdentityStressOptions): IdentityStressReport {
  const columns = options.columns ?? 8
  const pointsPerEdge = options.pointsPerEdge
  const depth = options.depth
  const baseSeed = options.baseSeed ?? 1
  const pattern = generateHexPattern({ columns, pointsPerEdge })
  const tuning = { pointsPerEdge }
  const designGraph = buildProximityGraph(pattern.points, tuning)
  const pose = {
    ...DEFAULT_VIEW,
    yawDeg: options.yawDeg ?? 0,
    pitchDeg: options.pitchDeg ?? 0,
    cameraDistance: options.cameraDistance ?? DEFAULT_VIEW.cameraDistance,
    sphericalDistortion: options.sphericalDistortion ?? 0,
  }
  const projected = projectPattern(pattern.points, pose)
  const observationMask = {
    ...DEFAULT_OBSERVATION_MASK,
    dropoutPercent: options.dropoutPercent ?? 0,
    dropoutSeed: options.dropoutSeed ?? 1,
    cropLeft: options.cropLeft ?? 0,
    cropRight: options.cropRight ?? 0,
    cropTop: options.cropTop ?? 0,
    cropBottom: options.cropBottom ?? 0,
  }
  const jitterPercent = options.jitterPercent ?? 0
  const perRunObservation = observationMask.dropoutPercent > 0 || jitterPercent > 0
  const fixedObservedPoints = perRunObservation
    ? undefined
    : applyObservationMask(projected, observationMask)
  const fixedObservedGraph =
    fixedObservedPoints === undefined ? undefined : buildProximityGraph(fixedObservedPoints, tuning)

  let projectedMinX = Number.POSITIVE_INFINITY
  let projectedMaxX = Number.NEGATIVE_INFINITY
  let projectedMinY = Number.POSITIVE_INFINITY
  let projectedMaxY = Number.NEGATIVE_INFINITY
  for (const point of projected) {
    projectedMinX = Math.min(projectedMinX, point.x)
    projectedMaxX = Math.max(projectedMaxX, point.x)
    projectedMinY = Math.min(projectedMinY, point.y)
    projectedMaxY = Math.max(projectedMaxY, point.y)
  }

  let accuracySum = 0
  let minAccuracy = 1
  let maxAccuracy = 0
  let perfectRuns = 0
  let atLeast99Runs = 0
  let falseMatchRuns = 0
  let totalFalseMatches = 0
  let designCollisionSum = 0
  let maxDesignCollisions = 0
  let detectedPointSum = 0

  let recoveryAcceptedRuns = 0
  let recoveryRejectedRuns = 0
  let refinedAccuracySum = 0
  let minRefinedAccuracy = 1
  let acceptedCorrectSum = 0
  let acceptedObservedSum = 0
  let minAcceptedRefinedAccuracy = 1
  let refinedFalseMatches = 0
  let recoveredSum = 0
  let predictedMissingSum = 0
  let predictedMissingFalse = 0
  let anchorSum = 0
  let inlierSum = 0
  let pairSum = 0
  let designRmsSum = 0
  let conflictSum = 0
  const rejectionReasons: Record<string, number> = {}

  const maxProofDepth = Math.min(12, Math.max(depth, 8))
  for (let run = 0; run < options.runs; run += 1) {
    const colors = randomBinaryColors(pattern.points.length, baseSeed + run)
    const indexesByDepth =
      options.recovery === true
        ? createIdentityIndexesByDepth(designGraph, colors, maxProofDepth)
        : []
    const designIndex =
      indexesByDepth[depth - 1] ?? createIdentityIndex(designGraph, colors, depth)
    const baseObservedPoints = perRunObservation
      ? applyObservationMask(projected, {
          ...observationMask,
          dropoutSeed: observationMask.dropoutSeed + run,
        })
      : fixedObservedPoints!
    let observedPoints = baseObservedPoints
    if (jitterPercent > 0) {
      const random = createRandom((observationMask.dropoutSeed + run + 0x51ed270b) >>> 0)
      const spacing = designGraph.stats.localScaleMedian
      const amplitude = spacing * (jitterPercent / 100)
      observedPoints = observedPoints.map((point) => ({
        ...point,
        x: point.x + (random() * 2 - 1) * amplitude,
        y: point.y + (random() * 2 - 1) * amplitude,
      }))
    }
    const observedGraph = perRunObservation
      ? buildProximityGraph(observedPoints, tuning)
      : fixedObservedGraph!
    const fingerprintDepth = options.recovery === true ? maxProofDepth : depth
    const observedFingerprints = computeFingerprints(observedGraph, colors, fingerprintDepth)
    const sharedFingerprints = fingerprintDepth === depth ? observedFingerprints : undefined
    const evaluation = evaluateIdentity(
      designGraph,
      observedGraph,
      designIndex,
      colors,
      depth,
      sharedFingerprints,
    )
    const summary = evaluation.summary
    accuracySum += summary.accuracy
    minAccuracy = Math.min(minAccuracy, summary.accuracy)
    maxAccuracy = Math.max(maxAccuracy, summary.accuracy)
    if (summary.accuracy === 1) {
      perfectRuns += 1
    }
    if (summary.accuracy >= 0.99) {
      atLeast99Runs += 1
    }
    if (summary.falseMatch > 0) {
      falseMatchRuns += 1
      totalFalseMatches += summary.falseMatch
    }
    designCollisionSum += summary.designCollisions
    maxDesignCollisions = Math.max(maxDesignCollisions, summary.designCollisions)
    detectedPointSum += observedPoints.length

    if (options.recovery === true) {
      const observedColors = observedGraph.points.map((point) => colors[point.index] ?? 0)
      const observedColorByGraphIndex = [...colors]
      const baseline = matchObservedIdentity(
        indexesByDepth[depth - 1]!,
        observedGraph,
        observedColorByGraphIndex,
        depth,
        { fingerprints: sharedFingerprints },
      )
      const refined = refineObservedIdentityWithLocalIcp({
        designPattern: pattern,
        designColors: colors,
        designGraph,
        designIndexesByDepth: indexesByDepth,
        observedGraph,
        observedColors,
        observedFingerprints,
        baseline,
        options: createLocalIcpOptions({
          identityDepth: depth,
          maxProofDepth,
          minProofPower: options.minProofPower ?? 7,
          maxLocalIterations: options.maxLocalIterations ?? 3,
          hopRadius: options.hopRadius ?? 8,
          frameBounds: {
            minX: projectedMinX,
            minY: projectedMinY,
            maxX: projectedMaxX,
            maxY: projectedMaxY,
          },
        }),
      })

      if (refined.accepted) {
        recoveryAcceptedRuns += 1
        designRmsSum += refined.metrics.designRms
      } else {
        recoveryRejectedRuns += 1
        const reason = refined.rejectionReason ?? 'unknown'
        rejectionReasons[reason] = (rejectionReasons[reason] ?? 0) + 1
      }

      let refinedCorrect = 0
      let refinedFalse = 0
      for (let position = 0; position < refined.points.length; position += 1) {
        const point = refined.points[position]!
        const truth = observedGraph.points[position]!.index
        if (point.status === IDENTITY_RECOVERED) {
          recoveredSum += 1
        }
        if (point.designId === undefined) {
          continue
        }
        if (point.designId === truth) {
          refinedCorrect += 1
        } else {
          refinedFalse += 1
        }
      }
      const refinedAccuracy =
        observedGraph.points.length === 0 ? 0 : refinedCorrect / observedGraph.points.length
      refinedAccuracySum += refinedAccuracy
      minRefinedAccuracy = Math.min(minRefinedAccuracy, refinedAccuracy)
      if (refined.accepted) {
        acceptedCorrectSum += refinedCorrect
        acceptedObservedSum += observedGraph.points.length
        minAcceptedRefinedAccuracy = Math.min(minAcceptedRefinedAccuracy, refinedAccuracy)
      }
      refinedFalseMatches += refinedFalse
      anchorSum += refined.metrics.anchors
      inlierSum += refined.metrics.initialInliers
      pairSum += refined.metrics.pairs
      conflictSum += refined.metrics.geometryConflicts
      predictedMissingSum += refined.predictedMissing.length
      const truthIds = new Set(observedGraph.points.map((point) => point.index))
      for (const missing of refined.predictedMissing) {
        if (truthIds.has(missing.designId)) {
          predictedMissingFalse += 1
        }
      }
    }
  }

  const report: IdentityStressReport = {
    runs: options.runs,
    depth,
    pointsPerEdge,
    view: {
      yawDeg: pose.yawDeg,
      pitchDeg: pose.pitchDeg,
      cameraDistance: pose.cameraDistance,
      sphericalDistortion: pose.sphericalDistortion,
    },
    meanAccuracy: accuracySum / options.runs,
    minAccuracy,
    maxAccuracy,
    perfectRuns,
    atLeast99Runs,
    falseMatchRuns,
    totalFalseMatches,
    meanDesignCollisions: designCollisionSum / options.runs,
    maxDesignCollisions,
    meanDetectedPoints: detectedPointSum / options.runs,
  }

  if (options.recovery === true) {
    report.recovery = {
      acceptedRuns: recoveryAcceptedRuns,
      rejectedRuns: recoveryRejectedRuns,
      meanRefinedAccuracy: refinedAccuracySum / options.runs,
      minRefinedAccuracy,
      meanAcceptedRefinedAccuracy:
        acceptedObservedSum === 0 ? 0 : acceptedCorrectSum / acceptedObservedSum,
      minAcceptedRefinedAccuracy:
        recoveryAcceptedRuns === 0 ? 0 : minAcceptedRefinedAccuracy,
      refinedFalseMatches,
      meanRecovered: recoveredSum / options.runs,
      meanPredictedMissing: predictedMissingSum / options.runs,
      predictedMissingFalse,
      meanAnchors: anchorSum / options.runs,
      meanInliers: inlierSum / options.runs,
      meanPairs: pairSum / options.runs,
      meanDesignRms: recoveryAcceptedRuns === 0 ? 0 : designRmsSum / recoveryAcceptedRuns,
      meanConflicts: conflictSum / options.runs,
      rejectionReasons,
    }
  }

  return report
}
