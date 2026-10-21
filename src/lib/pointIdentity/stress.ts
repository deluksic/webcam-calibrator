import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { createIdentityIndex, evaluateIdentity } from '@/lib/pointIdentity/identity'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'

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
  const dropoutActive = observationMask.dropoutPercent > 0
  const fixedObservedPoints = dropoutActive ? undefined : applyObservationMask(projected, observationMask)
  const fixedObservedGraph =
    fixedObservedPoints === undefined ? undefined : buildProximityGraph(fixedObservedPoints, tuning)

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

  for (let run = 0; run < options.runs; run += 1) {
    const colors = randomBinaryColors(pattern.points.length, baseSeed + run)
    const designIndex = createIdentityIndex(designGraph, colors, depth)
    const observedPoints = dropoutActive
      ? applyObservationMask(projected, {
          ...observationMask,
          dropoutSeed: observationMask.dropoutSeed + run,
        })
      : fixedObservedPoints!
    const observedGraph = dropoutActive ? buildProximityGraph(observedPoints, tuning) : fixedObservedGraph!
    const evaluation = evaluateIdentity(designGraph, observedGraph, designIndex, colors, depth)
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
  }

  return {
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
}
