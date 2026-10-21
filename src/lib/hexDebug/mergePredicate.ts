export type MergeTuning = {
  baseTolerance: number
  z: number
  maxTolerance: number
}

export type MergeStats = {
  mean: number
  variance: number
  area: number
}

/**
 * Merge adjacent components only when their colors agree closely enough for
 * the less noisy of the two components. A uniform component therefore stays
 * strict; two noisy components may merge more freely.
 */
export function shouldMergeComponents(first: MergeStats, second: MergeStats, tuning: MergeTuning): boolean {
  if (first.area <= 0 || second.area <= 0) {
    return false
  }
  const varianceFloor = Math.min(Math.max(first.variance, 0), Math.max(second.variance, 0))
  const tolerance = Math.min(tuning.maxTolerance, tuning.baseTolerance + tuning.z * Math.sqrt(varianceFloor))
  return Math.abs(first.mean - second.mean) <= tolerance
}

export const DEFAULT_MERGE_TUNING: MergeTuning = {
  baseTolerance: 0.05,
  z: 1.5,
  maxTolerance: 0.2,
}

export function grayMean(sumGray: number, area: number, grayScale = 127): number {
  return area > 0 ? sumGray / (area * grayScale) : 0
}

export function grayVariance(sumGray: number, sumGraySq: number, area: number, grayScale = 127): number {
  if (area <= 0) {
    return 0
  }
  const mean = sumGray / (area * grayScale)
  const secondMoment = sumGraySq / (area * grayScale * grayScale)
  return Math.max(0, secondMoment - mean * mean)
}
