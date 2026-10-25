/**
 * CPU neighbour-consistency filter for refined ROI peaks.
 *
 * This is the exact post-GPU stage: the GPU already produced subpixel peaks and
 * their cubic/Hessian features. For each point we compare `peak`, `lambda_1`
 * and `lambda_2` with the same-polarity spatial neighbours, trim neighbour
 * outliers at 2 sigma, recompute the neighbour statistics once, then accept the
 * point only if all three z-scores are inside 4 sigma.
 */

export type RoiRefinedPeak = {
  x: number
  y: number
  /** 0 = black/dark, 1 = white/bright. */
  polarity: 0 | 1
  /** Cubic surface value at the refined stationary point. */
  peak: number
  /** Smaller physical Hessian eigenvalue. */
  eigenvalue1: number
  /** Larger physical Hessian eigenvalue. */
  eigenvalue2: number
  /** Physical Hessian off-diagonal term. */
  hessianXY: number
}

export type RoiDetectedPoint = {
  x: number
  y: number
  color: 0 | 1
}

export type RoiNeighbourFilterOptions = {
  /** Same-polarity spatial neighbours per point. */
  neighbours?: number
  /** Neighbour values farther than this many sigma are trimmed. */
  trimSigma?: number
  /** Point z-scores must be inside this many sigma to be kept. */
  acceptSigma?: number
  /**
   * Divide Hessian features by the peak contrast `|peak - 0.5|` before the
   * neighbour test. Only brightness is removed; target size is retained.
   */
  normalizeHessian?: boolean
}

export type RoiNeighbourFilterResult = {
  /** Aligned with the input order. */
  keep: boolean[]
  kept: RoiDetectedPoint[]
  /** Point-major, feature-minor; NaN when no neighbour test ran. */
  zScores: Float64Array
  /** Number of neighbours trimmed for each point and feature. */
  trimmedCounts: Uint8Array
}

export const ROI_NEIGHBOUR_COUNT = 40
export const ROI_TRIM_SIGMA = 2
export const ROI_ACCEPT_SIGMA = 4
/** Peak must pass; at least this many of the three Hessian features must pass. */
export const ROI_MIN_PASSING_HESSIAN_FEATURES = 2
export const ROI_NORMALIZE_HESSIAN = true
export const ROI_CONTRAST_FLOOR = 1e-6
export const ROI_PEAK_FEATURE_LABELS = ['peak', 'λ1/c', 'λ2/c', 'hxy/c'] as const

const FEATURE_COUNT = 4
const PEAK_FEATURE = 0
const FEATURE_KEYS = ['peak', 'eigenvalue1', 'eigenvalue2', 'hessianXY'] as const
const MIN_POLARITY_POINTS = 4
const MIN_NEIGHBOURS = 3
const MIN_TRIM_INLIERS = 3

export type RoiPeakFilterFeatureKey = (typeof FEATURE_KEYS)[number]
export type RoiPeakFilterFeatureLabel = (typeof ROI_PEAK_FEATURE_LABELS)[number]

export type RoiPeakFeatureInspection = {
  key: RoiPeakFilterFeatureKey
  label: RoiPeakFilterFeatureLabel
  value: number
  neighbourValues: number[]
  initialMean: number
  initialStd: number
  mean: number
  std: number
  zScore: number
  /** Neighbour values outside the trim band, if trimming was applied. */
  trimmed: number[]
  inlierCount: number
}

export type RoiPeakInspection = {
  index: number
  point: RoiRefinedPeak
  tested: boolean
  kept: boolean
  neighbourCount: number
  note?: 'fewer-than-four-same-polarity' | 'fewer-than-three-neighbours'
  features: RoiPeakFeatureInspection[]
}

function hessianScale(point: RoiRefinedPeak): number {
  return Math.max(Math.abs(point.peak - 0.5), ROI_CONTRAST_FLOOR)
}

function featureAt(point: RoiRefinedPeak, feature: number, normalizeHessian: boolean): number {
  if (feature === 0) {
    return point.peak
  }
  const value = feature === 1 ? point.eigenvalue1 : feature === 2 ? point.eigenvalue2 : point.hessianXY
  return normalizeHessian ? value / hessianScale(point) : value
}

function sortByPosition(indices: number[], peaks: readonly RoiRefinedPeak[]): void {
  indices.sort((first, second) => {
    const firstPoint = peaks[first]!
    const secondPoint = peaks[second]!
    if (firstPoint.y !== secondPoint.y) {
      return firstPoint.y - secondPoint.y
    }
    if (firstPoint.x !== secondPoint.x) {
      return firstPoint.x - secondPoint.x
    }
    return first - second
  })
}

function indicesForPolarity(peaks: readonly RoiRefinedPeak[], polarity: 0 | 1): number[] {
  const indices: number[] = []
  for (let index = 0; index < peaks.length; index += 1) {
    if (peaks[index]!.polarity === polarity) {
      indices.push(index)
    }
  }
  sortByPosition(indices, peaks)
  return indices
}

function heapIsWorse(distance: number, candidate: number, otherDistance: number, otherCandidate: number): boolean {
  return distance > otherDistance || (distance === otherDistance && candidate > otherCandidate)
}

function siftUp(distances: Float64Array, result: Int32Array, start: number): void {
  let child = start
  while (child > 0) {
    const parent = (child - 1) >> 1
    if (!heapIsWorse(distances[child]!, result[child]!, distances[parent]!, result[parent]!)) {
      break
    }
    const distance = distances[parent]!
    const candidate = result[parent]!
    distances[parent] = distances[child]!
    result[parent] = result[child]!
    distances[child] = distance
    result[child] = candidate
    child = parent
  }
}

function siftDown(distances: Float64Array, result: Int32Array, size: number, start: number): void {
  let parent = start
  for (;;) {
    const left = parent * 2 + 1
    if (left >= size) {
      break
    }
    const right = left + 1
    let worst = left
    if (right < size && heapIsWorse(distances[right]!, result[right]!, distances[left]!, result[left]!)) {
      worst = right
    }
    if (!heapIsWorse(distances[worst]!, result[worst]!, distances[parent]!, result[parent]!)) {
      break
    }
    const distance = distances[parent]!
    const candidate = result[parent]!
    distances[parent] = distances[worst]!
    result[parent] = result[worst]!
    distances[worst] = distance
    result[worst] = candidate
    parent = worst
  }
}

/**
 * Fill `result` with the `neighbourCount` nearest same-polarity candidates.
 *
 * A max-heap of the current worst neighbours keeps this O(N log K) instead of
 * the previous insertion-sort O(N K). Ties use the sorted candidate order so
 * the selected neighbour set stays deterministic.
 */
function fillNearestNeighbours(
  indices: readonly number[],
  peaks: readonly RoiRefinedPeak[],
  localIndex: number,
  neighbourCount: number,
  distances: Float64Array,
  result: Int32Array,
): void {
  const point = peaks[indices[localIndex]!]!
  let filled = 0
  for (let candidate = 0; candidate < indices.length; candidate += 1) {
    if (candidate === localIndex) {
      continue
    }
    const other = peaks[indices[candidate]!]!
    const dx = other.x - point.x
    const dy = other.y - point.y
    const distance = dx * dx + dy * dy
    if (filled < neighbourCount) {
      distances[filled] = distance
      result[filled] = candidate
      siftUp(distances, result, filled)
      filled += 1
      continue
    }
    if (!heapIsWorse(distances[0]!, result[0]!, distance, candidate)) {
      continue
    }
    distances[0] = distance
    result[0] = candidate
    siftDown(distances, result, filled, 0)
  }
}

function neighbourStatistics(values: Float64Array, count: number): { mean: number; standardDeviation: number } {
  let sum = 0
  for (let index = 0; index < count; index += 1) {
    sum += values[index]!
  }
  const mean = sum / count
  let squares = 0
  for (let index = 0; index < count; index += 1) {
    const difference = values[index]! - mean
    squares += difference * difference
  }
  return {
    mean,
    standardDeviation: Math.sqrt(squares / Math.max(1, count - 1)),
  }
}

function inspectFeature(
  key: RoiPeakFeatureInspection['key'],
  label: RoiPeakFeatureInspection['label'],
  value: number,
  neighbourValues: Float64Array,
  neighbourCount: number,
  trimSigma: number,
): RoiPeakFeatureInspection {
  const initial = neighbourStatistics(neighbourValues, neighbourCount)
  const floor = 1e-8 * Math.max(1, Math.abs(initial.mean))
  const safeInitial = Math.max(initial.standardDeviation, floor)
  const trimLimit = trimSigma * safeInitial
  const inlierValues = new Float64Array(neighbourCount)
  const trimmed: number[] = []
  let inlierCount = 0
  for (let neighbour = 0; neighbour < neighbourCount; neighbour += 1) {
    const neighbourValue = neighbourValues[neighbour]!
    if (Math.abs(neighbourValue - initial.mean) <= trimLimit) {
      inlierValues[inlierCount] = neighbourValue
      inlierCount += 1
    } else {
      trimmed.push(neighbourValue)
    }
  }
  let mean = initial.mean
  let standardDeviation = initial.standardDeviation
  let effectiveInliers = neighbourCount
  if (inlierCount >= MIN_TRIM_INLIERS) {
    const trimmedStats = neighbourStatistics(inlierValues, inlierCount)
    mean = trimmedStats.mean
    standardDeviation = trimmedStats.standardDeviation
    effectiveInliers = inlierCount
  } else {
    trimmed.length = 0
  }
  const safeFinal = Math.max(standardDeviation, floor)
  return {
    key,
    label,
    value,
    neighbourValues: Array.from(neighbourValues),
    initialMean: initial.mean,
    initialStd: initial.standardDeviation,
    mean,
    std: standardDeviation,
    zScore: (value - mean) / safeFinal,
    trimmed,
    inlierCount: effectiveInliers,
  }
}

/**
 * Apply the same-polarity trim/recompute/accept rule. The result is per point:
 * a feature whose trim would leave fewer than three inliers is left untrimmed,
 * but the point is still accepted or rejected by the final 4 sigma test.
 */
export function filterRefinedPeaks(
  peaks: readonly RoiRefinedPeak[],
  options: RoiNeighbourFilterOptions = {},
): RoiNeighbourFilterResult {
  const neighbours = Math.max(0, Math.floor(options.neighbours ?? ROI_NEIGHBOUR_COUNT))
  const trimSigma = options.trimSigma ?? ROI_TRIM_SIGMA
  const acceptSigma = options.acceptSigma ?? ROI_ACCEPT_SIGMA
  const normalizeHessian = options.normalizeHessian ?? ROI_NORMALIZE_HESSIAN
  const keep = Array.from({ length: peaks.length }, () => false)
  const zScores = new Float64Array(peaks.length * FEATURE_COUNT)
  zScores.fill(Number.NaN)
  const trimmedCounts = new Uint8Array(peaks.length * FEATURE_COUNT)
  const featureValues = new Float64Array(peaks.length * FEATURE_COUNT)
  for (let index = 0; index < peaks.length; index += 1) {
    const point = peaks[index]!
    for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
      featureValues[index * FEATURE_COUNT + feature] = featureAt(point, feature, normalizeHessian)
    }
  }

  for (const polarity of [0, 1] as const) {
    const indices = indicesForPolarity(peaks, polarity)
    if (indices.length < MIN_POLARITY_POINTS) {
      for (const index of indices) {
        keep[index] = true
      }
      continue
    }
    const neighbourCount = Math.min(neighbours, indices.length - 1)
    if (neighbourCount < MIN_NEIGHBOURS) {
      for (const index of indices) {
        keep[index] = true
      }
      continue
    }
    const localNeighbours = new Int32Array(neighbourCount)
    const nearestDistances = new Float64Array(neighbourCount)
    const neighbourValues = new Float64Array(neighbourCount)
    const inlierValues = new Float64Array(neighbourCount)
    for (let localIndex = 0; localIndex < indices.length; localIndex += 1) {
      const globalIndex = indices[localIndex]!
      fillNearestNeighbours(indices, peaks, localIndex, neighbourCount, nearestDistances, localNeighbours)
      let peakPass = false
      let hessianPassing = 0
      for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
        for (let neighbour = 0; neighbour < neighbourCount; neighbour += 1) {
          neighbourValues[neighbour] = featureValues[indices[localNeighbours[neighbour]!]! * FEATURE_COUNT + feature]!
        }
        const initial = neighbourStatistics(neighbourValues, neighbourCount)
        const meanAbs = Math.abs(initial.mean)
        const floor = 1e-8 * Math.max(1, meanAbs)
        const safeInitial = Math.max(initial.standardDeviation, floor)
        const trimLimit = trimSigma * safeInitial
        let inlierCount = 0
        for (let neighbour = 0; neighbour < neighbourCount; neighbour += 1) {
          if (Math.abs(neighbourValues[neighbour]! - initial.mean) <= trimLimit) {
            inlierValues[inlierCount] = neighbourValues[neighbour]!
            inlierCount += 1
          }
        }
        let mean = initial.mean
        let standardDeviation = initial.standardDeviation
        if (inlierCount >= MIN_TRIM_INLIERS) {
          const trimmed = neighbourStatistics(inlierValues, inlierCount)
          mean = trimmed.mean
          standardDeviation = trimmed.standardDeviation
          trimmedCounts[globalIndex * FEATURE_COUNT + feature] = neighbourCount - inlierCount
        }
        const safeFinal = Math.max(standardDeviation, floor)
        const z = (featureValues[globalIndex * FEATURE_COUNT + feature]! - mean) / safeFinal
        zScores[globalIndex * FEATURE_COUNT + feature] = z
        if (Number.isFinite(z) && Math.abs(z) <= acceptSigma) {
          if (feature === PEAK_FEATURE) {
            peakPass = true
          } else {
            hessianPassing += 1
          }
        }
      }
      keep[globalIndex] = peakPass && hessianPassing >= ROI_MIN_PASSING_HESSIAN_FEATURES
    }
  }

  const kept: RoiDetectedPoint[] = []
  for (let index = 0; index < peaks.length; index += 1) {
    if (!keep[index]) {
      continue
    }
    const point = peaks[index]!
    kept.push({ x: point.x, y: point.y, color: point.polarity })
  }
  return { keep, kept, zScores, trimmedCounts }
}

/**
 * Recompute the neighbour statistics for one point for the ROI Lab inspector.
 * This follows the same ordering and trim/recompute rules as
 * `filterRefinedPeaks`, but keeps the neighbour values so they can be plotted.
 */
export function describeRefinedPeak(
  peaks: readonly RoiRefinedPeak[],
  index: number,
  options: RoiNeighbourFilterOptions = {},
): RoiPeakInspection | undefined {
  if (index < 0 || index >= peaks.length) {
    return undefined
  }
  const point = peaks[index]!
  const neighbours = Math.max(0, Math.floor(options.neighbours ?? ROI_NEIGHBOUR_COUNT))
  const trimSigma = options.trimSigma ?? ROI_TRIM_SIGMA
  const acceptSigma = options.acceptSigma ?? ROI_ACCEPT_SIGMA
  const normalizeHessian = options.normalizeHessian ?? ROI_NORMALIZE_HESSIAN
  const indices = indicesForPolarity(peaks, point.polarity)
  if (indices.length < MIN_POLARITY_POINTS) {
    return {
      index,
      point,
      tested: false,
      kept: true,
      neighbourCount: 0,
      note: 'fewer-than-four-same-polarity',
      features: [],
    }
  }
  const localIndex = indices.indexOf(index)
  if (localIndex < 0) {
    return undefined
  }
  const neighbourCount = Math.min(neighbours, indices.length - 1)
  if (neighbourCount < MIN_NEIGHBOURS) {
    return {
      index,
      point,
      tested: false,
      kept: true,
      neighbourCount,
      note: 'fewer-than-three-neighbours',
      features: [],
    }
  }
  const localNeighbours = new Int32Array(neighbourCount)
  const nearestDistances = new Float64Array(neighbourCount)
  fillNearestNeighbours(indices, peaks, localIndex, neighbourCount, nearestDistances, localNeighbours)

  const inspect = (feature: number): RoiPeakFeatureInspection => {
    const neighbourValues = new Float64Array(neighbourCount)
    for (let neighbour = 0; neighbour < neighbourCount; neighbour += 1) {
      neighbourValues[neighbour] = featureAt(peaks[indices[localNeighbours[neighbour]!]!]!, feature, normalizeHessian)
    }
    return inspectFeature(
      FEATURE_KEYS[feature]!,
      ROI_PEAK_FEATURE_LABELS[feature]!,
      featureAt(point, feature, normalizeHessian),
      neighbourValues,
      neighbourCount,
      trimSigma,
    )
  }

  const features = Array.from({ length: FEATURE_COUNT }, (_, feature) => inspect(feature))
  let peakPass = false
  let hessianPassing = 0
  for (const [feature, inspected] of features.entries()) {
    if (!Number.isFinite(inspected.zScore) || Math.abs(inspected.zScore) > acceptSigma) {
      continue
    }
    if (feature === PEAK_FEATURE) {
      peakPass = true
    } else {
      hessianPassing += 1
    }
  }
  return {
    index,
    point,
    tested: true,
    kept: peakPass && hessianPassing >= ROI_MIN_PASSING_HESSIAN_FEATURES,
    neighbourCount,
    features,
  }
}
