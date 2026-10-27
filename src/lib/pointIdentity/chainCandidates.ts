import type { PatternPoint, Vec2 } from '@/lib/pointIdentity/types'

const SEED_MIN_DISTANCE_FACTOR = 0.55
const SEED_MAX_DISTANCE_FACTOR = 1.8
const DEFAULT_PERPENDICULAR_TOLERANCE = 0.24
const BROAD_SEED_TOLERANCE = 0.35
const DEFAULT_MAX_STRAIGHTNESS = 0.12
const MAX_SPACING_VARIATION = 0.7
const MAX_GAP_RATIO = 2.4
const MAX_GAP_TO_LOCAL_SCALE = 1.7

export type ChainCandidate = {
  /** Points ordered along the chain axis. */
  pointIndices: number[]
  /** Lower is better. */
  score: number
  spacingVariation: number
  straightness: number
}

export type ChainCandidateOptions = {
  /** Maximum perpendicular distance from the seed line, as a fraction of local point spacing. */
  perpendicularTolerance?: number
  /** Maximum max-perpendicular / mean-gap allowed for a chain. */
  maxStraightness?: number
}

function distance(a: Vec2, b: Vec2): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return Math.sqrt(dx * dx + dy * dy)
}

function unitVector(from: Vec2, to: Vec2): Vec2 | undefined {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const length = Math.sqrt(dx * dx + dy * dy)
  if (length < 1e-9) {
    return undefined
  }
  return { x: dx / length, y: dy / length }
}

function dot(first: Vec2, second: Vec2): number {
  return first.x * second.x + first.y * second.y
}

/**
 * A point on the interior of a lattice edge has two same-side neighbors that
 * are nearly opposite and roughly equidistant. Their direction is the local
 * chain axis. Corner points intentionally have no chain axis, because their
 * near neighbors sit on different cell sides.
 *
 * The line grower below does not require this seed, but callers that inspect a
 * single point's chain axis still use it.
 */
export function computeChainDirections(
  points: readonly PatternPoint[],
  neighborLists: ReadonlyArray<readonly number[]>,
): Array<Vec2 | undefined> {
  const CHAIN_OPPOSITE_DOT = -0.97
  const CHAIN_DISTANCE_BALANCE = 0.6
  const directions: Array<Vec2 | undefined> = []

  for (let index = 0; index < points.length; index += 1) {
    const origin = points[index]!
    const neighbors = neighborLists[index]!
    let bestScore = 0
    let bestLengthSum = Number.POSITIVE_INFINITY
    let bestDirection: Vec2 | undefined

    for (let firstIndex = 0; firstIndex < neighbors.length; firstIndex += 1) {
      const first = neighbors[firstIndex]!
      const firstUnit = unitVector(origin, points[first]!)
      if (firstUnit === undefined) {
        continue
      }
      const firstLength = distance(origin, points[first]!)

      for (let secondIndex = firstIndex + 1; secondIndex < neighbors.length; secondIndex += 1) {
        const second = neighbors[secondIndex]!
        const secondUnit = unitVector(origin, points[second]!)
        if (secondUnit === undefined) {
          continue
        }
        const secondLength = distance(origin, points[second]!)
        const opposition = dot(firstUnit, secondUnit)
        if (opposition > CHAIN_OPPOSITE_DOT) {
          continue
        }
        const longest = Math.max(firstLength, secondLength)
        if (longest < 1e-9) {
          continue
        }
        const balance = Math.min(firstLength, secondLength) / longest
        if (balance < CHAIN_DISTANCE_BALANCE) {
          continue
        }
        const score = -opposition * balance
        const lengthSum = firstLength + secondLength
        if (score > bestScore + 1e-9 || (Math.abs(score - bestScore) <= 1e-9 && lengthSum < bestLengthSum)) {
          const dx = firstUnit.x - secondUnit.x
          const dy = firstUnit.y - secondUnit.y
          const length = Math.sqrt(dx * dx + dy * dy)
          if (length > 1e-9) {
            bestScore = score
            bestLengthSum = lengthSum
            bestDirection = { x: dx / length, y: dy / length }
          }
        }
      }
    }

    directions.push(bestDirection)
  }

  return directions
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[sorted.length >> 1] ?? 0
}

/**
 * Upper median of the first `count` entries, sorted in place. Matches
 * `median()` (which takes `sorted[length >> 1]`, not the mean of the middle
 * pair) without allocating.
 */
function medianInPlace(values: Float64Array, count: number): number {
  for (let index = 1; index < count; index += 1) {
    const value = values[index]!
    let slot = index - 1
    while (slot >= 0 && values[slot]! > value) {
      values[slot + 1] = values[slot]!
      slot -= 1
    }
    values[slot + 1] = value
  }
  return values[count >> 1] ?? 0
}

/** Sorted `a:b:c` key for candidate de-duplication. */
function candidateKey(index: Int32Array, count: number, scratch: Int32Array): string {
  for (let slot = 0; slot < count; slot += 1) {
    scratch[slot] = index[slot]!
  }
  for (let slot = 1; slot < count; slot += 1) {
    const value = scratch[slot]!
    let position = slot - 1
    while (position >= 0 && scratch[position]! > value) {
      scratch[position + 1] = scratch[position]!
      position -= 1
    }
    scratch[position + 1] = value
  }
  let key = ''
  for (let slot = 0; slot < count; slot += 1) {
    key += slot === 0 ? `${scratch[slot]!}` : `:${scratch[slot]!}`
  }
  return key
}

/**
 * Finds candidate chains of exactly `pointsPerEdge` collinear, similarly
 * spaced points. Rather than relying on one exact opposite-neighbor seed,
 * every nearby pair proposes a seed direction. Inliers are projected onto that
 * axis, and each length-N window is validated against its own principal-axis
 * fit, so seed-direction noise cannot reject or admit a chain by itself. The
 * caller selects non-overlapping candidates.
 *
 * Every per-window temporary lives in a preallocated scratch buffer: the
 * windows dominate the cost (63% of this function) and each one previously
 * allocated around ten arrays and two sort closures.
 */
export function findChainCandidates(
  points: readonly PatternPoint[],
  neighborLists: ReadonlyArray<readonly number[]>,
  localScales: readonly number[],
  pointsPerEdge: number,
  options: ChainCandidateOptions = {},
): ChainCandidate[] {
  const count = points.length
  if (pointsPerEdge < 2 || count < pointsPerEdge) {
    return []
  }

  const perpendicularTolerance = Math.max(0.01, options.perpendicularTolerance ?? DEFAULT_PERPENDICULAR_TOLERANCE)
  const maxStraightness = Math.max(0.001, options.maxStraightness ?? DEFAULT_MAX_STRAIGHTNESS)
  const candidates = new Map<string, ChainCandidate>()
  const fallbackScale = Math.max(median(localScales), 1e-6)
  // The kNN local scales are inflated by off-edge neighbors. Use the nearest
  // neighbor distance as the actual point spacing for seed admission and gap
  // caps, so one skipped sample cannot masquerade as a valid gap.
  const pointSpacings = points.map((point, index) => {
    let nearest = Number.POSITIVE_INFINITY
    for (const neighbor of neighborLists[index]!) {
      nearest = Math.min(nearest, distance(point, points[neighbor]!))
    }
    return Number.isFinite(nearest) ? Math.max(nearest, 1e-6) : fallbackScale
  })
  const globalScale = Math.max(median(pointSpacings), 1e-6)
  const seedPairs = new Set<number>()

  const inliers: number[] = []
  const inlierT = new Float64Array(count)
  const inlierOrder = new Int32Array(count)
  const windowIdx = new Int32Array(pointsPerEdge)
  const projIdx = new Int32Array(pointsPerEdge)
  const projT = new Float64Array(pointsPerEdge)
  const projPerp = new Float64Array(pointsPerEdge)
  const permutation = new Int32Array(pointsPerEdge)
  const gaps = new Float64Array(pointsPerEdge)
  const spacings = new Float64Array(pointsPerEdge)
  const keyScratch = new Int32Array(pointsPerEdge)
  const gapCount = pointsPerEdge - 1

  const addCandidate = (
    pointIndices: number[],
    quality: { score: number; spacingVariation: number; straightness: number },
  ): void => {
    const key = candidateKey(Int32Array.from(pointIndices), pointIndices.length, keyScratch)
    const existing = candidates.get(key)
    if (existing === undefined || quality.score < existing.score) {
      candidates.set(key, {
        pointIndices,
        score: quality.score,
        spacingVariation: quality.spacingVariation,
        straightness: quality.straightness,
      })
    }
  }

  const considerLine = (originX: number, originY: number, dirX: number, dirY: number, tolerance: number): void => {
    // Inlier scan: the cross product is inlined so no Vec2 temporary is
    // allocated per point.
    inliers.length = 0
    for (let index = 0; index < count; index += 1) {
      const point = points[index]!
      const rx = point.x - originX
      const ry = point.y - originY
      if (Math.abs(rx * dirY - ry * dirX) <= tolerance) {
        inliers.push(index)
      }
    }
    const inlierCount = inliers.length
    if (inlierCount < pointsPerEdge) {
      return
    }

    for (let position = 0; position < inlierCount; position += 1) {
      const index = inliers[position]!
      const point = points[index]!
      inlierT[position] = (point.x - originX) * dirX + (point.y - originY) * dirY
      inlierOrder[position] = position
    }
    for (let slot = 1; slot < inlierCount; slot += 1) {
      const value = inlierOrder[slot]!
      const valueT = inlierT[value]!
      const valueIndex = inliers[value]!
      let position = slot - 1
      while (position >= 0) {
        const other = inlierOrder[position]!
        const otherT = inlierT[other]!
        const otherIndex = inliers[other]!
        const before = Math.abs(valueT - otherT) < 1e-9 ? valueIndex < otherIndex : valueT < otherT
        if (!before) {
          break
        }
        inlierOrder[position + 1] = other
        position -= 1
      }
      inlierOrder[position + 1] = value
    }

    const startMax = inlierCount - pointsPerEdge
    for (let start = 0; start <= startMax; start += 1) {
      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        windowIdx[slot] = inliers[inlierOrder[start + slot]!]!
      }

      // Principal-axis fit of the window.
      let meanX = 0
      let meanY = 0
      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        const point = points[windowIdx[slot]!]!
        meanX += point.x
        meanY += point.y
      }
      meanX /= pointsPerEdge
      meanY /= pointsPerEdge
      let covarianceX = 0
      let covarianceY = 0
      let covarianceCross = 0
      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        const point = points[windowIdx[slot]!]!
        const dx = point.x - meanX
        const dy = point.y - meanY
        covarianceX += dx * dx
        covarianceY += dy * dy
        covarianceCross += dx * dy
      }
      if (covarianceX + covarianceY < 1e-18) {
        continue
      }
      const angle = 0.5 * Math.atan2(2 * covarianceCross, covarianceX - covarianceY)
      const axisX = Math.cos(angle)
      const axisY = Math.sin(angle)

      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        const index = windowIdx[slot]!
        const point = points[index]!
        const dx = point.x - meanX
        const dy = point.y - meanY
        projIdx[slot] = index
        projT[slot] = dx * axisX + dy * axisY
        projPerp[slot] = Math.abs(dx * axisY - dy * axisX)
      }

      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        permutation[slot] = slot
      }
      for (let slot = 1; slot < pointsPerEdge; slot += 1) {
        const value = permutation[slot]!
        const valueT = projT[value]!
        const valueIndex = projIdx[value]!
        let position = slot - 1
        while (position >= 0) {
          const other = permutation[position]!
          const delta = valueT - projT[other]!
          const before = Math.abs(delta) < 1e-9 ? valueIndex < projIdx[other]! : delta < 0
          if (!before) {
            break
          }
          permutation[position + 1] = other
          position -= 1
        }
        permutation[position + 1] = value
      }

      let minGap = Number.POSITIVE_INFINITY
      let maxGap = Number.NEGATIVE_INFINITY
      let gapSum = 0
      for (let slot = 1; slot < pointsPerEdge; slot += 1) {
        const gap = projT[permutation[slot]!]! - projT[permutation[slot - 1]!]!
        gaps[slot - 1] = gap
        if (gap < minGap) {
          minGap = gap
        }
        if (gap > maxGap) {
          maxGap = gap
        }
        gapSum += gap
      }
      if (minGap < 1e-9) {
        continue
      }
      const meanGap = gapSum / gapCount
      if (meanGap < 1e-9) {
        continue
      }
      let spacingVariation = 0
      for (let slot = 0; slot < gapCount; slot += 1) {
        const deviation = Math.abs(gaps[slot]! - meanGap) / meanGap
        if (deviation > spacingVariation) {
          spacingVariation = deviation
        }
      }
      let maxPerpendicular = 0
      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        if (projPerp[slot]! > maxPerpendicular) {
          maxPerpendicular = projPerp[slot]!
        }
      }
      const straightness = maxPerpendicular / meanGap
      if (
        spacingVariation > MAX_SPACING_VARIATION ||
        maxGap / minGap > MAX_GAP_RATIO ||
        straightness > maxStraightness
      ) {
        continue
      }
      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        spacings[slot] = pointSpacings[projIdx[slot]!]!
      }
      const localScale = Math.min(medianInPlace(spacings, pointsPerEdge), globalScale * 1.5)
      if (maxGap > Math.max(localScale, 1e-6) * MAX_GAP_TO_LOCAL_SCALE) {
        continue
      }

      for (let slot = 0; slot < pointsPerEdge; slot += 1) {
        windowIdx[slot] = projIdx[permutation[slot]!]!
      }
      if (windowIdx[0]! > windowIdx[pointsPerEdge - 1]!) {
        for (let low = 0, high = pointsPerEdge - 1; low < high; low += 1, high -= 1) {
          const swap = windowIdx[low]!
          windowIdx[low] = windowIdx[high]!
          windowIdx[high] = swap
        }
      }
      addCandidate(Array.from(windowIdx), {
        score: spacingVariation + straightness * 1.5,
        spacingVariation,
        straightness,
      })
    }
  }

  for (let first = 0; first < count; first += 1) {
    const neighbors = neighborLists[first]!
    for (const second of neighbors) {
      if (second === first) {
        continue
      }
      const key = first < second ? first * count + second : second * count + first
      if (seedPairs.has(key)) {
        continue
      }
      seedPairs.add(key)

      const firstPoint = points[first]!
      const secondPoint = points[second]!
      const seedLength = distance(firstPoint, secondPoint)
      const seedScale = Math.max(Math.min(pointSpacings[first]!, pointSpacings[second]!), 1e-6)
      if (seedLength < seedScale * SEED_MIN_DISTANCE_FACTOR || seedLength > seedScale * SEED_MAX_DISTANCE_FACTOR) {
        continue
      }
      const direction = unitVector(firstPoint, secondPoint)
      if (direction === undefined) {
        continue
      }
      const tolerance = Math.max(seedLength * BROAD_SEED_TOLERANCE, seedScale * perpendicularTolerance, 1e-6)
      considerLine(firstPoint.x, firstPoint.y, direction.x, direction.y, tolerance)
    }
  }

  return [...candidates.values()].sort((first, second) => first.score - second.score)
}
