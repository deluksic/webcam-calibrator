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

type Line = {
  origin: Vec2
  direction: Vec2
}

function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

function unitVector(from: Vec2, to: Vec2): Vec2 | undefined {
  const dx = to.x - from.x
  const dy = to.y - from.y
  const length = Math.hypot(dx, dy)
  if (length < 1e-9) {
    return undefined
  }
  return { x: dx / length, y: dy / length }
}

function dot(first: Vec2, second: Vec2): number {
  return first.x * second.x + first.y * second.y
}

function cross(first: Vec2, second: Vec2): number {
  return first.x * second.y - first.y * second.x
}

function perpendicularDistance(point: Vec2, line: Line): number {
  return Math.abs(cross({ x: point.x - line.origin.x, y: point.y - line.origin.y }, line.direction))
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
          const length = Math.hypot(dx, dy)
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

function chainScore(
  gaps: readonly number[],
  perpendiculars: readonly number[],
  maxStraightness: number,
  localScale: number,
):
  | {
      score: number
      spacingVariation: number
      straightness: number
    }
  | undefined {
  if (gaps.length === 0) {
    return undefined
  }
  const minGap = Math.min(...gaps)
  const maxGap = Math.max(...gaps)
  if (minGap < 1e-9) {
    return undefined
  }
  const meanGap = gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length
  if (meanGap < 1e-9) {
    return undefined
  }
  const spacingVariation = Math.max(...gaps.map((gap) => Math.abs(gap - meanGap) / meanGap))
  const gapRatio = maxGap / minGap
  const straightness = Math.max(...perpendiculars) / meanGap
  if (
    spacingVariation > MAX_SPACING_VARIATION ||
    gapRatio > MAX_GAP_RATIO ||
    straightness > maxStraightness ||
    maxGap > Math.max(localScale, 1e-6) * MAX_GAP_TO_LOCAL_SCALE
  ) {
    return undefined
  }
  return {
    score: spacingVariation + straightness * 1.5,
    spacingVariation,
    straightness,
  }
}

/** Principal-axis fit of a set of point positions. */
function fitLine(points: readonly PatternPoint[], indices: readonly number[]): Line | undefined {
  const count = indices.length
  if (count < 2) {
    return undefined
  }

  let meanX = 0
  let meanY = 0
  for (const index of indices) {
    const point = points[index]!
    meanX += point.x
    meanY += point.y
  }
  meanX /= count
  meanY /= count

  let covarianceX = 0
  let covarianceY = 0
  let covarianceCross = 0
  for (const index of indices) {
    const point = points[index]!
    const dx = point.x - meanX
    const dy = point.y - meanY
    covarianceX += dx * dx
    covarianceY += dy * dy
    covarianceCross += dx * dy
  }

  if (covarianceX + covarianceY < 1e-18) {
    return undefined
  }

  const angle = 0.5 * Math.atan2(2 * covarianceCross, covarianceX - covarianceY)
  return {
    origin: { x: meanX, y: meanY },
    direction: { x: Math.cos(angle), y: Math.sin(angle) },
  }
}

function collectInliers(points: readonly PatternPoint[], line: Line, tolerance: number, inliers: number[]): number[] {
  inliers.length = 0
  for (let index = 0; index < points.length; index += 1) {
    if (perpendicularDistance(points[index]!, line) <= tolerance) {
      inliers.push(index)
    }
  }
  return inliers
}

function pairKey(first: number, second: number): string {
  return first < second ? `${first}:${second}` : `${second}:${first}`
}

/**
 * Finds candidate chains of exactly `pointsPerEdge` collinear, similarly
 * spaced points. Rather than relying on one exact opposite-neighbor seed,
 * every nearby pair proposes a seed direction. Inliers are projected onto that
 * axis, and each length-N window is validated against its own principal-axis
 * fit, so seed-direction noise cannot reject or admit a chain by itself. The
 * caller selects non-overlapping candidates.
 */
export function findChainCandidates(
  points: readonly PatternPoint[],
  neighborLists: ReadonlyArray<readonly number[]>,
  localScales: readonly number[],
  pointsPerEdge: number,
  options: ChainCandidateOptions = {},
): ChainCandidate[] {
  if (pointsPerEdge < 2 || points.length < pointsPerEdge) {
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
  const seedPairs = new Set<string>()
  const inliers: number[] = []
  const ordered: Array<{ index: number; t: number }> = []

  const addCandidate = (
    pointIndices: number[],
    quality: { score: number; spacingVariation: number; straightness: number },
  ): void => {
    const key = [...pointIndices].sort((a, b) => a - b).join(':')
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

  const considerLine = (seedLine: Line, tolerance: number): void => {
    collectInliers(points, seedLine, tolerance, inliers)
    if (inliers.length < pointsPerEdge) {
      return
    }

    ordered.length = 0
    for (const index of inliers) {
      const point = points[index]!
      ordered.push({
        index,
        t: dot({ x: point.x - seedLine.origin.x, y: point.y - seedLine.origin.y }, seedLine.direction),
      })
    }
    ordered.sort((first, second) =>
      Math.abs(first.t - second.t) < 1e-9 ? first.index - second.index : first.t - second.t,
    )

    const startMax = ordered.length - pointsPerEdge
    for (let start = 0; start <= startMax; start += 1) {
      const window = ordered.slice(start, start + pointsPerEdge)
      const pointIndices = window.map((item) => item.index)

      // Validate each window against its own principal-axis fit. A window is
      // only a chain if the points themselves are straight and evenly spaced;
      // the noisy seed direction must not decide that.
      const fitted = fitLine(points, pointIndices)
      if (fitted === undefined) {
        continue
      }
      const projected = pointIndices.map((index) => {
        const point = points[index]!
        return {
          index,
          t: dot({ x: point.x - fitted.origin.x, y: point.y - fitted.origin.y }, fitted.direction),
          perpendicular: perpendicularDistance(point, fitted),
        }
      })
      projected.sort((first, second) =>
        Math.abs(first.t - second.t) < 1e-9 ? first.index - second.index : first.t - second.t,
      )

      const gaps: number[] = []
      for (let index = 1; index < projected.length; index += 1) {
        gaps.push(projected[index]!.t - projected[index - 1]!.t)
      }
      const localScale = Math.min(median(projected.map((item) => pointSpacings[item.index]!)), globalScale * 1.5)
      const quality = chainScore(
        gaps,
        projected.map((item) => item.perpendicular),
        maxStraightness,
        localScale,
      )
      if (quality === undefined) {
        continue
      }
      const orderedIndices = projected.map((item) => item.index)
      if (orderedIndices[0]! > orderedIndices[orderedIndices.length - 1]!) {
        orderedIndices.reverse()
      }
      addCandidate(orderedIndices, quality)
    }
  }

  for (let first = 0; first < points.length; first += 1) {
    const neighbors = neighborLists[first]!
    for (const second of neighbors) {
      if (second === first) {
        continue
      }
      const key = pairKey(first, second)
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
      const line: Line = { origin: firstPoint, direction }
      const tolerance = Math.max(seedLength * BROAD_SEED_TOLERANCE, seedScale * perpendicularTolerance, 1e-6)
      considerLine(line, tolerance)
    }
  }

  return [...candidates.values()].sort((first, second) => first.score - second.score)
}
