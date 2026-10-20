import type { PatternPoint, Vec2 } from '@/lib/pointIdentity/types'

/** Two same-side neighbors must be almost exactly opposite to define a chain. */
const CHAIN_OPPOSITE_DOT = -0.97
const CHAIN_DISTANCE_BALANCE = 0.6
const DEFAULT_PERPENDICULAR_TOLERANCE = 0.22
const DEFAULT_MAX_STRAIGHTNESS = 0.12
const MAX_SPACING_VARIATION = 0.7
const MAX_GAP_RATIO = 2.4

export type ChainCandidate = {
  /** Points ordered along the chain axis. */
  pointIndices: number[]
  /** Lower is better. */
  score: number
  spacingVariation: number
  straightness: number
}

export type ChainCandidateOptions = {
  /** Maximum perpendicular distance from the seed line, as a fraction of local scale. */
  perpendicularTolerance?: number
  /** Maximum max-perpendicular / mean-gap allowed for a chain. */
  maxStraightness?: number
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

/**
 * A point on the interior of a lattice edge has two same-side neighbors that
 * are nearly opposite and roughly equidistant. Their direction is the local
 * chain axis. Corner points intentionally have no chain axis, because their
 * near neighbors sit on different cell sides.
 */
export function computeChainDirections(
  points: readonly PatternPoint[],
  neighborLists: ReadonlyArray<readonly number[]>,
): Array<Vec2 | undefined> {
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

function chainScore(
  gaps: readonly number[],
  perpendiculars: readonly number[],
  maxStraightness: number,
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
  if (spacingVariation > MAX_SPACING_VARIATION || gapRatio > MAX_GAP_RATIO || straightness > maxStraightness) {
    return undefined
  }
  return {
    score: spacingVariation + straightness * 1.5,
    spacingVariation,
    straightness,
  }
}

/**
 * Finds candidate chains of exactly `pointsPerEdge` collinear, similarly
 * spaced points. Windows are grown around interior points that have a reliable
 * local chain axis. The caller selects non-overlapping candidates.
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
  const directions = computeChainDirections(points, neighborLists)
  const candidates = new Map<string, ChainCandidate>()

  for (let seed = 0; seed < points.length; seed += 1) {
    const direction = directions[seed]
    if (direction === undefined) {
      continue
    }
    const origin = points[seed]!
    const tolerance = Math.max(localScales[seed]! * perpendicularTolerance, 1e-6)
    const along: Array<{ index: number; t: number; perpendicular: number }> = [{ index: seed, t: 0, perpendicular: 0 }]

    for (let other = 0; other < points.length; other += 1) {
      if (other === seed) {
        continue
      }
      const point = points[other]!
      const dx = point.x - origin.x
      const dy = point.y - origin.y
      const t = dx * direction.x + dy * direction.y
      const perpendicular = Math.abs(dx * direction.y - dy * direction.x)
      if (perpendicular <= tolerance) {
        along.push({ index: other, t, perpendicular })
      }
    }

    along.sort((first, second) => first.t - second.t)
    const seedPosition = along.findIndex((item) => item.index === seed)
    if (seedPosition < 0) {
      continue
    }
    const startMin = Math.max(0, seedPosition - pointsPerEdge + 1)
    const startMax = Math.min(seedPosition, along.length - pointsPerEdge)

    for (let start = startMin; start <= startMax; start += 1) {
      const window = along.slice(start, start + pointsPerEdge)
      const gaps: number[] = []
      for (let index = 1; index < window.length; index += 1) {
        gaps.push(window[index]!.t - window[index - 1]!.t)
      }
      const quality = chainScore(
        gaps,
        window.map((item) => item.perpendicular),
        maxStraightness,
      )
      if (quality === undefined) {
        continue
      }
      const pointIndices = window.map((item) => item.index)
      if (pointIndices[0]! > pointIndices[pointIndices.length - 1]!) {
        pointIndices.reverse()
      }
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
  }

  return [...candidates.values()].sort((first, second) => first.score - second.score)
}
