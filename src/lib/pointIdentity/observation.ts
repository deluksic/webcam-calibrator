import { createRandom } from '@/lib/pointIdentity/random'
import type { PatternPoint } from '@/lib/pointIdentity/types'

export type ObservationMask = {
  /** Percent of the projected field removed from each side. */
  cropLeft: number
  cropRight: number
  cropTop: number
  cropBottom: number
  /** Percent of the remaining points removed at random. */
  dropoutPercent: number
  dropoutSeed: number
}

export const DEFAULT_OBSERVATION_MASK: ObservationMask = {
  cropLeft: 0,
  cropRight: 0,
  cropTop: 0,
  cropBottom: 0,
  dropoutPercent: 0,
  dropoutSeed: 1,
}

/**
 * Removes points outside a rectangular crop and then randomly drops a
 * percentage of the survivors. Original point indices are preserved so
 * identity comparisons remain in design space.
 */
export function applyObservationMask(
  points: readonly PatternPoint[],
  mask: ObservationMask = DEFAULT_OBSERVATION_MASK,
): PatternPoint[] {
  if (points.length === 0 || mask.dropoutPercent >= 100) {
    return []
  }

  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const point of points) {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }
  const width = maxX - minX
  const height = maxY - minY
  const left = minX + (width * Math.max(0, Math.min(100, mask.cropLeft))) / 100
  const right = maxX - (width * Math.max(0, Math.min(100, mask.cropRight))) / 100
  const top = minY + (height * Math.max(0, Math.min(100, mask.cropTop))) / 100
  const bottom = maxY - (height * Math.max(0, Math.min(100, mask.cropBottom))) / 100

  const random = createRandom(mask.dropoutSeed)
  const kept: PatternPoint[] = []
  for (const point of points) {
    const roll = random()
    if (point.x < left || point.x > right || point.y < top || point.y > bottom) {
      continue
    }
    if (roll * 100 < mask.dropoutPercent) {
      continue
    }
    kept.push(point)
  }
  return kept
}
