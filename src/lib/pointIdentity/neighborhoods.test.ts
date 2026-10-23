import { describe, expect, it } from 'vitest'

import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { applyObservationMask } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { computeNeighborhoods } from '@/lib/pointIdentity/proximityGraph'
import { createRandom } from '@/lib/pointIdentity/random'
import type { PatternPoint } from '@/lib/pointIdentity/types'

const NEIGHBOR_SAMPLE = 12
const LOCAL_SCALE_SAMPLE = 6

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

/** The original O(n²) implementation, kept as the reference oracle. */
function reference(points: readonly PatternPoint[]): {
  neighborLists: number[][]
  localScales: number[]
} {
  const count = points.length
  const neighborLists: number[][] = []
  const localScales: number[] = []
  const bestDistances = new Float64Array(NEIGHBOR_SAMPLE)
  const bestIndices = new Int32Array(NEIGHBOR_SAMPLE)

  for (let i = 0; i < count; i += 1) {
    const origin = points[i]!
    bestDistances.fill(Number.POSITIVE_INFINITY)
    bestIndices.fill(-1)
    for (let j = 0; j < count; j += 1) {
      if (j === i) {
        continue
      }
      const length = distance(origin, points[j]!)
      let position = NEIGHBOR_SAMPLE
      for (let slot = 0; slot < NEIGHBOR_SAMPLE; slot += 1) {
        const slotDistance = bestDistances[slot]!
        const tied = Math.abs(length - slotDistance) < 1e-6
        if (tied ? j < bestIndices[slot]! : length < slotDistance) {
          position = slot
          break
        }
      }
      if (position >= NEIGHBOR_SAMPLE) {
        continue
      }
      for (let slot = NEIGHBOR_SAMPLE - 1; slot > position; slot -= 1) {
        bestDistances[slot] = bestDistances[slot - 1]!
        bestIndices[slot] = bestIndices[slot - 1]!
      }
      bestDistances[position] = length
      bestIndices[position] = j
    }

    const neighborCount = Math.min(NEIGHBOR_SAMPLE, Math.max(0, count - 1))
    const neighbors: number[] = []
    for (let slot = 0; slot < neighborCount; slot += 1) {
      neighbors.push(bestIndices[slot]!)
    }
    neighborLists.push(neighbors)

    const scaleCount = Math.min(LOCAL_SCALE_SAMPLE, neighborCount)
    let localScale = 0
    if (scaleCount > 0) {
      const middle = scaleCount >> 1
      localScale =
        scaleCount % 2 === 1
          ? bestDistances[middle]!
          : (bestDistances[middle - 1]! + bestDistances[middle]!) / 2
    }
    localScales.push(Math.max(localScale, 1e-6))
  }
  return { neighborLists, localScales }
}

function toPoints(raw: ReadonlyArray<{ x: number; y: number }>): PatternPoint[] {
  return raw.map((point, index) => ({ x: point.x, y: point.y, index }))
}

function expectSame(points: readonly PatternPoint[]): void {
  const expected = reference(points)
  const actual = computeNeighborhoods(points)
  expect(actual.neighborLists).toEqual(expected.neighborLists)
  expect(actual.localScales).toEqual(expected.localScales)
}

function projectedTarget(options: {
  yawDeg?: number
  pitchDeg?: number
  sphericalDistortion?: number
  dropoutPercent?: number
  pointsPerEdge?: number
}): PatternPoint[] {
  const pattern = generateHexPattern({
    columns: 8,
    pointsPerEdge: options.pointsPerEdge ?? 4,
  })
  const projected = projectPattern(pattern.points, {
    ...DEFAULT_VIEW,
    yawDeg: options.yawDeg ?? 0,
    pitchDeg: options.pitchDeg ?? 0,
    sphericalDistortion: options.sphericalDistortion ?? 0,
  })
  const masked = applyObservationMask(projected, {
    cropLeft: 0,
    cropRight: 0,
    cropTop: 0,
    cropBottom: 0,
    dropoutPercent: options.dropoutPercent ?? 0,
    dropoutSeed: 7,
  })
  return toPoints(masked)
}

describe('grid neighbourhoods', () => {
  it('matches brute force on a flat projected target', () => {
    expectSame(projectedTarget({}))
  })

  it('matches brute force across perspective and distortion', () => {
    for (const yawDeg of [0, 17, -29]) {
      for (const pitchDeg of [0, 12]) {
        for (const sphericalDistortion of [0, 0.4]) {
          expectSame(projectedTarget({ yawDeg, pitchDeg, sphericalDistortion }))
        }
      }
    }
  })

  it('matches brute force with dropped points', () => {
    for (const dropoutPercent of [5, 20, 45]) {
      expectSame(projectedTarget({ dropoutPercent }))
    }
  })

  it('matches brute force on a denser target', () => {
    expectSame(projectedTarget({ pointsPerEdge: 5 }))
  })

  it('matches brute force on uniform, clustered and collinear fields', () => {
    for (const seed of [1, 2, 3]) {
      const random = createRandom(seed)
      const uniform = toPoints(
        Array.from({ length: 400 }, () => ({ x: random() * 640, y: random() * 480 })),
      )
      expectSame(uniform)

      // Two clusters at very different densities exercise the ring expansion.
      const clustered = toPoints(
        Array.from({ length: 400 }, () =>
          random() < 0.85
            ? { x: random() * 80, y: random() * 60 }
            : { x: 400 + random() * 600, y: 300 + random() * 500 },
        ),
      )
      expectSame(clustered)

      const collinear = toPoints(
        Array.from({ length: 200 }, (_, index) => ({ x: index * 3, y: 0 })),
      )
      expectSame(collinear)
    }
  })

  it('matches brute force with exact duplicates and ties', () => {
    const random = createRandom(9)
    const points: PatternPoint[] = []
    for (let index = 0; index < 200; index += 1) {
      // A coarse grid of positions so exact duplicate distances are frequent.
      points.push({
        x: Math.round(random() * 4) * 10,
        y: Math.round(random() * 4) * 10,
        index,
      })
    }
    expectSame(points)
  })

  it('matches brute force for tiny inputs', () => {
    for (const count of [0, 1, 2, 3, 12, 13]) {
      const random = createRandom(count + 1)
      expectSame(
        toPoints(Array.from({ length: count }, () => ({ x: random() * 100, y: random() * 100 }))),
      )
    }
  })
})
