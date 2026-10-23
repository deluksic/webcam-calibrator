import { describe, expect, it } from 'vitest'

import { findChainCandidates } from '@/lib/pointIdentity/chainCandidates'
import { createRandom } from '@/lib/pointIdentity/random'
import type { PatternPoint } from '@/lib/pointIdentity/types'

function linePoints(count: number): PatternPoint[] {
  return Array.from({ length: count }, (_, index) => ({ x: index * 10, y: 0, index }))
}

function neighborLists(points: readonly PatternPoint[]): number[][] {
  return points.map((point) =>
    points
      .filter((other) => other.index !== point.index)
      .sort((first, second) => {
        const firstDistance = Math.hypot(first.x - point.x, first.y - point.y)
        const secondDistance = Math.hypot(second.x - point.x, second.y - point.y)
        return firstDistance === secondDistance ? first.index - second.index : firstDistance - secondDistance
      })
      .map((other) => other.index),
  )
}

describe('findChainCandidates', () => {
  it('finds one exact chain on a four-point line', () => {
    const points = linePoints(4)
    const candidates = findChainCandidates(
      points,
      neighborLists(points),
      points.map(() => 10),
      4,
    )
    expect(candidates).toHaveLength(1)
    expect(candidates[0]!.pointIndices).toEqual([0, 1, 2, 3])
  })

  it('produces overlapping windows on a longer line', () => {
    const points = linePoints(5)
    const candidates = findChainCandidates(
      points,
      neighborLists(points),
      points.map(() => 10),
      4,
    )
    expect(candidates).toHaveLength(2)
    expect(candidates.map((candidate) => candidate.pointIndices).sort()).toEqual([
      [0, 1, 2, 3],
      [1, 2, 3, 4],
    ])
  })

  it('returns nothing when the line is shorter than N', () => {
    const points = linePoints(3)
    expect(
      findChainCandidates(
        points,
        neighborLists(points),
        points.map(() => 10),
        4,
      ),
    ).toEqual([])
  })
})

describe('findChainCandidates under position noise', () => {
  function jittered(points: readonly PatternPoint[], amplitude: number, seed: number): PatternPoint[] {
    const random = createRandom(seed)
    return points.map((point) => ({
      ...point,
      x: point.x + (random() * 2 - 1) * amplitude,
      y: point.y + (random() * 2 - 1) * amplitude,
    }))
  }

  it('recovers a jittered four-point line', () => {
    const points = jittered(linePoints(4), 1.2, 7)
    const candidates = findChainCandidates(
      points,
      neighborLists(points),
      points.map(() => 10),
      4,
    )
    expect(candidates).toHaveLength(1)
    expect(candidates[0]!.pointIndices).toEqual([0, 1, 2, 3])
  })

  it('does not bridge two parallel jittered lines', () => {
    const points = jittered(
      [
        { x: 0, y: 0, index: 0 },
        { x: 10, y: 0, index: 1 },
        { x: 20, y: 0, index: 2 },
        { x: 30, y: 0, index: 3 },
        { x: 0, y: 40, index: 4 },
        { x: 10, y: 40, index: 5 },
        { x: 20, y: 40, index: 6 },
        { x: 30, y: 40, index: 7 },
      ],
      1.5,
      11,
    )
    const candidates = findChainCandidates(
      points,
      neighborLists(points),
      points.map(() => 10),
      4,
    )
    expect(candidates.length).toBeGreaterThanOrEqual(2)
    for (const candidate of candidates) {
      const inFirstLine = candidate.pointIndices.every((index) => index < 4)
      const inSecondLine = candidate.pointIndices.every((index) => index >= 4)
      expect(inFirstLine || inSecondLine).toBe(true)
    }
  })
})
