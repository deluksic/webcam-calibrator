import { describe, expect, it } from 'vitest'

import { findChainCandidates } from '@/lib/pointIdentity/chainCandidates'
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
