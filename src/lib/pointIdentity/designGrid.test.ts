import { describe, expect, it } from 'vitest'

import {
  computeDesignLatticeMetrics,
  createDesignGrid,
  createDesignPositionMap,
} from '@/lib/pointIdentity/designGrid'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { PatternPoint, Vec2 } from '@/lib/pointIdentity/types'

function bruteForceWithin(points: readonly PatternPoint[], center: Vec2, radius: number): number[] {
  const result: number[] = []
  for (let position = 0; position < points.length; position += 1) {
    const point = points[position]!
    if (Math.hypot(point.x - center.x, point.y - center.y) <= radius) {
      result.push(position)
    }
  }
  return result.sort((first, second) => first - second)
}

describe('designGrid', () => {
  const pattern = generateHexPattern({ columns: 4, pointsPerEdge: 2 })
  const colors = randomBinaryColors(pattern.points.length, 3)
  const grid = createDesignGrid(pattern.points, colors, 12)

  it('matches brute-force radius queries', () => {
    for (const center of [
      { x: 0, y: 0 },
      { x: 37, y: -22 },
      { x: -90, y: 61 },
      { x: 200, y: 200 },
    ]) {
      for (const radius of [4, 12, 25]) {
        expect(grid.within(center, radius)).toEqual(bruteForceWithin(pattern.points, center, radius))
      }
    }
  })

  it('matches brute-force nearest queries and color filters', () => {
    for (const center of [
      { x: 0, y: 0 },
      { x: 37, y: -22 },
      { x: -90, y: 61 },
    ]) {
      for (const color of [0, 1] as const) {
        const nearest = grid.nearest(center, 40, color)
        let expected: number | undefined
        let expectedDistance = Number.POSITIVE_INFINITY
        for (let position = 0; position < pattern.points.length; position += 1) {
          const point = pattern.points[position]!
          if ((colors[position] ?? 0) !== color) {
            continue
          }
          const distance = Math.hypot(point.x - center.x, point.y - center.y)
          if (distance <= 40 && distance < expectedDistance) {
            expected = position
            expectedDistance = distance
          }
        }
        expect(nearest).toBe(expected)
      }
    }
  })

  it('computes uniform lattice metrics and the design-id position map', () => {
    const points: PatternPoint[] = [
      { x: 0, y: 0, index: 10 },
      { x: 10, y: 0, index: 11 },
      { x: 0, y: 10, index: 12 },
    ]
    const metrics = computeDesignLatticeMetrics(points)
    expect(metrics.medianNearest).toBe(10)
    expect(metrics.minNearest).toBe(10)
    const map = createDesignPositionMap(points)
    expect(map.get(10)).toBe(0)
    expect(map.get(11)).toBe(1)
    expect(map.get(12)).toBe(2)
  })
})
