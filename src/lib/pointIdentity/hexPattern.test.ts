import { describe, expect, it } from 'vitest'

import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'

const BASE = {
  columns: 8,
  pointsPerEdge: 2,
} as const

describe('generateHexPattern', () => {
  it('is deterministic', () => {
    const first = generateHexPattern(BASE)
    const second = generateHexPattern(BASE)
    expect(first.points).toEqual(second.points)
    expect(first.guides).toEqual(second.guides)
    expect(first.stats).toEqual(second.stats)
  })

  it('builds a honeycomb with equal-length cell edges', () => {
    const pattern = generateHexPattern(BASE)
    expect(pattern.stats.cells).toBeGreaterThan(0)
    expect(pattern.stats.candidateEdges).toBe(pattern.guides.length)
    expect(pattern.stats.candidateEdges).toBeLessThan(pattern.stats.cells * 6)

    const first = pattern.guides[0]
    expect(first).toBeDefined()
    if (first === undefined) {
      return
    }
    const sideLength = Math.hypot(first.b.x - first.a.x, first.b.y - first.a.y)
    for (const guide of pattern.guides) {
      expect(Math.hypot(guide.b.x - guide.a.x, guide.b.y - guide.a.y)).toBeCloseTo(sideLength, 9)
    }
  })

  it('samples points at half-step offsets away from cell corners', () => {
    const pattern = generateHexPattern({ ...BASE, pointsPerEdge: 2 })
    expect(pattern.stats.points).toBe(pattern.stats.sampledEdges * 2)

    for (const point of pattern.points) {
      let closest = Number.POSITIVE_INFINITY
      let closestT = 0
      for (const guide of pattern.guides) {
        const dx = guide.b.x - guide.a.x
        const dy = guide.b.y - guide.a.y
        const lengthSquared = dx * dx + dy * dy
        const t = Math.max(0, Math.min(1, ((point.x - guide.a.x) * dx + (point.y - guide.a.y) * dy) / lengthSquared))
        const distance = Math.hypot(point.x - (guide.a.x + dx * t), point.y - (guide.a.y + dy * t))
        if (distance < closest) {
          closest = distance
          closestT = t
        }
      }
      expect(closest).toBeLessThan(1e-6)
      expect(closestT).toBeGreaterThan(0.2)
      expect(closestT).toBeLessThan(0.8)
    }
  })

  it('respects the points-per-edge setting', () => {
    const dense = generateHexPattern({ ...BASE, pointsPerEdge: 4 })
    const sparse = generateHexPattern({ ...BASE, pointsPerEdge: 1 })
    expect(dense.stats.points).toBe(dense.stats.sampledEdges * 4)
    expect(sparse.stats.points).toBe(sparse.stats.sampledEdges)
  })

  it('centres the generated field bounds', () => {
    const pattern = generateHexPattern(BASE)
    expect(pattern.bounds.minX + pattern.bounds.maxX).toBeCloseTo(0, 6)
    expect(pattern.bounds.minY + pattern.bounds.maxY).toBeCloseTo(0, 6)
  })
})
