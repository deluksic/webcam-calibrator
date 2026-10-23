import { describe, expect, it } from 'vitest'

import { DEFAULT_TARGET_CONFIG } from '@/lib/hexTarget/config'
import { createHexTargetDesign, defaultHexTargetModel } from '@/lib/hexTarget/target'

describe('hex target model', () => {
  it('builds the default production target', () => {
    const model = defaultHexTargetModel()
    expect(model.pattern.points.length).toBeGreaterThan(100)
    expect(model.objectPoints).toHaveLength(model.pattern.points.length)
    expect(model.indexesByDepth).toHaveLength(DEFAULT_TARGET_CONFIG.identityDepth)
    expect(model.spacing).toBeGreaterThan(0)
    for (const point of model.objectPoints) {
      expect(point.z).toBe(0)
      expect(Number.isFinite(point.x)).toBe(true)
      expect(Number.isFinite(point.y)).toBe(true)
    }
    expect(model.pointsPerEdge).toBe(DEFAULT_TARGET_CONFIG.pointsPerEdge)
    expect(model.pattern.stats.rows).toBe(DEFAULT_TARGET_CONFIG.rows)
  })

  it('is deterministic for the same parameters', () => {
    const first = createHexTargetDesign(DEFAULT_TARGET_CONFIG)
    const second = createHexTargetDesign(DEFAULT_TARGET_CONFIG)
    expect(first.colors).toEqual(second.colors)
    expect(first.objectPoints).toEqual(second.objectPoints)
  })

  it('honors explicit row counts', () => {
    const model = createHexTargetDesign({ ...DEFAULT_TARGET_CONFIG, rows: 6 })
    expect(model.pattern.stats.rows).toBe(6)
  })

  it('normalizes object coordinates by the median lattice spacing', () => {
    const model = defaultHexTargetModel()
    const distances: number[] = []
    for (let i = 0; i < Math.min(32, model.objectPoints.length); i += 1) {
      let nearest = Number.POSITIVE_INFINITY
      const a = model.objectPoints[i]!
      for (let j = 0; j < model.objectPoints.length; j += 1) {
        if (i === j) {
          continue
        }
        const b = model.objectPoints[j]!
        nearest = Math.min(nearest, Math.hypot(a.x - b.x, a.y - b.y))
      }
      if (Number.isFinite(nearest)) {
        distances.push(nearest)
      }
    }
    expect(distances.length).toBeGreaterThan(0)
    distances.sort((a, b) => a - b)
    const median = distances[Math.floor(distances.length / 2)]!
    expect(median).toBeGreaterThan(0.5)
    expect(median).toBeLessThan(1.5)
  })
})
