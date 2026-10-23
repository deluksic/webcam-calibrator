import { describe, expect, it } from 'vitest'

import { matchHexTargetPoints } from '@/lib/hexTarget/match'
import { defaultHexTargetModel } from '@/lib/hexTarget/target'

describe('matchHexTargetPoints', () => {
  it('resolves a clean, uniformly scaled target view', () => {
    const model = defaultHexTargetModel()
    const observed = model.pattern.points.map((point) => ({
      x: point.x * 0.5 + 120,
      y: point.y * 0.5 + 80,
      color: model.colors[point.index] ?? 0,
    }))
    const match = matchHexTargetPoints(model, observed, {
      recovery: true,
      depth: 12,
    })
    expect(match.summary.total).toBe(observed.length)
    expect(match.summary.resolved).toBeGreaterThan(observed.length * 0.8)
    expect(match.summary.conflicts).toBe(0)
    expect(match.edges.length).toBeGreaterThan(0)
  })
})
