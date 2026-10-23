import { describe, expect, it } from 'vitest'

import { runIdentityStress } from '@/lib/pointIdentity/stress'

describe('identity stress recovery', () => {
  it('reports refined metrics for a cropped field', () => {
    const report = runIdentityStress({
      runs: 3,
      depth: 12,
      pointsPerEdge: 3,
      columns: 6,
      cropLeft: 20,
      recovery: true,
    })
    expect(report.recovery).toBeDefined()
    expect(report.recovery!.meanRefinedAccuracy).toBeGreaterThan(0.5)
    expect(report.recovery!.refinedFalseMatches).toBe(0)
  }, 30000)

  it('is backward compatible when recovery is disabled', () => {
    const report = runIdentityStress({
      runs: 2,
      depth: 5,
      pointsPerEdge: 3,
      columns: 6,
    })
    expect(report.recovery).toBeUndefined()
    expect(report.runs).toBe(2)
  }, 30000)
})
