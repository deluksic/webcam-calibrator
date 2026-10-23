import { describe, expect, it } from 'vitest'

import {
  createIdentityDesignBundle,
  identityDesignKey,
  runIdentityPipeline,
  type IdentityDesignSpec,
} from '@/lib/pointIdentity/identityPipeline'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'

const SPEC: IdentityDesignSpec = { columns: 8, pointsPerEdge: 4, colorSeed: 1 }

function observedPoints(dropoutPercent: number) {
  const design = createIdentityDesignBundle(SPEC)
  const projected = projectPattern(design.pattern.points, DEFAULT_VIEW)
  const masked = applyObservationMask(projected, {
    ...DEFAULT_OBSERVATION_MASK,
    dropoutPercent,
  })
  return {
    design,
    points: masked.map((point) => ({
      x: point.x,
      y: point.y,
      color: (design.colors[point.index] ?? 0) as 0 | 1,
    })),
  }
}

describe('identity pipeline', () => {
  it('keys the design by every setting that shapes it', () => {
    expect(identityDesignKey(SPEC)).toBe('8:4:1')
    expect(identityDesignKey({ ...SPEC, colorSeed: 2 })).not.toBe(identityDesignKey(SPEC))
    expect(identityDesignKey({ ...SPEC, pointsPerEdge: 3 })).not.toBe(identityDesignKey(SPEC))
  })

  it('resolves every point when nothing is dropped', () => {
    const { design, points } = observedPoints(0)
    const { graph, metrics } = runIdentityPipeline({
      design,
      points,
      depth: 12,
      recovery: true,
      frameId: 7,
    })
    expect(metrics.total).toBe(points.length)
    expect(metrics.resolved).toBe(points.length)
    expect(metrics.unresolved).toBe(0)
    expect(metrics.frameId).toBe(7)
    expect(graph.points).toHaveLength(metrics.total)
    expect(graph.edges).toHaveLength(metrics.edges)
  })

  it('recovers identity from support anchors after dropout', () => {
    const { design, points } = observedPoints(10)
    const { metrics } = runIdentityPipeline({
      design,
      points,
      depth: 12,
      recovery: true,
      frameId: 1,
    })
    expect(metrics.total).toBe(points.length)
    expect(metrics.recovered).toBeGreaterThan(0)
    expect(metrics.resolved + metrics.recovered).toBeGreaterThan(0)
  })

  it('reuses a cached loose index for the non-recovery path', () => {
    const { design, points } = observedPoints(5)
    const first = runIdentityPipeline({ design, points, depth: 12, recovery: false, frameId: 1 })
    const second = runIdentityPipeline({ design, points, depth: 12, recovery: false, frameId: 2 })
    expect(design.looseIndexesByDepth.size).toBe(1)
    expect(first.graph).toEqual(second.graph)
  })
})
