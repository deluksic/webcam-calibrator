import { describe, expect, it } from 'vitest'

import { applyHomography, type Mat3 } from '@/lib/geometry'
import type { Anchor } from '@/lib/pointIdentity/anchors'
import { runDesignSpaceIcp, type IcpOptions } from '@/lib/pointIdentity/designSpaceIcp'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { BinaryColor, PatternPoint, Vec2 } from '@/lib/pointIdentity/types'

const TRANSFORM: Mat3 = [1.1, 0.08, 120, -0.04, 1.05, 90, 0.0004, 0.00015, 1]

const OPTIONS: IcpOptions = {
  maxIterations: 10,
  designGateFactor: 0.4,
  anchorGateFactor: 0.2,
  minInitialInliers: 6,
  minInitialInlierRatio: 0.5,
  minPairCoverage: 0.5,
  maxDesignRmsFactor: 0.35,
  maxColorRejectRate: 0.1,
  holdoutAnchorFraction: 0.2,
  minObservedSpacingPx: 0,
  seed: 1,
}

type Scene = {
  designPoints: PatternPoint[]
  colors: BinaryColor[]
  observedPoints: Vec2[]
  observedColors: BinaryColor[]
  anchors: Anchor[]
  truthByObserved: number[]
}

function scene(options?: { skipPosition?: number; flipColorPosition?: number }): Scene {
  const pattern = generateHexPattern({ columns: 5, pointsPerEdge: 3 })
  const colors = randomBinaryColors(pattern.points.length, 11)
  const anchors: Anchor[] = []
  const observedPoints: Vec2[] = []
  const observedColors: BinaryColor[] = []
  const truthByObserved: number[] = []

  for (let position = 0; position < pattern.points.length; position += 1) {
    if (options?.skipPosition === position) {
      continue
    }
    const design = pattern.points[position]!
    const projected = applyHomography(TRANSFORM, design.x, design.y)
    const observedIndex = observedPoints.length
    observedPoints.push({ x: projected.x, y: projected.y })
    const color = options?.flipColorPosition === position ? ((colors[position] ?? 0) === 0 ? 1 : 0) : (colors[position] ?? 0)
    observedColors.push(color)
    truthByObserved.push(design.index)
    anchors.push({
      observedPosition: observedIndex,
      designId: design.index,
      proofPower: 6,
      design: { x: design.x, y: design.y },
      observed: { x: projected.x, y: projected.y },
      color: colors[position] ?? 0,
    })
  }

  return {
    designPoints: pattern.points,
    colors,
    observedPoints,
    observedColors,
    anchors,
    truthByObserved,
  }
}

function run(current: Scene, anchors = current.anchors) {
  return runDesignSpaceIcp({
    designPoints: current.designPoints,
    designColors: current.colors,
    observedPoints: current.observedPoints,
    observedColors: current.observedColors,
    anchors,
    options: OPTIONS,
  })
}

describe('runDesignSpaceIcp', () => {
  it('recovers a clean perspective lattice', () => {
    const current = scene()
    const result = run(current)
    expect(result.accepted).toBe(true)
    expect(result.pairs).toHaveLength(current.designPoints.length)
    expect(result.designRms).toBeLessThan(1e-4)
    expect(result.colorRejects).toBe(0)
    for (const pair of result.pairs) {
      expect(pair.designId).toBe(current.truthByObserved[pair.observedPosition])
    }
  })

  it('detects a skipped point in design space instead of stealing a neighbor', () => {
    const current = scene({ skipPosition: 20 })
    const result = run(current)
    expect(result.accepted).toBe(true)
    expect(result.pairs).toHaveLength(current.designPoints.length - 1)
    const skippedDesignId = current.designPoints[20]!.index
    expect(result.pairs.some((pair) => pair.designId === skippedDesignId)).toBe(false)
    for (const pair of result.pairs) {
      expect(pair.designId).toBe(current.truthByObserved[pair.observedPosition])
    }
  })

  it('does not pair a color-mismatched observation', () => {
    const current = scene({ flipColorPosition: 12 })
    // Leave the flipped point out of the forced-anchor set so the color gate is
    // actually exercised rather than bypassed by a trusted anchor.
    const result = run(
      current,
      current.anchors.filter((_, index) => index % 2 === 1),
    )
    expect(result.accepted).toBe(true)
    expect(result.pairs).toHaveLength(current.designPoints.length - 1)
    expect(result.colorRejects).toBeGreaterThan(0)
    for (const pair of result.pairs) {
      expect(pair.designId).toBe(current.truthByObserved[pair.observedPosition])
    }
  })

  it('rejects a mismatched anchor set instead of inventing a transform', () => {
    const current = scene()
    const shifted = current.anchors.map((anchor, index) => ({
      ...anchor,
      designId: current.designPoints[(index + 13) % current.designPoints.length]!.index,
    }))
    const result = run(current, shifted)
    expect(result.accepted).toBe(false)
  })
})
