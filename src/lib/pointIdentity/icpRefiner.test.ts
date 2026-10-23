import { describe, expect, it } from 'vitest'

import type { IdentityMatchResult } from '@/lib/pointIdentity/match'
import { matchObservedIdentity } from '@/lib/pointIdentity/match'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { createIdentityIndexesByDepth } from '@/lib/pointIdentity/identity'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { projectPattern, DEFAULT_VIEW } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { createRandom, randomBinaryColors } from '@/lib/pointIdentity/random'
import type { BinaryColor, ProximityGraph, ViewPose } from '@/lib/pointIdentity/types'
import {
  createIcpRefineOptions,
  refineObservedIdentityWithIcp,
  type IcpRefineResult,
} from '@/lib/pointIdentity/icpRefiner'

const IDENTITY_DEPTH = 6
const MAX_PROOF_DEPTH = 8
const PATTERN = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
const DESIGN_GRAPH = buildProximityGraph(PATTERN.points, { pointsPerEdge: 4 })

type ScenarioOptions = {
  seed?: number
  cropLeft?: number
  cropTop?: number
  dropoutPercent?: number
  yawDeg?: number
  pitchDeg?: number
  sphericalDistortion?: number
  jitterPercent?: number
  wrongSeed?: number
}

type Scenario = {
  result: IcpRefineResult
  observedGraph: ProximityGraph
  baseline: IdentityMatchResult
  truthIds: Set<number>
  correct: number
  falseMatches: number
  recoveredFalse: number
  predictedMissingFalse: number
  conflicts: number
  total: number
}

function scenario(options: ScenarioOptions = {}): Scenario {
  const seed = options.seed ?? 1
  const targetColors = randomBinaryColors(PATTERN.points.length, seed)
  const colors =
    options.wrongSeed === undefined
      ? targetColors
      : randomBinaryColors(PATTERN.points.length, options.wrongSeed)
  const indexes = createIdentityIndexesByDepth(DESIGN_GRAPH, colors, MAX_PROOF_DEPTH)
  const pose: ViewPose = {
    ...DEFAULT_VIEW,
    yawDeg: options.yawDeg ?? 0,
    pitchDeg: options.pitchDeg ?? 0,
    sphericalDistortion: options.sphericalDistortion ?? 0,
  }
  const projected = projectPattern(PATTERN.points, pose)
  const mask = {
    ...DEFAULT_OBSERVATION_MASK,
    cropLeft: options.cropLeft ?? 0,
    cropTop: options.cropTop ?? 0,
    dropoutPercent: options.dropoutPercent ?? 0,
  }
  let observed = applyObservationMask(projected, mask)
  if ((options.jitterPercent ?? 0) > 0) {
    const random = createRandom((DEFAULT_OBSERVATION_MASK.dropoutSeed + 0x51ed270b) >>> 0)
    const spacing = DESIGN_GRAPH.stats.localScaleMedian
    const amplitude = spacing * ((options.jitterPercent ?? 0) / 100)
    observed = observed.map((point) => ({
      ...point,
      x: point.x + (random() * 2 - 1) * amplitude,
      y: point.y + (random() * 2 - 1) * amplitude,
    }))
  }
  const observedGraph = buildProximityGraph(observed, { pointsPerEdge: 4 })
  const observedColors = observedGraph.points.map((point) => targetColors[point.index] ?? 0)
  const baseline = matchObservedIdentity(
    indexes[IDENTITY_DEPTH - 1]!,
    observedGraph,
    observedColors,
    IDENTITY_DEPTH,
  )

  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const point of projected) {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }

  const result = refineObservedIdentityWithIcp({
    designPattern: PATTERN,
    designColors: colors as BinaryColor[],
    designGraph: DESIGN_GRAPH,
    designIndexesByDepth: indexes,
    observedGraph,
    observedColors,
    baseline,
    options: createIcpRefineOptions({
      identityDepth: IDENTITY_DEPTH,
      maxProofDepth: MAX_PROOF_DEPTH,
      frameBounds: { minX, minY, maxX, maxY },
    }),
  })

  const truthIds = new Set(observedGraph.points.map((point) => point.index))
  let correct = 0
  let falseMatches = 0
  let recoveredFalse = 0
  let conflicts = 0
  for (let position = 0; position < result.points.length; position += 1) {
    const point = result.points[position]!
    const truth = observedGraph.points[position]!.index
    if (point.status === 3) {
      conflicts += 1
    }
    if (point.designId === undefined) {
      continue
    }
    if (point.designId === truth) {
      correct += 1
    } else {
      falseMatches += 1
      if (point.status === 4) {
        recoveredFalse += 1
      }
    }
  }
  let predictedMissingFalse = 0
  for (const missing of result.predictedMissing) {
    if (truthIds.has(missing.designId)) {
      predictedMissingFalse += 1
    }
  }

  return {
    result,
    observedGraph,
    baseline,
    truthIds,
    correct,
    falseMatches,
    recoveredFalse,
    predictedMissingFalse,
    conflicts,
    total: observedGraph.points.length,
  }
}

describe('icpRefiner', () => {
  it('leaves a clean scene fully resolved', () => {
    const current = scenario()
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBe(0)
    expect(current.correct).toBe(current.total)
    expect(current.falseMatches).toBe(0)
  }, 30000)

  it('recovers a left crop without false matches', () => {
    const current = scenario({ cropLeft: 20 })
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBeGreaterThan(0)
    expect(current.falseMatches).toBe(0)
    expect(current.recoveredFalse).toBe(0)
    expect(current.correct).toBeGreaterThanOrEqual(current.total - 1)
    expect(current.predictedMissingFalse).toBe(0)
  }, 30000)

  it('recovers dropout without false matches', () => {
    const current = scenario({ dropoutPercent: 10 })
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBeGreaterThan(0)
    expect(current.falseMatches).toBe(0)
    expect(current.recoveredFalse).toBe(0)
    expect(current.correct).toBe(current.total)
  }, 30000)

  it('handles sparse dropout by recovering or safely rejecting', () => {
    for (const seed of [1, 2, 3]) {
      const current = scenario({ seed, dropoutPercent: 25 })
      if (current.result.accepted) {
        expect(current.falseMatches).toBe(0)
        expect(current.recoveredFalse).toBe(0)
        expect(current.correct).toBeGreaterThanOrEqual(current.total - 1)
      } else {
        expect(current.result.metrics.recovered).toBe(0)
        expect(current.recoveredFalse).toBe(0)
      }
    }
  }, 30000)

  it('recovers under strong perspective and edge crop', () => {
    const current = scenario({ cropLeft: 10, yawDeg: 40 })
    expect(current.result.accepted).toBe(true)
    expect(current.falseMatches).toBe(0)
    expect(current.recoveredFalse).toBe(0)
    expect(current.correct).toBe(current.total)
  }, 30000)

  it('recovers under pitch and top crop', () => {
    const current = scenario({ cropTop: 10, pitchDeg: 30 })
    expect(current.result.accepted).toBe(true)
    expect(current.falseMatches).toBe(0)
    expect(current.recoveredFalse).toBe(0)
    expect(current.correct).toBe(current.total)
  }, 30000)

  it('recovers with position jitter and crop', () => {
    const current = scenario({ cropLeft: 20, jitterPercent: 2 })
    expect(current.result.accepted).toBe(true)
    expect(current.falseMatches).toBe(0)
    expect(current.recoveredFalse).toBe(0)
    expect(current.correct).toBeGreaterThanOrEqual(current.total - 1)
  }, 30000)

  it('rejects a wrong color seed without recovering false identities', () => {
    for (const seed of [1, 2, 3]) {
      const current = scenario({ seed, cropLeft: 20, wrongSeed: seed + 100 })
      if (current.result.accepted) {
        expect(current.recoveredFalse).toBe(0)
      } else {
        expect(current.result.metrics.recovered).toBe(0)
      }
      expect(current.recoveredFalse).toBe(0)
    }
  }, 30000)

  it('is stable across seeds for crop recovery', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const current = scenario({ seed, cropLeft: 20 })
      expect(current.result.accepted).toBe(true)
      expect(current.falseMatches).toBe(0)
      expect(current.recoveredFalse).toBe(0)
      expect(current.correct).toBeGreaterThanOrEqual(current.total - 1)
    }
  }, 60000)
})
