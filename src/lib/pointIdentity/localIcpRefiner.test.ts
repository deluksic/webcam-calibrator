import { describe, expect, it } from 'vitest'

import { createIdentityIndexesByDepth } from '@/lib/pointIdentity/identity'
import { createLocalIcpOptions, refineObservedIdentityWithLocalIcp } from '@/lib/pointIdentity/localIcpRefiner'
import {
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  matchObservedIdentity,
} from '@/lib/pointIdentity/match'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { applyObservationMask, DEFAULT_OBSERVATION_MASK } from '@/lib/pointIdentity/observation'
import { projectPattern, DEFAULT_VIEW } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { BinaryColor, ProximityGraph, ViewPose } from '@/lib/pointIdentity/types'

const IDENTITY_DEPTH = 12
const MAX_PROOF_DEPTH = 12
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
  hopRadius?: number
}

type Scenario = {
  result: ReturnType<typeof refineObservedIdentityWithLocalIcp>
  observedGraph: ProximityGraph
  baseline: ReturnType<typeof matchObservedIdentity>
  correct: number
  falseMatches: number
  recoveredFalse: number
  ambiguousBaseline: number
  total: number
}

function scenario(options: ScenarioOptions = {}): Scenario {
  const seed = options.seed ?? 1
  const colors: BinaryColor[] = randomBinaryColors(PATTERN.points.length, seed)
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
  const observed = applyObservationMask(projected, mask)
  const observedGraph = buildProximityGraph(observed, { pointsPerEdge: 4 })
  const observedColors = observedGraph.points.map((point) => colors[point.index] ?? 0)
  // matchObservedIdentity indexes colors by graph point `.index` inside its
  // fingerprint pass; in Point Lab that index is the design id.
  const baseline = matchObservedIdentity(
    indexes[IDENTITY_DEPTH - 1]!,
    observedGraph,
    colors,
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

  const result = refineObservedIdentityWithLocalIcp({
    designPattern: PATTERN,
    designColors: colors,
    designGraph: DESIGN_GRAPH,
    designIndexesByDepth: indexes,
    observedGraph,
    observedColors,
    baseline,
    options: createLocalIcpOptions({
      identityDepth: IDENTITY_DEPTH,
      maxProofDepth: MAX_PROOF_DEPTH,
      hopRadius: options.hopRadius ?? 7,
      frameBounds: { minX, minY, maxX, maxY },
    }),
  })

  let correct = 0
  let falseMatches = 0
  let recoveredFalse = 0
  let ambiguousBaseline = 0
  for (let position = 0; position < baseline.points.length; position += 1) {
    const baselinePoint = baseline.points[position]!
    if (baselinePoint.status === 1 || baselinePoint.status === 2) {
      ambiguousBaseline += 1
    }
  }
  for (let position = 0; position < result.points.length; position += 1) {
    const point = result.points[position]!
    const truth = observedGraph.points[position]!.index
    if (point.designId === undefined) {
      continue
    }
    if (point.designId === truth) {
      correct += 1
    } else {
      falseMatches += 1
      if (point.status === IDENTITY_RECOVERED) {
        recoveredFalse += 1
      }
    }
  }

  return {
    result,
    observedGraph,
    baseline,
    correct,
    falseMatches,
    recoveredFalse,
    ambiguousBaseline,
    total: observedGraph.points.length,
  }
}

describe('local hop-graph ICP', () => {
  it('is a no-op when the hash has no ambiguous points', () => {
    const current = scenario()
    expect(current.result.accepted).toBe(false)
    expect(current.result.rejectionReason).toBe('no-ambiguous-points')
    expect(current.falseMatches).toBe(0)
  }, 30000)

  it('locks anchors and refines points from support anchors', () => {
    const current = scenario({ cropLeft: 20 })
    expect(current.ambiguousBaseline).toBeGreaterThan(0)
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBeGreaterThan(0)
    expect(current.falseMatches).toBe(0)
    expect(current.recoveredFalse).toBe(0)
    expect(current.result.anchors.length).toBeGreaterThan(0)
    for (const anchor of current.result.anchors) {
      expect(current.result.points[anchor.observedPosition]!.status).toBe(IDENTITY_RESOLVED)
    }
  }, 30000)

  it('refines a larger crop from support anchors', () => {
    const current = scenario({ cropLeft: 30 })
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBeGreaterThan(0)
    expect(current.falseMatches).toBe(0)
  }, 30000)

  it('never modifies points that the hash already resolved', () => {
    const current = scenario({ cropLeft: 20 })
    for (let position = 0; position < current.baseline.points.length; position += 1) {
      const baselinePoint = current.baseline.points[position]!
      if (baselinePoint.designId === undefined) {
        continue
      }
      expect(current.result.points[position]!.designId).toBe(baselinePoint.designId)
    }
  }, 30000)

  it('refines dropout points that still have support anchors', () => {
    const current = scenario({ dropoutPercent: 10 })
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBeGreaterThan(0)
    expect(current.falseMatches).toBe(0)
  }, 30000)

  it('refines distortion points that still have support anchors', () => {
    const current = scenario({ cropLeft: 20, sphericalDistortion: 0.4 })
    expect(current.result.accepted).toBe(true)
    expect(current.result.metrics.recovered).toBeGreaterThan(0)
    expect(current.falseMatches).toBe(0)
  }, 30000)
})
