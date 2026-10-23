import { describe, expect, it } from 'vitest'

import { collectAnchors, computeProofPower } from '@/lib/pointIdentity/anchors'
import { createIdentityIndexesByDepth, computeFingerprints } from '@/lib/pointIdentity/identity'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import {
  applyObservationMask,
  DEFAULT_OBSERVATION_MASK,
  type ObservationMask,
} from '@/lib/pointIdentity/observation'
import { projectPattern, DEFAULT_VIEW } from '@/lib/pointIdentity/projection'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { BinaryColor, PatternPoint, ProximityGraph } from '@/lib/pointIdentity/types'

const IDENTITY_DEPTH = 6
const MAX_PROOF_DEPTH = 8

type Scene = {
  pattern: ReturnType<typeof generateHexPattern>
  colors: BinaryColor[]
  designGraph: ProximityGraph
  indexes: ReturnType<typeof createIdentityIndexesByDepth>
  observedGraph: ProximityGraph
  observedColors: BinaryColor[]
  fingerprints: ReturnType<typeof computeFingerprints>
  designPointsByIndex: Map<number, PatternPoint>
}

function scene(mask: Partial<ObservationMask>): Scene {
  const pattern = generateHexPattern({ columns: 8, pointsPerEdge: 4 })
  const colors = randomBinaryColors(pattern.points.length, 1)
  const tuning = { pointsPerEdge: 4 }
  const designGraph = buildProximityGraph(pattern.points, tuning)
  const indexes = createIdentityIndexesByDepth(designGraph, colors, MAX_PROOF_DEPTH)
  const projected = projectPattern(pattern.points, { ...DEFAULT_VIEW, yawDeg: 0, pitchDeg: 0 })
  const observed = applyObservationMask(projected, { ...DEFAULT_OBSERVATION_MASK, ...mask })
  const observedGraph = buildProximityGraph(observed, tuning)
  const observedColors = observedGraph.points.map((point) => colors[point.index] ?? 0)
  const colorByGraphIndex: BinaryColor[] = []
  for (let position = 0; position < observedGraph.points.length; position += 1) {
    colorByGraphIndex[observedGraph.points[position]!.index] = observedColors[position]!
  }
  const fingerprints = computeFingerprints(observedGraph, colorByGraphIndex, MAX_PROOF_DEPTH)
  const designPointsByIndex = new Map<number, PatternPoint>()
  for (const point of pattern.points) {
    designPointsByIndex.set(point.index, point)
  }
  return {
    pattern,
    colors,
    designGraph,
    indexes,
    observedGraph,
    observedColors,
    fingerprints,
    designPointsByIndex,
  }
}

describe('proof power and anchors', () => {
  it('matches the design id for every point on a clean graph', () => {
    const current = scene({})
    const indexes = current.indexes
    for (let position = 0; position < current.fingerprints.length; position += 1) {
      const proof = computeProofPower(current.fingerprints[position]!, indexes, MAX_PROOF_DEPTH)
      expect(proof.designId).toBe(current.observedGraph.points[position]!.index)
      expect(proof.depth).toBeGreaterThan(0)
      expect(proof.unstable).toBe(false)
    }
  })

  it('collects enough anchors after a left crop', () => {
    const current = scene({ cropLeft: 20 })
    const result = collectAnchors(
      current.observedGraph,
      current.observedColors,
      current.fingerprints,
      current.indexes,
      {
        identityDepth: IDENTITY_DEPTH,
        maxProofDepth: MAX_PROOF_DEPTH,
        minProofPower: 4,
        minAnchors: 6,
        designPointsByIndex: current.designPointsByIndex,
      },
    )
    expect(result.anchors.length).toBeGreaterThan(100)
    expect(result.usedProofPower).toBe(4)
    expect(result.fallbackUsed).toBe(false)
  })

  it('falls back one proof-power hop when the requested threshold is too strict', () => {
    const current = scene({ cropLeft: 20 })
    const result = collectAnchors(
      current.observedGraph,
      current.observedColors,
      current.fingerprints,
      current.indexes,
      {
        identityDepth: IDENTITY_DEPTH,
        maxProofDepth: MAX_PROOF_DEPTH,
        minProofPower: 6,
        minAnchors: 400,
        designPointsByIndex: current.designPointsByIndex,
      },
    )
    expect(result.usedProofPower).toBeLessThan(6)
    expect(result.fallbackUsed).toBe(true)
    expect(result.anchors.length).toBeGreaterThan(0)
  })

  it('returns no anchors when the threshold cannot be met', () => {
    const current = scene({ cropLeft: 20 })
    const result = collectAnchors(
      current.observedGraph,
      current.observedColors,
      current.fingerprints,
      current.indexes,
      {
        identityDepth: IDENTITY_DEPTH,
        maxProofDepth: MAX_PROOF_DEPTH,
        minProofPower: 9,
        minAnchors: 1000,
        allowProofFallback: false,
        designPointsByIndex: current.designPointsByIndex,
      },
    )
    expect(result.anchors).toHaveLength(0)
    expect(result.usedProofPower).toBe(9)
  })
})
