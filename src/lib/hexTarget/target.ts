import type { TargetObjectPoint } from '@/lib/calibrationTypes'
import {
  DEFAULT_TARGET_CONFIG,
  normalizeHexTargetConfig,
  type HexTargetConfig,
} from '@/lib/hexTarget/config'
import { computeDesignLatticeMetrics } from '@/lib/pointIdentity/designGrid'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { createIdentityIndexesByDepth } from '@/lib/pointIdentity/identity'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { BinaryColor, HexPattern, IdentityIndex, ProximityGraph } from '@/lib/pointIdentity/types'

export type HexTargetModel = {
  config: HexTargetConfig
  pattern: HexPattern
  colors: readonly BinaryColor[]
  graph: ProximityGraph
  indexesByDepth: readonly IdentityIndex[]
  pointsPerEdge: number
  /** Median nearest-neighbor distance in design units. */
  spacing: number
  /** Normalized object-plane points (`z = 0`, one lattice spacing ≈ 1). */
  objectPoints: readonly TargetObjectPoint[]
}

export function createHexTargetDesign(input: HexTargetConfig): HexTargetModel {
  const config = normalizeHexTargetConfig(input)
  const pointsPerEdge = config.pointsPerEdge
  const pattern = generateHexPattern({
    columns: config.columns,
    rows: config.rows,
    pointsPerEdge,
  })
  const colors = randomBinaryColors(pattern.points.length, config.colorSeed)
  const graph = buildProximityGraph(pattern.points, { pointsPerEdge })
  const indexesByDepth = createIdentityIndexesByDepth(graph, colors, config.identityDepth)
  const spacing = computeDesignLatticeMetrics(pattern.points).medianNearest
  const scale = spacing > 1e-9 ? 1 / spacing : 1
  const objectPoints: TargetObjectPoint[] = pattern.points.map((point) => ({
    designId: point.index,
    x: point.x * scale,
    y: point.y * scale,
    z: 0,
  }))
  return { config, pattern, colors, graph, indexesByDepth, pointsPerEdge, spacing, objectPoints }
}

/** Default model used by the demo calibration and as a first-render fallback. */
export function defaultHexTargetModel(): HexTargetModel {
  return createHexTargetDesign(DEFAULT_TARGET_CONFIG)
}
