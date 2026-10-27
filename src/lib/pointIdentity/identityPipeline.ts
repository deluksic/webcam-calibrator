import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import {
  computeFingerprints,
  createIdentityIndexesByDepth,
  truncateFingerprints,
} from '@/lib/pointIdentity/identity'
import {
  createLocalIcpDesign,
  createLocalIcpOptions,
  MAX_FINGERPRINT_DEPTH,
  refineObservedIdentityWithLocalIcp,
  type LocalIcpDesign,
} from '@/lib/pointIdentity/localIcpRefiner'
import {
  createColorIdentityIndex,
  IDENTITY_AMBIGUOUS,
  IDENTITY_CONFLICT,
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  IDENTITY_UNRESOLVED,
  matchObservedIdentity,
  type ColorIdentityIndex,
} from '@/lib/pointIdentity/match'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type {
  BinaryColor,
  HexPattern,
  IdentityIndex,
  ProximityGraph,
} from '@/lib/pointIdentity/types'

/** A detected component centroid, as read back from the GPU. */
export type IdentityObservedPoint = {
  x: number
  y: number
  color: 0 | 1
}

export type IdentityGraphPoint = {
  x: number
  y: number
  designId: number
  status: number
}

export type IdentityGraphEdge = {
  a: number
  b: number
}

/** Structurally identical to the renderer's `RoiGraphData`. */
export type IdentityGraph = {
  points: IdentityGraphPoint[]
  edges: IdentityGraphEdge[]
}

export type IdentityMetrics = {
  total: number
  resolved: number
  ambiguous: number
  unresolved: number
  conflicts: number
  edges: number
  components: number
  recovered: number
  frameId: number
}

/** Everything the identity pipeline needs to rebuild the design lattice. */
export type IdentityDesignSpec = {
  columns: number
  pointsPerEdge: number
  colorSeed: number
}

export type IdentityDesignBundle = {
  spec: IdentityDesignSpec
  pattern: HexPattern
  colors: BinaryColor[]
  graph: ProximityGraph
  indexesByDepth: IdentityIndex[]
  designCache: LocalIcpDesign
  /** Loose color-only indexes, built on first use per depth. */
  looseIndexesByDepth: Map<number, ColorIdentityIndex>
}

/**
 * Builds the design-side structures exactly as the ROI lab used to build them
 * on the main thread. Deterministic from `spec`, so the main thread and the
 * worker agree on ids and colors.
 */
export function createIdentityDesignBundle(spec: IdentityDesignSpec): IdentityDesignBundle {
  const pattern = generateHexPattern({
    columns: spec.columns,
    pointsPerEdge: spec.pointsPerEdge,
  })
  const colors = randomBinaryColors(pattern.points.length, spec.colorSeed)
  const graph = buildProximityGraph(pattern.points, { pointsPerEdge: spec.pointsPerEdge })
  const indexesByDepth = createIdentityIndexesByDepth(graph, colors, MAX_FINGERPRINT_DEPTH)
  return {
    spec,
    pattern,
    colors,
    graph,
    indexesByDepth,
    designCache: createLocalIcpDesign(pattern),
    looseIndexesByDepth: new Map(),
  }
}

export function identityDesignKey(spec: IdentityDesignSpec): string {
  return `${spec.columns}:${spec.pointsPerEdge}:${spec.colorSeed}`
}

function looseIndexFor(design: IdentityDesignBundle, depth: number): ColorIdentityIndex {
  const cached = design.looseIndexesByDepth.get(depth)
  if (cached !== undefined) {
    return cached
  }
  const built = createColorIdentityIndex(design.graph, design.colors, depth)
  design.looseIndexesByDepth.set(depth, built)
  return built
}

export type IdentityRunInput = {
  design: IdentityDesignBundle
  points: readonly IdentityObservedPoint[]
  /** Fingerprint depth; clamped to the indexed range. */
  depth: number
  /** Enable local ICP recovery for ambiguous/unresolved points. */
  recovery: boolean
  frameId: number
}

export type IdentityRunOutput = {
  graph: IdentityGraph
  metrics: IdentityMetrics
}

/**
 * The whole `runIdentity` stage: observed graph -> fingerprints -> hash match ->
 * optional local ICP recovery. Pure with respect to the design bundle, so it can
 * run in a worker without touching the DOM, GPU, or Solid signals.
 */
export function runIdentityPipeline(input: IdentityRunInput): IdentityRunOutput {
  const { design } = input
  const points = input.points
  const depth = Math.min(MAX_FINGERPRINT_DEPTH, Math.max(1, Math.round(input.depth)))
  const pointsPerEdge = design.spec.pointsPerEdge

  const observedPoints = points.map((point, index) => ({
    x: point.x,
    y: point.y,
    index,
  }))
  const observedColors = points.map((point) => point.color)
  const observedGraph = buildProximityGraph(observedPoints, { pointsPerEdge })
  // Recovery needs MAX-depth fingerprints for proof power; the matcher needs
  // depth-limited ones. Both come from a single traversal — the full-depth rings
  // are the exact per-hop rings every shallower depth would produce.
  const recoveryFingerprints = input.recovery
    ? computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
    : undefined
  const baselineFingerprints =
    recoveryFingerprints === undefined
      ? computeFingerprints(observedGraph, observedColors, depth)
      : depth === MAX_FINGERPRINT_DEPTH
        ? recoveryFingerprints
        : truncateFingerprints(recoveryFingerprints, depth)
  const baseline = matchObservedIdentity(
    design.indexesByDepth[depth - 1]!,
    observedGraph,
    observedColors,
    depth,
    input.recovery
      ? { fingerprints: baselineFingerprints }
      : {
          looseByKey: looseIndexFor(design, depth),
          fingerprints: baselineFingerprints,
        },
  )

  let finalPoints = baseline.points
  if (input.recovery && recoveryFingerprints !== undefined) {
    finalPoints = refineObservedIdentityWithLocalIcp({
      designPattern: design.pattern,
      designColors: design.colors,
      designGraph: design.graph,
      designIndexesByDepth: design.indexesByDepth,
      observedGraph,
      observedColors,
      observedFingerprints: recoveryFingerprints,
      baseline,
      designCache: design.designCache,
      options: createLocalIcpOptions({
        identityDepth: depth,
        maxProofDepth: MAX_FINGERPRINT_DEPTH,
        minProofPower: 7,
        hopRadius: 8,
      }),
    }).points
  }

  let resolved = 0
  let ambiguous = 0
  let unresolved = 0
  let conflicts = 0
  let recovered = 0
  for (const point of finalPoints) {
    switch (point.status) {
      case IDENTITY_RESOLVED:
        resolved += 1
        break
      case IDENTITY_AMBIGUOUS:
        ambiguous += 1
        break
      case IDENTITY_UNRESOLVED:
        unresolved += 1
        break
      case IDENTITY_CONFLICT:
        conflicts += 1
        break
      case IDENTITY_RECOVERED:
        recovered += 1
        break
    }
  }

  return {
    graph: {
      points: finalPoints.map((point) => ({
        x: point.x,
        y: point.y,
        designId: point.designId ?? 0,
        status: point.status,
      })),
      edges: observedGraph.edges.map(([a, b]) => ({ a, b })),
    },
    metrics: {
      total: finalPoints.length,
      resolved,
      ambiguous,
      unresolved,
      conflicts,
      edges: observedGraph.edges.length,
      components: observedGraph.stats.components,
      recovered,
      frameId: input.frameId,
    },
  }
}
