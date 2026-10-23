/** Shared value types for the point-identity lab. */

export type Vec2 = {
  x: number
  y: number
}

export type BinaryColor = 0 | 1

/** A design-space point with a stable identity. */
export type PatternPoint = Vec2 & {
  index: number
}

export type PatternGuide = {
  a: Vec2
  b: Vec2
}

export type HexPatternSpec = {
  /** Number of hexagonal cells across. */
  columns: number
  /** Number of hexagonal cell rows. Defaults to the count that best fills the design height. */
  rows?: number
  /** Points sampled on each cell edge, at half-step offsets. */
  pointsPerEdge: number
}

export type PatternStats = {
  cells: number
  rows: number
  candidateEdges: number
  sampledEdges: number
  rawPoints: number
  points: number
}

export type HexPattern = {
  points: PatternPoint[]
  guides: PatternGuide[]
  stats: PatternStats
  /** Design-space bounds before fitting to the canvas. */
  bounds: {
    minX: number
    maxX: number
    minY: number
    maxY: number
  }
}

export type ViewPose = {
  /** Rotation of the target around the vertical axis, in degrees. */
  yawDeg: number
  /** Rotation of the target around the horizontal axis, in degrees. */
  pitchDeg: number
  /** Camera distance along the optical axis, in design units. */
  cameraDistance: number
  /** Pinhole focal length in design units. */
  focalLength: number
  /** Field-relative radial lens warp; 0 is a perfect pinhole. */
  sphericalDistortion: number
}

export type GraphTuning = {
  /** Known design count per cell side. */
  pointsPerEdge: number
}

export type GraphRejections = {
  degree: number
  crossing: number
}

export type GraphStats = {
  inputPoints: number
  candidates: number
  edges: number
  meanDegree: number
  isolated: number
  components: number
  chainEdges: number
  junctionEdges: number
  rejections: GraphRejections
  localScaleMedian: number
}

export type EdgeKind = 'chain' | 'junction'

export type ProximityGraph = {
  points: readonly PatternPoint[]
  /** Neighbor indices for each point, in insertion order. */
  neighbors: ReadonlyArray<readonly number[]>
  edges: ReadonlyArray<readonly [number, number]>
  /** Admission role of each edge, parallel to `edges`. */
  edgeKinds: ReadonlyArray<EdgeKind>
  stats: GraphStats
}

export type RingDescriptor = {
  /** Number of points exactly this many hops from the root. */
  size: number
  /** Ring colors in angular order, canonically rotated to the smallest string. */
  colors: string
  /** First-appearance chain-partition labels, aligned with the color rotation. */
  partition: string
  /** Chain-endpoint bits, aligned with the canonical color rotation. */
  endpoints: string
}

export type PointFingerprint = {
  index: number
  color: BinaryColor
  endpoint: boolean
  rings: RingDescriptor[]
  /** Unambiguous serialization used for lookup. */
  key: string
}

export type IdentityIndex = {
  byKey: Map<string, number[]>
  fingerprints: PointFingerprint[]
  keyCount: number
  uniqueKeys: number
  duplicateKeyCount: number
  collidingPointCount: number
}

export type MatchOutcome =
  | { kind: 'correct'; matchedIndex: number }
  | { kind: 'false-match'; matchedIndex: number }
  | { kind: 'ambiguous'; candidateIndices: number[] }
  | { kind: 'unresolved' }

export type PointEvaluation = {
  index: number
  fingerprint: PointFingerprint
  outcome: MatchOutcome
}

export type EvaluationSummary = {
  total: number
  correct: number
  falseMatch: number
  safeFail: number
  unresolved: number
  ambiguous: number
  accuracy: number
  falseMatchRate: number
  designUniquePoints: number
  designCollisions: number
  edgeJaccard: number
  degreeChanged: number
}

export type IdentityEvaluation = {
  summary: EvaluationSummary
  perPoint: PointEvaluation[]
  observedFingerprints: PointFingerprint[]
}
