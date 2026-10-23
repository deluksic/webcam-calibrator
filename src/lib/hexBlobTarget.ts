import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import type { BinaryColor, PatternPoint } from '@/lib/pointIdentity/types'

export type HexTargetSpec = {
  widthPx: number
  heightPx: number
  columns: number
  /** Optional explicit cell-row count; omitted keeps the automatic fill. */
  rows?: number
  pointsPerEdge: number
  markerScale: number
  colorSeed: number
  /** Minimum blank border as a fraction of each target dimension. Default 0.10. */
  paddingFraction?: number
}

export type HexTargetQuad = {
  /** Stable design-point index from the source hex pattern; unique per marker. */
  designId: number
  color: BinaryColor
  left: number
  top: number
  right: number
  bottom: number
}

export type HexTargetLayout = {
  widthPx: number
  heightPx: number
  paddingXPx: number
  paddingYPx: number
  columns: number
  rows: number
  pointsPerEdge: number
  spacingPx: number
  sidePx: number
  quads: HexTargetQuad[]
  /** All source pattern points, including markers clipped from the image. */
  points: readonly PatternPoint[]
  /** Color for every source point, indexed by designId. */
  colors: readonly BinaryColor[]
}

const MIN_PADDING_PX = 8
const DEFAULT_PADDING_FRACTION = 0.1
const DEFAULT_MARKER_SCALE = 0.4
const MIN_GAP_FRACTION = 0.06

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

/**
 * Lay out one axis-aligned square per sampled Point Lab honeycomb edge point.
 *
 * All quad bounds are integer, half-open device-pixel rectangles. The GPU
 * renderer maps those boundaries directly to clip space, so no fractional
 * vertex positions are introduced.
 */
export function buildHexTargetLayout(spec: HexTargetSpec): HexTargetLayout {
  const widthPx = Math.max(0, Math.floor(spec.widthPx))
  const heightPx = Math.max(0, Math.floor(spec.heightPx))
  const columns = Math.max(2, Math.round(spec.columns))
  const pointsPerEdge = Math.max(1, Math.round(spec.pointsPerEdge))
  const markerScale = clamp(spec.markerScale, 0.2, 0.7)
  const paddingFraction = clamp(spec.paddingFraction ?? DEFAULT_PADDING_FRACTION, 0, 0.4)
  const paddingXPx = Math.max(MIN_PADDING_PX, Math.round(widthPx * paddingFraction))
  const paddingYPx = Math.max(MIN_PADDING_PX, Math.round(heightPx * paddingFraction))

  const base: HexTargetLayout = {
    widthPx,
    heightPx,
    paddingXPx,
    paddingYPx,
    columns,
    rows: 0,
    pointsPerEdge,
    spacingPx: 0,
    sidePx: 0,
    quads: [],
    points: [],
    colors: [],
  }
  if (widthPx < 2 || heightPx < 2 || widthPx - paddingXPx * 2 < 2 || heightPx - paddingYPx * 2 < 2) {
    return base
  }

  const pattern = generateHexPattern({ columns, rows: spec.rows, pointsPerEdge })
  if (pattern.points.length === 0) {
    return base
  }
  const patternWidth = pattern.bounds.maxX - pattern.bounds.minX
  const patternHeight = pattern.bounds.maxY - pattern.bounds.minY
  if (patternWidth <= 0 || patternHeight <= 0) {
    return base
  }

  const scale = Math.min((widthPx - paddingXPx * 2) / patternWidth, (heightPx - paddingYPx * 2) / patternHeight)
  const offsetX = widthPx / 2 - ((pattern.bounds.minX + pattern.bounds.maxX) / 2) * scale
  const offsetY = heightPx / 2 - ((pattern.bounds.minY + pattern.bounds.maxY) / 2) * scale
  const centers = pattern.points.map((point) => ({
    x: Math.round(point.x * scale + offsetX),
    y: Math.round(point.y * scale + offsetY),
    sourceIndex: point.index,
  }))

  let minPairDistance = Number.POSITIVE_INFINITY
  const nearestDistances: number[] = []
  for (let first = 0; first < centers.length; first += 1) {
    let nearest = Number.POSITIVE_INFINITY
    const a = centers[first]!
    for (let second = first + 1; second < centers.length; second += 1) {
      const b = centers[second]!
      const distance = Math.hypot(a.x - b.x, a.y - b.y)
      if (distance < nearest) {
        nearest = distance
      }
      if (distance < minPairDistance) {
        minPairDistance = distance
      }
    }
    if (Number.isFinite(nearest)) {
      nearestDistances.push(nearest)
    }
  }
  if (nearestDistances.length === 0 || !Number.isFinite(minPairDistance)) {
    return base
  }
  nearestDistances.sort((first, second) => first - second)
  const medianNearest = nearestDistances[Math.floor(nearestDistances.length / 2)] ?? minPairDistance
  const minGap = Math.max(1, Math.round(medianNearest * MIN_GAP_FRACTION))
  const desiredSide = Math.max(1, Math.floor(medianNearest * markerScale))
  const sidePx = Math.max(1, Math.min(desiredSide, minPairDistance - minGap * 2))
  if (sidePx < 1 || minPairDistance < minGap * 2 + 2) {
    return base
  }

  const colors = randomBinaryColors(pattern.points.length, spec.colorSeed)
  const quads: HexTargetQuad[] = []
  for (const center of centers) {
    const left = center.x - Math.floor(sidePx / 2)
    const top = center.y - Math.floor(sidePx / 2)
    const right = left + sidePx
    const bottom = top + sidePx
    if (left < 0 || top < 0 || right > widthPx || bottom > heightPx) {
      continue
    }
    quads.push({
      designId: center.sourceIndex,
      color: colors[center.sourceIndex] ?? 0,
      left,
      top,
      right,
      bottom,
    })
  }

  return {
    ...base,
    rows: pattern.stats.rows,
    spacingPx: Math.round(medianNearest),
    sidePx,
    quads,
    points: pattern.points,
    colors,
  }
}

export { DEFAULT_MARKER_SCALE }
