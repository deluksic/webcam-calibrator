import type { DetectedQuad } from '@/gpu/contour'
import { length, type Corners } from '@/lib/geometry'

const { min, max, abs } = Math

/** Stub thresholds — tune after BA exists. */
export const CALIB_MIN_MIN_R2 = 0.75
export const CALIB_MIN_MIN_EDGE_PX = 12
export const CALIB_MIN_AREA_PX = 400

function quadMinEdgePx(corners: Corners): number {
  const [tl, tr, bl, br] = corners
  const e = (a: { x: number; y: number }, b: { x: number; y: number }) => length(b.x - a.x, b.y - a.y)
  return min(e(tl, tr), e(tr, br), e(br, bl), e(bl, tl))
}

export function quadAreaPx(corners: Corners): number {
  const [tl, tr, bl, br] = corners
  // Shoelace on convex walk TL → TR → BR → BL → TL (strip order is not a polygon walk)
  return abs(
    (tl.x * (tr.y - bl.y) +
      tr.x * (br.y - tl.y) +
      br.x * (bl.y - tr.y) +
      bl.x * (tl.y - br.y)) /
      2,
  )
}

export function calibrationQuadScore(q: DetectedQuad): number {
  const minR2 = q.cornerDebug?.minR2 ?? 0
  const minEdge = quadMinEdgePx(q.corners)
  return minR2 * max(minEdge, 1e-6)
}

export function acceptQuadForCalibration(q: DetectedQuad): boolean {
  if (q.decodedTagId === undefined) {
    return false
  }
  if (!q.hasCorners) {
    return false
  }
  if (!q || q.cornerDebug?.failureCode !== 0) {
    return false
  }
  const minR2 = q.cornerDebug.minR2
  if (minR2 < CALIB_MIN_MIN_R2) {
    return false
  }
  const minEdge = quadMinEdgePx(q.corners)
  if (minEdge < CALIB_MIN_MIN_EDGE_PX) {
    return false
  }
  if (quadAreaPx(q.corners) < CALIB_MIN_AREA_PX) {
    return false
  }
  return true
}

/**
 * If two+ decoded quads share the same `tagId`, reject the whole frame (likely bit flip).
 */
export function frameHasDuplicateDecodedTagIds(quads: DetectedQuad[]): boolean {
  const seen = new Set<number>()
  for (const q of quads) {
    if (typeof q.decodedTagId !== 'number') {
      continue
    }
    if (seen.has(q.decodedTagId)) {
      return true
    }
    seen.add(q.decodedTagId)
  }
  return false
}
