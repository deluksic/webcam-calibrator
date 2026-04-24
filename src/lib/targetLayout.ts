// Learn target plane tag positions from a multi-tag frame; anchor = lowest tagId at unit square.

import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import { applyHomography, tryComputeHomography, type Corners, type Mat3, type Point } from '@/lib/geometry'
import type { TagObservation } from '@/lib/calibrationTypes'

const UNIT_SQUARE: Corners = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: 1, y: 1 },
]

export type TargetLayout = ReadonlyMap<number, Corners>
export type TargetLayoutEntry = { tagId: number; corners: Corners }

/**
 * Map each tag to its four corner positions in the anchor tag's plane (Z=0, xy).
 * Requires ≥2 tags. Anchor (lowest id) is fixed at the unit square.
 */
export function learnLayoutFromFrame(tags: readonly TagObservation[]): TargetLayout | undefined {
  if (tags.length < 2) {
    return undefined
  }
  const sorted = [...tags].sort((a, b) => a.tagId - b.tagId)
  const anchor = sorted[0]!
  const hAnchor = tryComputeHomography(anchor.corners) as Mat3
  const hInv = invertMat3RowMajor(hAnchor)
  if (!hInv) {
    return undefined
  }
  const m = new Map<number, Corners>()
  m.set(anchor.tagId, UNIT_SQUARE)
  for (let i = 1; i < sorted.length; i++) {
    const t = sorted[i]!
    const c = t.corners
    const corners: Corners = [mapPt(hInv, c[0]!), mapPt(hInv, c[1]!), mapPt(hInv, c[2]!), mapPt(hInv, c[3]!)]
    m.set(t.tagId, corners)
  }
  return m
}

function mapPt(h: Mat3, p: Point): Point {
  return applyHomography(h, p.x, p.y)
}
