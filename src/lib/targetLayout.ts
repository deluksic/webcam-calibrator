// Learn target plane tag positions from a multi-tag frame; anchor = lowest tagId at unit square.

import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import type { TagObservation, LabeledPoint } from '@/lib/calibrationTypes'
import { applyHomography, tryComputeHomography, type Mat3, type Point, type Point3 } from '@/lib/geometry'

/** Corners with required z-coordinate for 3D object model. */
export type TargetCorners = [Point3, Point3, Point3, Point3]

const UNIT_SQUARE: TargetCorners = [
  { x: 0, y: 0, z: 0 },
  { x: 1, y: 0, z: 0 },
  { x: 0, y: 1, z: 0 },
  { x: 1, y: 1, z: 0 },
]

export type TargetLayout = Map<number, TargetCorners>
export type TargetLayoutEntry = { tagId: number; corners: TargetCorners }

const POINT_ID_MULTIPLIER = 10000

/**
 * Map each tag to its four corner positions in the anchor tag's plane (Z=0, xy).
 * Requires ≥2 tags. Anchor (lowest id) is fixed at the unit square.
 */
export function learnLayoutFromFrame(tags: TagObservation[]): TargetLayout | undefined {
  if (tags.length < 2) {
    return undefined
  }
  const sorted = [...tags].sort((a, b) => a.tagId - b.tagId)
  const anchor = sorted[0]!
  const hAnchor = tryComputeHomography(anchor.corners) as Mat3
  const hInv = invertMat3RowMajor(hAnchor)
  if (!hInv) {
    console.log(`[layout] Cannot invert anchor tag ${anchor.tagId} homography - singular matrix`)
    return undefined
  }
  const m = new Map<number, TargetCorners>()
  m.set(anchor.tagId, UNIT_SQUARE)

  // Map all other tags
  for (let i = 1; i < sorted.length; i++) {
    const t = sorted[i]!
    const c = t.corners
    const mapped: TargetCorners = [
      mapPt(hInv, c[0]!),
      mapPt(hInv, c[1]!),
      mapPt(hInv, c[2]!),
      mapPt(hInv, c[3]!),
    ]
    m.set(t.tagId, mapped)
  }

  return m
}

/**
 * Convert layout tags into labeled points with unique pointIds.
 * Each tag has 4 corners labeled as tagId * 10000 + cornerId.
 */
export function layoutToLabeledPoints(layout: TargetLayout): LabeledPoint[] {
  const result: LabeledPoint[] = []
  for (const [tagId, corners] of layout.entries()) {
    for (let cornerId = 0; cornerId < 4; cornerId++) {
      result.push({
        pointId: tagId * POINT_ID_MULTIPLIER + cornerId,
        position: corners[cornerId]!,
      })
    }
  }
  return result
}

function mapPt(h: Mat3, p: Point): Point3 {
  return { ...applyHomography(h, p.x, p.y), z: 0 }
}
