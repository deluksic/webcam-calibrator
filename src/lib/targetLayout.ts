// Learn target plane tag positions from a multi-tag frame; anchor = lowest tagId mapped to unit square,
// then the whole layout is translated in-plane so all corners have zero mean (xy).

import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import type { Corners3, ObjectTag, TagObservation } from '@/lib/calibrationTypes'
import type { Point3, Corners, Mat3, Point } from '@/lib/geometry'
import { applyHomography, tryComputeHomography } from '@/lib/geometry'

/** Corners with required z-coordinate for 3D object model. */
export type TargetCorners = [Point3, Point3, Point3, Point3]

const UNIT_SQUARE: TargetCorners = [
  { x: 0, y: 0, z: 0 },
  { x: 1, y: 0, z: 0 },
  { x: 0, y: 1, z: 0 },
  { x: 1, y: 1, z: 0 },
]

/** Single-plane layout today: lift **`Corners`** to **`Corners3`** with **`z = 0`** (multi-plane layouts can build **`Corners3`** with varying **`z`** later). */
function layoutPlaneCornersToCorners3(corners: Corners): Corners3 {
  return [
    { x: corners[0].x, y: corners[0].y, z: 0 },
    { x: corners[1].x, y: corners[1].y, z: 0 },
    { x: corners[2].x, y: corners[2].y, z: 0 },
    { x: corners[3].x, y: corners[3].y, z: 0 },
  ]
}

export type TargetLayout = Map<number, Corners>
export type TargetLayoutEntry = { tagId: number; corners: Corners }

/**
 * Map each tag to its four corner positions in the anchor tag's plane (Z=0, xy).
 * Requires ≥2 tags. Anchor (lowest id) is mapped to the unit square, then the layout is shifted so the
 * mean of all corner **(x, y)** is zero.
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
    const mapped: TargetCorners = [mapPt(hInv, c[0]!), mapPt(hInv, c[1]!), mapPt(hInv, c[2]!), mapPt(hInv, c[3]!)]
    m.set(t.tagId, mapped)
  }

  return layoutWithZeroMeanInPlane(m)
}

/** One **`ObjectTag`** per layout entry (**`Corners3`**, planar prior with **`z = 0`**). */
export function layoutToObjectTags(layout: TargetLayout): ObjectTag[] {
  const result: ObjectTag[] = []
  for (const [tagId, corners] of layout.entries()) {
    result.push({ tagId, corners: layoutPlaneCornersToCorners3(corners) })
  }
  return result.sort((a, b) => a.tagId - b.tagId)
}

function mapPt(h: Mat3, p: Point): Point3 {
  return { ...applyHomography(h, p.x, p.y), z: 0 }
}

/** Translate every corner by **`-(mean x, mean y)`** over all layout corners (planar z unchanged when lifted). */
function layoutWithZeroMeanInPlane(layout: TargetLayout): TargetLayout {
  let sx = 0
  let sy = 0
  let n = 0
  for (const corners of layout.values()) {
    for (let i = 0; i < 4; i++) {
      const p = corners[i]!
      sx += p.x
      sy += p.y
      n++
    }
  }
  if (n === 0) {
    return layout
  }
  const mx = sx / n
  const my = sy / n
  const out = new Map<number, Corners>()
  for (const [tagId, corners] of layout.entries()) {
    out.set(tagId, [
      { x: corners[0]!.x - mx, y: corners[0]!.y - my },
      { x: corners[1]!.x - mx, y: corners[1]!.y - my },
      { x: corners[2]!.x - mx, y: corners[2]!.y - my },
      { x: corners[3]!.x - mx, y: corners[3]!.y - my },
    ])
  }
  return out
}
