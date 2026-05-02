// Learn target plane tag positions from a multi-tag frame; anchor = lowest tagId mapped to unit square,
// then the whole layout is translated in-plane so all corners have zero mean (xy), **z** preserved per corner.

import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import type { Corners3, ObjectTag, Point3, TagObservation } from '@/lib/calibrationTypes'
import type { Mat3, Point } from '@/lib/geometry'
import { applyHomography, tryComputeHomography } from '@/lib/geometry'

const UNIT_SQUARE: Corners3 = [
  { x: 0, y: 0, z: 0 },
  { x: 1, y: 0, z: 0 },
  { x: 0, y: 1, z: 0 },
  { x: 1, y: 1, z: 0 },
]

/** Board / object model layout: each tag is a full **`Corners3`** (planar tags use **`z = 0`** on every corner). */
export type TargetLayout = Map<number, Corners3>
export type TargetLayoutEntry = { tagId: number; corners: Corners3 }

/**
 * Map each tag to its four corner positions in the anchor tag's plane (Z=0, xy).
 * Requires ≥2 tags. Anchor (lowest id) is mapped to the unit square, then the layout is shifted so the
 * mean of all corner **(x, y)** is zero (**z** unchanged).
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
  const m = new Map<number, Corners3>()
  m.set(anchor.tagId, UNIT_SQUARE)

  for (let i = 1; i < sorted.length; i++) {
    const t = sorted[i]!
    const c = t.corners
    const mapped: Corners3 = [mapPt(hInv, c[0]!), mapPt(hInv, c[1]!), mapPt(hInv, c[2]!), mapPt(hInv, c[3]!)]
    m.set(t.tagId, mapped)
  }

  return layoutWithZeroMeanInPlane(m)
}

export function layoutToObjectTags(layout: TargetLayout): ObjectTag[] {
  const result: ObjectTag[] = []
  for (const [tagId, corners] of layout.entries()) {
    result.push({ tagId, corners })
  }
  return result.sort((a, b) => a.tagId - b.tagId)
}

function mapPt(h: Mat3, p: Point): Point3 {
  const q = applyHomography(h, p.x, p.y)
  return { x: q.x, y: q.y, z: 0 }
}

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
  const out = new Map<number, Corners3>()
  for (const [tagId, corners] of layout.entries()) {
    out.set(tagId, [
      { x: corners[0]!.x - mx, y: corners[0]!.y - my, z: corners[0]!.z },
      { x: corners[1]!.x - mx, y: corners[1]!.y - my, z: corners[1]!.z },
      { x: corners[2]!.x - mx, y: corners[2]!.y - my, z: corners[2]!.z },
      { x: corners[3]!.x - mx, y: corners[3]!.y - my, z: corners[3]!.z },
    ])
  }
  return out
}
