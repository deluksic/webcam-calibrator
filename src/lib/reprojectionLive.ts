// Build per-frame reprojection overlay geometry for a 2D canvas (image pixel space).

import { solveHomographyDLT, type Correspondence } from '@/lib/dltHomography'
import { extrinsicsFromHomography, type Mat3R } from '@/lib/zhangCalibration'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { TargetLayout } from '@/lib/targetLayout'
import { applyHomography, type Point } from '@/lib/geometry'
import type { Vec3 } from '@/lib/zhangCalibration'
import type { DetectedQuad } from '@/gpu/contour'

const { hypot, acos, min: minf, max: maxf } = Math

export type ReprojectionDrawOp =
  | { t: 'ring'; c: Point; r: number; color: string }
  | { t: 'dot'; c: Point; r: number; color: string }
  | { t: 'line'; a: Point; b: Point; color: string; w: number }

function residualColor(d: number): string {
  if (d < 0.5) {
    return 'rgba(0,255,120,0.85)'
  }
  if (d < 1) {
    return 'rgba(255,220,0,0.85)'
  }
  if (d < 3) {
    return 'rgba(255,120,0,0.9)'
  }
  return 'rgba(255,40,40,0.95)'
}

function dist(a: Point, b: Point) {
  return hypot(a.x - b.x, a.y - b.y)
}

/**
 * @returns null if not all decoded tags are in `layout` or DLT/ extrinsics fail.
 * Residuals use the DLT `H` via {@link applyHomography} (same model as the fit). Pinhole
 * reprojection from decomposed (R, t) can disagree with that H when K̂H→R/t is not exact.
 */
export function buildReprojectionDrawOps(
  layout: TargetLayout,
  k: CameraIntrinsics,
  quads: readonly DetectedQuad[],
  imageWidth: number,
  imageHeight: number,
): { ops: ReprojectionDrawOp[]; rms: number; R: Mat3R; t: Vec3; tagCount: number } | null {
  const pairs: Correspondence[] = []
  const tags: { id: number; corners: [Point, Point, Point, Point] }[] = []
  for (const q of quads) {
    if (typeof q.decodedTagId !== 'number' || !q.hasCorners) {
      continue
    }
    const pl = layout.get(q.decodedTagId)
    if (!pl) {
      return null
    }
    for (let j = 0; j < 4; j++) {
      pairs.push({ plane: { x: pl[j]!.x, y: pl[j]!.y }, image: q.corners[j]! })
    }
    tags.push({ id: q.decodedTagId, corners: [q.corners[0]!, q.corners[1]!, q.corners[2]!, q.corners[3]!] })
  }
  if (tags.length < 2 || pairs.length < 8) {
    return null
  }
  const h = solveHomographyDLT(pairs)
  if (!h) {
    return null
  }
  const ex = extrinsicsFromHomography(h, k)
  if (!ex) {
    return null
  }
  const { R, t } = ex
  const ops: ReprojectionDrawOp[] = []
  const errSq: number[] = []
  for (const tg of tags) {
    const pl = layout.get(tg.id)!
    for (let j = 0; j < 4; j++) {
      const pred = applyHomography(h, pl[j]!.x, pl[j]!.y)
      const im = tg.corners[j]!
      const d = dist(pred, im)
      errSq.push(d * d)
      ops.push({ t: 'ring', c: im, r: 4, color: 'rgba(0,200,255,0.5)' })
      ops.push({ t: 'dot', c: pred, r: 3, color: 'rgba(255,0,200,0.9)' })
      ops.push({ t: 'line', a: im, b: pred, color: residualColor(d), w: 1.5 })
    }
  }
  const rms = errSq.length > 0 ? Math.sqrt(errSq.reduce((a, b) => a + b, 0) / errSq.length) : 0
  void imageWidth
  void imageHeight
  return { ops, rms, R, t, tagCount: tags.length }
}

export function cameraTiltDegFromR(R: Mat3R): number {
  const nx = R[2]!
  const ny = R[5]!
  const nz = R[8]!
  const L = hypot(nx, hypot(ny, nz)) + 1e-12
  return (acos(minf(1, maxf(-1, nz / L))) * 180) / Math.PI
}

export function cameraDistanceFromT(t: Vec3): number {
  return hypot(t.x, hypot(t.y, t.z))
}
