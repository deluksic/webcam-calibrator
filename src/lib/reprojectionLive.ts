// Build per-frame reprojection overlay geometry for a 2D canvas (image pixel space).

import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { TargetLayout } from '@/lib/targetLayout'
import { solveHomographyDLT } from '@/lib/dltHomography'
import { projectPlanePoint } from '@/lib/reprojectionError'
import type { Point } from '@/lib/geometry'
import type { Mat3 } from '@/lib/geometry'
import { extrinsicsFromHomography, type Vec3, type Mat3R } from '@/lib/zhangCalibration'
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
  // Compute live pose from current detections
  const livePose = livePoseFromDetections(layout, quads, k)
  if (!livePose) {
    return null
  }

  const { R, t } = livePose
  const tags: { id: number; corners: [Point, Point, Point, Point] }[] = []
  const pairs: Array<{ plane: Point; image: Point }> = []

  // Collect all valid tags and build correspondences
  for (const q of quads) {
    if (typeof q.decodedTagId !== 'number' || !q.hasCorners || q.cornerDebug?.failureCode !== 0) {
      continue
    }
    const pl = layout.get(q.decodedTagId)
    if (!pl) {
      continue
    }
    tags.push({ id: q.decodedTagId, corners: [q.corners[0]!, q.corners[1]!, q.corners[2]!, q.corners[3]!] })
    for (let j = 0; j < 4; j++) {
      pairs.push({
        plane: { x: pl[j]!.x, y: pl[j]!.y },
        image: q.corners[j]!,
      })
    }
  }
  if (tags.length < 2) {
    return null
  }

  // Compute reprojection errors and build visualization ops
  const ops: ReprojectionDrawOp[] = []
  const errSq: number[] = []
  for (const { plane, image } of pairs) {
    const pred = projectPlanePoint(k, R, t, plane.x, plane.y)
    const d = dist(pred, image)
    errSq.push(d * d)
    ops.push({ t: 'ring', c: image, r: 4, color: 'rgba(0,200,255,0.5)' })
    ops.push({ t: 'dot', c: pred, r: 3, color: 'rgba(255,0,200,0.9)' })
    ops.push({ t: 'line', a: image, b: pred, color: residualColor(d), w: 1.5 })
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

/**
 * Compute live pose (R, t) from current tag detections.
 * For each visible tag, build a homography from layout positions → image corners.
 * Use the first valid homography to get extrinsics (assuming the camera is essentially
 * at a single position for the current frame).
 *
 * Returns {R, t} if we have enough correspondences, null otherwise.
 */
export function livePoseFromDetections(
  layout: TargetLayout,
  quads: readonly DetectedQuad[],
  k: CameraIntrinsics,
): { R: Mat3R; t: Vec3 } | null {
  // Build correspondences from visible tags to their layout positions
  const pairs: Array<{ plane: Point; image: Point }> = []
  for (const q of quads) {
    if (typeof q.decodedTagId !== 'number' || !q.hasCorners || q.cornerDebug?.failureCode !== 0) {
      continue
    }
    const pl = layout.get(q.decodedTagId)
    if (!pl) {
      continue
    }
    // Use all 4 corners from this tag
    for (let j = 0; j < 4; j++) {
      if (q.corners[j]) {
        pairs.push({
          plane: { x: pl[j]!.x, y: pl[j]!.y },
          image: q.corners[j]!,
        })
      }
    }
  }

  if (pairs.length < 8) {
    return null
  }

  // Compute homography from layout → image
  const h = solveHomographyDLT(pairs)
  if (!h) {
    return null
  }

  // Get extrinsics from homography
  const ex = extrinsicsFromHomography(h, k)
  return ex ?? null
}
