// Build per-frame reprojection overlay geometry for a 2D canvas (image pixel space).

import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { TargetLayout } from '@/lib/targetLayout'
import { initCalibrator, DEFAULT_WASM_MODULE_PATH, type Calibrator } from '@deluksic/opencv-calibration-wasm'
import { extrinsicsFromHomography, matrixToRvec, type Vec3, type Mat3 } from '@/lib/opencvCalibration'
import type { Point } from '@/lib/geometry'
import type { DetectedQuad } from '@/gpu/contour'

let calibratorPromise: Promise<Calibrator> | null = null

function getCalibrator(): Promise<Calibrator> {
  if (!calibratorPromise) {
    calibratorPromise = initCalibrator({ modulePath: DEFAULT_WASM_MODULE_PATH })
  }
  return calibratorPromise
}

const { hypot } = Math

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

export async function buildReprojectionDrawOps(
  layout: TargetLayout,
  k: CameraIntrinsics,
  quads: readonly DetectedQuad[],
  imageWidth: number,
  imageHeight: number,
): Promise<{ ops: ReprojectionDrawOp[]; rms: number; R: Mat3; t: Vec3; tagCount: number } | null> {
  const livePose = buildLivePose(layout, quads, k)
  if (!livePose) {
    return null
  }
  const { R, t, objectPoints, imagePoints, tagCount } = livePose

  const rvec = matrixToRvec(R)
  const cameraMatrix: [[number, number, number], [number, number, number], [number, number, number]] = [
    [k.fx, 0, k.cx],
    [0, k.fy, k.cy],
    [0, 0, 1],
  ]

  const calibrator = await getCalibrator()
  const projected = calibrator.projectPoints({
    objectPoints: [objectPoints],
    rvecs: [[rvec[0], rvec[1], rvec[2]]],
    tvecs: [[t.x, t.y, t.z]],
    cameraMatrix,
  })

  const projectedPoints = projected.projectedImagePoints[0]!
  const ops: ReprojectionDrawOp[] = []
  const errSq: number[] = []

  for (let i = 0; i < objectPoints.length; i++) {
    const image = { x: imagePoints[i]![0]!, y: imagePoints[i]![1]! }
    const pred = { x: projectedPoints[i]![0]!, y: projectedPoints[i]![1]! }
    const d = dist(pred, image)
    errSq.push(d * d)
    ops.push({ t: 'ring', c: image, r: 4, color: 'rgba(0,200,255,0.5)' })
    ops.push({ t: 'dot', c: pred, r: 3, color: 'rgba(255,0,200,0.9)' })
    ops.push({ t: 'line', a: image, b: pred, color: residualColor(d), w: 1.5 })
  }
  const rms = errSq.length > 0 ? Math.sqrt(errSq.reduce((a, b) => a + b, 0) / errSq.length) : 0
  void imageWidth
  void imageHeight

  return { ops, rms, R, t, tagCount }
}

export function cameraTiltDegFromR(R: Mat3): number {
  const nx = R[2]!
  const ny = R[5]!
  const nz = R[8]!
  const L = hypot(nx, hypot(ny, nz)) + 1e-12
  return (Math.acos(Math.min(1, Math.max(-1, nz / L))) * 180) / Math.PI
}

export function cameraDistanceFromT(t: Vec3): number {
  return hypot(t.x, hypot(t.y, t.z))
}

interface LivePose {
  R: Mat3
  t: Vec3
  objectPoints: [number, number, number][]
  imagePoints: [number, number][]
  tagCount: number
}

/**
 * Compute live pose (R, t) from current tag detections.
 * Builds DLT correspondences from layout → image, extracts pose via homography.
 */
function buildLivePose(
  layout: TargetLayout,
  quads: readonly DetectedQuad[],
  k: CameraIntrinsics,
): LivePose | null {
  const pairs: Array<{ plane: Point; image: Point }> = []
  const objectPoints: [number, number, number][] = []
  const imagePoints: [number, number][] = []

  for (const q of quads) {
    if (typeof q.decodedTagId !== 'number' || !q.hasCorners || q.cornerDebug?.failureCode !== 0) {
      continue
    }
    const pl = layout.get(q.decodedTagId)
    if (!pl) {
      continue
    }
    for (let j = 0; j < 4; j++) {
      if (q.corners[j]) {
        const plane = { x: pl[j]!.x, y: pl[j]!.y }
        pairs.push({ plane, image: q.corners[j]! })
        objectPoints.push([plane.x, plane.y, 0])
        imagePoints.push([q.corners[j]!.x, q.corners[j]!.y])
      }
    }
  }

  if (pairs.length < 4) {
    return null
  }

  const h = solveHomographySimple(pairs)
  if (!h) {
    return null
  }

  const ex = extrinsicsFromHomography(h, k)
  if (!ex) {
    return null
  }

  return { R: ex.R, t: ex.t, objectPoints, imagePoints, tagCount: quads.length }
}

/**
 * Simple DLT via normal equations + Gaussian elimination.
 */
function solveHomographySimple(pairs: Array<{ plane: Point; image: Point }>): Mat3 | null {
  let AtA: number[][] = Array.from({ length: 8 }, () => new Array(8).fill(0))
  let Atb: number[] = new Array(8).fill(0)

  for (const p of pairs) {
    const X = p.plane.x
    const Y = p.plane.y
    const u = p.image.x
    const v = p.image.y

    const r0 = [-X, -Y, -1, 0, 0, 0, u * X, u * Y]
    const r1 = [0, 0, 0, -X, -Y, -1, v * X, v * Y]

    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        AtA[i]![j]! += r0[i]! * r0[j]!
        AtA[i]![j]! += r1[i]! * r1[j]!
      }
      Atb[i]! += r0[i]! * (-u)
      Atb[i]! += r1[i]! * (-v)
    }
  }

  const N = 8
  for (let col = 0; col < N; col++) {
    let maxRow = col
    for (let row = col + 1; row < N; row++) {
      if (Math.abs(AtA[row]![col]!) > Math.abs(AtA[maxRow]![col]!)) {
        maxRow = row
      }
    }
    ;[AtA[col], AtA[maxRow]] = [AtA[maxRow]!, AtA[col]!]
    ;[Atb[col], Atb[maxRow]] = [Atb[maxRow]!, Atb[col]!]

    const pivot = AtA[col]![col]!
    if (Math.abs(pivot) < 1e-12) return null

    const pivotRow = AtA[col]!
    for (let j = col; j < N; j++) pivotRow[j]! /= pivot
    Atb[col]! /= pivot

    for (let row = 0; row < N; row++) {
      if (row !== col) {
        const factor = AtA[row]![col]!
        if (factor !== 0) {
          for (let j = col; j < N; j++) AtA[row]![j]! -= factor * pivotRow[j]!
          Atb[row]! -= factor * Atb[col]!
        }
      }
    }
  }

  return [Atb[0]!, Atb[1]!, Atb[2]!, Atb[3]!, Atb[4]!, Atb[5]!, Atb[6]!, Atb[7]!, 1]
}
