// Build per-frame reprojection overlay geometry for a 2D canvas (image pixel space).

import { initCalibrator } from '@deluksic/opencv-calibration-wasm'
import USE_WASM_MODULE from '@deluksic/opencv-calibration-wasm/wasm/calibrate.wasm?url'
void USE_WASM_MODULE

import type { DetectedQuad } from '@/gpu/contour'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { Point } from '@/lib/geometry'
import type { Vec3, Mat3 } from '@/lib/opencvCalibration'
import type { TargetLayout } from '@/lib/targetLayout'

const calibrator = await initCalibrator()

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

export function buildReprojectionDrawOps(
  layout: TargetLayout,
  k: CameraIntrinsics,
  distortion: RationalDistortion8 | undefined,
  quads: DetectedQuad[],
  imageWidth: number,
  imageHeight: number,
): { ops: ReprojectionDrawOp[]; rms: number; R: Mat3; t: Vec3; tagCount: number } | null {
  const livePose = buildLivePose(layout, quads, k, distortion)
  if (!livePose) {
    return null
  }
  const { R, t, rvec, objectPoints, imagePoints, tagCount } = livePose
  const cameraMatrix: [[number, number, number], [number, number, number], [number, number, number]] = [
    [k.fx, 0, k.cx],
    [0, k.fy, k.cy],
    [0, 0, 1],
  ]

  const projected = calibrator.projectPoints({
    objectPoints: [objectPoints],
    rvecs: [[rvec[0], rvec[1], rvec[2]]],
    tvecs: [[t.x, t.y, t.z]],
    cameraMatrix,
    distortionCoefficients: distortion,
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
  rvec: [number, number, number]
  objectPoints: [number, number, number][]
  imagePoints: [number, number][]
  tagCount: number
}

/**
 * Compute live pose (R, t) from current tag detections via OpenCV solvePnP.
 */
function buildLivePose(
  layout: TargetLayout,
  quads: DetectedQuad[],
  k: CameraIntrinsics,
  distortion: RationalDistortion8 | undefined,
): LivePose | null {
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
        const observed = q.corners[j]!
        const plane = { x: pl[j]!.x, y: pl[j]!.y }
        objectPoints.push([plane.x, plane.y, 0])
        imagePoints.push([observed.x, observed.y])
      }
    }
  }

  if (objectPoints.length < 4) {
    return null
  }
  const cameraMatrix: [[number, number, number], [number, number, number], [number, number, number]] = [
    [k.fx, 0, k.cx],
    [0, k.fy, k.cy],
    [0, 0, 1],
  ]
  const pnp = calibrator.solvePnP({
    objectPoints,
    imagePoints,
    cameraMatrix,
    distortionCoefficients: distortion,
  })
  if (!pnp.success) {
    return null
  }
  const [rx, ry, rz] = pnp.rvec
  const [tx, ty, tz] = pnp.tvec
  const R = rvecToMatrix([rx, ry, rz])
  return { R, t: { x: tx, y: ty, z: tz }, rvec: [rx, ry, rz], objectPoints, imagePoints, tagCount: quads.length }
}

function rvecToMatrix(rvec: [number, number, number]): Mat3 {
  const theta = Math.hypot(rvec[0], rvec[1], rvec[2])
  if (theta < 1e-12) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  }
  const ux = rvec[0] / theta
  const uy = rvec[1] / theta
  const uz = rvec[2] / theta
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const t = 1 - c
  return [
    t * ux * ux + c,
    t * ux * uy - s * uz,
    t * ux * uz + s * uy,
    t * ux * uy + s * uz,
    t * uy * uy + c,
    t * uy * uz - s * ux,
    t * ux * uz - s * uy,
    t * uy * uz + s * ux,
    t * uz * uz + c,
  ]
}
