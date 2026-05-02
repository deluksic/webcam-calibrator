// Live reprojection: solvePnP + projectPoints for GPU overlay pairs and UI metrics.

import { initCalibrator } from '@deluksic/opencv-calibration-wasm'
import CALIBRATE_WASM_PATH from '@deluksic/opencv-calibration-wasm/wasm/calibrate.wasm?url'

import type { DetectedQuad } from '@/gpu/contour'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { Point } from '@/lib/geometry'
import type { Vec3, Mat3 } from '@/lib/opencvCalibration'
import type { TargetLayout } from '@/lib/targetLayout'

const calibrator = await initCalibrator({ wasmPath: CALIBRATE_WASM_PATH })

const { hypot } = Math

export interface ReprojectionOverlayPair {
  original: Point
  reprojected: Point
}

export interface ReprojectionOverlayResult {
  pairs: ReprojectionOverlayPair[]
  count: number
  rms: number
  R: Mat3
  t: Vec3
  tagCount: number
}

function dist(a: Point, b: Point) {
  return hypot(a.x - b.x, a.y - b.y)
}

export function buildReprojectionOverlayPairs(
  layout: TargetLayout,
  k: CameraIntrinsics,
  distortion: RationalDistortion8 | undefined,
  quads: DetectedQuad[],
  _imageWidth: number,
  _imageHeight: number,
): ReprojectionOverlayResult | undefined {
  void _imageWidth
  void _imageHeight
  const livePose = buildLivePose(layout, quads, k, distortion)
  if (!livePose) {
    return undefined
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
  const pairs: ReprojectionOverlayPair[] = []
  const errSq: number[] = []

  for (let i = 0; i < objectPoints.length; i++) {
    const image = { x: imagePoints[i]![0]!, y: imagePoints[i]![1]! }
    const pred = { x: projectedPoints[i]![0]!, y: projectedPoints[i]![1]! }
    const d = dist(pred, image)
    errSq.push(d * d)
    pairs.push({ original: image, reprojected: pred })
  }
  const rms = errSq.length > 0 ? Math.sqrt(errSq.reduce((a, b) => a + b, 0) / errSq.length) : 0

  return { pairs, count: pairs.length, rms, R, t, tagCount }
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
): LivePose | undefined {
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
        const p = pl[j]!
        objectPoints.push([p.x, p.y, p.z])
        imagePoints.push([observed.x, observed.y])
      }
    }
  }

  if (objectPoints.length < 4) {
    return undefined
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
    return undefined
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
