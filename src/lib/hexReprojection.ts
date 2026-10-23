// Live reprojection: solvePnP from identified hex-target points, then project them back.

import { initCalibrator } from '@deluksic/opencv-calibration-wasm'
import CALIBRATE_WASM_PATH from '@deluksic/opencv-calibration-wasm/wasm/calibrate.wasm?url'

import type { CalibrationPointObservation } from '@/lib/calibrationTypes'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import { calibratorInitOptions } from '@/lib/calibratorWasmInit'
import type { HexTargetModel } from '@/lib/hexTarget/target'
import {
  cameraDistanceFromT,
  cameraTiltDegFromR,
  rvecToMatrix,
  type Mat3,
  type Vec3,
} from '@/lib/poseMath'

const calibrator = await initCalibrator(calibratorInitOptions(CALIBRATE_WASM_PATH))

export type ReprojectionOverlayPair = {
  original: { x: number; y: number }
  reprojected: { x: number; y: number }
}

export type HexReprojectionResult = {
  pairs: ReprojectionOverlayPair[]
  count: number
  rms: number
  R: Mat3
  t: Vec3
  pointCount: number
  tiltDeg: number
  dist: number
}

export function buildHexReprojection(
  design: HexTargetModel,
  k: CameraIntrinsics,
  distortion: RationalDistortion8 | undefined,
  observed: readonly CalibrationPointObservation[],
): HexReprojectionResult | undefined {
  const objectById = new Map(design.objectPoints.map((point) => [point.designId, point] as const))
  const objectPoints: [number, number, number][] = []
  const imagePoints: [number, number][] = []
  for (const point of observed) {
    const object = objectById.get(point.designId)
    if (!object) {
      continue
    }
    objectPoints.push([object.x, object.y, object.z])
    imagePoints.push([point.x, point.y])
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
  const projected = calibrator.projectPoints({
    objectPoints: [objectPoints],
    rvecs: [pnp.rvec],
    tvecs: [pnp.tvec],
    cameraMatrix,
    distortionCoefficients: distortion,
  })
  const projectedPoints = projected.projectedImagePoints[0]!
  const pairs: ReprojectionOverlayPair[] = []
  const squaredErrors: number[] = []
  for (let index = 0; index < objectPoints.length; index += 1) {
    const original = { x: imagePoints[index]![0]!, y: imagePoints[index]![1]! }
    const reprojected = {
      x: projectedPoints[index]![0]!,
      y: projectedPoints[index]![1]!,
    }
    const dx = reprojected.x - original.x
    const dy = reprojected.y - original.y
    squaredErrors.push(dx * dx + dy * dy)
    pairs.push({ original, reprojected })
  }
  const rms =
    squaredErrors.length > 0
      ? Math.sqrt(squaredErrors.reduce((sum, value) => sum + value, 0) / squaredErrors.length)
      : 0
  const R = rvecToMatrix(pnp.rvec)
  const t: Vec3 = { x: pnp.tvec[0], y: pnp.tvec[1], z: pnp.tvec[2] }
  return {
    pairs,
    count: pairs.length,
    rms,
    R,
    t,
    pointCount: observed.length,
    tiltDeg: cameraTiltDegFromR(R),
    dist: cameraDistanceFromT(t),
  }
}
