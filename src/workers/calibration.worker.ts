import { initCalibrator } from '@deluksic/opencv-calibration-wasm'
import USE_WASM_MODULE from '@deluksic/opencv-calibration-wasm/wasm/calibrate.wasm?url'
void USE_WASM_MODULE

import * as Comlink from 'comlink'

import type { CalibrationFrameObservation, LabeledPoint } from '@/lib/calibrationTypes'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { Point3 } from '@/lib/calibrationTypes'

export interface CalibWorkerApi {
  solveCalibration(
    labeledPoints: LabeledPoint[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
  ): Promise<CalibrationResult>
}

export type Mat3 = [number, number, number, number, number, number, number, number, number]
export type Vec3 = { x: number; y: number; z: number }

export type CalibrationOk = {
  kind: 'ok'
  K: CameraIntrinsics
  distortion: RationalDistortion8
  extrinsics: { frameId: number; R: Mat3; t: Vec3 }[]
  updatedTargetPoints: { pointId: number; position: Point3 }[]
  rmsPx: number
  perFrameRmsPx: [number, number][]
}

export type CalibrationErr = {
  kind: 'error'
  reason: 'too-few-views' | 'singular' | 'non-physical'
  details?: string
}

export type CalibrationResult = CalibrationOk | CalibrationErr

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
    t * uy * uz + s * uy,
    t * uz * uz + c,
  ]
}

function cameraIntrinsicsFromMatrix(
  cm: [[number, number, number], [number, number, number], [number, number, number]],
): CameraIntrinsics {
  return { fx: cm[0]![0]!, fy: cm[1]![1]!, cx: cm[0]![2]!, cy: cm[1]![2]! }
}

function padDistortion(coeffs: number[]): RationalDistortion8 {
  return [
    coeffs[0] ?? 0,
    coeffs[1] ?? 0,
    coeffs[2] ?? 0,
    coeffs[3] ?? 0,
    coeffs[4] ?? 0,
    coeffs[5] ?? 0,
    coeffs[6] ?? 0,
    coeffs[7] ?? 0,
  ]
}

function buildWasmInput(
  layoutPoints: LabeledPoint[],
  frames: CalibrationFrameObservation[],
):
  | { objectPoints: [number, number, number][][]; imagePoints: [number, number][][]; frameIds: number[]; pointIds: number[] }
  | { error: string }
  | undefined {
  const layoutById = new Map(layoutPoints.map((lp) => [lp.pointId, lp] as const))
  const validFrames = frames
    .map((frame) => ({
      frameId: frame.frameId,
      pointsById: new Map(frame.framePoints.filter((fp) => layoutById.has(fp.pointId)).map((fp) => [fp.pointId, fp] as const)),
    }))
    .filter((frame) => frame.pointsById.size >= 6)

  if (validFrames.length < 3) {
    return { error: `need >=3 valid frames (have ${validFrames.length})` }
  }

  // `calibrateCameraRO` requires identical object-point templates per view.
  // Use ALL valid views and intersect shared points across all of them.
  let sharedPointIds = [...validFrames[0]!.pointsById.keys()]
  for (let i = 1; i < validFrames.length; i++) {
    const ids = validFrames[i]!.pointsById
    sharedPointIds = sharedPointIds.filter((id) => ids.has(id))
  }
  if (sharedPointIds.length < 6) {
    return { error: `need >=6 shared corners across all views (have ${sharedPointIds.length})` }
  }
  sharedPointIds.sort((a, b) => a - b)

  const objectTemplate: [number, number, number][] = sharedPointIds.map((pointId) => {
    const lp = layoutById.get(pointId)!
    return [lp.position.x, lp.position.y, 0]
  })

  const objectPoints: [number, number, number][][] = validFrames.map(() => objectTemplate)
  const imagePoints: [number, number][][] = validFrames.map((frame) =>
    sharedPointIds.map((pointId) => {
      const fp = frame.pointsById.get(pointId)!
      return [fp.imagePoint.x, fp.imagePoint.y]
    }),
  )
  const frameIds = validFrames.map((frame) => frame.frameId)

  return { objectPoints, imagePoints, frameIds, pointIds: sharedPointIds }
}

const api: CalibWorkerApi = {
  async solveCalibration(
    layoutPoints: LabeledPoint[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
  ): Promise<CalibrationResult> {
    try {
      const wasmInput = buildWasmInput(layoutPoints, frames)
      if (!wasmInput) {
        return { kind: 'error', reason: 'too-few-views', details: 'input builder returned undefined' }
      }
      if ('error' in wasmInput) {
        return { kind: 'error', reason: 'too-few-views', details: wasmInput.error }
      }
      if (wasmInput.objectPoints.length < 3) {
        return { kind: 'error', reason: 'too-few-views', details: `need >=3 solve views (have ${wasmInput.objectPoints.length})` }
      }

      const { objectPoints, imagePoints, frameIds, pointIds } = wasmInput

      const calibrator = await initCalibrator()

      const wasmResult = calibrator.calibrateCameraRO({
        objectPoints,
        imagePoints,
        imageSize,
      })

      const K = cameraIntrinsicsFromMatrix(wasmResult.cameraMatrix)
      const distortion = padDistortion(wasmResult.distortionCoefficients)
      const refinedObjPoints =
        Array.isArray(wasmResult.newObjPoints) && wasmResult.newObjPoints.length === pointIds.length
          ? wasmResult.newObjPoints
          : objectPoints[0]!
      const updatedTargetPoints = pointIds.map((pointId, i) => {
        const p = refinedObjPoints[i]!
        return { pointId, position: { x: p[0]!, y: p[1]!, z: p[2]! } }
      })
      const refinedObjectPoints: [number, number, number][][] = imagePoints.map(() => refinedObjPoints)

    // Compute per-frame RMS via projectPoints
      const projected = calibrator.projectPoints({
        objectPoints: refinedObjectPoints,
        rvecs: wasmResult.rvecs,
        tvecs: wasmResult.tvecs,
        cameraMatrix: wasmResult.cameraMatrix,
        distortionCoefficients: wasmResult.distortionCoefficients,
      })

      const perFrameRmsPx: [number, number][] = []
      let allSq: number[] = []
      for (let view = 0; view < frameIds.length; view++) {
        const proj = projected.projectedImagePoints[view]!
        let sq: number[] = []
        for (let p = 0; p < proj.length; p++) {
          const dx = proj[p]![0]! - imagePoints[view]![p]![0]!
          const dy = proj[p]![1]! - imagePoints[view]![p]![1]!
          sq.push(dx * dx + dy * dy)
        }
        if (sq.length > 0) {
          perFrameRmsPx.push([frameIds[view]!, Math.sqrt(sq.reduce((a, b) => a + b, 0) / sq.length)])
          allSq.push(...sq)
        }
      }
      const rmsPx = allSq.length > 0 ? Math.sqrt(allSq.reduce((a, b) => a + b, 0) / allSq.length) : 0

    // Convert extrinsics
      const extrinsics: { frameId: number; R: Mat3; t: Vec3 }[] = []
      for (let i = 0; i < wasmResult.rvecs.length; i++) {
        const rvec = wasmResult.rvecs[i]!
        const tvec = wasmResult.tvecs[i]!
        const frameId = frameIds[i] ?? i
        extrinsics.push({
          frameId,
          R: rvecToMatrix(rvec),
          t: { x: tvec[0]!, y: tvec[1]!, z: tvec[2]! },
        })
      }

      return {
        kind: 'ok',
        K,
        distortion,
        extrinsics,
        updatedTargetPoints,
        rmsPx,
        perFrameRmsPx,
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase()
      if (msg.includes('too-few') || msg.includes('not enough') || msg.includes('bad argument')) {
        return { kind: 'error', reason: 'too-few-views', details: String(err) }
      }
      if (msg.includes('singular') || msg.includes('degenerate')) {
        return { kind: 'error', reason: 'singular', details: String(err) }
      }
      return { kind: 'error', reason: 'non-physical', details: String(err) }
    }
  },
}

Comlink.expose(api)
