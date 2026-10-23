import { initCalibrator } from '@deluksic/opencv-calibration-wasm'
import CALIBRATE_WASM_PATH from '@deluksic/opencv-calibration-wasm/wasm/calibrate.wasm?url'
import * as Comlink from 'comlink'

import { buildCalibrationInput } from '@/lib/calibrationInput'
import type { CalibrationFrameObservation, TargetObjectPoint } from '@/lib/calibrationTypes'
import { calibratorInitOptions } from '@/lib/calibratorWasmInit'
import type { HexTargetConfig } from '@/lib/hexTarget/config'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import { rvecToMatrix, type Mat3, type Vec3 } from '@/lib/poseMath'

export type { Mat3, Vec3 } from '@/lib/poseMath'

export interface CalibWorkerApi {
  solveCalibration(
    targetPoints: TargetObjectPoint[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
    cameraId: string,
  ): Promise<CalibrationResult>
}

export type CalibrationOk = {
  kind: 'ok'
  K: CameraIntrinsics
  distortion: RationalDistortion8
  extrinsics: { frameId: number; R: Mat3; t: Vec3 }[]
  /** Refined target points that appear in every solve view. */
  updatedTargetPoints: TargetObjectPoint[]
  rmsPx: number
  perFrameRmsPx: [number, number][]
  imageSize: { width: number; height: number }
  cameraId: string
  /** Target parameters used by this solve (attached by the session context). */
  targetConfig?: HexTargetConfig
  /** PNG filenames keyed by frameId string. Files are in the same directory as the JSON. */
  frameImages?: Record<string, string>
  /** Per-frame identified target points with image file reference. */
  observations?: {
    frameId: number
    imageFile: string
    points: { designId: number; x: number; y: number }[]
  }[]
}

export type CalibrationErr = {
  kind: 'error'
  reason: 'too-few-views' | 'singular' | 'non-physical'
  details?: string
}

export type CalibrationResult = CalibrationOk | CalibrationErr

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

const api: CalibWorkerApi = {
  async solveCalibration(
    targetPoints: TargetObjectPoint[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
    cameraId: string,
  ): Promise<CalibrationResult> {
    try {
      const built = buildCalibrationInput(targetPoints, frames)
      if (built.kind === 'error') {
        return { kind: 'error', reason: 'too-few-views', details: built.details }
      }
      const { objectPoints, imagePoints, frameIds, sharedDesignIds } = built.input

      const calibrator = await initCalibrator(calibratorInitOptions(CALIBRATE_WASM_PATH))
      const wasmResult = calibrator.calibrateCameraRO({
        objectPoints,
        imagePoints,
        imageSize,
      })

      const K = cameraIntrinsicsFromMatrix(wasmResult.cameraMatrix)
      const distortion = padDistortion(wasmResult.distortionCoefficients)

      const refined = wasmResult.newObjPoints
      const refinedComplete = refined.length >= sharedDesignIds.length
      const updatedTargetPoints: TargetObjectPoint[] = sharedDesignIds.map((designId, index) => {
        const point = refinedComplete ? refined[index]! : objectPoints[0]![index]!
        return { designId, x: point[0], y: point[1], z: point[2] }
      })
      const refinedObjectPoints: [number, number, number][] = updatedTargetPoints.map((point) => [
        point.x,
        point.y,
        point.z,
      ])
      const refinedObjPointsAligned = imagePoints.map(() => refinedObjectPoints)

      const projected = calibrator.projectPoints({
        objectPoints: refinedObjPointsAligned,
        rvecs: wasmResult.rvecs,
        tvecs: wasmResult.tvecs,
        cameraMatrix: wasmResult.cameraMatrix,
        distortionCoefficients: wasmResult.distortionCoefficients,
      })

      const perFrameRmsPx: [number, number][] = []
      const allSq: number[] = []
      for (let view = 0; view < frameIds.length; view += 1) {
        const proj = projected.projectedImagePoints[view]!
        const sq: number[] = []
        for (let pointIndex = 0; pointIndex < proj.length; pointIndex += 1) {
          const dx = proj[pointIndex]![0]! - imagePoints[view]![pointIndex]![0]!
          const dy = proj[pointIndex]![1]! - imagePoints[view]![pointIndex]![1]!
          sq.push(dx * dx + dy * dy)
        }
        if (sq.length > 0) {
          perFrameRmsPx.push([
            frameIds[view]!,
            Math.sqrt(sq.reduce((sum, value) => sum + value, 0) / sq.length),
          ])
          allSq.push(...sq)
        }
      }
      const rmsPx =
        allSq.length > 0
          ? Math.sqrt(allSq.reduce((sum, value) => sum + value, 0) / allSq.length)
          : 0

      const extrinsics: { frameId: number; R: Mat3; t: Vec3 }[] = []
      for (let index = 0; index < wasmResult.rvecs.length; index += 1) {
        const rvec = wasmResult.rvecs[index]!
        const tvec = wasmResult.tvecs[index]!
        extrinsics.push({
          frameId: frameIds[index] ?? index,
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
        imageSize,
        cameraId,
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
