import { initCalibrator } from '@deluksic/opencv-calibration-wasm'
import CALIBRATE_WASM_PATH from '@deluksic/opencv-calibration-wasm/wasm/calibrate.wasm?url'
import * as Comlink from 'comlink'

import type { CalibrationFrameObservation, Corners3, ObjectTag } from '@/lib/calibrationTypes'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { Point } from '@/lib/geometry'

export interface CalibWorkerApi {
  solveCalibration(
    objectTags: ObjectTag[],
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
  updatedTargets: ObjectTag[]
  rmsPx: number
  perFrameRmsPx: [number, number][]
}

export type CalibrationErr = {
  kind: 'error'
  reason: 'too-few-views' | 'singular' | 'non-physical'
  details?: string
}

export type CalibrationResult = CalibrationOk | CalibrationErr

/** > max corner index (3). Row key = `tagId * STRIDE + cornerId` (same order as `(tagId, cornerId)` lexicographic). */
const OBJECT_POINT_ROW_KEY_STRIDE = 4

function objectPointRowKey(tagId: number, cornerId: number): number {
  return tagId * OBJECT_POINT_ROW_KEY_STRIDE + cornerId
}

function unpackObjectPointRowKey(rowKey: number): { tagId: number; cornerId: number } {
  return {
    tagId: Math.trunc(rowKey / OBJECT_POINT_ROW_KEY_STRIDE),
    cornerId: rowKey % OBJECT_POINT_ROW_KEY_STRIDE,
  }
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
  objectTags: ObjectTag[],
  frames: CalibrationFrameObservation[],
):
  | {
      objectPoints: [number, number, number][][]
      imagePoints: [number, number][][]
      frameIds: number[]
      sharedObjectPointRowKeysSorted: number[]
    }
  | { error: string }
  | undefined {
  const layoutByTagId = new Map(objectTags.map((t) => [t.tagId, t.corners] as const))

  type FrameCorners = Map<number /* objectPointRowKey */, Point /* imagePoint */>
  const validFrames = frames
    .map((frame) => {
      const corners: FrameCorners = new Map()
      for (const ft of frame.tags) {
        const layoutCorners = layoutByTagId.get(ft.tagId)
        if (!layoutCorners) {
          continue
        }
        for (let cornerId = 0; cornerId < 4; cornerId++) {
          const rowKey = objectPointRowKey(ft.tagId, cornerId)
          corners.set(rowKey, ft.corners[cornerId]!)
        }
      }
      if (corners.size >= 6) {
        return { frameId: frame.frameId, corners }
      }
      return undefined
    })
    .filter((f) => f !== undefined)

  if (validFrames.length < 3) {
    return { error: `need >=3 valid frames (have ${validFrames.length})` }
  }

  // `calibrateCameraRO` requires identical object-point templates per view.
  let sharedObjectPointRowKeysSorted = [...validFrames[0]!.corners.keys()]
  for (let i = 1; i < validFrames.length; i++) {
    const rowKeys = validFrames[i]!.corners
    sharedObjectPointRowKeysSorted = sharedObjectPointRowKeysSorted.filter((k) => rowKeys.has(k))
  }

  // Map key order follows the first frame's tag iteration order. The solver
  // needs the same row order in every view's object/image point lists; numeric row keys sort correctly.
  sharedObjectPointRowKeysSorted.sort((a, b) => a - b)

  if (sharedObjectPointRowKeysSorted.length < 6) {
    return { error: `need >=6 shared corners across all views (have ${sharedObjectPointRowKeysSorted.length})` }
  }

  const objectTemplate: [number, number, number][] = sharedObjectPointRowKeysSorted.map((rowKey) => {
    const { tagId, cornerId } = unpackObjectPointRowKey(rowKey)
    const plane = layoutByTagId.get(tagId)![cornerId]!
    return [plane.x, plane.y, plane.z]
  })

  const objectPoints: [number, number, number][][] = validFrames.map(() => objectTemplate)

  const imagePoints: [number, number][][] = validFrames.map((frame) =>
    sharedObjectPointRowKeysSorted.map((k) => {
      const pt = frame.corners.get(k)!
      return [pt.x, pt.y]
    }),
  )

  const frameIds = validFrames.map((frame) => frame.frameId)

  return { objectPoints, imagePoints, frameIds, sharedObjectPointRowKeysSorted }
}

const api: CalibWorkerApi = {
  async solveCalibration(
    objectTags: ObjectTag[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
  ): Promise<CalibrationResult> {
    try {
      const wasmInput = buildWasmInput(objectTags, frames)
      if (!wasmInput) {
        return { kind: 'error', reason: 'too-few-views', details: 'input builder returned undefined' }
      }
      if ('error' in wasmInput) {
        return { kind: 'error', reason: 'too-few-views', details: wasmInput.error }
      }
      if (wasmInput.objectPoints.length < 3) {
        return {
          kind: 'error',
          reason: 'too-few-views',
          details: `need >=3 solve views (have ${wasmInput.objectPoints.length})`,
        }
      }

      const { objectPoints, imagePoints, frameIds, sharedObjectPointRowKeysSorted } = wasmInput

      const calibrator = await initCalibrator({ wasmPath: CALIBRATE_WASM_PATH })

      const wasmResult = calibrator.calibrateCameraRO({
        objectPoints,
        imagePoints,
        imageSize,
      })

      const K = cameraIntrinsicsFromMatrix(wasmResult.cameraMatrix)
      const distortion = padDistortion(wasmResult.distortionCoefficients)

      const refinedObjectPoints = wasmResult.newObjPoints
      const updatedTargetsMap = new Map<number, Corners3>(objectTags.map((t) => [t.tagId, [...t.corners]]))
      for (let i = 0; i < sharedObjectPointRowKeysSorted.length; ++i) {
        const { tagId, cornerId } = unpackObjectPointRowKey(sharedObjectPointRowKeysSorted[i]!)
        const [x, y, z] = refinedObjectPoints[i]!
        const corners = updatedTargetsMap.get(tagId)
        if (corners) {
          corners[cornerId] = { x, y, z }
        }
      }
      const updatedTargets = [...updatedTargetsMap.entries()].map(([tagId, corners]): ObjectTag => ({ tagId, corners }))

      const refinedObjPointsAligned = imagePoints.map(() => refinedObjectPoints)

      // Compute per-frame RMS via projectPoints
      const projected = calibrator.projectPoints({
        objectPoints: refinedObjPointsAligned,
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
        updatedTargets,
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
