import * as Comlink from 'comlink'
import { calibrateCameraRO, projectPoints } from '@deluksic/opencv-calibration-wasm'

const WASM_MODULE_PATH = new URL(
  '@deluksic/opencv-calibration-wasm/wasm/calibrate.mjs',
  import.meta.url,
).href
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { CalibrationFrameObservation, LabeledPoint } from '@/lib/calibrationTypes'
import type { TargetLayout } from '@/lib/targetLayout'

export interface CalibWorkerApi {
  solveCalibration(
    layout: TargetLayout,
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
  rmsPx: number
  perFrameRmsPx: [number, number][]
}

export type CalibrationErr = { kind: 'error'; reason: 'too-few-views' | 'singular' | 'non-physical' }

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
    t * uy * uz + s * ux,
    t * uz * uz + c,
  ]
}

function cameraIntrinsicsFromMatrix(
  cm: [[number, number, number], [number, number, number], [number, number, number]],
): CameraIntrinsics {
  return { fx: cm[0]![0]!, fy: cm[1]![1]!, cx: cm[0]![2]!, cy: cm[1]![2]! }
}

function padDistortion(coeffs: number[]): RationalDistortion8 {
  return [coeffs[0] ?? 0, coeffs[1] ?? 0, coeffs[2] ?? 0, coeffs[3] ?? 0, coeffs[4] ?? 0, coeffs[5] ?? 0, coeffs[6] ?? 0, coeffs[7] ?? 0]
}

function buildWasmInput(
  labeledPoints: LabeledPoint[],
  frames: CalibrationFrameObservation[],
): { objectPoints: [number, number, number][][]; imagePoints: [number, number][][]; sharedPointIds: number[] } | undefined {
  const framePointSets = frames.map((f) => new Set(f.framePoints.map((fp) => fp.pointId)))
  const sharedPointIds = [...framePointSets[0]!].filter((pid) => framePointSets.every((s) => s.has(pid)))

  if (sharedPointIds.length < 6) {
    return undefined
  }

  const objectPoints: [number, number, number][][] = frames.map(() =>
    sharedPointIds.map((pid) => {
      const lp = labeledPoints.find((l) => l.pointId === pid)
      return [lp!.plane.x, lp!.plane.y, 0]
    }),
  )

  const imagePoints: [number, number][][] = frames.map((f) =>
    sharedPointIds.map((pid) => {
      const fp = f.framePoints.find((p) => p.pointId === pid)!
      return [fp.imagePoint.x, fp.imagePoint.y]
    }),
  )

  return { objectPoints, imagePoints, sharedPointIds }
}

const api: CalibWorkerApi = {
  async solveCalibration(
    layout: TargetLayout,
    labeledPoints: LabeledPoint[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
  ): Promise<CalibrationResult> {
    const wasmInput = buildWasmInput(labeledPoints, frames)
    if (!wasmInput || wasmInput.objectPoints.length < 3) {
      return { kind: 'error', reason: 'too-few-views' }
    }

    const { objectPoints, imagePoints, sharedPointIds } = wasmInput

    const wasmResult = await calibrateCameraRO({
      objectPoints,
      imagePoints,
      imageSize,
    }, { modulePath: WASM_MODULE_PATH })

    const K = cameraIntrinsicsFromMatrix(wasmResult.cameraMatrix)
    const distortion = padDistortion(wasmResult.distortionCoefficients)

    // Compute per-frame RMS via projectPoints
    const projected = await projectPoints({
      objectPoints,
      rvecs: wasmResult.rvecs,
      tvecs: wasmResult.tvecs,
      cameraMatrix: wasmResult.cameraMatrix,
      distortionCoefficients: wasmResult.distortionCoefficients,
    }, { modulePath: WASM_MODULE_PATH })

    const perFrameRmsPx: [number, number][] = []
    let allSq: number[] = []
    for (let view = 0; view < frames.length; view++) {
      const fo = frames[view]!
      const proj = projected.projectedImagePoints[view]!
      let sq: number[] = []
      for (let p = 0; p < sharedPointIds.length; p++) {
        const dx = proj[p]![0]! - imagePoints[view]![p]![0]!
        const dy = proj[p]![1]! - imagePoints[view]![p]![1]!
        sq.push(dx * dx + dy * dy)
      }
      if (sq.length > 0) {
        perFrameRmsPx.push([fo.frameId, Math.sqrt(sq.reduce((a, b) => a + b, 0) / sq.length)])
        allSq.push(...sq)
      }
    }
    const rmsPx = allSq.length > 0 ? Math.sqrt(allSq.reduce((a, b) => a + b, 0) / allSq.length) : 0

    // Convert extrinsics
    const extrinsics: { frameId: number; R: Mat3; t: Vec3 }[] = []
    for (let i = 0; i < wasmResult.rvecs.length; i++) {
      const rvec = wasmResult.rvecs[i]!
      const tvec = wasmResult.tvecs[i]!
      const frameId = frames[i]?.frameId ?? i
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
      rmsPx,
      perFrameRmsPx,
    }
  },
}

Comlink.expose(api)
