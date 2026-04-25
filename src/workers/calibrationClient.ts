import * as Comlink from 'comlink'
import type { CalibWorkerApi, CalibrationResult, CalibrationOk, CalibrationErr } from './calibration.worker'
import type { TargetLayout } from '@/lib/targetLayout'
import type { LabeledPoint, CalibrationFrameObservation } from '@/lib/calibrationTypes'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { Mat3, Vec3 } from './calibration.worker'

let worker: Worker | null = null
let proxy: Comlink.Remote<CalibWorkerApi> | null = null

function getWorker(): Comlink.Remote<CalibWorkerApi> {
  if (!proxy) {
    worker = new Worker(new URL('./calibration.worker.ts', import.meta.url), { type: 'module' })
    proxy = Comlink.wrap<CalibWorkerApi>(worker)
  }
  return proxy
}

// Re-export types for consumers
export type { CalibrationResult, CalibrationOk, CalibrationErr }
export { getWorker }

export interface CalibApi {
  solveCalibration(
    layout: TargetLayout,
    labeledPoints: LabeledPoint[],
    frames: CalibrationFrameObservation[],
    imageSize?: { width: number; height: number },
  ): Promise<CalibrationResult>
}

export const calibApi: CalibApi = {
  async solveCalibration(
    layout: TargetLayout,
    labeledPoints: LabeledPoint[],
    frames: CalibrationFrameObservation[],
    imageSize = { width: 1280, height: 720 },
  ): Promise<CalibrationResult> {
    const w = getWorker()
    return w.solveCalibration(layout, labeledPoints, frames, imageSize)
  },
}
