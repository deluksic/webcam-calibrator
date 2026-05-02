import * as Comlink from 'comlink'

import type { CalibrationFrameObservation, ObjectTag } from '@/lib/calibrationTypes'

import type { CalibWorkerApi, CalibrationResult, CalibrationOk, CalibrationErr } from './calibration.worker'

let worker: Worker | undefined
let proxy: Comlink.Remote<CalibWorkerApi> | undefined

function getWorker(): Comlink.Remote<CalibWorkerApi> {
  if (proxy === undefined) {
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
    objectTags: ObjectTag[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
  ): Promise<CalibrationResult>
}

export const calibApi: CalibApi = {
  async solveCalibration(
    objectTags: ObjectTag[],
    frames: CalibrationFrameObservation[],
    imageSize: { width: number; height: number },
  ): Promise<CalibrationResult> {
    const w = getWorker()
    return w.solveCalibration(objectTags, frames, imageSize)
  },
}
