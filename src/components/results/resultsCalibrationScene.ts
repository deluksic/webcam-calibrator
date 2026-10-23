import { markerCenterWritesForGpu } from '@/gpu/pipelines/resultsMarkerPipeline'
import { orthoExtentYForPoints } from '@/gpu/pipelines/resultsSceneCpu'
import type { CalibrationOk } from '@/workers/calibration.worker'

export type ResultsCalibrationScene = {
  ok: CalibrationOk
  baseOrthoExtentY: number
  centerWrites: ReturnType<typeof markerCenterWritesForGpu>
}

export function buildResultsCalibrationScene(ok: CalibrationOk): ResultsCalibrationScene {
  return {
    ok,
    baseOrthoExtentY: orthoExtentYForPoints(ok),
    centerWrites: markerCenterWritesForGpu(ok),
  }
}
