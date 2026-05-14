import { markerCenterWritesForGpu } from '@/gpu/pipelines/resultsMarkerPipeline'
import { orthoExtentYForPoints } from '@/gpu/pipelines/resultsSceneCpu'
import { tagQuadDrawCountForGpu, tagQuadWritesForGpu } from '@/gpu/pipelines/resultsTagQuadsPipeline'
import type { CalibrationOk } from '@/workers/calibration.worker'

export type ResultsCalibrationScene = {
  ok: CalibrationOk
  baseOrthoExtentY: number
  centerWrites: ReturnType<typeof markerCenterWritesForGpu>
  tagQuadWrites: ReturnType<typeof tagQuadWritesForGpu>
  tagQuadDrawCount: number
}

export function buildResultsCalibrationScene(ok: CalibrationOk): ResultsCalibrationScene {
  return {
    ok,
    baseOrthoExtentY: orthoExtentYForPoints(ok),
    centerWrites: markerCenterWritesForGpu(ok),
    tagQuadWrites: tagQuadWritesForGpu(ok),
    tagQuadDrawCount: tagQuadDrawCountForGpu(ok),
  }
}
