import { d } from 'typegpu'

import { MAX_INSTANCES, type GridVizFailInterrogateMode } from '@/gpu/pipelines/gridVizPipeline'
import type { ReprojPairGpu } from '@/gpu/pipelines/reprojectionOverlayPipeline'
import type { ReprojectionOverlayPair } from '@/lib/reprojectionLive'

import type { CameraPipeline } from './cameraPipeline'

const { min } = Math

/** Upload `{ original, reprojected }` pairs for the GPU reprojection overlay; pads to `MAX_INSTANCES`. */
export function updateReprojectionOverlayBuffer(
  pipeline: CameraPipeline,
  pairs: ReprojectionOverlayPair[],
  count: number,
): boolean {
  const capped = min(count, MAX_INSTANCES)
  if (capped === 0 && pipeline.reproj.reprojOverlayDrawState.instanceCount === 0) {
    return false
  }
  const data: ReprojPairGpu[] = []
  for (let i = 0; i < capped; i++) {
    const p = pairs[i]!
    data.push({
      original: d.vec2f(p.original.x, p.original.y),
      reprojected: d.vec2f(p.reprojected.x, p.reprojected.y),
    })
  }
  const dead = d.vec2f(-1, -1)
  for (let i = capped; i < MAX_INSTANCES; i++) {
    data.push({ original: dead, reprojected: dead })
  }
  pipeline.reproj.reprojOverlayBuffer.write(data)
  pipeline.reproj.reprojOverlayDrawState.instanceCount = capped
  return true
}

/** Grid overlay: 0 = legacy fail colors, 1 = red highlights insufficient-edge failures, 2 = blue highlights line-fit failures. */
export function setGridVizFailInterrogate(pipeline: CameraPipeline, mode: GridVizFailInterrogateMode): void {
  pipeline.grid.gridVizDebugModeBuffer.write(mode)
}
