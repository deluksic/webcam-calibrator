import { d } from 'typegpu'

import type { DetectedQuad } from '@/gpu/contour'
import {
  DECODED_TAG_ID_UNKNOWN,
  type QuadData,
  MAX_INSTANCES,
  type GridVizFailInterrogateMode,
} from '@/gpu/pipelines/gridVizPipeline'
import type { ReprojPairGpu } from '@/gpu/pipelines/reprojectionOverlayPipeline'
import { tryComputeHomography } from '@/lib/geometry'
import type { ReprojectionOverlayPair } from '@/lib/reprojectionLive'

import type { CameraPipeline } from './cameraPipeline'

const { min } = Math

/** Write homography matrix (mat3x3) + debug data per quad to the GPU buffer.
 * H maps unit square → detected quad. Vertex shader applies: (x,y,z) = H * (u,v,1).
 * H is column-major mat3x3f: [c0.x, c0.y, c0.z, 0, c1..., c2...]
 */
export function updateQuadCornersBuffer(
  pipeline: CameraPipeline,
  quads: DetectedQuad[],
  showFallbacks: boolean = true,
): void {
  const filtered = showFallbacks ? quads : quads.filter((q) => q.hasCorners && typeof q.decodedTagId === 'number')
  const count = min(filtered.length, MAX_INSTANCES)

  const data: QuadData[] = []
  for (let i = 0; i < count; i++) {
    const quad = filtered[i]!
    const H = tryComputeHomography(quad.corners)
    const debug = quad.cornerDebug
    const tagId = quad.vizTagId !== undefined ? quad.vizTagId >>> 0 : DECODED_TAG_ID_UNKNOWN

    data.push({
      homography: H
        ? d.mat3x3f(H[0], H[3], H[6], H[1], H[4], H[7], H[2], H[5], H[8])
        : d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: debug ? debug.failureCode : 0,
        edgePixelCount: debug ? debug.edgePixelCount / 100 : 0,
        minR2: debug ? debug.minR2 : 0,
        intersectionCount: debug ? debug.intersectionCount : 0,
      },
      decodedTagId: d.u32(tagId),
    })
  }

  for (let i = count; i < MAX_INSTANCES; i++) {
    data.push({
      homography: d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: 0,
        edgePixelCount: 0,
        minR2: 0,
        intersectionCount: 0,
      },
      decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
    })
  }

  pipeline.grid.quadCornersBuffer.write(data)
}

/** Upload `{ original, reprojected }` pairs for the GPU reprojection overlay; pads to `MAX_INSTANCES`. */
export function updateReprojectionOverlayBuffer(
  pipeline: CameraPipeline,
  pairs: ReprojectionOverlayPair[],
  count: number,
): void {
  const capped = min(count, MAX_INSTANCES)
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
}

/** Grid overlay: 0 = legacy fail colors, 1 = red highlights insufficient-edge failures, 2 = blue highlights line-fit failures. */
export function setGridVizFailInterrogate(pipeline: CameraPipeline, mode: GridVizFailInterrogateMode): void {
  pipeline.grid.gridVizDebugModeBuffer.write(mode)
}
