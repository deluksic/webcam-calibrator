// Pack grid quad fields needed on CPU (no homography) after tag decode.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'

import {
  GridDataSchema,
  MAX_INSTANCES,
  QuadDebug,
  type GridVizQuadBuffer,
} from '@/gpu/pipelines/gridVizPipeline'

const WORKGROUP_SIZE = 64

/** CPU/calibration readback — corners, debug, decode fields only (no mat3x3f homography). */
export const HostQuadCorners = d.arrayOf(d.vec2f, 4)

export const HostQuadReadbackGpu = d.struct({
  screenCorners: HostQuadCorners,
  debug: QuadDebug,
  decodedTagId: d.u32,
  decodedRotation: d.u32,
})

export type HostQuadReadback = d.Infer<typeof HostQuadReadbackGpu>

export const HostQuadReadbackSchema = d.arrayOf(HostQuadReadbackGpu, MAX_INSTANCES)

function createHostQuadPackLayouts() {
  const layout = tgpu.bindGroupLayout({
    quadData: { storage: GridDataSchema, access: 'readonly' },
    hostOut: { storage: HostQuadReadbackSchema, access: 'mutable' },
  })
  return { layout }
}

function createHostQuadPackPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createHostQuadPackLayouts>['layout'],
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const quadId = d.u32(input.gid.x)
    if (quadId >= d.u32(MAX_INSTANCES)) {
      return
    }

    const src = layout.$.quadData[quadId]!
    layout.$.hostOut[quadId] = HostQuadReadbackGpu({
      screenCorners: src.screenCorners,
      debug: src.debug,
      decodedTagId: src.decodedTagId,
      decodedRotation: src.decodedRotation,
    })
  })

  return root.createComputePipeline({ compute: kernel })
}

export function createHostQuadReadbackStage(root: TgpuRoot, quadDataBuffer: GridVizQuadBuffer) {
  const hostQuadReadbackBuffer = root.createBuffer(HostQuadReadbackSchema).$usage('storage')
  const layouts = createHostQuadPackLayouts()
  const pipeline = createHostQuadPackPipeline(root, layouts.layout)

  const bindGroup = root.createBindGroup(layouts.layout, {
    quadData: quadDataBuffer,
    hostOut: hostQuadReadbackBuffer,
  })

  const wg = Math.ceil(MAX_INSTANCES / WORKGROUP_SIZE)

  const encodePack = (pass: GPUComputePassEncoder) => {
    pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wg)
  }

  return { hostQuadReadbackBuffer, encodePack }
}

export type HostQuadReadbackBuffer = ReturnType<typeof createHostQuadReadbackStage>['hostQuadReadbackBuffer']
