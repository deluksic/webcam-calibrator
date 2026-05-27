// Pack grid quad fields needed on CPU (no homography) after tag decode.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'

import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { GridDataSchema, QuadDebug, type GridVizQuadBuffer } from '@/gpu/pipelines/gridVizPipeline'

const WORKGROUP_SIZE = 64

/** CPU/calibration readback — corners, debug, decode fields only (no mat3x3f homography). */
export const HostQuadCorners = d.arrayOf(d.vec2f, 4)

export const HostQuadReadbackGpu = d.struct({
  screenCorners: HostQuadCorners,
  debug: QuadDebug,
  decodedTagId: d.i32,
  decodedRotation: d.u32,
  tagKind: d.u32,
})

export type HostQuadReadback = d.Infer<typeof HostQuadReadbackGpu>

export const HostQuadReadbackSchema = d.arrayOf(HostQuadReadbackGpu, MAX_QUADS)

function createHostQuadPackLayouts() {
  const layout = tgpu.bindGroupLayout({
    quadData: { storage: GridDataSchema, access: 'readonly' },
    hostOut: { storage: HostQuadReadbackSchema, access: 'mutable' },
  }).$name('host-quad-readback-bgl')
  return { layout }
}

function createHostQuadPackPipeline(root: TgpuRoot, layout: ReturnType<typeof createHostQuadPackLayouts>['layout']) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const quadId = d.u32(input.gid.x)
    if (quadId >= d.u32(MAX_QUADS)) {
      return
    }

    const src = layout.$.quadData[quadId]!
    layout.$.hostOut[quadId] = HostQuadReadbackGpu({
      screenCorners: src.screenCorners,
      debug: src.debug,
      decodedTagId: src.decodedTagId,
      decodedRotation: src.decodedRotation,
      tagKind: src.tagKind,
    })
  })

  return root.createComputePipeline({ compute: kernel }).$name('host-quad-readback-compute')
}

export function createHostQuadReadbackStage(root: TgpuRoot, quadDataBuffer: GridVizQuadBuffer) {
  const hostQuadReadbackBuffer = root.createBuffer(HostQuadReadbackSchema).$name('host-quad-readback').$usage('storage')
  const layouts = createHostQuadPackLayouts()
  const pipeline = createHostQuadPackPipeline(root, layouts.layout)

  const bindGroup = root.createBindGroup(layouts.layout, {
    quadData: quadDataBuffer,
    hostOut: hostQuadReadbackBuffer,
  })

  const encodePack = (pass: GPUComputePassEncoder, quadCount: number) => {
    const n = Math.max(0, Math.min(quadCount, MAX_QUADS))
    if (n < 1) {
      return
    }
    const wg = Math.ceil(n / WORKGROUP_SIZE)
    pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wg)
  }

  return { hostQuadReadbackBuffer, encodePack }
}

export type HostQuadReadbackBuffer = ReturnType<typeof createHostQuadReadbackStage>['hostQuadReadbackBuffer']
