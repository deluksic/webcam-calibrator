// Scatter label-level line fits to flat quad×edge slots for profile/overlay.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { atomicAdd, atomicStore } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import {
  MAX_EDGES_PER_LABEL,
  type QuadCountBuffer,
  type QuadPeakEdgeBuffer,
  type QuadSourceLabelIdBuffer,
} from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import type { LabelLineOutBuffer } from '@/gpu/pipelines/labelLineFitPipeline'

const WORKGROUP_SIZE = 16

export const EDGE_MIN_SPAN_PX = 8
export const PROFILE_BUCKET_COUNT = 64
export const PROFILE_NEIGHBORHOOD_HALF = 2.5

export const EdgeLineEntry = d.struct({
  sumGx: d.f32,
  sumGy: d.f32,
  count: d.u32,
  /** Inlier pixels after coarse segment filter (used to rank peaks for quads). */
  inlierCount: d.u32,
  tMin: d.f32,
  tMax: d.f32,
  tSampleMin: d.f32,
  tSampleMax: d.f32,
  /** Mean dot(p, normal) over edge pixels — reference for signed normal distance. */
  nDotMean: d.f32,
  p0x: d.f32,
  p0y: d.f32,
  p1x: d.f32,
  p1y: d.f32,
  valid: d.u32,
})

function createScatterLayouts() {
  const resetLayout = tgpu.bindGroupLayout({
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
    validEdgeCount: { storage: d.atomic(d.u32), access: 'mutable' },
  })
  const scatterLayout = tgpu.bindGroupLayout({
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    quadPeakEdge: { storage: d.arrayOf(d.u32), access: 'readonly' },
    quadSourceLabelId: { storage: d.arrayOf(d.u32), access: 'readonly' },
    quadCount: { storage: d.arrayOf(d.u32, 1), access: 'readonly' },
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
    validEdgeCount: { storage: d.atomic(d.u32), access: 'mutable' },
  })
  return { resetLayout, scatterLayout }
}

export function createEdgeLineFitStage(
  root: TgpuRoot,
  maxFlatEdges: number,
  labelLineOut: LabelLineOutBuffer,
  quadPeakEdge: QuadPeakEdgeBuffer,
  quadSourceLabelId: QuadSourceLabelIdBuffer,
  quadCount: QuadCountBuffer,
) {
  const layouts = createScatterLayouts()
  const lineOut = root.createBuffer(d.arrayOf(EdgeLineEntry, maxFlatEdges)).$usage('storage')
  const validEdgeCount = root.createBuffer(d.atomic(d.u32)).$usage('storage')

  const resetPipeline = createScatterResetPipeline(root, layouts.resetLayout, maxFlatEdges)
  const scatterPipeline = createScatterQuadLinesPipeline(root, layouts.scatterLayout, maxFlatEdges)

  const resetBindGroup = root.createBindGroup(layouts.resetLayout, {
    lineOut,
    validEdgeCount,
  })
  const scatterBindGroup = root.createBindGroup(layouts.scatterLayout, {
    labelLineOut,
    quadPeakEdge,
    quadSourceLabelId,
    quadCount,
    lineOut,
    validEdgeCount,
  })

  const flatWg = Math.ceil(maxFlatEdges / WORKGROUP_SIZE)

  const encodeCompute = (pass: GPUComputePassEncoder) => {
    resetPipeline.with(pass).with(resetBindGroup).dispatchWorkgroups(flatWg)
    scatterPipeline.with(pass).with(scatterBindGroup).dispatchWorkgroups(flatWg)
  }

  return {
    lineOut,
    validEdgeCount,
    encodeCompute,
  }
}

function createScatterResetPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createScatterLayouts>['resetLayout'],
  maxFlatEdges: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const fid = d.u32(input.gid.x)
    if (fid >= d.u32(maxFlatEdges)) {
      return
    }
    layout.$.lineOut[fid] = EdgeLineEntry({
      sumGx: d.f32(0),
      sumGy: d.f32(0),
      count: d.u32(0),
      inlierCount: d.u32(0),
      tMin: d.f32(0),
      tMax: d.f32(0),
      tSampleMin: d.f32(0),
      tSampleMax: d.f32(0),
      nDotMean: d.f32(0),
      p0x: d.f32(0),
      p0y: d.f32(0),
      p1x: d.f32(0),
      p1y: d.f32(0),
      valid: d.u32(0),
    })
    if (fid === d.u32(0)) {
      atomicStore(layout.$.validEdgeCount, d.u32(0))
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

function createScatterQuadLinesPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createScatterLayouts>['scatterLayout'],
  maxFlatEdges: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const flatSlot = d.u32(input.gid.x)
    if (flatSlot >= d.u32(maxFlatEdges)) {
      return
    }

    const quadId = d.u32(flatSlot / d.u32(MAX_EDGES_PER_LABEL))
    const edgeId = d.u32(flatSlot % d.u32(MAX_EDGES_PER_LABEL))
    const nQuads = layout.$.quadCount[d.u32(0)]!

    if (quadId >= nQuads) {
      return
    }

    const labelId = layout.$.quadSourceLabelId[quadId]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const quadBase = quadId * d.u32(MAX_EDGES_PER_LABEL)
    const peakEdge = layout.$.quadPeakEdge[quadBase + edgeId]!
    if (peakEdge === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const labelSlot = labelId * d.u32(MAX_EDGES_PER_LABEL) + peakEdge
    const src = layout.$.labelLineOut[labelSlot]!
    layout.$.lineOut[flatSlot] = EdgeLineEntry({
      sumGx: src.sumGx,
      sumGy: src.sumGy,
      count: src.count,
      inlierCount: src.inlierCount,
      tMin: src.tMin,
      tMax: src.tMax,
      tSampleMin: src.tSampleMin,
      tSampleMax: src.tSampleMax,
      nDotMean: src.nDotMean,
      p0x: src.p0x,
      p0y: src.p0y,
      p1x: src.p1x,
      p1y: src.p1y,
      valid: src.valid,
    })

    if (src.valid !== d.u32(0)) {
      atomicAdd(layout.$.validEdgeCount, d.u32(1))
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

export type EdgeLineOutBuffer = ReturnType<typeof createEdgeLineFitStage>['lineOut']
