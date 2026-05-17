// Per-label line fit: sum gradients, extreme tangent projections, finalize span ≥ 6px.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { atomicAdd, atomicLoad, atomicMax, atomicMin, atomicStore, length, sqrt } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'

const WORKGROUP_SIZE = 16

export const EDGE_MIN_SPAN_PX = 6
export const PROFILE_BUCKET_COUNT = 64
export const PROFILE_NEIGHBORHOOD_HALF = 2.5

/** Fixed-point scales for atomic i32 reductions (WGSL has no atomic f32). */
const GRAD_FIXED_SCALE = 4096
const T_FIXED_SCALE = 16
const NDOT_FIXED_SCALE = 4096
/** Initial tMin (atomicMin): larger than any plausible fixed tangent. */
const T_MIN_INIT_FIXED = 2_147_483_647
/** Initial tMax (atomicMax): smaller than any plausible fixed tangent (not i32::MIN — invalid WGSL literal). */
const T_MAX_INIT_FIXED = -2_147_483_647

export const EdgeLineEntry = d.struct({
  sumGx: d.f32,
  sumGy: d.f32,
  count: d.u32,
  tMin: d.f32,
  tMax: d.f32,
  tSampleMin: d.f32,
  tSampleMax: d.f32,
  /** Mean dot(p, normal) over edge pixels — reference for signed normal distance. */
  nDotMean: d.f32,
  valid: d.u32,
})

const LineReduceAtomic = d.struct({
  sumGxFixed: d.atomic(d.i32),
  sumGyFixed: d.atomic(d.i32),
  count: d.atomic(d.u32),
  tMinFixed: d.atomic(d.i32),
  tMaxFixed: d.atomic(d.i32),
  sumNDotFixed: d.atomic(d.i32),
})

function createLineFitLayouts() {
  const resetLayout = tgpu.bindGroupLayout({
    lineReduce: { storage: d.arrayOf(LineReduceAtomic), access: 'mutable' },
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
    validEdgeCount: { storage: d.atomic(d.u32), access: 'mutable' },
  })
  const reduceGradLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    lineReduce: { storage: d.arrayOf(LineReduceAtomic), access: 'mutable' },
  })
  const extremesLayout = tgpu.bindGroupLayout({
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    lineReduce: { storage: d.arrayOf(LineReduceAtomic), access: 'mutable' },
  })
  const finalizeLayout = tgpu.bindGroupLayout({
    lineReduce: { storage: d.arrayOf(LineReduceAtomic), access: 'mutable' },
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
    validEdgeCount: { storage: d.atomic(d.u32), access: 'mutable' },
  })
  return { resetLayout, reduceGradLayout, extremesLayout, finalizeLayout }
}

export function createEdgeLineFitStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxComponents: number,
  filteredBuffer: EdgeFilterBindResources['filteredBuffer'],
  compactLabels: CompactLabelMapBuffer,
) {
  const layouts = createLineFitLayouts()
  const lineReduce = root.createBuffer(d.arrayOf(LineReduceAtomic, maxComponents)).$usage('storage')
  const lineOut = root.createBuffer(d.arrayOf(EdgeLineEntry, maxComponents)).$usage('storage')
  const validEdgeCount = root.createBuffer(d.atomic(d.u32)).$usage('storage')

  const resetPipeline = createLineResetPipeline(root, layouts.resetLayout, maxComponents)
  const reduceGradPipeline = createLineReduceGradPipeline(root, layouts.reduceGradLayout, width, height)
  const extremesPipeline = createLineExtremesPipeline(root, layouts.extremesLayout, width, height)
  const finalizePipeline = createLineFinalizePipeline(root, layouts.finalizeLayout, maxComponents)

  const resetBindGroup = root.createBindGroup(layouts.resetLayout, {
    lineReduce,
    lineOut,
    validEdgeCount,
  })
  const reduceGradBindGroup = root.createBindGroup(layouts.reduceGradLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels,
    lineReduce,
  })
  const extremesBindGroup = root.createBindGroup(layouts.extremesLayout, {
    compactLabels,
    lineReduce,
  })
  const finalizeBindGroup = root.createBindGroup(layouts.finalizeLayout, {
    lineReduce,
    lineOut,
    validEdgeCount,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const labelWg = Math.ceil(maxComponents / WORKGROUP_SIZE)

  const encodeCompute = (pass: GPUComputePassEncoder) => {
    resetPipeline.with(pass).with(resetBindGroup).dispatchWorkgroups(labelWg)
    reduceGradPipeline.with(pass).with(reduceGradBindGroup).dispatchWorkgroups(wgX, wgY)
    extremesPipeline.with(pass).with(extremesBindGroup).dispatchWorkgroups(wgX, wgY)
    finalizePipeline.with(pass).with(finalizeBindGroup).dispatchWorkgroups(labelWg)
  }

  return {
    lineOut,
    lineReduce,
    validEdgeCount,
    encodeCompute,
  }
}

function createLineResetPipeline(
  root: TgpuRoot,
  resetLayout: ReturnType<typeof createLineFitLayouts>['resetLayout'],
  maxComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const lid = d.u32(input.gid.x)
    if (lid >= d.u32(maxComponents)) {
      return
    }
    atomicStore(resetLayout.$.lineReduce[lid]!.sumGxFixed, d.i32(0))
    atomicStore(resetLayout.$.lineReduce[lid]!.sumGyFixed, d.i32(0))
    atomicStore(resetLayout.$.lineReduce[lid]!.count, d.u32(0))
    atomicStore(resetLayout.$.lineReduce[lid]!.tMinFixed, d.i32(T_MIN_INIT_FIXED))
    atomicStore(resetLayout.$.lineReduce[lid]!.tMaxFixed, d.i32(T_MAX_INIT_FIXED))
    atomicStore(resetLayout.$.lineReduce[lid]!.sumNDotFixed, d.i32(0))
    resetLayout.$.lineOut[lid] = EdgeLineEntry({
      sumGx: d.f32(0),
      sumGy: d.f32(0),
      count: d.u32(0),
      tMin: d.f32(0),
      tMax: d.f32(0),
      tSampleMin: d.f32(0),
      tSampleMax: d.f32(0),
      nDotMean: d.f32(0),
      valid: d.u32(0),
    })
    if (lid === d.u32(0)) {
      atomicStore(resetLayout.$.validEdgeCount, d.u32(0))
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLineReduceGradPipeline(
  root: TgpuRoot,
  reduceGradLayout: ReturnType<typeof createLineFitLayouts>['reduceGradLayout'],
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const w = d.i32(width)
    const h = d.i32(height)
    if (x >= w || y >= h) {
      return
    }

    const idx = d.u32(y * w + x)
    const g = reduceGradLayout.$.edgeBuffer[idx]!
    if (length(g) <= d.f32(0)) {
      return
    }

    const label = reduceGradLayout.$.compactLabels[idx]!
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const slot = reduceGradLayout.$.lineReduce[label]!
    atomicAdd(slot.sumGxFixed, d.i32(g.x * d.f32(GRAD_FIXED_SCALE)))
    atomicAdd(slot.sumGyFixed, d.i32(g.y * d.f32(GRAD_FIXED_SCALE)))
    atomicAdd(slot.count, d.u32(1))
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLineExtremesPipeline(
  root: TgpuRoot,
  extremesLayout: ReturnType<typeof createLineFitLayouts>['extremesLayout'],
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const w = d.i32(width)
    const h = d.i32(height)
    if (x >= w || y >= h) {
      return
    }

    const idx = d.u32(y * w + x)
    const label = extremesLayout.$.compactLabels[idx]!
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const reduce = extremesLayout.$.lineReduce[label]!
    const count = atomicLoad(reduce.count)
    if (count === d.u32(0)) {
      return
    }

    const sumGx = d.f32(atomicLoad(reduce.sumGxFixed)) / d.f32(GRAD_FIXED_SCALE)
    const sumGy = d.f32(atomicLoad(reduce.sumGyFixed)) / d.f32(GRAD_FIXED_SCALE)
    const gLen = sqrt(sumGx * sumGx + sumGy * sumGy)
    if (gLen < d.f32(1e-8)) {
      return
    }

    const nx = sumGx / gLen
    const ny = sumGy / gLen
    const tx = -ny
    const ty = nx
    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const t = px * tx + py * ty
    const nDot = px * nx + py * ny
    const tFixed = d.i32(t * d.f32(T_FIXED_SCALE))
    const nDotFixed = d.i32(nDot * d.f32(NDOT_FIXED_SCALE))

    atomicMin(reduce.tMinFixed, tFixed)
    atomicMax(reduce.tMaxFixed, tFixed)
    atomicAdd(reduce.sumNDotFixed, nDotFixed)
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLineFinalizePipeline(
  root: TgpuRoot,
  finalizeLayout: ReturnType<typeof createLineFitLayouts>['finalizeLayout'],
  maxComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const lid = d.u32(input.gid.x)
    if (lid >= d.u32(maxComponents)) {
      return
    }

    const reduce = finalizeLayout.$.lineReduce[lid]!
    const count = atomicLoad(reduce.count)
    const sumGx = d.f32(atomicLoad(reduce.sumGxFixed)) / d.f32(GRAD_FIXED_SCALE)
    const sumGy = d.f32(atomicLoad(reduce.sumGyFixed)) / d.f32(GRAD_FIXED_SCALE)
    const tMin = d.f32(atomicLoad(reduce.tMinFixed)) / d.f32(T_FIXED_SCALE)
    const tMax = d.f32(atomicLoad(reduce.tMaxFixed)) / d.f32(T_FIXED_SCALE)
    const sumNDot = d.f32(atomicLoad(reduce.sumNDotFixed)) / d.f32(NDOT_FIXED_SCALE)
    const span = tMax - tMin
    const invCount = d.f32(1) / d.f32(count)
    const nDotMean = sumNDot * invCount

    let valid = d.u32(0)
    let tSampleMin = d.f32(0)
    let tSampleMax = d.f32(0)
    if (count > d.u32(0) && span >= d.f32(EDGE_MIN_SPAN_PX)) {
      valid = d.u32(1)
      tSampleMin = tMin
      tSampleMax = tMax
      atomicAdd(finalizeLayout.$.validEdgeCount, d.u32(1))
    }

    finalizeLayout.$.lineOut[lid] = EdgeLineEntry({
      sumGx,
      sumGy,
      count,
      tMin,
      tMax,
      tSampleMin,
      tSampleMax,
      nDotMean,
      valid,
    })
  })
  return root.createComputePipeline({ compute: kernel })
}

export type EdgeLineOutBuffer = ReturnType<typeof createEdgeLineFitStage>['lineOut']
