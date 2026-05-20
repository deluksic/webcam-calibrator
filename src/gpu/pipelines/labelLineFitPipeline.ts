// Per labelId×edgeId (4 slots): accumulate moments → TLS fit → extent → trimmed endpoints.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { atomicAdd, atomicMin, atomicMax, atomicStore, abs, length, max } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import {
  LINE_EXTENT_TRIM_FRAC,
  LINE_INLIER_DIST_PX,
  LINE_MIN_INLIER_RATIO,
  LINE_MIN_PEAK_HIST_COUNT,
  LINE_FIT_PEAKDIR_COS_MIN,
  LINE_MIN_REFINE_INLIERS,
  LINE_MIN_SLOT_COUNT,
  MAX_EDGES_PER_LABEL,
} from '@/gpu/lineFitThresholds'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import { LabelOrientClusterReadonly, type LabelOrientClusterBuffer } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { EdgeLineEntry, EDGE_MIN_SPAN_PX } from '@/gpu/pipelines/edgeLineFitPipeline'
import { lineSegmentEndpoints, lineDirDot, tlsNormalFromMoments } from '@/gpu/shaders/linePca'
import { ORIENT_ASSIGN_MAX_BIN_DIST } from '@/gpu/lineFitThresholds'
import { assignPeakEdgeId } from '@/gpu/shaders/orientPeakAssign'

const WORKGROUP_SIZE = 16
const T_FIXED_SCALE = 16
/** i32 moments: (px*S)^2 must fit in i32; S=4 is safe for tag-sized regions at ~1k px/slot. */
const POS_FIXED_SCALE = 4
const T_MIN_INIT_FIXED = 2_147_483_647
const T_MAX_INIT_FIXED = -2_147_483_647

export { LINE_INLIER_DIST_PX, LINE_MIN_SLOT_COUNT as LINE_MIN_COARSE_COUNT } from '@/gpu/lineFitThresholds'

const LabelLineReduceAtomic = d.struct({
  count: d.atomic(d.u32),
})

const LabelLineReduce = d.struct({
  count: d.u32,
})

const LabelInlierStatsAtomic = d.struct({
  count: d.atomic(d.u32),
  sumXFixed: d.atomic(d.i32),
  sumYFixed: d.atomic(d.i32),
  sumXXFixed: d.atomic(d.i32),
  sumXYFixed: d.atomic(d.i32),
  sumYYFixed: d.atomic(d.i32),
  tMinFixed: d.atomic(d.i32),
  tMaxFixed: d.atomic(d.i32),
})

const LabelInlierStats = d.struct({
  count: d.u32,
  sumXFixed: d.i32,
  sumYFixed: d.i32,
  sumXXFixed: d.i32,
  sumXYFixed: d.i32,
  sumYYFixed: d.i32,
  tMinFixed: d.i32,
  tMaxFixed: d.i32,
})

function createLabelLineFitLayouts() {
  const resetLayout = tgpu.bindGroupLayout({
    labelLineReduce: { storage: d.arrayOf(LabelLineReduceAtomic), access: 'mutable' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
  }).$name('label-line-fit-reset-bgl')
  const accumLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineReduce: { storage: d.arrayOf(LabelLineReduceAtomic), access: 'mutable' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
  }).$name('label-line-fit-accum-bgl')
  const fitLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineReduce: { storage: d.arrayOf(LabelLineReduce), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStats), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
  }).$name('label-line-fit-fit-bgl')
  const extentLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
  }).$name('label-line-fit-extent-bgl')
  const refineResetLayout = tgpu.bindGroupLayout({
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
  }).$name('label-line-fit-refine-reset-bgl')
  const refineAccumLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
  }).$name('label-line-fit-refine-accum-bgl')
  const scatterLayout = tgpu.bindGroupLayout({
    labelInlierStats: { storage: d.arrayOf(LabelInlierStats), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
  }).$name('label-line-fit-scatter-bgl')
  return { resetLayout, accumLayout, fitLayout, extentLayout, refineResetLayout, refineAccumLayout, scatterLayout }
}

export function createLabelLineFitStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxLabels: number,
  filteredBuffer: EdgeFilterBindResources['filteredBuffer'],
  compactLabels: CompactLabelMapBuffer,
  labelClusters: LabelOrientClusterBuffer,
) {
  const maxSlots = maxLabels * MAX_EDGES_PER_LABEL
  const layouts = createLabelLineFitLayouts()
  const labelLineReduce = root.createBuffer(d.arrayOf(LabelLineReduceAtomic, maxSlots)).$usage('storage')
  const labelInlierStats = root.createBuffer(d.arrayOf(LabelInlierStatsAtomic, maxSlots)).$usage('storage')
  const labelLineOut = root.createBuffer(d.arrayOf(EdgeLineEntry, maxSlots)).$usage('storage')

  const resetPipeline = createLabelLineResetPipeline(root, layouts.resetLayout, maxSlots)
  const accumPipeline = createLabelAccumPipeline(root, layouts.accumLayout, width, height)
  const fitPipeline = createLabelFitPipeline(root, layouts.fitLayout, maxSlots)
  const extentPipeline = createLabelExtentPipeline(root, layouts.extentLayout, width, height)
  const refineResetPipeline = createLabelRefineResetPipeline(root, layouts.refineResetLayout, maxSlots)
  const refineAccumPipeline = createLabelRefineAccumPipeline(root, layouts.refineAccumLayout, width, height)
  const extentResetPipeline = createLabelExtentResetPipeline(root, layouts.refineResetLayout, maxSlots)
  const scatterPipeline = createLabelScatterPipeline(root, layouts.scatterLayout, maxSlots)

  const resetBindGroup = root.createBindGroup(layouts.resetLayout, {
    labelLineReduce,
    labelInlierStats,
    labelLineOut,
  })
  const accumBindGroup = root.createBindGroup(layouts.accumLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels,
    labelClusters,
    labelLineReduce,
    labelInlierStats,
  })
  const fitBindGroup = root.createBindGroup(layouts.fitLayout, {
    labelClusters,
    labelLineReduce,
    labelInlierStats,
    labelLineOut,
  })
  const extentBindGroup = root.createBindGroup(layouts.extentLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels,
    labelClusters,
    labelLineOut,
    labelInlierStats,
  })
  const refineResetBindGroup = root.createBindGroup(layouts.refineResetLayout, {
    labelLineOut,
    labelInlierStats,
  })
  const refineAccumBindGroup = root.createBindGroup(layouts.refineAccumLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels,
    labelClusters,
    labelLineOut,
    labelInlierStats,
  })
  const scatterBindGroup = root.createBindGroup(layouts.scatterLayout, {
    labelInlierStats,
    labelLineOut,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const slotWg = Math.ceil(maxSlots / WORKGROUP_SIZE)

  const encodeLabelLineFit = (pass: GPUComputePassEncoder) => {
    resetPipeline.with(pass).with(resetBindGroup).dispatchWorkgroups(slotWg)
    accumPipeline.with(pass).with(accumBindGroup).dispatchWorkgroups(wgX, wgY)
    fitPipeline.with(pass).with(fitBindGroup).dispatchWorkgroups(slotWg)
    refineResetPipeline.with(pass).with(refineResetBindGroup).dispatchWorkgroups(slotWg)
    refineAccumPipeline.with(pass).with(refineAccumBindGroup).dispatchWorkgroups(wgX, wgY)
    fitPipeline.with(pass).with(fitBindGroup).dispatchWorkgroups(slotWg)
    extentResetPipeline.with(pass).with(refineResetBindGroup).dispatchWorkgroups(slotWg)
    extentPipeline.with(pass).with(extentBindGroup).dispatchWorkgroups(wgX, wgY)
    scatterPipeline.with(pass).with(scatterBindGroup).dispatchWorkgroups(slotWg)
  }

  return {
    labelLineOut,
    labelLineReduce,
    labelInlierStats,
    encodeLabelLineFit,
  }
}

export type LabelLineOutBuffer = ReturnType<typeof createLabelLineFitStage>['labelLineOut']
export type LabelLineReduceBuffer = ReturnType<typeof createLabelLineFitStage>['labelLineReduce']
export type LabelInlierStatsBuffer = ReturnType<typeof createLabelLineFitStage>['labelInlierStats']

function createLabelLineResetPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['resetLayout'],
  maxSlots: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const sid = d.u32(input.gid.x)
    if (sid >= d.u32(maxSlots)) {
      return
    }
    atomicStore(layout.$.labelLineReduce[sid]!.count, 0)
    const ir = layout.$.labelInlierStats[sid]!
    atomicStore(ir.count, 0)
    atomicStore(ir.sumXFixed, 0)
    atomicStore(ir.sumYFixed, 0)
    atomicStore(ir.sumXXFixed, 0)
    atomicStore(ir.sumXYFixed, 0)
    atomicStore(ir.sumYYFixed, 0)
    atomicStore(ir.tMinFixed, T_MIN_INIT_FIXED)
    atomicStore(ir.tMaxFixed, T_MAX_INIT_FIXED)
    layout.$.labelLineOut[sid] = EdgeLineEntry({
      sumGx: 0,
      sumGy: 0,
      count: 0,
      inlierCount: 0,
      tMin: 0,
      tMax: 0,
      tSampleMin: 0,
      tSampleMax: 0,
      nDotMean: 0,
      p0x: 0,
      p0y: 0,
      p1x: 0,
      p1y: 0,
      valid: 0,
    })
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelAccumPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['accumLayout'],
  width: number,
  height: number,
) {
  const maxBinDist = d.u32(ORIENT_ASSIGN_MAX_BIN_DIST)
  const minPeakHist = d.u32(LINE_MIN_PEAK_HIST_COUNT)
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
    const g = layout.$.edgeBuffer[idx]!
    const gLen = length(g)
    if (gLen <= d.f32(0)) {
      return
    }

    const labelId = layout.$.compactLabels[idx]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const cluster = layout.$.labelClusters[labelId]!
    if (cluster.peakCount === d.u32(0)) {
      return
    }

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g, maxBinDist)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const peakBin = cluster.peakBins[edgeId]!
    if (peakBin === d.u32(COMPONENT_LABEL_INVALID) || cluster.orientationHistogram[peakBin]! < minPeakHist) {
      return
    }

    const peakDir = cluster.peakDirs[edgeId]!
    const peakLen = length(peakDir)
    if (peakLen <= d.f32(1e-6)) {
      return
    }

    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
    const xFixed = d.i32(px * d.f32(POS_FIXED_SCALE))
    const yFixed = d.i32(py * d.f32(POS_FIXED_SCALE))

    atomicAdd(layout.$.labelLineReduce[slot]!.count, 1)
    const ir = layout.$.labelInlierStats[slot]!
    atomicAdd(ir.sumXFixed, xFixed)
    atomicAdd(ir.sumYFixed, yFixed)
    atomicAdd(ir.sumXXFixed, xFixed * xFixed)
    atomicAdd(ir.sumXYFixed, xFixed * yFixed)
    atomicAdd(ir.sumYYFixed, yFixed * yFixed)
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelFitPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['fitLayout'],
  maxSlots: number,
) {
  const invPos = d.f32(1) / d.f32(POS_FIXED_SCALE)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const sid = d.u32(input.gid.x)
    if (sid >= d.u32(maxSlots)) {
      return
    }

    const labelId = d.u32(sid / d.u32(MAX_EDGES_PER_LABEL))
    const edgeId = d.u32(sid % d.u32(MAX_EDGES_PER_LABEL))
    const cluster = layout.$.labelClusters[labelId]!
    const peakBin = cluster.peakBins[edgeId]!
    const peakDir = cluster.peakDirs[edgeId]!
    const peakLen = length(peakDir)
    const slotCount = layout.$.labelLineReduce[sid]!.count
    const inlier = layout.$.labelInlierStats[sid]!
    const refinedCount = inlier.count
    let fitCount = slotCount
    let minFitCount = d.u32(LINE_MIN_SLOT_COUNT)
    if (refinedCount > d.u32(0)) {
      fitCount = refinedCount
      minFitCount = d.u32(LINE_MIN_REFINE_INLIERS)
    }

    let valid = d.u32(0)
    let sumGx = d.f32(0)
    let sumGy = d.f32(0)
    let nDotMean = d.f32(0)

    if (
      peakBin !== d.u32(COMPONENT_LABEL_INVALID) &&
      peakLen > d.f32(1e-6) &&
      slotCount >= d.u32(LINE_MIN_SLOT_COUNT) &&
      fitCount >= minFitCount
    ) {
      // AprilTag border: peakDir points outward (dark→light away from tag interior).
      // Normal should follow peakDir. TLS refines direction when covariance is well-conditioned.
      const refNx = peakDir.x / peakLen
      const refNy = peakDir.y / peakLen

      const sumX = d.f32(inlier.sumXFixed) * invPos
      const sumY = d.f32(inlier.sumYFixed) * invPos
      const sumXX = d.f32(inlier.sumXXFixed) * invPos * invPos
      const sumXY = d.f32(inlier.sumXYFixed) * invPos * invPos
      const sumYY = d.f32(inlier.sumYYFixed) * invPos * invPos

      const cosMin = d.f32(LINE_FIT_PEAKDIR_COS_MIN)
      const tls = tlsNormalFromMoments(fitCount, sumX, sumY, sumXX, sumXY, sumYY)
      let nx = refNx
      let ny = refNy
      if (tls.ok !== d.u32(0)) {
        const dotTls = tls.nx * refNx + tls.ny * refNy
        if (abs(dotTls) >= cosMin) {
          nx = tls.nx
          ny = tls.ny
          if (dotTls < d.f32(0)) {
            nx = -nx
            ny = -ny
          }
        }
        valid = d.u32(1)
      } else {
        valid = d.u32(1)
      }
      const invN = d.f32(1) / d.f32(fitCount)
      nDotMean = sumX * invN * nx + sumY * invN * ny
      sumGx = nx
      sumGy = ny
    }

    layout.$.labelLineOut[sid] = EdgeLineEntry({
      sumGx,
      sumGy,
      count: slotCount,
      inlierCount: d.u32(0),
      tMin: d.f32(0),
      tMax: d.f32(0),
      tSampleMin: d.f32(0),
      tSampleMax: d.f32(0),
      nDotMean,
      p0x: d.f32(0),
      p0y: d.f32(0),
      p1x: d.f32(0),
      p1y: d.f32(0),
      valid,
    })
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelRefineResetPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['refineResetLayout'],
  maxSlots: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const sid = d.u32(input.gid.x)
    if (sid >= d.u32(maxSlots)) {
      return
    }
    if (layout.$.labelLineOut[sid]!.valid === d.u32(0)) {
      return
    }

    const ir = layout.$.labelInlierStats[sid]!
    atomicStore(ir.count, 0)
    atomicStore(ir.sumXFixed, 0)
    atomicStore(ir.sumYFixed, 0)
    atomicStore(ir.sumXXFixed, 0)
    atomicStore(ir.sumXYFixed, 0)
    atomicStore(ir.sumYYFixed, 0)
    atomicStore(ir.tMinFixed, T_MIN_INIT_FIXED)
    atomicStore(ir.tMaxFixed, T_MAX_INIT_FIXED)
  })
  return root.createComputePipeline({ compute: kernel })
}

/** Clears extent counters before final inlier recount (moments unchanged). */
function createLabelExtentResetPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['refineResetLayout'],
  maxSlots: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const sid = d.u32(input.gid.x)
    if (sid >= d.u32(maxSlots)) {
      return
    }
    if (layout.$.labelLineOut[sid]!.valid === d.u32(0)) {
      return
    }

    const ir = layout.$.labelInlierStats[sid]!
    atomicStore(ir.count, 0)
    atomicStore(ir.tMinFixed, T_MIN_INIT_FIXED)
    atomicStore(ir.tMaxFixed, T_MAX_INIT_FIXED)
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelRefineAccumPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['refineAccumLayout'],
  width: number,
  height: number,
) {
  const maxBinDist = d.u32(ORIENT_ASSIGN_MAX_BIN_DIST)
  const inlierDist = d.f32(LINE_INLIER_DIST_PX)

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
    const g = layout.$.edgeBuffer[idx]!
    if (length(g) <= d.f32(0)) {
      return
    }

    const labelId = layout.$.compactLabels[idx]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const cluster = layout.$.labelClusters[labelId]!
    if (cluster.peakCount === d.u32(0)) {
      return
    }

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g, maxBinDist)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
    const line = layout.$.labelLineOut[slot]!
    if (line.valid === d.u32(0)) {
      return
    }

    const nx = line.sumGx
    const ny = line.sumGy
    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const s = px * nx + py * ny - line.nDotMean
    if (abs(s) >= inlierDist) {
      return
    }

    const xFixed = d.i32(px * d.f32(POS_FIXED_SCALE))
    const yFixed = d.i32(py * d.f32(POS_FIXED_SCALE))
    const ir = layout.$.labelInlierStats[slot]!
    atomicAdd(ir.sumXFixed, xFixed)
    atomicAdd(ir.sumYFixed, yFixed)
    atomicAdd(ir.sumXXFixed, xFixed * xFixed)
    atomicAdd(ir.sumXYFixed, xFixed * yFixed)
    atomicAdd(ir.sumYYFixed, yFixed * yFixed)
    atomicAdd(ir.count, d.u32(1))

    const tAlong = lineDirDot(px, py, nx, ny)
    const tFixed = d.i32(tAlong * d.f32(T_FIXED_SCALE))
    atomicMin(ir.tMinFixed, tFixed)
    atomicMax(ir.tMaxFixed, tFixed)
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelExtentPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['extentLayout'],
  width: number,
  height: number,
) {
  const maxBinDist = d.u32(ORIENT_ASSIGN_MAX_BIN_DIST)
  const inlierDist = d.f32(LINE_INLIER_DIST_PX)

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
    const g = layout.$.edgeBuffer[idx]!
    if (length(g) <= d.f32(0)) {
      return
    }

    const labelId = layout.$.compactLabels[idx]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const cluster = layout.$.labelClusters[labelId]!
    if (cluster.peakCount === d.u32(0)) {
      return
    }

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g, maxBinDist)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
    const line = layout.$.labelLineOut[slot]!
    if (line.valid === d.u32(0)) {
      return
    }

    const nx = line.sumGx
    const ny = line.sumGy
    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const s = px * nx + py * ny - line.nDotMean
    if (abs(s) >= inlierDist) {
      return
    }

    const tAlong = lineDirDot(px, py, nx, ny)
    const ir = layout.$.labelInlierStats[slot]!
    const tFixed = d.i32(tAlong * d.f32(T_FIXED_SCALE))
    atomicAdd(ir.count, d.u32(1))
    atomicMin(ir.tMinFixed, tFixed)
    atomicMax(ir.tMaxFixed, tFixed)
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelScatterPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['scatterLayout'],
  maxSlots: number,
) {
  const invT = d.f32(1) / d.f32(T_FIXED_SCALE)
  const trimFrac = d.f32(LINE_EXTENT_TRIM_FRAC)
  const minRatio = d.f32(LINE_MIN_INLIER_RATIO)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const sid = d.u32(input.gid.x)
    if (sid >= d.u32(maxSlots)) {
      return
    }

    const inlier = layout.$.labelInlierStats[sid]!
    const line = layout.$.labelLineOut[sid]!
    const slotCount = line.count
    const inlierCount = inlier.count

    let valid = line.valid
    let sumGx = line.sumGx
    let sumGy = line.sumGy
    let nDotMean = line.nDotMean
    let tMin = d.f32(0)
    let tMax = d.f32(0)
    let tSampleMin = d.f32(0)
    let tSampleMax = d.f32(0)
    let p0x = d.f32(0)
    let p0y = d.f32(0)
    let p1x = d.f32(0)
    let p1y = d.f32(0)

    if (valid !== d.u32(0) && inlierCount > d.u32(0)) {
      const ratio = d.f32(inlierCount) / d.f32(max(d.u32(1), slotCount))
      if (ratio < minRatio) {
        valid = d.u32(0)
      } else {
        tSampleMin = d.f32(inlier.tMinFixed) * invT
        tSampleMax = d.f32(inlier.tMaxFixed) * invT
        const span = tSampleMax - tSampleMin
        const trim = span * trimFrac
        tSampleMin = tSampleMin + trim
        tSampleMax = tSampleMax - trim
        if (tSampleMax - tSampleMin < d.f32(EDGE_MIN_SPAN_PX)) {
          valid = d.u32(0)
        } else {
          const ends = lineSegmentEndpoints(sumGx, sumGy, nDotMean, tSampleMin, tSampleMax)
          p0x = ends.p0.x
          p0y = ends.p0.y
          p1x = ends.p1.x
          p1y = ends.p1.y
          tMin = d.f32(0)
          tMax = tSampleMax - tSampleMin
        }
      }
    } else {
      valid = d.u32(0)
    }

    layout.$.labelLineOut[sid] = EdgeLineEntry({
      sumGx,
      sumGy,
      count: slotCount,
      inlierCount,
      tMin,
      tMax,
      tSampleMin,
      tSampleMax,
      nDotMean,
      p0x,
      p0y,
      p1x,
      p1y,
      valid,
    })
  })
  return root.createComputePipeline({ compute: kernel })
}
