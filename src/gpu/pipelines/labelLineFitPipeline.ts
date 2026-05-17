import { randf } from '@typegpu/noise'
// Per labelId×edgeId: reservoir → RANSAC hypothesis → inlier moments → TLS fit → extent → labelLineOut.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicAdd, atomicMax, atomicMin, abs, length, min } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import {
  LabelOrientClusterReadonly,
  MAX_EDGES_PER_LABEL,
  type LabelOrientClusterBuffer,
} from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { EdgeLineEntry, EDGE_MIN_SPAN_PX } from '@/gpu/pipelines/edgeLineFitPipeline'
import { lineFromSegment, lineSegmentEndpoints, tlsAgreesWithNormal, tlsNormalFromMoments } from '@/gpu/shaders/linePca'
import { assignPeakEdgeId, gradientPeakAlign, ORIENT_ASSIGN_COS_THRESHOLD } from '@/gpu/shaders/orientPeakAssign'
import { RANSAC_RESERVOIR_CAP, scoreSegmentOnPoints } from '@/gpu/shaders/ransacLine'

const WORKGROUP_SIZE = 16

export const LINE_INLIER_DIST_PX = 4.0
export const LINE_MIN_INLIER_RATIO = 0.8
/** Min edge pixels in a label×edge slot before a line is admitted. */
export const LINE_MIN_COARSE_COUNT = 10
const T_FIXED_SCALE = 16
const POS_FIXED_SCALE = 512
const T_MIN_INIT_FIXED = 2_147_483_647
const T_MAX_INIT_FIXED = -2_147_483_647
const P0_UNSET_X_FIXED = T_MIN_INIT_FIXED
const P1_UNSET_X_FIXED = T_MAX_INIT_FIXED
/** Inlier moments are summed in pixel space relative to RANSAC P0 (no extra fixed scale). */

const RESERVOIR_CAP = RANSAC_RESERVOIR_CAP
const RANSAC_ITERATIONS = 32
const RANSAC_MIN_SCORE_FRAC = 0.4
/** TLS line direction must agree with peak (|dot|); output sign follows peak. */
const TLS_REF_COS_MAX_ANGLE = 0.85

const LabelLineReservoir = d.struct({
  points: d.arrayOf(d.vec2f, RESERVOIR_CAP),
})

/** Slot population (atomic) + RANSAC segment endpoints (plain). */
const LabelLineReduceAtomic = d.struct({
  count: d.atomic(d.u32),
  p0xFixed: d.i32,
  p0yFixed: d.i32,
  p1xFixed: d.i32,
  p1yFixed: d.i32,
})

const LabelLineReduce = d.struct({
  count: d.u32,
  p0xFixed: d.i32,
  p0yFixed: d.i32,
  p1xFixed: d.i32,
  p1yFixed: d.i32,
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
    labelLineReduce: { storage: d.arrayOf(LabelLineReduce), access: 'mutable' },
    labelLineReservoir: { storage: d.arrayOf(LabelLineReservoir), access: 'mutable' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStats), access: 'mutable' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
  })
  const sampleLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineReduce: { storage: d.arrayOf(LabelLineReduceAtomic), access: 'mutable' },
    labelLineReservoir: { storage: d.arrayOf(LabelLineReservoir), access: 'mutable' },
  })
  const ransacLayout = tgpu.bindGroupLayout({
    labelLineReduce: { storage: d.arrayOf(LabelLineReduce), access: 'mutable' },
    labelLineReservoir: { storage: d.arrayOf(LabelLineReservoir), access: 'readonly' },
  })
  const filterLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineReduce: { storage: d.arrayOf(LabelLineReduce), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
  })
  const fitLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineReduce: { storage: d.arrayOf(LabelLineReduce), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStats), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
  })
  const extentLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsAtomic), access: 'mutable' },
  })
  const scatterLayout = tgpu.bindGroupLayout({
    labelInlierStats: { storage: d.arrayOf(LabelInlierStats), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'mutable' },
  })
  return {
    resetLayout,
    sampleLayout,
    ransacLayout,
    filterLayout,
    fitLayout,
    extentLayout,
    scatterLayout,
  }
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
  const labelLineReservoir = root.createBuffer(d.arrayOf(LabelLineReservoir, maxSlots)).$usage('storage')
  const labelInlierStats = root.createBuffer(d.arrayOf(LabelInlierStatsAtomic, maxSlots)).$usage('storage')
  const labelLineOut = root.createBuffer(d.arrayOf(EdgeLineEntry, maxSlots)).$usage('storage')

  const resetPipeline = createLabelLineResetPipeline(root, layouts.resetLayout, maxSlots)
  const samplePipeline = createLabelSamplePipeline(root, layouts.sampleLayout, width, height)
  const ransacPipeline = createLabelRansacPipeline(root, layouts.ransacLayout, maxSlots)
  const filterPipeline = createLabelFilterPipeline(root, layouts.filterLayout, width, height)
  const fitPipeline = createLabelFitPipeline(root, layouts.fitLayout, maxSlots)
  const extentPipeline = createLabelExtentPipeline(root, layouts.extentLayout, width, height)
  const scatterPipeline = createLabelScatterPipeline(root, layouts.scatterLayout, maxSlots)

  const resetBindGroup = root.createBindGroup(layouts.resetLayout, {
    labelLineReduce,
    labelLineReservoir,
    labelInlierStats,
    labelLineOut,
  })
  const sampleBindGroup = root.createBindGroup(layouts.sampleLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels,
    labelClusters,
    labelLineReduce,
    labelLineReservoir,
  })
  const ransacBindGroup = root.createBindGroup(layouts.ransacLayout, {
    labelLineReduce,
    labelLineReservoir,
  })
  const filterBindGroup = root.createBindGroup(layouts.filterLayout, {
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
  const scatterBindGroup = root.createBindGroup(layouts.scatterLayout, {
    labelInlierStats,
    labelLineOut,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const slotWg = Math.ceil(maxSlots / WORKGROUP_SIZE)

  const encodeLabelLineFit = (pass: GPUComputePassEncoder) => {
    resetPipeline.with(pass).with(resetBindGroup).dispatchWorkgroups(slotWg)
    samplePipeline.with(pass).with(sampleBindGroup).dispatchWorkgroups(wgX, wgY)
    ransacPipeline.with(pass).with(ransacBindGroup).dispatchWorkgroups(slotWg)
    filterPipeline.with(pass).with(filterBindGroup).dispatchWorkgroups(wgX, wgY)
    fitPipeline.with(pass).with(fitBindGroup).dispatchWorkgroups(slotWg)
    extentPipeline.with(pass).with(extentBindGroup).dispatchWorkgroups(wgX, wgY)
    scatterPipeline.with(pass).with(scatterBindGroup).dispatchWorkgroups(slotWg)
  }

  return {
    labelLineOut,
    labelLineReduce,
    encodeLabelLineFit,
  }
}

export type LabelLineOutBuffer = ReturnType<typeof createLabelLineFitStage>['labelLineOut']

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
    const lr = layout.$.labelLineReduce[sid]!
    lr.count = d.u32(0)
    lr.p0xFixed = d.i32(P0_UNSET_X_FIXED)
    lr.p0yFixed = d.i32(0)
    lr.p1xFixed = d.i32(P1_UNSET_X_FIXED)
    lr.p1yFixed = d.i32(0)
    const ir = layout.$.labelInlierStats[sid]!
    ir.count = d.u32(0)
    ir.sumXFixed = d.i32(0)
    ir.sumYFixed = d.i32(0)
    ir.sumXXFixed = d.i32(0)
    ir.sumXYFixed = d.i32(0)
    ir.sumYYFixed = d.i32(0)
    ir.tMinFixed = d.i32(T_MIN_INIT_FIXED)
    ir.tMaxFixed = d.i32(T_MAX_INIT_FIXED)
    layout.$.labelLineOut[sid] = EdgeLineEntry({
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
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelSamplePipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['sampleLayout'],
  width: number,
  height: number,
) {
  const cap = d.u32(RESERVOIR_CAP)

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

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const peakBin = cluster.peakBins[edgeId]!
    if (peakBin === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }
    if (gradientPeakAlign(cluster.peakDirs[edgeId]!, g) < d.f32(ORIENT_ASSIGN_COS_THRESHOLD)) {
      return
    }

    const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const p = d.vec2f(px, py)

    const reduce = layout.$.labelLineReduce[slot]!
    const seen = atomicAdd(reduce.count, d.u32(1))
    const reservoir = layout.$.labelLineReservoir[slot]!

    if (seen < cap) {
      reservoir.points[seen] = d.vec2f(p)
    } else {
      randf.seed2(d.vec2f(d.f32(x) * d.f32(1e-3), d.f32(y) * d.f32(1e-3) + d.f32(slot) * d.f32(1e-5)))
      const j = d.u32(randf.sample() * d.f32(RESERVOIR_CAP)) % cap
      reservoir.points[j] = d.vec2f(p)
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelRansacPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['ransacLayout'],
  maxSlots: number,
) {
  const inlierDist = d.f32(LINE_INLIER_DIST_PX)
  const minScoreFrac = d.f32(RANSAC_MIN_SCORE_FRAC)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const sid = d.u32(input.gid.x)
    if (sid >= d.u32(maxSlots)) {
      return
    }

    const reduce = layout.$.labelLineReduce[sid]!
    const totalCount = reduce.count
    const reservoir = layout.$.labelLineReservoir[sid]!
    const sampleCount = min(totalCount, d.u32(RESERVOIR_CAP))

    if (sampleCount < d.u32(2) || totalCount < d.u32(LINE_MIN_COARSE_COUNT)) {
      return
    }

    let bestScore = d.u32(0)
    let bestSpan = d.f32(0)
    let bestP0 = d.vec2f(0, 0)
    let bestP1 = d.vec2f(0, 0)

    for (const iter of std.range(0, RANSAC_ITERATIONS - 1)) {
      randf.seed(d.f32(sid) * d.f32(1e-3) + d.f32(iter) * d.f32(1e-5))
      const n = sampleCount
      let i = d.u32(randf.sample() * d.f32(n))
      let j = d.u32(randf.sample() * d.f32(n))
      if (i === j) {
        j = (j + d.u32(1)) % n
      }

      const P0 = reservoir.points[i]!
      const P1 = reservoir.points[j]!
      const geom = lineFromSegment(P0, P1)
      if (geom.ok === d.u32(0)) {
        continue
      }

      const score = scoreSegmentOnPoints(sampleCount, reservoir.points, P0, geom.n, inlierDist)

      const better = score > bestScore || (score === bestScore && geom.span > bestSpan)
      if (better) {
        bestScore = score
        bestSpan = geom.span
        bestP0 = d.vec2f(P0)
        bestP1 = d.vec2f(P1)
      }
    }

    // bestScore is counted on the reservoir (≤ RESERVOIR_CAP), not the full slot population.
    const minScore = d.u32(d.f32(sampleCount) * minScoreFrac)
    if (bestScore >= minScore) {
      reduce.p0xFixed = d.i32(bestP0.x * d.f32(POS_FIXED_SCALE))
      reduce.p0yFixed = d.i32(bestP0.y * d.f32(POS_FIXED_SCALE))
      reduce.p1xFixed = d.i32(bestP1.x * d.f32(POS_FIXED_SCALE))
      reduce.p1yFixed = d.i32(bestP1.y * d.f32(POS_FIXED_SCALE))
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelFilterPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['filterLayout'],
  width: number,
  height: number,
) {
  const invPos = d.f32(1) / d.f32(POS_FIXED_SCALE)

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

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const peakBin = cluster.peakBins[edgeId]!
    if (peakBin === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }
    if (gradientPeakAlign(cluster.peakDirs[edgeId]!, g) < d.f32(ORIENT_ASSIGN_COS_THRESHOLD)) {
      return
    }

    const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
    const coarse = layout.$.labelLineReduce[slot]!
    if (coarse.count === d.u32(0) || coarse.p0xFixed === d.i32(P0_UNSET_X_FIXED)) {
      return
    }

    const P0 = d.vec2f(d.f32(coarse.p0xFixed) * invPos, d.f32(coarse.p0yFixed) * invPos)
    const P1 = d.vec2f(d.f32(coarse.p1xFixed) * invPos, d.f32(coarse.p1yFixed) * invPos)
    const geom = lineFromSegment(P0, P1)
    if (geom.ok === d.u32(0)) {
      return
    }

    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const relX = px - P0.x
    const relY = py - P0.y
    const s = relX * geom.n.x + relY * geom.n.y
    if (abs(s) >= LINE_INLIER_DIST_PX) {
      return
    }

    const ir = layout.$.labelInlierStats[slot]!
    const relXi = d.i32(relX)
    const relYi = d.i32(relY)
    atomicAdd(ir.count, d.u32(1))
    atomicAdd(ir.sumXFixed, relXi)
    atomicAdd(ir.sumYFixed, relYi)
    atomicAdd(ir.sumXXFixed, relXi * relXi)
    atomicAdd(ir.sumXYFixed, relXi * relYi)
    atomicAdd(ir.sumYYFixed, relYi * relYi)
  })
  return root.createComputePipeline({ compute: kernel })
}

function createLabelFitPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['fitLayout'],
  maxSlots: number,
) {
  const invPos = d.f32(1) / d.f32(POS_FIXED_SCALE)
  const cosRef = d.f32(TLS_REF_COS_MAX_ANGLE)

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
    const coarse = layout.$.labelLineReduce[sid]!
    const coarseCount = coarse.count
    const inlier = layout.$.labelInlierStats[sid]!
    const inlierCount = inlier.count

    let valid = d.u32(0)
    let sumGx = d.f32(0)
    let sumGy = d.f32(0)
    let nDotMean = d.f32(0)

    if (
      peakBin !== d.u32(COMPONENT_LABEL_INVALID) &&
      peakLen > d.f32(1e-6) &&
      coarseCount > d.u32(0) &&
      coarse.p0xFixed !== d.i32(P0_UNSET_X_FIXED) &&
      inlierCount >= d.u32(2)
    ) {
      const P0 = d.vec2f(d.f32(coarse.p0xFixed) * invPos, d.f32(coarse.p0yFixed) * invPos)
      const P1 = d.vec2f(d.f32(coarse.p1xFixed) * invPos, d.f32(coarse.p1yFixed) * invPos)
      const ransacSeg = lineFromSegment(P0, P1)
      const refNx = peakDir.x / peakLen
      const refNy = peakDir.y / peakLen

      const sumX = d.f32(inlier.sumXFixed)
      const sumY = d.f32(inlier.sumYFixed)
      const sumXX = d.f32(inlier.sumXXFixed)
      const sumXY = d.f32(inlier.sumXYFixed)
      const sumYY = d.f32(inlier.sumYYFixed)

      const tls = tlsNormalFromMoments(inlierCount, sumX, sumY, sumXX, sumXY, sumYY)
      if (
        tls.ok !== d.u32(0) &&
        ransacSeg.ok !== d.u32(0) &&
        tlsAgreesWithNormal(tls.nx, tls.ny, refNx, refNy, cosRef) !== d.u32(0)
      ) {
        let nx = tls.nx
        let ny = tls.ny
        if (nx * refNx + ny * refNy < d.f32(0)) {
          nx = -nx
          ny = -ny
        }

        const invN = d.f32(1) / d.f32(inlierCount)
        const cxRel = sumX * invN
        const cyRel = sumY * invN
        nDotMean = (P0.x + cxRel) * nx + (P0.y + cyRel) * ny
        sumGx = nx
        sumGy = ny

        const inlierRatio = d.f32(inlierCount) / d.f32(coarseCount)
        if (
          coarseCount >= d.u32(LINE_MIN_COARSE_COUNT) &&
          ransacSeg.span >= d.f32(EDGE_MIN_SPAN_PX) &&
          inlierRatio >= d.f32(LINE_MIN_INLIER_RATIO)
        ) {
          valid = d.u32(1)
        }
      }
    }

    layout.$.labelLineOut[sid] = EdgeLineEntry({
      sumGx,
      sumGy,
      count: coarseCount,
      inlierCount,
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

function createLabelExtentPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createLabelLineFitLayouts>['extentLayout'],
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

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const peakBin = cluster.peakBins[edgeId]!
    if (peakBin === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }
    if (gradientPeakAlign(cluster.peakDirs[edgeId]!, g) < d.f32(ORIENT_ASSIGN_COS_THRESHOLD)) {
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
    if (abs(s) >= LINE_INLIER_DIST_PX) {
      return
    }

    const tAlong = px * ny - py * nx
    const ir = layout.$.labelInlierStats[slot]!
    const tFixed = d.i32(tAlong * d.f32(T_FIXED_SCALE))
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

    let valid = d.u32(0)
    let sumGx = d.f32(0)
    let sumGy = d.f32(0)
    let count = d.u32(0)
    let inlierCount = d.u32(0)
    let tMin = d.f32(0)
    let tMax = d.f32(0)
    let tSampleMin = d.f32(0)
    let tSampleMax = d.f32(0)
    let nDotMean = d.f32(0)
    let p0x = d.f32(0)
    let p0y = d.f32(0)
    let p1x = d.f32(0)
    let p1y = d.f32(0)

    const line = layout.$.labelLineOut[sid]!
    valid = line.valid
    sumGx = line.sumGx
    sumGy = line.sumGy
    nDotMean = line.nDotMean
    count = line.count
    inlierCount = line.inlierCount

    if (valid !== d.u32(0) && inlierCount > d.u32(0)) {
      tSampleMin = d.f32(inlier.tMinFixed) * invT
      tSampleMax = d.f32(inlier.tMaxFixed) * invT
      tMin = d.f32(0)
      tMax = tSampleMax - tSampleMin
      const ends = lineSegmentEndpoints(sumGx, sumGy, nDotMean, tSampleMin, tSampleMax)
      p0x = ends.p0.x
      p0y = ends.p0.y
      p1x = ends.p1.x
      p1y = ends.p1.y
    }

    layout.$.labelLineOut[sid] = EdgeLineEntry({
      sumGx,
      sumGy,
      count,
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
