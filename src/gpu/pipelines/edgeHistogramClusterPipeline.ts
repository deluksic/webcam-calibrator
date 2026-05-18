// Per compact labelId: 64-bin oriented histogram → 4 peaks → per-pixel edgeId.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicAdd, atomicLoad, atomicStore, length } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import {
  MAX_EDGES_PER_LABEL,
  MIN_PEAK_BIN_SEPARATION,
  MIN_QUAD_EDGE_INLIERS,
  MIN_QUAD_VALID_EDGES,
  ORIENT_PEAK_MIN_COUNT,
} from '@/gpu/lineFitThresholds'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import { createLabelLineFitStage } from '@/gpu/pipelines/labelLineFitPipeline'
import {
  assignPeakEdgeId,
  circularBinDist,
  gradientOrientationBin,
  ORIENT_ASSIGN_MAX_BIN_DIST,
  ORIENT_HIST_BINS,
  peakDirFromLocalBins,
} from '@/gpu/shaders/orientPeakAssign'

const WORKGROUP_SIZE = 16

export { MAX_EDGES_PER_LABEL, ORIENT_HIST_BINS, ORIENT_ASSIGN_MAX_BIN_DIST } from '@/gpu/shaders/orientPeakAssign'
export {
  MIN_PEAK_BIN_SEPARATION,
  MIN_QUAD_EDGE_INLIERS,
  MIN_QUAD_VALID_EDGES,
  ORIENT_PEAK_MIN_COUNT,
} from '@/gpu/lineFitThresholds'

export const MAX_QUADS = 2 << 9
export const MAX_FLAT_EDGES = MAX_QUADS * MAX_EDGES_PER_LABEL

export const LabelOrientCluster = d.struct({
  orientationHistogram: d.arrayOf(d.atomic(d.u32), ORIENT_HIST_BINS),
  peakBins: d.arrayOf(d.u32, MAX_EDGES_PER_LABEL),
  peakDirs: d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL),
  peakCount: d.u32,
})

export const LabelOrientClusterReadonly = d.struct({
  orientationHistogram: d.arrayOf(d.u32, ORIENT_HIST_BINS),
  peakBins: d.arrayOf(d.u32, MAX_EDGES_PER_LABEL),
  peakDirs: d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL),
  peakCount: d.u32,
})

function createEdgeHistogramClusterLayouts() {
  const histResetLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientCluster), access: 'mutable' },
    quadCount: { storage: d.arrayOf(d.atomic(d.u32), 1), access: 'mutable' },
  })
  const histAccumLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientCluster), access: 'mutable' },
  })
  const findPeaksLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientCluster), access: 'mutable' },
  })
  const compactQuadsLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    quadPeakEdge: { storage: d.arrayOf(d.u32), access: 'mutable' },
    labelToQuadId: { storage: d.arrayOf(d.u32), access: 'mutable' },
    quadSourceLabelId: { storage: d.arrayOf(d.u32), access: 'mutable' },
    quadCount: { storage: d.arrayOf(d.atomic(d.u32), 1), access: 'mutable' },
  })
  const writeQuadLabelMapLayout = tgpu.bindGroupLayout({
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelToQuadId: { storage: d.arrayOf(d.u32), access: 'readonly' },
    quadLabelBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  })
  const assignEdgesLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelToQuadId: { storage: d.arrayOf(d.u32), access: 'readonly' },
    packedEdgeLabels: { storage: d.arrayOf(d.u32), access: 'mutable' },
  })
  return {
    histResetLayout,
    histAccumLayout,
    findPeaksLayout,
    compactQuadsLayout,
    writeQuadLabelMapLayout,
    assignEdgesLayout,
  }
}

function createHistResetPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['histResetLayout'],
  maxComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const labelId = d.u32(input.gid.x)
    if (labelId >= d.u32(maxComponents)) {
      return
    }

    const slot = layout.$.labelClusters[labelId]!
    for (const b of tgpu.unroll(std.range(0, ORIENT_HIST_BINS))) {
      atomicStore(slot.orientationHistogram[d.u32(b)]!, d.u32(0))
    }
    for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      slot.peakBins[d.u32(k)] = d.u32(COMPONENT_LABEL_INVALID)
      slot.peakDirs[d.u32(k)] = d.vec2f(0, 0)
    }
    slot.peakCount = d.u32(0)
    if (labelId === d.u32(0)) {
      atomicStore(layout.$.quadCount[d.u32(0)]!, d.u32(0))
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

function createHistAccumPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['histAccumLayout'],
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

    const bin = gradientOrientationBin(g)
    atomicAdd(layout.$.labelClusters[labelId]!.orientationHistogram[bin]!, d.u32(1))
  })
  return root.createComputePipeline({ compute: kernel })
}

function createFindPeaksPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['findPeaksLayout'],
  maxComponents: number,
) {
  const minPeakCount = d.u32(ORIENT_PEAK_MIN_COUNT)
  const minPeakSep = d.u32(MIN_PEAK_BIN_SEPARATION)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const labelId = d.u32(input.gid.x)
    if (labelId >= d.u32(maxComponents)) {
      return
    }

    const slot = layout.$.labelClusters[labelId]!
    slot.peakCount = d.u32(0)

    for (const _ of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      let bestBin = d.u32(COMPONENT_LABEL_INVALID)
      let bestCount = d.u32(0)
      const peakCount = slot.peakCount

      for (const b of tgpu.unroll(std.range(0, ORIENT_HIST_BINS))) {
        const bi = d.u32(b)
        const binCount = atomicLoad(slot.orientationHistogram[bi]!)
        if (binCount >= minPeakCount) {
          const prevB = (bi + d.u32(ORIENT_HIST_BINS - 1)) % d.u32(ORIENT_HIST_BINS)
          const nextB = (bi + d.u32(1)) % d.u32(ORIENT_HIST_BINS)
          const prevCount = atomicLoad(slot.orientationHistogram[prevB]!)
          const nextCount = atomicLoad(slot.orientationHistogram[nextB]!)
          if (binCount >= prevCount && binCount >= nextCount) {
            let tooClose = false
            for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
              if (k < peakCount) {
                const pk = slot.peakBins[d.u32(k)]!
                if (pk !== d.u32(COMPONENT_LABEL_INVALID) && circularBinDist(bi, pk) < minPeakSep) {
                  tooClose = true
                }
              }
            }
            if (!tooClose && binCount > bestCount) {
              bestCount = binCount
              bestBin = bi
            }
          }
        }
      }

      if (bestBin !== d.u32(COMPONENT_LABEL_INVALID)) {
        slot.peakBins[peakCount] = bestBin
        slot.peakCount = peakCount + d.u32(1)
      }
    }

    const peakCount = slot.peakCount
    for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (k < peakCount) {
        const peakBin = slot.peakBins[d.u32(k)]!
        if (peakBin !== d.u32(COMPONENT_LABEL_INVALID)) {
          const prevB = (peakBin + d.u32(ORIENT_HIST_BINS - 1)) % d.u32(ORIENT_HIST_BINS)
          const nextB = (peakBin + d.u32(1)) % d.u32(ORIENT_HIST_BINS)
          const wPrev = d.f32(atomicLoad(slot.orientationHistogram[prevB]!))
          const wCenter = d.f32(atomicLoad(slot.orientationHistogram[peakBin]!))
          const wNext = d.f32(atomicLoad(slot.orientationHistogram[nextB]!))
          slot.peakDirs[d.u32(k)] = peakDirFromLocalBins(peakBin, wPrev, wCenter, wNext)
        }
      }
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

function createCompactQuadsPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['compactQuadsLayout'],
  maxComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const labelId = d.u32(input.gid.x)
    if (labelId >= d.u32(maxComponents)) {
      return
    }

    layout.$.labelToQuadId[labelId] = d.u32(COMPONENT_LABEL_INVALID)

    const cluster = layout.$.labelClusters[labelId]!
    const peakCount = cluster.peakCount
    let validSides = d.u32(0)

    for (const q of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (q < peakCount) {
        const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + q
        const line = layout.$.labelLineOut[slot]!
        if (line.valid !== d.u32(0) && line.inlierCount >= d.u32(MIN_QUAD_EDGE_INLIERS)) {
          validSides = validSides + d.u32(1)
        }
      }
    }

    if (peakCount !== d.u32(MAX_EDGES_PER_LABEL) || validSides !== d.u32(MIN_QUAD_VALID_EDGES)) {
      return
    }

    // Reject two parallel pairs (four peaks but only two orientations).
    const minPeakSep = d.u32(MIN_PEAK_BIN_SEPARATION)
    for (const ki of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (ki < peakCount) {
        const bi = cluster.peakBins[ki]!
        let farOthers = d.u32(0)
        for (const kj of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
          if (kj < peakCount && kj !== ki) {
            if (circularBinDist(bi, cluster.peakBins[kj]!) >= minPeakSep) {
              farOthers = farOthers + d.u32(1)
            }
          }
        }
        if (farOthers < d.u32(3)) {
          return
        }
      }
    }

    const quadId = atomicAdd(layout.$.quadCount[d.u32(0)]!, d.u32(1))
    if (quadId >= MAX_QUADS) {
      return
    }

    const quadBase = quadId * d.u32(MAX_EDGES_PER_LABEL)
    for (const q of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      layout.$.quadPeakEdge[quadBase + d.u32(q)] = d.u32(COMPONENT_LABEL_INVALID)
    }
    for (const q of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (q < peakCount) {
        const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + q
        const line = layout.$.labelLineOut[slot]!
        if (line.valid !== d.u32(0) && line.inlierCount >= d.u32(MIN_QUAD_EDGE_INLIERS)) {
          layout.$.quadPeakEdge[quadBase + q] = q
        }
      }
    }
    layout.$.labelToQuadId[labelId] = quadId
    layout.$.quadSourceLabelId[quadId] = labelId
  })

  return root.createComputePipeline({ compute: kernel })
}

function createWriteQuadLabelMapPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['writeQuadLabelMapLayout'],
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
    const labelId = layout.$.compactLabels[idx]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      layout.$.quadLabelBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    layout.$.quadLabelBuffer[idx] = layout.$.labelToQuadId[labelId]!
  })

  return root.createComputePipeline({ compute: kernel })
}

function createAssignEdgesPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['assignEdgesLayout'],
  width: number,
  height: number,
) {
  const maxBinDist = d.u32(ORIENT_ASSIGN_MAX_BIN_DIST)

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
      layout.$.packedEdgeLabels[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    const labelId = layout.$.compactLabels[idx]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      layout.$.packedEdgeLabels[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    const quadId = layout.$.labelToQuadId[labelId]!
    if (quadId === d.u32(COMPONENT_LABEL_INVALID)) {
      layout.$.packedEdgeLabels[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    const cluster = layout.$.labelClusters[labelId]!
    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g, maxBinDist)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      layout.$.packedEdgeLabels[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    layout.$.packedEdgeLabels[idx] = quadId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
  })
  return root.createComputePipeline({ compute: kernel })
}

export function createEdgeHistogramClusterStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxComponents: number,
  filteredBuffer: EdgeFilterBindResources['filteredBuffer'],
  compactLabelBuffer: CompactLabelMapBuffer,
) {
  const area = width * height
  const labelClusters = root.createBuffer(d.arrayOf(LabelOrientCluster, maxComponents)).$usage('storage')
  const packedEdgeLabels = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const labelToQuadId = root.createBuffer(d.arrayOf(d.u32, maxComponents)).$usage('storage')
  const quadPeakEdge = root.createBuffer(d.arrayOf(d.u32, MAX_QUADS * MAX_EDGES_PER_LABEL)).$usage('storage')
  const quadSourceLabelId = root.createBuffer(d.arrayOf(d.u32, MAX_QUADS)).$usage('storage')
  const quadCount = root.createBuffer(d.arrayOf(d.atomic(d.u32), 1)).$usage('storage')
  const quadLabelBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const layouts = createEdgeHistogramClusterLayouts()
  const histResetPipeline = createHistResetPipeline(root, layouts.histResetLayout, maxComponents)
  const histAccumPipeline = createHistAccumPipeline(root, layouts.histAccumLayout, width, height)
  const findPeaksPipeline = createFindPeaksPipeline(root, layouts.findPeaksLayout, maxComponents)
  const compactQuadsPipeline = createCompactQuadsPipeline(root, layouts.compactQuadsLayout, maxComponents)
  const writeQuadLabelMapPipeline = createWriteQuadLabelMapPipeline(
    root,
    layouts.writeQuadLabelMapLayout,
    width,
    height,
  )
  const assignEdgesPipeline = createAssignEdgesPipeline(root, layouts.assignEdgesLayout, width, height)
  const labelLineFit = createLabelLineFitStage(
    root,
    width,
    height,
    maxComponents,
    filteredBuffer,
    compactLabelBuffer,
    labelClusters,
  )

  const histResetBindGroup = root.createBindGroup(layouts.histResetLayout, {
    labelClusters,
    quadCount,
  })
  const histAccumBindGroup = root.createBindGroup(layouts.histAccumLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels: compactLabelBuffer,
    labelClusters,
  })
  const findPeaksBindGroup = root.createBindGroup(layouts.findPeaksLayout, { labelClusters })
  const compactQuadsBindGroup = root.createBindGroup(layouts.compactQuadsLayout, {
    labelClusters,
    labelLineOut: labelLineFit.labelLineOut,
    quadPeakEdge,
    labelToQuadId,
    quadSourceLabelId,
    quadCount,
  })
  const writeQuadLabelMapBindGroup = root.createBindGroup(layouts.writeQuadLabelMapLayout, {
    compactLabels: compactLabelBuffer,
    labelToQuadId,
    quadLabelBuffer,
  })
  const assignEdgesBindGroup = root.createBindGroup(layouts.assignEdgesLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels: compactLabelBuffer,
    labelClusters,
    labelToQuadId,
    packedEdgeLabels,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const labelWg = Math.ceil(maxComponents / WORKGROUP_SIZE)

  const encodeCompute = (pass: GPUComputePassEncoder) => {
    histResetPipeline.with(pass).with(histResetBindGroup).dispatchWorkgroups(labelWg)
    histAccumPipeline.with(pass).with(histAccumBindGroup).dispatchWorkgroups(wgX, wgY)
    findPeaksPipeline.with(pass).with(findPeaksBindGroup).dispatchWorkgroups(labelWg)
    labelLineFit.encodeLabelLineFit(pass)
    compactQuadsPipeline.with(pass).with(compactQuadsBindGroup).dispatchWorkgroups(labelWg)
    writeQuadLabelMapPipeline.with(pass).with(writeQuadLabelMapBindGroup).dispatchWorkgroups(wgX, wgY)
    assignEdgesPipeline.with(pass).with(assignEdgesBindGroup).dispatchWorkgroups(wgX, wgY)
  }

  return {
    labelClusters,
    labelLineOut: labelLineFit.labelLineOut,
    labelLineReduce: labelLineFit.labelLineReduce,
    quadPeakEdge,
    labelToQuadId,
    quadSourceLabelId,
    quadCount,
    quadLabelBuffer,
    packedEdgeLabels,
    encodeCompute,
  }
}

export type { LabelLineOutBuffer } from '@/gpu/pipelines/labelLineFitPipeline'

export type PackedEdgeLabelBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['packedEdgeLabels']
export type LabelOrientClusterBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['labelClusters']
export type QuadPeakEdgeBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadPeakEdge']
export type LabelToQuadIdBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['labelToQuadId']
export type QuadSourceLabelIdBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadSourceLabelId']
export type QuadCountBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadCount']
export type QuadLabelMapBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadLabelBuffer']
