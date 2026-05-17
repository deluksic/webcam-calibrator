// Per compact labelId: 32-bin circular gradient histogram → up to 6 peaks → per-pixel edgeId.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicAdd, atomicLoad, atomicStore, length, select } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import { createLabelLineFitStage } from '@/gpu/pipelines/labelLineFitPipeline'
import {
  circularBinDist,
  gradientOrientationBin,
  gradientPeakAlign,
  MAX_EDGES_PER_LABEL,
  ORIENT_ASSIGN_COS_THRESHOLD,
  ORIENT_HIST_BINS,
  peakDirFromLocalBins,
} from '@/gpu/shaders/orientPeakAssign'

const WORKGROUP_SIZE = 16

export { ORIENT_HIST_BINS, MAX_EDGES_PER_LABEL } from '@/gpu/shaders/orientPeakAssign'
export const ORIENT_PEAK_MIN_COUNT = 3
/** Min circular bin distance between accepted peaks. */
export const MIN_PEAK_BIN_SEPARATION = 3
export { ORIENT_ASSIGN_COS_THRESHOLD } from '@/gpu/shaders/orientPeakAssign'
/** Max quad edge slots in flat buffers (profiles / packed labels). */
export const REQUIRED_ORIENTATION_PEAK_COUNT = 4
/** Min valid fitted peaks to compact a label into a quad (top-N by inlier, N ≤ 4). */
export const MIN_QUAD_VALID_EDGES = 2
export const MAX_QUADS = 2 << 9
export const MAX_FLAT_EDGES = MAX_QUADS * MAX_EDGES_PER_LABEL

export const LabelOrientCluster = d.struct({
  orientationHistogram: d.arrayOf(d.atomic(d.u32), ORIENT_HIST_BINS),
  peakBins: d.arrayOf(d.u32, MAX_EDGES_PER_LABEL),
  peakDirs: d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL),
  peakCount: d.u32,
})

/** Same layout as {@link LabelOrientCluster}; plain histogram for readonly binds. */
export const LabelOrientClusterReadonly = d.struct({
  orientationHistogram: d.arrayOf(d.u32, ORIENT_HIST_BINS),
  peakBins: d.arrayOf(d.u32, MAX_EDGES_PER_LABEL),
  peakDirs: d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL),
  peakCount: d.u32,
})

function createEdgeHistogramClusterLayouts() {
  const histResetLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientCluster), access: 'mutable' },
    quadCount: { storage: d.arrayOf(d.u32, 1), access: 'mutable' },
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
    quadPeakEdge: { storage: d.arrayOf(d.u32), access: 'readonly' },
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
      layout.$.quadCount[d.u32(0)] = d.u32(0)
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
    const histSlot = layout.$.labelClusters[labelId]!.orientationHistogram[bin]!
    atomicAdd(histSlot, d.u32(1))
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

    for (const b of tgpu.unroll(std.range(0, ORIENT_HIST_BINS))) {
      const bi = d.u32(b)
      const count = atomicLoad(slot.orientationHistogram[bi]!)
      if (count >= minPeakCount) {
        const prevB = (bi + d.u32(ORIENT_HIST_BINS - 1)) % d.u32(ORIENT_HIST_BINS)
        const nextB = (bi + d.u32(1)) % d.u32(ORIENT_HIST_BINS)
        const prevCount = atomicLoad(slot.orientationHistogram[prevB]!)
        const nextCount = atomicLoad(slot.orientationHistogram[nextB]!)
        if (count > prevCount && count > nextCount) {
          let tooClose = false
          const peakCount = slot.peakCount
          for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
            if (k < peakCount) {
              const pk = slot.peakBins[d.u32(k)]!
              if (pk !== d.u32(COMPONENT_LABEL_INVALID)) {
                if (circularBinDist(bi, pk) < minPeakSep) {
                  tooClose = true
                }
              }
            }
          }
          if (!tooClose) {
            if (peakCount < d.u32(MAX_EDGES_PER_LABEL)) {
              slot.peakBins[peakCount] = bi
              slot.peakCount = peakCount + d.u32(1)
            } else {
              let minIdx = d.u32(0)
              let minCount = atomicLoad(slot.orientationHistogram[slot.peakBins[d.u32(0)]!]!)
              for (const k of tgpu.unroll(std.range(1, MAX_EDGES_PER_LABEL))) {
                const pk = slot.peakBins[d.u32(k)]!
                const pc = atomicLoad(slot.orientationHistogram[pk]!)
                if (pc < minCount) {
                  minCount = pc
                  minIdx = d.u32(k)
                }
              }
              if (count > minCount) {
                slot.peakBins[minIdx] = bi
              }
            }
          }
        }
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

    let s0 = d.u32(0)
    let s1 = d.u32(0)
    let s2 = d.u32(0)
    let s3 = d.u32(0)
    let s4 = d.u32(0)
    let s5 = d.u32(0)

    for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (k < peakCount) {
        const peakBin = cluster.peakBins[d.u32(k)]!
        if (peakBin !== d.u32(COMPONENT_LABEL_INVALID)) {
          const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + d.u32(k)
          const line = layout.$.labelLineOut[slot]!
          const score = select(d.u32(0), line.inlierCount, line.valid !== d.u32(0))
          if (k === d.u32(0)) {
            s0 = score
          } else if (k === d.u32(1)) {
            s1 = score
          } else if (k === d.u32(2)) {
            s2 = score
          } else if (k === d.u32(3)) {
            s3 = score
          } else if (k === d.u32(4)) {
            s4 = score
          } else {
            s5 = score
          }
        }
      }
    }

    let pick0 = d.u32(COMPONENT_LABEL_INVALID)
    let pick1 = d.u32(COMPONENT_LABEL_INVALID)
    let pick2 = d.u32(COMPONENT_LABEL_INVALID)
    let pick3 = d.u32(COMPONENT_LABEL_INVALID)
    let numPicked = d.u32(0)
    let canPick = true

    for (const _rank of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (canPick) {
        let bestPeak = d.u32(COMPONENT_LABEL_INVALID)
        let bestScore = d.u32(0)
        if (s0 > bestScore) {
          bestScore = s0
          bestPeak = d.u32(0)
        }
        if (s1 > bestScore) {
          bestScore = s1
          bestPeak = d.u32(1)
        }
        if (s2 > bestScore) {
          bestScore = s2
          bestPeak = d.u32(2)
        }
        if (s3 > bestScore) {
          bestScore = s3
          bestPeak = d.u32(3)
        }
        if (s4 > bestScore) {
          bestScore = s4
          bestPeak = d.u32(4)
        }
        if (s5 > bestScore) {
          bestScore = s5
          bestPeak = d.u32(5)
        }
        if (bestPeak === d.u32(COMPONENT_LABEL_INVALID)) {
          canPick = false
        } else {
          if (numPicked === d.u32(0)) {
            pick0 = bestPeak
          } else if (numPicked === d.u32(1)) {
            pick1 = bestPeak
          } else if (numPicked === d.u32(2)) {
            pick2 = bestPeak
          } else {
            pick3 = bestPeak
          }
          numPicked = numPicked + d.u32(1)
          if (bestPeak === d.u32(0)) {
            s0 = d.u32(0)
          } else if (bestPeak === d.u32(1)) {
            s1 = d.u32(0)
          } else if (bestPeak === d.u32(2)) {
            s2 = d.u32(0)
          } else if (bestPeak === d.u32(3)) {
            s3 = d.u32(0)
          } else if (bestPeak === d.u32(4)) {
            s4 = d.u32(0)
          } else {
            s5 = d.u32(0)
          }
        }
      }
    }

    if (numPicked < d.u32(MIN_QUAD_VALID_EDGES)) {
      return
    }

    const quadId = atomicAdd(layout.$.quadCount[d.u32(0)]!, d.u32(1))
    if (quadId >= MAX_QUADS) {
      return
    }

    const quadBase = quadId * d.u32(REQUIRED_ORIENTATION_PEAK_COUNT)
    for (const q of tgpu.unroll(std.range(0, REQUIRED_ORIENTATION_PEAK_COUNT))) {
      layout.$.quadPeakEdge[quadBase + d.u32(q)] = d.u32(COMPONENT_LABEL_INVALID)
    }
    if (numPicked > d.u32(0)) {
      layout.$.quadPeakEdge[quadBase + d.u32(0)] = pick0
    }
    if (numPicked > d.u32(1)) {
      layout.$.quadPeakEdge[quadBase + d.u32(1)] = pick1
    }
    if (numPicked > d.u32(2)) {
      layout.$.quadPeakEdge[quadBase + d.u32(2)] = pick2
    }
    if (numPicked > d.u32(3)) {
      layout.$.quadPeakEdge[quadBase + d.u32(3)] = pick3
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
    const quadBase = quadId * d.u32(REQUIRED_ORIENTATION_PEAK_COUNT)
    let quadEdge = d.u32(COMPONENT_LABEL_INVALID)
    let bestDot = d.f32(-1)

    for (const q of tgpu.unroll(std.range(0, REQUIRED_ORIENTATION_PEAK_COUNT))) {
      const peakK = layout.$.quadPeakEdge[quadBase + d.u32(q)]!
      if (peakK !== d.u32(COMPONENT_LABEL_INVALID)) {
        const peakBin = cluster.peakBins[peakK]!
        if (peakBin !== d.u32(COMPONENT_LABEL_INVALID)) {
          const align = gradientPeakAlign(cluster.peakDirs[peakK]!, g)
          if (align >= ORIENT_ASSIGN_COS_THRESHOLD && align > bestDot) {
            bestDot = align
            quadEdge = d.u32(q)
          }
        }
      }
    }

    if (quadEdge === d.u32(COMPONENT_LABEL_INVALID)) {
      layout.$.packedEdgeLabels[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    layout.$.packedEdgeLabels[idx] = quadId * d.u32(MAX_EDGES_PER_LABEL) + quadEdge
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
  const quadPeakEdge = root
    .createBuffer(d.arrayOf(d.u32, MAX_QUADS * REQUIRED_ORIENTATION_PEAK_COUNT))
    .$usage('storage')
  const quadSourceLabelId = root.createBuffer(d.arrayOf(d.u32, MAX_QUADS)).$usage('storage')
  const quadCount = root.createBuffer(d.arrayOf(d.u32, 1)).$usage('storage')
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
    quadPeakEdge,
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
/** Per compact quad: peak index (0..5) for each quad edge slot (0..3). */
export type QuadPeakEdgeBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadPeakEdge']
export type LabelToQuadIdBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['labelToQuadId']
export type QuadSourceLabelIdBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadSourceLabelId']
export type QuadCountBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadCount']
export type QuadLabelMapBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadLabelBuffer']
