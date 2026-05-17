// Per compact labelId: 32-bin circular gradient histogram → up to 6 peaks → per-pixel edgeId.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicAdd, atomicLoad, atomicStore, atan2, cos, dot, length, sin } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import { MAX_EXTENT_COMPONENTS } from '@/gpu/pipelines/extentTrackingPipeline'

const WORKGROUP_SIZE = 16

export const ORIENT_HIST_BINS = 32
export const MAX_EDGES_PER_LABEL = 6
export const ORIENT_PEAK_MIN_COUNT = 3
/** Min circular bin distance between accepted peaks (~45° at 32 bins). */
export const MIN_PEAK_BIN_SEPARATION = 4
/** Pixel must align with peak direction; dot(n_px, n_peak) >= this (0.707 ≈ 45° cone). */
export const ORIENT_ASSIGN_COS_THRESHOLD = 0.707
/** Components must have exactly this many orientation peaks to become a quad. */
export const REQUIRED_ORIENTATION_PEAK_COUNT = 4
export const MAX_QUADS = MAX_EXTENT_COMPONENTS
export const MAX_FLAT_EDGES = MAX_QUADS * MAX_EDGES_PER_LABEL

export const LabelOrientCluster = d.struct({
  orientationHistogram: d.arrayOf(d.atomic(d.u32), ORIENT_HIST_BINS),
  peakBins: d.arrayOf(d.u32, MAX_EDGES_PER_LABEL),
  peakCount: d.u32,
})

/** Same layout as {@link LabelOrientCluster}; plain histogram for readonly binds. */
export const LabelOrientClusterReadonly = d.struct({
  orientationHistogram: d.arrayOf(d.u32, ORIENT_HIST_BINS),
  peakBins: d.arrayOf(d.u32, MAX_EDGES_PER_LABEL),
  peakCount: d.u32,
})

const PI_F32 = d.f32(3.14159265)

const gradientOrientationBin = tgpu.fn([d.f32, d.f32], d.u32)((gx, gy) => {
  'use gpu'
  const theta = atan2(gy, gx)
  const scaled = (theta / PI_F32 + d.f32(1)) * d.f32(16)
  let bin = d.u32(std.floor(scaled))
  if (bin >= d.u32(ORIENT_HIST_BINS)) {
    bin = d.u32(0)
  }
  return bin
})

const circularBinDist = tgpu.fn([d.u32, d.u32], d.u32)((a, b) => {
  'use gpu'
  const bins = d.u32(ORIENT_HIST_BINS)
  const forward = (a + bins - b) % bins
  const backward = (b + bins - a) % bins
  return std.min(forward, backward)
})

/** Unit normal from histogram bin center (matches {@link gradientOrientationBin}). */
const orientationBinToUnit = tgpu.fn([d.u32], d.vec2f)((bin) => {
  'use gpu'
  const theta = (d.f32(bin) + d.f32(0.5)) / d.f32(16) * PI_F32 - PI_F32
  return d.vec2f(cos(theta), sin(theta))
})

function createEdgeHistogramClusterLayouts() {
  const histResetLayout = tgpu.bindGroupLayout({
    labelClusters: { storage: d.arrayOf(LabelOrientCluster), access: 'mutable' },
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
    labelToQuadId: { storage: d.arrayOf(d.u32), access: 'mutable' },
    quadSourceLabelId: { storage: d.arrayOf(d.u32), access: 'mutable' },
    quadCount: { storage: d.arrayOf(d.u32, 1), access: 'mutable' },
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
    }
    slot.peakCount = d.u32(0)
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

    const bin = gradientOrientationBin(g.x, g.y)
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
    for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      slot.peakBins[d.u32(k)] = d.u32(COMPONENT_LABEL_INVALID)
    }
    slot.peakCount = d.u32(0)

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
  })
  return root.createComputePipeline({ compute: kernel })
}

function createCompactQuadsPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createEdgeHistogramClusterLayouts>['compactQuadsLayout'],
  maxComponents: number,
) {
  const requiredPeaks = d.u32(REQUIRED_ORIENTATION_PEAK_COUNT)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu'
    if (input.gid.x !== d.u32(0)) {
      return
    }

    layout.$.quadCount[d.u32(0)] = d.u32(0)
    const maxLabels = d.u32(maxComponents)

    for (let labelId = d.u32(0); labelId < maxLabels; labelId = labelId + d.u32(1)) {
      layout.$.labelToQuadId[labelId] = d.u32(COMPONENT_LABEL_INVALID)
    }

    for (let labelId = d.u32(0); labelId < maxLabels; labelId = labelId + d.u32(1)) {
      const cluster = layout.$.labelClusters[labelId]!
      if (cluster.peakCount !== requiredPeaks) {
        continue
      }

      let total = d.u32(0)
      for (const b of tgpu.unroll(std.range(0, ORIENT_HIST_BINS))) {
        total = total + cluster.orientationHistogram[d.u32(b)]!
      }
      if (total === d.u32(0)) {
        continue
      }

      const quadId = layout.$.quadCount[d.u32(0)]!
      layout.$.labelToQuadId[labelId] = quadId
      layout.$.quadSourceLabelId[quadId] = labelId
      layout.$.quadCount[d.u32(0)] = quadId + d.u32(1)
    }
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
    const peakCount = cluster.peakCount
    const pixelBin = gradientOrientationBin(g.x, g.y)
    const gLen = length(g)
    const nx = g.x / gLen
    const ny = g.y / gLen
    const cosAssign = d.f32(ORIENT_ASSIGN_COS_THRESHOLD)

    let edgeId = d.u32(0)
    if (peakCount > d.u32(0)) {
      let bestDot = d.f32(-1)
      let bestDist = d.u32(ORIENT_HIST_BINS)
      for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
        if (k < peakCount) {
          const peakBin = cluster.peakBins[d.u32(k)]!
          if (peakBin !== d.u32(COMPONENT_LABEL_INVALID)) {
            const peakDir = orientationBinToUnit(peakBin)
            const align = dot(d.vec2f(nx, ny), peakDir)
            if (align >= cosAssign) {
              const dist = circularBinDist(pixelBin, peakBin)
              const pick = align > bestDot || (align === bestDot && dist < bestDist)
              if (pick) {
                bestDot = align
                bestDist = dist
                edgeId = d.u32(k)
              }
            }
          }
        }
      }
      if (bestDot < cosAssign) {
        for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
          if (k < peakCount) {
            const peakBin = cluster.peakBins[d.u32(k)]!
            if (peakBin !== d.u32(COMPONENT_LABEL_INVALID)) {
              const peakDir = orientationBinToUnit(peakBin)
              const align = dot(d.vec2f(nx, ny), peakDir)
              if (align > bestDot) {
                bestDot = align
                edgeId = d.u32(k)
              }
            }
          }
        }
      }
    }

    layout.$.packedEdgeLabels[idx] =
      quadId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
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

  const histResetBindGroup = root.createBindGroup(layouts.histResetLayout, { labelClusters })
  const histAccumBindGroup = root.createBindGroup(layouts.histAccumLayout, {
    edgeBuffer: filteredBuffer,
    compactLabels: compactLabelBuffer,
    labelClusters,
  })
  const findPeaksBindGroup = root.createBindGroup(layouts.findPeaksLayout, { labelClusters })
  const compactQuadsBindGroup = root.createBindGroup(layouts.compactQuadsLayout, {
    labelClusters,
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
    compactQuadsPipeline.with(pass).with(compactQuadsBindGroup).dispatchWorkgroups(1)
    writeQuadLabelMapPipeline.with(pass).with(writeQuadLabelMapBindGroup).dispatchWorkgroups(wgX, wgY)
    assignEdgesPipeline.with(pass).with(assignEdgesBindGroup).dispatchWorkgroups(wgX, wgY)
  }

  return {
    labelClusters,
    labelToQuadId,
    quadSourceLabelId,
    quadCount,
    quadLabelBuffer,
    packedEdgeLabels,
    encodeCompute,
  }
}

export type PackedEdgeLabelBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['packedEdgeLabels']
export type LabelOrientClusterBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['labelClusters']
export type LabelToQuadIdBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['labelToQuadId']
export type QuadSourceLabelIdBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadSourceLabelId']
export type QuadCountBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadCount']
export type QuadLabelMapBuffer = ReturnType<typeof createEdgeHistogramClusterStage>['quadLabelBuffer']
