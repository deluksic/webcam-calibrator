// Per-pixel line-fit failure stage for debugging assign / sample / TLS / quad drops.
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'
import { abs, clamp, floor, length } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import {
  LINE_INLIER_DIST_PX,
  LINE_MIN_PEAK_HIST_COUNT,
  LINE_MIN_SLOT_COUNT,
  MAX_EDGES_PER_LABEL,
  ORIENT_ASSIGN_MAX_BIN_DIST,
} from '@/gpu/lineFitThresholds'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import type { LabelOrientClusterBuffer } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { LabelOrientClusterReadonly } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import type {
  LabelInlierStatsBuffer,
  LabelLineOutBuffer,
  LabelLineReduceBuffer,
} from '@/gpu/pipelines/labelLineFitPipeline'
import {
  classifyInvalidLineFit,
  LabelInlierStatsReadonly,
  lineFitProbeFromSlot,
} from '@/gpu/shaders/lineFitRejectClassify'
import { assignPeakEdgeId } from '@/gpu/shaders/orientPeakAssign'

const WORKGROUP_SIZE = 16

export const LineFitDebugCode = {
  none: 0,
  noLabel: 1,
  noPeaks: 2,
  assign: 3,
  lowSample: 4,
  outlier: 5,
  ok: 8,
  histGate: 9,
  lineInvalidRatio: 10,
  lineInvalidSpan: 11,
  tlsIsotropy: 12,
  tlsPeak: 13,
  tlsDegenerate: 14,
  tlsRefineFail: 16,
} as const

export type LineFitDebugCodeValue = (typeof LineFitDebugCode)[keyof typeof LineFitDebugCode]

export const LINE_REJECT_LEGEND: ReadonlyArray<{ code: LineFitDebugCodeValue; label: string; color: string }> = [
  { code: LineFitDebugCode.ok, label: 'Valid line', color: '#38383d' },
  { code: LineFitDebugCode.assign, label: 'Peak assign', color: '#ff2bd6' },
  { code: LineFitDebugCode.histGate, label: 'Hist gate', color: '#ff6b35' },
  { code: LineFitDebugCode.lowSample, label: 'Slot < min px', color: '#f7b801' },
  { code: LineFitDebugCode.noPeaks, label: 'No peaks', color: '#4361ee' },
  { code: LineFitDebugCode.outlier, label: 'Inlier dist', color: '#ffee00' },
  { code: LineFitDebugCode.lineInvalidRatio, label: 'Inlier ratio', color: '#c1121f' },
  { code: LineFitDebugCode.lineInvalidSpan, label: 'Span short', color: '#70e000' },
  { code: LineFitDebugCode.tlsIsotropy, label: 'TLS spread', color: '#c77dff' },
  { code: LineFitDebugCode.tlsPeak, label: 'TLS vs peak', color: '#00bbf9' },
  { code: LineFitDebugCode.tlsDegenerate, label: 'TLS degenerate', color: '#9b5de5' },
  { code: LineFitDebugCode.tlsRefineFail, label: 'TLS refine', color: '#00f5d4' },
]

const LabelLineReduceReadonly = d.struct({
  count: d.u32,
})

const FrameSizeUniform = d.struct({
  width: d.u32,
  height: d.u32,
})

function createDebugLayouts() {
  const accumLayout = tgpu.bindGroupLayout({
    frame: { uniform: FrameSizeUniform },
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
    labelLineReduce: { storage: d.arrayOf(LabelLineReduceReadonly), access: 'readonly' },
    labelInlierStats: { storage: d.arrayOf(LabelInlierStatsReadonly), access: 'readonly' },
    labelLineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    lineFitDebug: { storage: d.arrayOf(d.u32), access: 'mutable' },
  })
  const renderLayout = tgpu.bindGroupLayout({
    frame: { uniform: FrameSizeUniform },
    lineFitDebug: { storage: d.arrayOf(d.u32), access: 'readonly' },
  })
  return { accumLayout, renderLayout }
}

export function createLineFitDebugStage(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
  filteredBuffer: EdgeFilterBindResources['filteredBuffer'],
  compactLabels: CompactLabelMapBuffer,
  labelClusters: LabelOrientClusterBuffer,
  labelLineReduce: LabelLineReduceBuffer,
  labelInlierStats: LabelInlierStatsBuffer,
  labelLineOut: LabelLineOutBuffer,
) {
  const area = width * height
  const lineFitDebug = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const frameUniform = root.createBuffer(FrameSizeUniform).$usage('uniform')

  const layouts = createDebugLayouts()

  const accumPipeline = createDebugAccumPipeline(
    root,
    layouts.accumLayout,
    ORIENT_ASSIGN_MAX_BIN_DIST,
    LINE_INLIER_DIST_PX,
  )
  const renderPipeline = createDebugRenderPipeline(root, layouts.renderLayout, width, height, presentationFormat)

  const accumBindGroup = root.createBindGroup(layouts.accumLayout, {
    frame: frameUniform,
    edgeBuffer: filteredBuffer,
    compactLabels,
    labelClusters,
    labelLineReduce,
    labelInlierStats,
    labelLineOut,
    lineFitDebug,
  })
  const renderBindGroup = root.createBindGroup(layouts.renderLayout, {
    frame: frameUniform,
    lineFitDebug,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)

  return {
    lineFitDebug,
    encodeCompute: (pass: GPUComputePassEncoder) => {
      frameUniform.write({ width: d.u32(width), height: d.u32(height) })
      accumPipeline.with(pass).with(accumBindGroup).dispatchWorkgroups(wgX, wgY)
    },
    encodeToCanvas: (enc: GPUCommandEncoder, colorAttachment: ColorAttachment) => {
      frameUniform.write({ width: d.u32(width), height: d.u32(height) })
      renderPipeline
        .with(enc)
        .withColorAttachment({ ...colorAttachment, loadOp: 'load', storeOp: 'store' })
        .with(renderBindGroup)
        .draw(3)
    },
  }
}

function createDebugAccumPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createDebugLayouts>['accumLayout'],
  maxBinDist: number,
  inlierDist: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const fw = d.i32(layout.$.frame.width)
    const fh = d.i32(layout.$.frame.height)
    if (x >= fw || y >= fh) {
      return
    }

    const idx = d.u32(y * fw + x)
    let code = d.u32(LineFitDebugCode.none)

    const g = layout.$.edgeBuffer[idx]!
    if (length(g) <= d.f32(0)) {
      layout.$.lineFitDebug[idx] = code
      return
    }

    const labelId = layout.$.compactLabels[idx]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      code = d.u32(LineFitDebugCode.noLabel)
      layout.$.lineFitDebug[idx] = code
      return
    }

    const cluster = layout.$.labelClusters[labelId]!
    if (cluster.peakCount === d.u32(0)) {
      code = d.u32(LineFitDebugCode.noPeaks)
      layout.$.lineFitDebug[idx] = code
      return
    }

    const edgeId = assignPeakEdgeId(cluster.peakCount, cluster.peakBins, cluster.peakDirs, g, maxBinDist)
    if (edgeId === d.u32(COMPONENT_LABEL_INVALID)) {
      code = d.u32(LineFitDebugCode.assign)
      layout.$.lineFitDebug[idx] = code
      return
    }

    const peakBin = cluster.peakBins[edgeId]!
    if (
      peakBin === d.u32(COMPONENT_LABEL_INVALID) ||
      cluster.orientationHistogram[peakBin]! < d.u32(LINE_MIN_PEAK_HIST_COUNT)
    ) {
      code = d.u32(LineFitDebugCode.histGate)
      layout.$.lineFitDebug[idx] = code
      return
    }

    const slot = labelId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
    const reduce = layout.$.labelLineReduce[slot]!
    const slotCount = reduce.count
    if (slotCount < d.u32(LINE_MIN_SLOT_COUNT)) {
      code = d.u32(LineFitDebugCode.lowSample)
      layout.$.lineFitDebug[idx] = code
      return
    }

    const line = layout.$.labelLineOut[slot]!
    if (line.valid !== d.u32(0)) {
      code = d.u32(LineFitDebugCode.ok)
      layout.$.lineFitDebug[idx] = code
      return
    }

    const peakDir = cluster.peakDirs[edgeId]!
    const inlier = layout.$.labelInlierStats[slot]!
    const probe = lineFitProbeFromSlot(inlier, peakDir, slotCount)
    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)
    const s = px * probe.nx + py * probe.ny - probe.nDotMean
    if (abs(s) >= inlierDist) {
      code = d.u32(LineFitDebugCode.outlier)
    } else {
      code = classifyInvalidLineFit(inlier, line, peakDir, slotCount)
    }
    layout.$.lineFitDebug[idx] = code
  })
  return root.createComputePipeline({ compute: kernel })
}

function createDebugRenderPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createDebugLayouts>['renderLayout'],
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const frag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const wi = d.i32(width)
    const hi = d.i32(height)
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), d.f32(wi - d.i32(1)))))
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), d.f32(hi - d.i32(1)))))
    const idx = py * d.u32(wi) + px
    const code = layout.$.lineFitDebug[idx]!

    if (code === d.u32(LineFitDebugCode.ok)) {
      return d.vec4f(0.22, 0.22, 0.24, 1)
    }
    if (code === d.u32(LineFitDebugCode.assign)) {
      return d.vec4f(1, 0.17, 0.84, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.histGate)) {
      return d.vec4f(1, 0.42, 0.21, 0.92)
    }
    if (code === d.u32(LineFitDebugCode.lowSample)) {
      return d.vec4f(0.97, 0.72, 0.0, 0.92)
    }
    if (code === d.u32(LineFitDebugCode.noPeaks)) {
      return d.vec4f(0.26, 0.38, 0.93, 0.9)
    }
    if (code === d.u32(LineFitDebugCode.outlier)) {
      return d.vec4f(1, 0.93, 0, 0.92)
    }
    if (code === d.u32(LineFitDebugCode.lineInvalidRatio)) {
      return d.vec4f(0.76, 0.07, 0.12, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.lineInvalidSpan)) {
      return d.vec4f(0.44, 0.88, 0, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.tlsIsotropy)) {
      return d.vec4f(0.78, 0.49, 1, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.tlsPeak)) {
      return d.vec4f(0, 0.73, 0.98, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.tlsDegenerate)) {
      return d.vec4f(0.61, 0.36, 0.9, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.tlsRefineFail)) {
      return d.vec4f(0, 0.96, 0.83, 0.95)
    }
    if (code === d.u32(LineFitDebugCode.noLabel)) {
      return d.vec4f(0.36, 0.36, 0.4, 0.75)
    }
    return d.vec4f(0, 0, 0, 0)
  })

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: {
      format: presentationFormat,
      blend: {
        color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      },
    },
  })
}
