// Per-component orientation histograms as literal 32×8 pixel bars (TypeGPU render).
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { common } from 'typegpu'
import { clamp, floor, max } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import {
  LabelOrientClusterReadonly,
  MAX_EDGES_PER_LABEL,
  ORIENT_HIST_BINS,
} from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import type {
  LabelOrientClusterBuffer,
  QuadCountBuffer,
  QuadSourceLabelIdBuffer,
} from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { stableHashToRgb01 } from '@/lib/hashStableColor'

/** Histogram strip width in pixels (one pixel per orientation bin). */
export const ORIENT_HIST_VIZ_WIDTH = ORIENT_HIST_BINS
/** Bar area height in pixels per histogram cell. */
export const ORIENT_HIST_VIZ_BIN_H = 8
/** Gap between histogram cells in pixels (horizontal and vertical). */
export const ORIENT_HIST_VIZ_GAP = 2
/** Histograms per grid row. */
export const ORIENT_HIST_GRID_COLS = 32
export const ORIENT_HIST_CELL_STRIDE_X = ORIENT_HIST_VIZ_WIDTH + ORIENT_HIST_VIZ_GAP
export const ORIENT_HIST_CELL_STRIDE_Y = ORIENT_HIST_VIZ_BIN_H + ORIENT_HIST_VIZ_GAP

const OrientHistVizParams = d.struct({
  canvasSize: d.vec2u,
})

const orientHistVizLayout = tgpu.bindGroupLayout({
  params: { uniform: OrientHistVizParams },
  rowCount: { storage: d.arrayOf(d.u32, 1), access: 'readonly' },
  rowLabelIds: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labelClusters: { storage: d.arrayOf(LabelOrientClusterReadonly), access: 'readonly' },
})

const orientHistPackLayout = tgpu.bindGroupLayout({
  quadCount: { storage: d.arrayOf(d.u32, 1), access: 'readonly' },
  quadSourceLabelId: { storage: d.arrayOf(d.u32), access: 'readonly' },
  rowCount: { storage: d.arrayOf(d.u32, 1), access: 'mutable' },
  rowLabelIds: { storage: d.arrayOf(d.u32), access: 'mutable' },
})

const BG = d.vec4f(d.f32(0.1), d.f32(0.1), d.f32(0.14), d.f32(1))
const GAP = d.vec4f(d.f32(0.06), d.f32(0.06), d.f32(0.08), d.f32(1))
const PEAK = d.vec4f(d.f32(1), d.f32(1), d.f32(0.92), d.f32(1))

export const ORIENT_HIST_CANVAS_WIDTH =
  ORIENT_HIST_GRID_COLS * ORIENT_HIST_CELL_STRIDE_X - ORIENT_HIST_VIZ_GAP

const ORIENT_HIST_GRID_ROWS = Math.ceil(MAX_QUADS / ORIENT_HIST_GRID_COLS)

/** Fixed GPU canvas; scroll via `.orientHistScroll` when content exceeds viewport. */
export const ORIENT_HIST_CANVAS_HEIGHT =
  ORIENT_HIST_GRID_ROWS * ORIENT_HIST_CELL_STRIDE_Y - ORIENT_HIST_VIZ_GAP

function createOrientHistPackPipeline(root: TgpuRoot, maxQuads: number) {
  const packRows = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu'
    if (input.gid.x !== d.u32(0)) {
      return
    }

    const quads = orientHistPackLayout.$.quadCount[d.u32(0)]!
    orientHistPackLayout.$.rowCount[d.u32(0)] = quads

    for (let quadId = d.u32(0); quadId < quads; quadId = quadId + d.u32(1)) {
      orientHistPackLayout.$.rowLabelIds[quadId] =
        orientHistPackLayout.$.quadSourceLabelId[quadId]!
    }

    for (let quadId = quads; quadId < d.u32(maxQuads); quadId = quadId + d.u32(1)) {
      orientHistPackLayout.$.rowLabelIds[quadId] = d.u32(COMPONENT_LABEL_INVALID)
    }
  })

  return root.createComputePipeline({ compute: packRows })
}

function createOrientHistRenderPipeline(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const frag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const size = orientHistVizLayout.$.params.canvasSize
    const wi = d.i32(size.x)
    const hi = d.i32(size.y)
    if (wi <= d.i32(0) || hi <= d.i32(0)) {
      return BG
    }

    const maxPx = d.f32(wi - d.i32(1))
    const maxPy = d.f32(hi - d.i32(1))
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)))
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)))

    const histCount = orientHistVizLayout.$.rowCount[d.u32(0)]!
    const gridCols = d.u32(ORIENT_HIST_GRID_COLS)
    const cellW = d.u32(ORIENT_HIST_VIZ_WIDTH)
    const cellH = d.u32(ORIENT_HIST_VIZ_BIN_H)
    const gap = d.u32(ORIENT_HIST_VIZ_GAP)
    const strideX = cellW + gap
    const strideY = cellH + gap

    const gridCol = d.u32(px / strideX)
    const gridRow = d.u32(py / strideY)
    const localX = px % strideX
    const localY = py % strideY

    if (gridCol >= gridCols) {
      return BG
    }
    if (localX >= cellW || localY >= cellH) {
      return GAP
    }

    const flatIdx = gridRow * gridCols + gridCol
    if (flatIdx >= histCount) {
      return BG
    }

    const labelId = orientHistVizLayout.$.rowLabelIds[flatIdx]!
    const cluster = orientHistVizLayout.$.labelClusters[labelId]!
    const bin = localX

    let maxCount = d.u32(0)
    for (const b of tgpu.unroll(std.range(0, ORIENT_HIST_BINS))) {
      const c = cluster.orientationHistogram[d.u32(b)]!
      maxCount = max(maxCount, c)
    }
    if (maxCount === d.u32(0)) {
      return BG
    }

    const count = cluster.orientationHistogram[bin]!
    let barH = d.u32(0)
    if (count > d.u32(0)) {
      const scaled = d.u32((count * cellH) / maxCount)
      barH = scaled
      if (scaled < d.u32(1)) {
        barH = d.u32(1)
      }
    }

    // py/uv increase downward: anchor bars on cell bottom (localY = cellH - 1).
    const barTop = cellH - barH
    if (localY < barTop) {
      return BG
    }

    let isPeak = d.bool(false)
    const peakCount = cluster.peakCount
    for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (k < peakCount) {
        const peakBin = cluster.peakBins[d.u32(k)]!
        if (peakBin !== d.u32(COMPONENT_LABEL_INVALID) && peakBin === bin) {
          isPeak = d.bool(true)
        }
      }
    }

    if (isPeak) {
      return PEAK
    }

    const rgb = stableHashToRgb01(labelId)
    return d.vec4f(rgb, d.f32(1))
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })

  return pipeline
}

export function createOrientHistVizStage(
  root: TgpuRoot,
  labelClusters: LabelOrientClusterBuffer,
  quadSourceLabelId: QuadSourceLabelIdBuffer,
  quadCount: QuadCountBuffer,
  presentationFormat: GPUTextureFormat,
) {
  const paramsBuffer = root.createBuffer(OrientHistVizParams).$usage('uniform')
  const rowCountBuffer = root.createBuffer(d.arrayOf(d.u32, 1)).$usage('storage')
  const rowLabelIds = root.createBuffer(d.arrayOf(d.u32, MAX_QUADS)).$usage('storage')

  const packPipeline = createOrientHistPackPipeline(root, MAX_QUADS)
  const renderPipeline = createOrientHistRenderPipeline(root, presentationFormat)

  const packBindGroup = root.createBindGroup(orientHistPackLayout, {
    quadCount,
    quadSourceLabelId,
    rowCount: rowCountBuffer,
    rowLabelIds,
  })
  const renderBindGroup = root.createBindGroup(orientHistVizLayout, {
    params: paramsBuffer,
    rowCount: rowCountBuffer,
    rowLabelIds,
    labelClusters,
  })

  const encodePackCompute = (pass: GPUComputePassEncoder) => {
    packPipeline.with(pass).with(packBindGroup).dispatchWorkgroups(1)
  }

  const encodeDisplay = (enc: GPUCommandEncoder, colorAttachment: ColorAttachment) => {
    paramsBuffer.write({
      canvasSize: d.vec2u(ORIENT_HIST_CANVAS_WIDTH, ORIENT_HIST_CANVAS_HEIGHT),
    })
    renderPipeline.with(enc).withColorAttachment(colorAttachment).with(renderBindGroup).draw(3)
  }

  return {
    rowCountBuffer,
    encodePackCompute,
    encodeDisplay,
  }
}

export type OrientHistVizStage = ReturnType<typeof createOrientHistVizStage>
