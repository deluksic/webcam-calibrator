import { tgpu, d } from 'typegpu'

import { createFrameSlotPool } from '@/gpu/frameSlotPool'
import type { FrameSlotPool } from '@/gpu/frameSlotPool'
import {
  createCompactLabelLayouts,
  createCanonicalResetPipeline,
  createCanonicalClaimPipeline,
  createRemapLabelPipeline,
} from '@/gpu/pipelines/compactLabelPipeline'
import {
  HISTOGRAM_BINS,
  POINTER_JUMP_ITERATIONS,
} from '@/gpu/pipelines/constants'
import { createCopyPipeline } from '@/gpu/pipelines/copyPipeline'
import { createEdgeDilatePipeline } from '@/gpu/pipelines/edgeDilatePipeline'
import { createEdgeFilterPipeline } from '@/gpu/pipelines/edgeFilterPipeline'
import { createEdgesPipeline } from '@/gpu/pipelines/edgesPipeline'
import {
  createExtentTrackingLayouts,
  createExtentResetPipeline,
  createExtentTrackPipeline,
  ExtentEntry,
} from '@/gpu/pipelines/extentTrackingPipeline'
import { createFilteredRenderPipeline } from '@/gpu/pipelines/filteredRenderPipeline'
import { createGrayPipeline } from '@/gpu/pipelines/grayPipeline'
import { createGrayRenderPipeline } from '@/gpu/pipelines/grayRenderPipeline'
import {
  createGridVizPipeline,
  createGridVizLayouts,
  GridDataSchema,
  MAX_INSTANCES,
} from '@/gpu/pipelines/gridVizPipeline'
import { createHistogramResetPipeline, createHistogramAccumulatePipeline } from '@/gpu/pipelines/histogramPipelines'
import { createHistogramRenderPipeline } from '@/gpu/pipelines/histogramRenderPipeline'
import { createLabelVizPipeline } from '@/gpu/pipelines/labelVizPipeline'
import { createLayouts } from '@/gpu/pipelines/layouts'
import {
  createPointerJumpLayouts,
  createPointerJumpInitPipeline,
  createPointerJumpStepPipeline,
  createPointerJumpLabelsToAtomicPipeline,
  createPointerJumpParentTightenPipeline,
  createPointerJumpAtomicToLabelsPipeline,
} from '@/gpu/pipelines/pointerJumpPipeline'
import { createSobelPipeline } from '@/gpu/pipelines/sobelPipeline'
import { createSobelRenderPipeline } from '@/gpu/pipelines/sobelRenderPipeline'

/** Max quads drawn in grid mode; must match `MAX_INSTANCES` in gridVizPipeline (buffer + draw). */
export const MAX_DETECTED_TAGS = MAX_INSTANCES

export type DisplayMode = 'edges' | 'nms' | 'labels' | 'grayscale' | 'debug' | 'grid'

export const MAX_EXTENT_COMPONENTS = 16384
export const MAX_COMPONENTS = 16384 // alias for CalibrationView readback
export const EXTENT_FIELDS = 4 // 4 fields per extent entry: minX, minY, maxX, maxY

export const MAX_U32 = 0xffffffff

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE FACTORY
// ═══════════════════════════════════════════════════════════════════════════
export function createCameraPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  canvas: HTMLCanvasElement,
  histCanvas: HTMLCanvasElement,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  // Configure contexts on canvases
  const context = root.configureContext({ canvas, alphaMode: 'premultiplied' })
  const histContext = root.configureContext({ canvas: histCanvas })

  // ═══════════════════════════════════════════════════════════════════════
  // RESOURCES
  // ═══════════════════════════════════════════════════════════════════════

  // Intermediate RGBA texture for external → usable format
  const grayTex = root
    .createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      dimension: '2d',
    })
    .$usage('storage', 'sampled', 'render')

  const sampler = root.createSampler({
    minFilter: 'linear',
    magFilter: 'linear',
  })

  const grayBuffer = root.createBuffer(d.arrayOf(d.f32, width * height)).$usage('storage')

  const sobelBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$usage('storage')

  const filteredBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$usage('storage')

  const dilatedEdgeBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$usage('storage')

  const thresholdBuffer = root.createBuffer(d.f32).$usage('uniform')

  const histogramSchema = d.arrayOf(d.atomic(d.u32), HISTOGRAM_BINS)
  const histogramBuffer = root.createBuffer(histogramSchema).$usage('storage')

  const thresholdBinBuffer = root.createBuffer(d.u32).$usage('uniform')

  const pointerJumpBuffer0 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage('storage')
  const pointerJumpBuffer1 = root.createBuffer(d.arrayOf(d.u32, width * height)).$usage('storage')

  const pointerJumpAtomicBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), width * height)).$usage('storage')

  const {
    initLayout: pointerJumpInitLayout,
    stepLayout: pointerJumpStepLayout,
    labelsToAtomicLayout: pointerJumpLabelsToAtomicLayout,
    parentTightenLayout: pointerJumpParentTightenLayout,
    atomicToLabelsLayout: pointerJumpAtomicToLabelsLayout,
  } = createPointerJumpLayouts()
  const pointerJumpInitPipeline = createPointerJumpInitPipeline(root, pointerJumpInitLayout, width, height)
  const pointerJumpStepPipeline = createPointerJumpStepPipeline(root, pointerJumpStepLayout, width, height)
  const pointerJumpLabelsToAtomicPipeline = createPointerJumpLabelsToAtomicPipeline(
    root,
    pointerJumpLabelsToAtomicLayout,
    width,
    height,
  )
  const pointerJumpParentTightenPipeline = createPointerJumpParentTightenPipeline(
    root,
    pointerJumpParentTightenLayout,
    width,
    height,
  )
  const pointerJumpAtomicToLabelsPipeline = createPointerJumpAtomicToLabelsPipeline(
    root,
    pointerJumpAtomicToLabelsLayout,
    width,
    height,
  )

  const pointerJumpInitBindGroup = root.createBindGroup(pointerJumpInitLayout, {
    edgeBuffer: filteredBuffer,
    labelBuffer: pointerJumpBuffer0,
  })

  const pointerJumpPingPongBindGroups = [
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: filteredBuffer,
      readBuffer: pointerJumpBuffer0,
      writeBuffer: pointerJumpBuffer1,
    }),
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: filteredBuffer,
      readBuffer: pointerJumpBuffer1,
      writeBuffer: pointerJumpBuffer0,
    }),
  ]

  const pointerJumpLabelsToAtomicBindGroups = [
    root.createBindGroup(pointerJumpLabelsToAtomicLayout, {
      source: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpLabelsToAtomicLayout, {
      source: pointerJumpBuffer1,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
  ]
  const pointerJumpParentTightenBindGroups = [
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: filteredBuffer,
      labelRead: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: filteredBuffer,
      labelRead: pointerJumpBuffer1,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
  ]
  const pointerJumpAtomicToLabelsBindGroups = [
    root.createBindGroup(pointerJumpAtomicToLabelsLayout, {
      atomicLabels: pointerJumpAtomicBuffer,
      dest: pointerJumpBuffer0,
    }),
    root.createBindGroup(pointerJumpAtomicToLabelsLayout, {
      atomicLabels: pointerJumpAtomicBuffer,
      dest: pointerJumpBuffer1,
    }),
  ]

  // ─── Extent tracking (per-component bounding boxes) ─────────────────────
  // Compact labeling: raw pixel-index labels → compact IDs (0..N-1).
  // canonicalRoot: one entry per possible root pixel index (area entries, max ~921K).
  // compactLabelBuffer: remapped labels after compact pass.
  // extentBuffer: sized for MAX_EXTENT_COMPONENTS entries (compact IDs < MAX_EXTENT_COMPONENTS).
  const area = width * height

  const canonicalRootBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), area)).$usage('storage')

  const compactLabelBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')

  const compactCounterBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')

  const { trackLayout: extentTrackLayout } = createExtentTrackingLayouts()

  const extentResetLayout = tgpu.bindGroupLayout({
    extentBuffer: { storage: d.arrayOf(ExtentEntry), access: 'mutable' },
  })

  const extentBuffer = root.createBuffer(d.arrayOf(ExtentEntry, MAX_EXTENT_COMPONENTS)).$usage('storage')

  const extentResetPipeline = createExtentResetPipeline(root, extentResetLayout, MAX_EXTENT_COMPONENTS)
  const extentTrackPipeline = createExtentTrackPipeline(root, extentTrackLayout, width, height, MAX_EXTENT_COMPONENTS)

  const extentResetBindGroup = root.createBindGroup(extentResetLayout, {
    extentBuffer,
  })
  // Extent tracking on compact labels
  const extentTrackBindGroup = root.createBindGroup(extentTrackLayout, {
    labelBuffer: compactLabelBuffer,
    extentBuffer,
  })

  // Compact labeling: remap raw pixel-index labels to compact IDs (0..N-1)
  const { resetLayout, claimLayout, remapLayout } = createCompactLabelLayouts()
  const compactResetPipeline = createCanonicalResetPipeline(root, resetLayout, area)
  const compactClaimPipeline = createCanonicalClaimPipeline(root, claimLayout, width, height, MAX_EXTENT_COMPONENTS)
  const compactRemapPipeline = createRemapLabelPipeline(root, remapLayout, width, height)

  // compactCounter: single u32 atomic counter for next compact ID
  const compactResetBindGroup = root.createBindGroup(resetLayout, {
    compactCounter: compactCounterBuffer,
    canonicalRoot: canonicalRootBuffer,
  })
  // POINTER_JUMP_ITERATIONS is required to be even, so pj === 0 after the loop
  // and pointerJumpBuffer0 always holds the converged labels here.
  const compactClaimBindGroup = root.createBindGroup(claimLayout, {
    labelBuffer: pointerJumpBuffer0,
    compactCounter: compactCounterBuffer,
    canonicalRoot: canonicalRootBuffer,
  })
  const compactRemapBindGroup = root.createBindGroup(remapLayout, {
    labelBuffer: pointerJumpBuffer0,
    compactLabelBuffer: compactLabelBuffer,
    canonicalRoot: canonicalRootBuffer,
  })

  // compactLabelBuffer is the remapped output used by extent tracking

  // ─── Grid visualization (AprilTag grid overlay) ─────────────────────────
  const quadCornersBuffer = root.createBuffer(GridDataSchema).$usage('storage')

  const { gridVizLayout } = createGridVizLayouts()

  const gridVizDebugModeBuffer = root.createBuffer(d.u32).$usage('uniform')
  gridVizDebugModeBuffer.write(0)

  const gridVizPipeline = createGridVizPipeline(root, gridVizLayout, width, height, presentationFormat)

  const gridVizBindGroup = root.createBindGroup(gridVizLayout, {
    quads: quadCornersBuffer,
    failInterrogate: gridVizDebugModeBuffer,
  })

  // ═══════════════════════════════════════════════════════════════════════
  // LAYOUTS & PIPELINES
  // ═══════════════════════════════════════════════════════════════════════

  const {
    copyLayout,
    grayTexToBufferLayout,
    sobelLayout,
    histogramLayout,
    histogramResetLayout,
    edgesLayout,
    histogramDisplayLayout,
    edgeFilterLayout,
    edgeDilateLayout,
    labelVizLayout,
    grayRenderLayout,
    sobelRenderLayout,
    filteredRenderLayout,
  } = createLayouts(histogramSchema)

  // ─── Frame slot pool (grid mode: 3 pinned gray+staging slots for frame pairing) ───
  const frameSlotPool: FrameSlotPool = createFrameSlotPool(root, { width, height, grayRenderLayout })

  const copyPipeline = createCopyPipeline(root, copyLayout)
  const grayPipeline = createGrayPipeline(root, grayTexToBufferLayout, width, height)
  const sobelPipeline = createSobelPipeline(root, sobelLayout, width, height)
  const histogramResetPipeline = createHistogramResetPipeline(root, histogramResetLayout)
  const histogramPipeline = createHistogramAccumulatePipeline(root, histogramLayout, width, height)
  const edgesPipeline = createEdgesPipeline(root, edgesLayout, width, height, presentationFormat)
  const edgeFilterPipeline = createEdgeFilterPipeline(root, edgeFilterLayout, width, height)
  const edgeDilatePipeline = createEdgeDilatePipeline(root, edgeDilateLayout, width, height)
  const histogramDisplayPipeline = createHistogramRenderPipeline(
    root,
    histogramDisplayLayout,
    presentationFormat,
    width * height,
  )
  const labelVizPipeline = createLabelVizPipeline(root, labelVizLayout, width, height, presentationFormat)
  const grayRenderPipeline = createGrayRenderPipeline(root, grayRenderLayout, width, height, presentationFormat)
  const sobelRenderPipeline = createSobelRenderPipeline(root, sobelRenderLayout, width, height, presentationFormat)
  const filteredRenderPipeline = createFilteredRenderPipeline(
    root,
    filteredRenderLayout,
    width,
    height,
    presentationFormat,
  )

  // ═══════════════════════════════════════════════════════════════════════
  // BIND GROUPS
  // ═══════════════════════════════════════════════════════════════════════

  // Copy (recreated per-frame for external texture)
  const copyLayoutTemplate = copyLayout

  const grayTexToBufferBindGroup = root.createBindGroup(grayTexToBufferLayout, {
    grayTex: grayTex,
    grayBuffer: grayBuffer,
  })

  const sobelBindGroup = root.createBindGroup(sobelLayout, {
    grayBuffer: grayBuffer,
    sobelBuffer: sobelBuffer,
  })

  const histogramResetBindGroup = root.createBindGroup(histogramResetLayout, {
    histogram: histogramBuffer,
  })

  const histogramComputeBindGroup = root.createBindGroup(histogramLayout, {
    sobelBuffer: sobelBuffer,
    histogram: histogramBuffer,
  })

  const edgesBindGroup = root.createBindGroup(edgesLayout, {
    sobelBuffer: sobelBuffer,
    filteredBuffer: filteredBuffer,
  })

  // Note: edgesDilatedBindGroup is the same as edgesBindGroup since we removed dilation.
  // It binds filteredBuffer (NMS output) for display — labels mode already reads filteredBuffer.
  const edgesDilatedBindGroup = edgesBindGroup

  const edgeFilterBindGroup = root.createBindGroup(edgeFilterLayout, {
    sobelBuffer: sobelBuffer,
    threshold: thresholdBuffer,
    filteredBuffer: filteredBuffer,
  })

  const edgeDilateBindGroup = root.createBindGroup(edgeDilateLayout, {
    src: filteredBuffer,
    grad: filteredBuffer,
    threshold: thresholdBuffer,
    dst: dilatedEdgeBuffer,
  })

  const histogramDisplayBindGroup = root.createBindGroup(histogramDisplayLayout, {
    histogram: histogramBuffer,
    thresholdBin: thresholdBinBuffer,
  })

  const grayRenderBindGroup = root.createBindGroup(grayRenderLayout, {
    grayBuffer: grayBuffer,
  })

  const sobelRenderBindGroup = root.createBindGroup(sobelRenderLayout, {
    sobelBuffer: sobelBuffer,
  })

  const filteredRenderBindGroup = root.createBindGroup(filteredRenderLayout, {
    filteredBuffer: filteredBuffer,
  })

  const labelVizBindGroup = root.createBindGroup(labelVizLayout, {
    labelBuffer: pointerJumpBuffer0,
  })

  return {
    context,
    histContext,
    grayTex,
    grayBuffer,
    sobelBuffer,
    filteredBuffer,
    dilatedEdgeBuffer,
    thresholdBuffer,
    thresholdBinBuffer,
    histogramBuffer,
    copyPipeline,
    copyLayoutTemplate,
    grayPipeline,
    grayTexToBufferBindGroup,
    sobelPipeline,
    sobelBindGroup,
    histogramResetPipeline,
    histogramResetBindGroup,
    histogramPipeline,
    histogramComputeBindGroup,
    edgesPipeline,
    edgesBindGroup,
    edgesDilatedBindGroup,
    edgeFilterPipeline,
    edgeFilterBindGroup,
    edgeDilatePipeline,
    edgeDilateBindGroup,
    histogramDisplayPipeline,
    histogramDisplayBindGroup,
    labelVizPipeline,
    labelVizBindGroup,
    labelVizLayout,
    sampler,
    width,
    height,
    histWidth: 512,
    histHeight: 120,
    // Grayscale render
    grayRenderPipeline,
    grayRenderBindGroup,
    grayRenderLayout,
    sobelRenderPipeline,
    sobelRenderBindGroup,
    sobelRenderLayout,
    filteredRenderPipeline,
    filteredRenderBindGroup,
    filteredRenderLayout,
    pointerJumpBuffer0,
    pointerJumpBuffer1,
    pointerJumpAtomicBuffer,
    pointerJumpInitPipeline,
    pointerJumpInitBindGroup,
    pointerJumpStepPipeline,
    pointerJumpPingPongBindGroups,
    pointerJumpLabelsToAtomicPipeline,
    pointerJumpLabelsToAtomicBindGroups,
    pointerJumpParentTightenPipeline,
    pointerJumpParentTightenBindGroups,
    pointerJumpAtomicToLabelsPipeline,
    pointerJumpAtomicToLabelsBindGroups,
    extentBuffer,
    extentResetPipeline,
    extentResetBindGroup,
    extentTrackPipeline,
    extentTrackBindGroup,
    canonicalRootBuffer,
    compactLabelBuffer,
    compactResetPipeline,
    compactResetBindGroup,
    compactClaimPipeline,
    compactClaimBindGroup,
    compactRemapPipeline,
    compactRemapBindGroup,
    // Grid visualization
    gridVizPipeline,
    gridVizLayout,
    gridVizBindGroup,
    quadCornersBuffer,
    gridVizDebugModeBuffer,
    // Frame slot pool for grid-mode frame pairing
    frameSlotPool,
  }
}

export type CameraPipeline = ReturnType<typeof createCameraPipeline>
