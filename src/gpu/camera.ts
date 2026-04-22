import { tgpu, d } from 'typegpu'

import { type DetectedQuad, extractRegions, validateAndFilterQuads } from '@/gpu/contour'
import { createFrameSlotPool } from '@/gpu/frameSlotPool'
import type { FrameSlot, FrameSlotPool } from '@/gpu/frameSlotPool'
import {
  createCompactLabelLayouts,
  createCanonicalResetPipeline,
  createCanonicalClaimPipeline,
  createRemapLabelPipeline,
} from '@/gpu/pipelines/compactLabelPipeline'
import {
  HISTOGRAM_BINS,
  computeDispatch2d,
  COMPUTE_WORKGROUP_SIZE,
  POINTER_JUMP_ITERATIONS,
  computeThreshold,
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
  DECODED_TAG_ID_UNKNOWN,
  GridDataSchema,
  type QuadData,
  MAX_INSTANCES,
  type GridVizFailInterrogateMode,
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
import { tryComputeHomography } from '@/lib/geometry'

const { min, ceil, round } = Math

/** Max quads drawn in grid mode; must match `MAX_INSTANCES` in gridVizPipeline (buffer + draw). */
export const MAX_DETECTED_TAGS = MAX_INSTANCES

export type DisplayMode = 'edges' | 'nms' | 'labels' | 'grayscale' | 'debug' | 'grid'

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
  // console.log(`[camera] presentationFormat=${presentationFormat}, canvas=${canvas.width}x${canvas.height}`);

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
  // ═══════════════════════════════���═══════════════════════════════════════

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

export const MAX_EXTENT_COMPONENTS = 16384
export const MAX_COMPONENTS = 16384 // alias for CalibrationView readback
export const EXTENT_FIELDS = 4 // 4 fields per extent entry: minX, minY, maxX, maxY

export const MAX_U32 = 0xffffffff

export type CameraPipeline = ReturnType<typeof createCameraPipeline>

// ═══════════════════════════════════════════════════════════════════════════
// PER-FRAME PROCESSING
// ═══════════════════════════════════════════════════════════════════════════

let nextFrameId = 0

/**
 * Append the external-texture copy and the full compute chain to `enc`.
 * Does not submit, does not draw to the canvas.
 * After this call `pipeline.compactLabelBuffer` and `pipeline.filteredBuffer`
 * contain the results for this frame.
 *
 * When `slot` is provided the matching copies are also appended so that slot's
 * pinned buffers capture this frame's data. The slot is transitioned to
 * `'inflight'` and assigned a monotonically-increasing `frameId`.
 *
 * IMPORTANT: `enc` must be submitted in the same task (before returning to the
 * event loop) to avoid the GPUExternalTexture expiring.
 */
export function runCompute(
  enc: GPUCommandEncoder,
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  threshold: number,
  slot?: FrameSlot,
) {
  const copyBindGroup = root.createBindGroup(pipeline.copyLayoutTemplate, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  })

  pipeline.thresholdBuffer.write(threshold)
  pipeline.thresholdBinBuffer.write(round(threshold * 255))

  // RENDER: Copy external → grayTex (MUST happen before compute)
  pipeline.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3)

  // COMPUTE: Gray → Sobel → Histogram + NMS filter + pointer-jump labeling → compact remap → extent tracking
  const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram + filter' })
  const [wgX, wgY] = computeDispatch2d(pipeline.width, pipeline.height)
  const area = pipeline.width * pipeline.height

  pipeline.grayPipeline.with(computePass).with(pipeline.grayTexToBufferBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.sobelPipeline.with(computePass).with(pipeline.sobelBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.histogramResetPipeline
    .with(computePass)
    .with(pipeline.histogramResetBindGroup)
    .dispatchWorkgroups(HISTOGRAM_BINS)
  pipeline.histogramPipeline.with(computePass).with(pipeline.histogramComputeBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.edgeFilterPipeline.with(computePass).with(pipeline.edgeFilterBindGroup).dispatchWorkgroups(wgX, wgY)

  // Pointer-jump connected component labeling
  pipeline.pointerJumpInitPipeline
    .with(computePass)
    .with(pipeline.pointerJumpInitBindGroup)
    .dispatchWorkgroups(wgX, wgY)
  let pj = 0
  for (let s = 0; s < POINTER_JUMP_ITERATIONS; s++) {
    pipeline.pointerJumpStepPipeline
      .with(computePass)
      .with(pipeline.pointerJumpPingPongBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pj ^= 1
    pipeline.pointerJumpLabelsToAtomicPipeline
      .with(computePass)
      .with(pipeline.pointerJumpLabelsToAtomicBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pipeline.pointerJumpParentTightenPipeline
      .with(computePass)
      .with(pipeline.pointerJumpParentTightenBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pipeline.pointerJumpAtomicToLabelsPipeline
      .with(computePass)
      .with(pipeline.pointerJumpAtomicToLabelsBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
  }

  // Compact labeling: 3-pass remap of pixel-index labels to compact IDs (0..N-1).
  // Always runs — needed for extent tracking (labels must fit in extent buffer).
  // Reads pointerJumpBuffer0, which holds the converged result because
  // POINTER_JUMP_ITERATIONS is even (pj === 0 on exit).
  pipeline.compactResetPipeline
    .with(computePass)
    .with(pipeline.compactResetBindGroup)
    .dispatchWorkgroups(ceil(area / COMPUTE_WORKGROUP_SIZE))
  pipeline.compactClaimPipeline.with(computePass).with(pipeline.compactClaimBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.compactRemapPipeline.with(computePass).with(pipeline.compactRemapBindGroup).dispatchWorkgroups(wgX, wgY)

  // Extent tracking on compact labels
  pipeline.extentResetPipeline
    .with(computePass)
    .with(pipeline.extentResetBindGroup)
    .dispatchWorkgroups(ceil(MAX_EXTENT_COMPONENTS / COMPUTE_WORKGROUP_SIZE))
  pipeline.extentTrackPipeline.with(computePass).with(pipeline.extentTrackBindGroup).dispatchWorkgroups(wgX, wgY)

  computePass.end()

  // If a slot was provided, pin this frame's buffers into it for async readback.
  if (slot !== undefined) {
    pipeline.frameSlotPool.enqueueCopiesForSlot(enc, pipeline, slot)
    slot.frameId = nextFrameId++
    slot.state = 'inflight'
  }
}

/**
 * Non-grid display modes that paint synchronously to the canvas.
 * `'grid'` is intentionally excluded — use `presentGridFrame` for that path,
 * which pairs the gray snapshot with its matching detection.
 */
export type NonGridDisplayMode = Exclude<DisplayMode, 'grid'>

/**
 * Append display-mode render passes and the histogram draw to `enc`.
 * After this the caller should submit `enc`.
 * `pipeline.compactLabelBuffer` is used for label/debug modes (always the
 * compact output after `runCompute`).
 *
 * Only non-grid modes are accepted. Grid mode is handled by `presentGridFrame`
 * so that the gray snapshot and the overlay always originate from the same frame.
 */
export function presentFrame(
  enc: GPUCommandEncoder,
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  displayMode: NonGridDisplayMode,
  onError?: (msg: string) => void,
) {
  if (displayMode === 'edges') {
    try {
      pipeline.sobelRenderPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.sobelRenderBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] sobelRender failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'nms') {
    try {
      pipeline.edgesPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.edgesBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] edgesPipeline (nms) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'labels' || displayMode === 'debug') {
    const labelVizBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: pipeline.compactLabelBuffer,
    })
    try {
      pipeline.labelVizPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(labelVizBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] labelVizPipeline (${displayMode}) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else {
    try {
      pipeline.grayRenderPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.grayRenderBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] grayRender fallback failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  }

  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS)
}

/**
 * Read the extent buffer. Call periodically (not every frame) to get bounding boxes.
 */
export type ExtentRow = d.Infer<typeof ExtentEntry>

export async function readExtentBuffer(pipeline: CameraPipeline): Promise<ExtentRow[]> {
  return pipeline.extentBuffer.read()
}

/**
 * Read extent buffer and filter to valid quad candidates.
 * Uses the same bboxes that appear in the debug view.
 */
export async function readExtentDataForQuads(pipeline: CameraPipeline): Promise<ExtentRow[]> {
  const all = await pipeline.extentBuffer.read()
  return all.filter((e) => e.minX !== MAX_U32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Update quad corners buffer for grid visualization (instanced rendering)
// ─────────────────────────────────────────────────────────────────────────────

/** Write homography matrix (mat3x3) + debug data per quad to the GPU buffer.
 * H maps unit square → detected quad. Vertex shader applies: (x,y,z) = H * (u,v,1).
 * H is column-major mat3x3f: [c0.x, c0.y, c0.z, 0, c1..., c2...]
 */
export function updateQuadCornersBuffer(
  pipeline: CameraPipeline,
  quads: DetectedQuad[],
  showFallbacks: boolean = true,
): void {
  const filtered = showFallbacks ? quads : quads.filter((q) => q.hasCorners && typeof q.decodedTagId === 'number')
  const count = min(filtered.length, MAX_INSTANCES)

  const data: QuadData[] = []
  for (let i = 0; i < count; i++) {
    const quad = filtered[i]
    const H = tryComputeHomography(quad.corners)
    const debug = quad.cornerDebug
    const tagId = quad.vizTagId !== undefined ? quad.vizTagId >>> 0 : DECODED_TAG_ID_UNKNOWN

    data.push({
      homography: H
        ? d.mat3x3f(
            // transpose the matrix
            H[0],
            H[3],
            H[6],
            H[1],
            H[4],
            H[7],
            H[2],
            H[5],
            1,
          )
        : d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: debug ? debug.failureCode : 0,
        edgePixelCount: debug ? debug.edgePixelCount / 100 : 0,
        minR2: debug ? debug.minR2 : 0,
        intersectionCount: debug ? debug.intersectionCount : 0,
      },
      decodedTagId: d.u32(tagId),
    })
  }

  // Pad remaining slots with zeros
  for (let i = count; i < MAX_INSTANCES; i++) {
    data.push({
      homography: d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: 0,
        edgePixelCount: 0,
        minR2: 0,
        intersectionCount: 0,
      },
      decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
    })
  }

  pipeline.quadCornersBuffer.write(data)
}

/** Grid overlay: 0 = legacy fail colors, 1 = red highlights insufficient-edge failures, 2 = blue highlights line-fit failures. */
export function setGridVizFailInterrogate(pipeline: CameraPipeline, mode: GridVizFailInterrogateMode): void {
  pipeline.gridVizDebugModeBuffer.write(mode)
}

/**
 * Append copies of `compactLabelBuffer` and `filteredBuffer` into the given
 * staging buffers. The encoder must be submitted by the caller before calling
 * `readDetection`.
 */
export function enqueueReadbackCopies(
  enc: GPUCommandEncoder,
  pipeline: CameraPipeline,
  labelStaging: GPUBuffer,
  filteredStaging: GPUBuffer,
) {
  const labelStorage = pipeline.compactLabelBuffer.buffer as GPUBuffer
  enc.copyBufferToBuffer(labelStorage, 0, labelStaging, 0, labelStorage.size)
  const edgeStorage = pipeline.filteredBuffer.buffer as GPUBuffer
  enc.copyBufferToBuffer(edgeStorage, 0, filteredStaging, 0, edgeStorage.size)
}

/**
 * After the encoder containing the readback copies has been submitted, await
 * GPU completion, map both staging buffers, run CPU region extraction + quad
 * fitting, and read the extent buffer. Returns the same shape as the old
 * `detectContours`.
 */
export async function readDetection(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  labelStaging: GPUBuffer,
  filteredStaging: GPUBuffer,
): Promise<{
  quads: DetectedQuad[]
  extentData: ExtentRow[]
  dilatedGradients: Float32Array
  labelData: Uint32Array
}> {
  await root.device.queue.onSubmittedWorkDone()

  await labelStaging.mapAsync(GPUMapMode.READ)
  const labelDataCopy = new Uint32Array(new Uint32Array(labelStaging.getMappedRange()))
  labelStaging.unmap()

  await filteredStaging.mapAsync(GPUMapMode.READ)
  const dilatedCopy = new Float32Array(new Float32Array(filteredStaging.getMappedRange()))
  filteredStaging.unmap()

  const regions = extractRegions(labelDataCopy, pipeline.width, pipeline.height, dilatedCopy)
  const maxArea = pipeline.width * pipeline.height * 0.5
  const quads = validateAndFilterQuads(regions, dilatedCopy, labelDataCopy, pipeline.width, 400, maxArea).filter(
    (q) => q.area < pipeline.width * pipeline.height * 0.25,
  )

  const extentData: ExtentRow[] = await pipeline.extentBuffer.read()

  return { quads, extentData, dilatedGradients: dilatedCopy, labelData: labelDataCopy }
}

/**
 * Read detection results from a slot whose copies were already enqueued by
 * `runCompute`. Returns the quads, extents, and the slot itself so the caller
 * can call `swapDisplaySlot` and `presentGridFrame` after.
 */
export async function detectForSlot(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  slot: FrameSlot,
): Promise<{
  quads: ReturnType<typeof validateAndFilterQuads>
  extentData: ExtentRow[]
  dilatedGradients: Float32Array
  labelData: Uint32Array
  frameId: number
  slot: FrameSlot
}> {
  const { quads, extentData, dilatedGradients, labelData } = await readDetection(
    root,
    pipeline,
    slot.labelStaging,
    slot.filteredStaging,
  )
  return { quads, extentData, dilatedGradients, labelData, frameId: slot.frameId, slot }
}

/**
 * Encode and submit a command buffer that renders the pinned gray snapshot
 * from `slot` plus the current grid viz overlay and histogram.
 * Call this right after `updateQuadCornersBuffer` so the write and draw are
 * in the same synchronous block.
 */
export function presentGridFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  slot: FrameSlot,
): void {
  const enc = root.device.createCommandEncoder({ label: 'grid frame present' })

  pipeline.grayRenderPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
    .with(slot.grayRenderBindGroup)
    .draw(3)

  try {
    pipeline.gridVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
      .with(pipeline.gridVizBindGroup)
      .draw(4, MAX_DETECTED_TAGS)
  } catch (e) {
    console.error('[presentGridFrame] gridViz failed:', e)
  }

  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS)

  root.device.queue.submit([enc.finish()])
}

export { computeThreshold }
