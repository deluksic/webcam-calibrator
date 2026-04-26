import type { TgpuRoot } from 'typegpu'
import { d } from 'typegpu'

import { createFrameSlotPool } from '@/gpu/frameSlotPool'
import type { FrameSlotPool } from '@/gpu/frameSlotPool'
import { createCompactLabelStage } from '@/gpu/pipelines/compactLabelPipeline'
import { createCopyIngest } from '@/gpu/pipelines/copyPipeline'
import { createEdgeDilateStage } from '@/gpu/pipelines/edgeDilatePipeline'
import { createEdgeFilterStage } from '@/gpu/pipelines/edgeFilterPipeline'
import { createEdgesPipeline } from '@/gpu/pipelines/edgesPipeline'
import { createExtentTrackingStage, MAX_EXTENT_COMPONENTS } from '@/gpu/pipelines/extentTrackingPipeline'
import { createFilteredRenderPipeline } from '@/gpu/pipelines/filteredRenderPipeline'
import { createGrayStage } from '@/gpu/pipelines/grayPipeline'
import { createGrayRenderPipeline } from '@/gpu/pipelines/grayRenderPipeline'
import { createGridVizStage } from '@/gpu/pipelines/gridVizPipeline'
import { createHistogramStage, HIST_HEIGHT, HIST_WIDTH } from '@/gpu/pipelines/histogramPipelines'
import { createLabelVizPipeline } from '@/gpu/pipelines/labelVizPipeline'
import { createPointerJumpLabeling } from '@/gpu/pipelines/pointerJumpPipeline'
import { createReprojectionOverlayStage } from '@/gpu/pipelines/reprojectionOverlayPipeline'
import { createSobelStage } from '@/gpu/pipelines/sobelPipeline'
import { createSobelRenderPipeline } from '@/gpu/pipelines/sobelRenderPipeline'

export type DisplayMode = 'edges' | 'nms' | 'labels' | 'grayscale' | 'debug' | 'grid'

/** Display modes that paint synchronously to the main canvas (not grid). */
export type NonGridDisplayMode = Exclude<DisplayMode, 'grid'>

// ═══════════════════════════════════════════════════════════════════════════
// PIPELINE FACTORY — stages allocate their outputs; downstream stages bind inputs
// ═══════════════════════════════════════════════════════════════════════════
export function createCameraPipeline(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  histCanvas: HTMLCanvasElement | undefined,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const context = root.configureContext({ canvas, alphaMode: 'premultiplied' })
  const grayRenderTimeBuffer = root.createBuffer(d.f32).$usage('uniform')

  const ingest = createCopyIngest(root, width, height)
  const gray = createGrayStage(root, width, height, ingest.grayTex)
  const sobel = createSobelStage(root, width, height, gray.buffer)
  const nms = createEdgeFilterStage(root, width, height, sobel.buffer)
  const dilate = createEdgeDilateStage(root, width, height, nms.filteredBuffer, nms.thresholdBuffer)
  const histogram = createHistogramStage(root, width, height, sobel.buffer, presentationFormat)
  const pointerJump = createPointerJumpLabeling(root, width, height, nms.filteredBuffer)
  const compact = createCompactLabelStage(root, width, height, MAX_EXTENT_COMPONENTS, pointerJump.pointerJumpBuffer0)
  const extent = createExtentTrackingStage(root, width, height, MAX_EXTENT_COMPONENTS, compact.compactLabelBuffer)
  const grid = createGridVizStage(root, width, height, presentationFormat)
  const reproj = createReprojectionOverlayStage(root, width, height, presentationFormat)

  const frameSlotPool: FrameSlotPool = createFrameSlotPool(root, { width, height, grayRenderTimeBuffer })

  const edges = createEdgesPipeline(root, width, height, presentationFormat, {
    sobelBuffer: sobel.buffer,
    filteredBuffer: nms.filteredBuffer,
  })
  const labelViz = createLabelVizPipeline(root, width, height, presentationFormat)
  const grayscale = createGrayRenderPipeline(root, width, height, presentationFormat, {
    grayBuffer: gray.buffer,
    timeSec: grayRenderTimeBuffer,
  })
  const sobelRender = createSobelRenderPipeline(root, width, height, presentationFormat, {
    sobelBuffer: sobel.buffer,
  })
  const filtered = createFilteredRenderPipeline(root, width, height, presentationFormat, {
    filteredBuffer: nms.filteredBuffer,
  })

  const histContext = histCanvas ? root.configureContext({ canvas: histCanvas }) : undefined

  return {
    context,
    histContext,
    width,
    height,
    histWidth: HIST_WIDTH,
    histHeight: HIST_HEIGHT,
    frameSlotPool,
    ingest,
    grayRenderTimeBuffer,
    gray,
    sobel,
    nms,
    dilate,
    histogram,
    pointerJump,
    compact,
    extent,
    grid,
    reproj,
    render: {
      edges,
      labelViz,
      grayscale,
      sobel: sobelRender,
      filtered,
    },
  }
}

export type CameraPipeline = ReturnType<typeof createCameraPipeline>
