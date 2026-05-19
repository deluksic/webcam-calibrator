import type { TgpuRoot } from 'typegpu'
import { d } from 'typegpu'

import { createFrameSlotPool } from '@/gpu/frameSlotPool'
import type { FrameSlotPool } from '@/gpu/frameSlotPool'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { createBoundaryFilterStage } from '@/gpu/pipelines/boundaryFilterPipeline'
import { createCompactLabelStage } from '@/gpu/pipelines/compactLabelPipeline'
import { createCopyIngest } from '@/gpu/pipelines/copyPipeline'
import { createEdgeFilterStage } from '@/gpu/pipelines/edgeFilterPipeline'
import { createEdgesPipeline } from '@/gpu/pipelines/edgesPipeline'
import { MAX_EXTENT_COMPONENTS } from '@/gpu/pipelines/compactLabelPipeline'
import { createFilteredRenderPipeline } from '@/gpu/pipelines/filteredRenderPipeline'
import { createGrayStage } from '@/gpu/pipelines/grayPipeline'
import { createGrayRenderPipeline, GrayRenderParams } from '@/gpu/pipelines/grayRenderPipeline'
import {
  createEdgeHistogramClusterStage,
  MAX_FLAT_EDGES,
} from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { createEdgeLineFitStage } from '@/gpu/pipelines/edgeLineFitPipeline'
import { createGridVizStage, createQuadCountPublishStage } from '@/gpu/pipelines/gridVizPipeline'
import { createHistogramStage, HIST_HEIGHT, HIST_WIDTH } from '@/gpu/pipelines/histogramPipelines'
import { createLabelVizPipeline } from '@/gpu/pipelines/labelVizPipeline'
import { createPointerJumpLabeling } from '@/gpu/pipelines/pointerJumpPipeline'
import { createQuadCornerHomographyStage } from '@/gpu/pipelines/quadCornerHomographyPipeline'
import { createReprojectionOverlayStage } from '@/gpu/pipelines/reprojectionOverlayPipeline'
import { createHostQuadReadbackStage } from '@/gpu/pipelines/hostQuadReadbackPipeline'
import { createTagDecodeStage } from '@/gpu/pipelines/tagDecodePipeline'
import { createSobelStage } from '@/gpu/pipelines/sobelPipeline'
import { createSobelRenderPipeline } from '@/gpu/pipelines/sobelRenderPipeline'
import {
  allocUndistortUniform,
  createUndistortPipeline,
} from '@/gpu/pipelines/undistortPipeline'

export type DisplayMode = 'edges' | 'nms' | 'labels' | 'grayscale' | 'debug' | 'grid' | 'undistort'

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
  function destroyGpuTexture(tex: GPUTexture | undefined) {
    tex?.destroy()
  }

  const context = root.configureContext({ canvas, alphaMode: 'premultiplied' })

  let msaaColorTex: GPUTexture | undefined
  function ensureMsaa(w: number, h: number) {
    if (msaaColorTex && msaaColorTex.width === w && msaaColorTex.height === h) {
      return
    }
    destroyGpuTexture(msaaColorTex)
    msaaColorTex = root.device.createTexture({
      label: 'camera-msaa',
      size: [w, h, 1],
      format: presentationFormat,
      sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
  }
  ensureMsaa(width, height)

  const grayRenderParamsBuffer = root.createBuffer(GrayRenderParams).$usage('uniform')

  const ingest = createCopyIngest(root, width, height)
  const gray = createGrayStage(root, width, height, ingest.grayTex)
  const sobel = createSobelStage(root, width, height, gray.buffer)
  const nms = createEdgeFilterStage(root, width, height, sobel.buffer)
  const histogram = createHistogramStage(root, width, height, sobel.buffer, presentationFormat)
  const pointerJump = createPointerJumpLabeling(root, width, height, nms.filteredBuffer)
  const compact = createCompactLabelStage(root, width, height, MAX_EXTENT_COMPONENTS, pointerJump.pointerJumpBuffer0)
  const boundaryFilter = createBoundaryFilterStage(
    root,
    width,
    height,
    MAX_EXTENT_COMPONENTS,
    compact.compactLabelBuffer,
  )
  const edgeHistogram = createEdgeHistogramClusterStage(
    root,
    width,
    height,
    MAX_EXTENT_COMPONENTS,
    nms.filteredBuffer,
    compact.compactLabelBuffer,
  )
  const lineFit = createEdgeLineFitStage(
    root,
    MAX_FLAT_EDGES,
    edgeHistogram.labelLineOut,
    edgeHistogram.quadPeakEdge,
    edgeHistogram.quadSourceLabelId,
    edgeHistogram.quadCount,
  )
  const grid = createGridVizStage(root, width, height, presentationFormat)
  const gridMsaa = createGridVizStage(root, width, height, presentationFormat, {
    sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
    quadCornersBuffer: grid.quadCornersBuffer,
    drawIndirectBuf: grid.drawIndirectBuf,
  })
  const grayTexView = ingest.grayTex.createView(d.texture2d(d.f32))
  const quadHomography = createQuadCornerHomographyStage(root, {
    lineOut: lineFit.lineOut,
    quadPeakEdge: edgeHistogram.quadPeakEdge,
    quadSourceLabelId: edgeHistogram.quadSourceLabelId,
    quadCount: edgeHistogram.quadCount,
    quadDataBuffer: grid.quadCornersBuffer,
  })
  const tagDecode = createTagDecodeStage(root, {
    grayTexView,
    quadDataBuffer: grid.quadCornersBuffer,
    width,
    height,
  })
  const publishQuadCount = createQuadCountPublishStage(
    root,
    edgeHistogram.quadCount,
    tagDecode.activeQuadCountBuf,
    grid.drawIndirectBuf,
  )
  const hostQuadReadback = createHostQuadReadbackStage(root, grid.quadCornersBuffer)
  const reproj = createReprojectionOverlayStage(root, width, height, presentationFormat)
  const reprojMsaa = createReprojectionOverlayStage(root, width, height, presentationFormat, {
    sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
    reprojBuffer: reproj.reprojOverlayBuffer,
    drawState: reproj.reprojOverlayDrawState,
  })

  /** One in-flight grid frame so GPU quads and CPU overlay read stay on the same frame. */
  const frameSlotPool: FrameSlotPool = createFrameSlotPool({ slotCount: 1 })

  const edges = createEdgesPipeline(root, width, height, presentationFormat, {
    sobelBuffer: sobel.buffer,
    filteredBuffer: nms.filteredBuffer,
  })
  const labelViz = createLabelVizPipeline(root, width, height, presentationFormat)
  const grayscale = createGrayRenderPipeline(root, width, height, presentationFormat, {
    grayBuffer: gray.buffer,
    params: grayRenderParamsBuffer,
  })
  const grayscaleMsaa = createGrayRenderPipeline(root, width, height, presentationFormat, {
    grayBuffer: gray.buffer,
    params: grayRenderParamsBuffer,
  }, { sampleCount: RESULTS_MSAA_SAMPLE_COUNT })
  const sobelRender = createSobelRenderPipeline(root, width, height, presentationFormat, {
    sobelBuffer: sobel.buffer,
  })
  const filtered = createFilteredRenderPipeline(root, width, height, presentationFormat, {
    filteredBuffer: nms.filteredBuffer,
  })

  const undistortUniform = allocUndistortUniform(root)
  const undistortSourceView = ingest.grayTex.createView(d.texture2d(d.f32))
  const undistort = createUndistortPipeline(root, undistortSourceView, presentationFormat, undistortUniform)

  const histContext = histCanvas ? root.configureContext({ canvas: histCanvas }) : undefined

  return {
    context,
    canvas,
    histContext,
    width,
    height,
    histWidth: HIST_WIDTH,
    histHeight: HIST_HEIGHT,
    frameSlotPool,
    ingest,
    grayRenderParamsBuffer,
    gray,
    sobel,
    nms,
    histogram,
    pointerJump,
    compact,
    boundaryFilter,
    edgeHistogram,
    lineFit,
    quadHomography,
    tagDecode,
    publishQuadCount,
    hostQuadReadback,
    grid,
    reproj,
    msaa: {
      grid: gridMsaa,
      reproj: reprojMsaa,
      /** Lazy MSAA texture. Call ensureMsaa before use if canvas may have resized. */
      get msaaColorTex() {
        return msaaColorTex
      },
      ensureMsaa,
    },
    render: {
      edges,
      labelViz,
      grayscale,
      grayscaleMsaa,
      sobel: sobelRender,
      filtered,
      undistort,
    },
    undistortUniform,
  }
}

export type CameraPipeline = ReturnType<typeof createCameraPipeline>
