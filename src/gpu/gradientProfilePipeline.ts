import type { TgpuRoot } from 'typegpu'
import { d } from 'typegpu'

import { MAX_EDGES_PER_LABEL } from '@/gpu/lineFitThresholds'
import { createCompactLabelStage } from '@/gpu/pipelines/compactLabelPipeline'
import { MAX_EXTENT_COMPONENTS } from '@/gpu/pipelines/compactLabelPipeline'
import { createCopyIngest } from '@/gpu/pipelines/copyPipeline'
import { createEdgeFilterStage } from '@/gpu/pipelines/edgeFilterPipeline'
import { createEdgeFittedLineOverlayStage } from '@/gpu/pipelines/edgeFittedLineOverlayPipeline'
import { createEdgeHistogramClusterStage, MAX_FLAT_EDGES } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { createEdgeLineFitStage } from '@/gpu/pipelines/edgeLineFitPipeline'
import { createBoundaryFilterStage } from '@/gpu/pipelines/boundaryFilterPipeline'
import { createEdgeProfileStage } from '@/gpu/pipelines/edgeProfilePipeline'
import { createEdgeProfilePlotStage } from '@/gpu/pipelines/edgeProfilePlotPipeline'
import { createEdgesPipeline } from '@/gpu/pipelines/edgesPipeline'
import { createFilteredRenderPipeline } from '@/gpu/pipelines/filteredRenderPipeline'
import { createGrayStage } from '@/gpu/pipelines/grayPipeline'
import { createGrayRenderPipeline, GrayRenderParams } from '@/gpu/pipelines/grayRenderPipeline'
import { createGridVizStage, createQuadCountPublishStage } from '@/gpu/pipelines/gridVizPipeline'
import { createHistogramStage, HIST_HEIGHT, HIST_WIDTH } from '@/gpu/pipelines/histogramPipelines'
import {
  createLabelVizPipeline,
  createQuadRejectVizPipeline,
  createQuadsLabelVizPipeline,
} from '@/gpu/pipelines/labelVizPipeline'
import { createLineFitDebugStage } from '@/gpu/pipelines/lineFitDebugPipeline'
import { createOrientHistVizStage } from '@/gpu/pipelines/orientHistVizPipeline'
import { createPointerJumpLabeling } from '@/gpu/pipelines/pointerJumpPipeline'
import { createQuadCornerHomographyStage } from '@/gpu/pipelines/quadCornerHomographyPipeline'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { createSobelStage } from '@/gpu/pipelines/sobelPipeline'
import { createSobelRenderPipeline } from '@/gpu/pipelines/sobelRenderPipeline'
import { createTagDecodeStage, createTagHistogramDisplayStage } from '@/gpu/pipelines/tagDecodePipeline'
import { createHostQuadReadbackStage } from '@/gpu/pipelines/hostQuadReadbackPipeline'
import { allocUndistortUniform, createUndistortPipeline } from '@/gpu/pipelines/undistortPipeline'

export type GradientProfileDisplayMode =
  | 'edges'
  | 'nms'
  | 'labels'
  | 'quads'
  | 'quadReject'
  | 'edgeLabels'
  | 'grayscale'
  | 'undistort'
  | 'fittedLines'
  | 'lineRejects'
  | 'quadGrid'

export type GradientProfileNonGridDisplayMode = GradientProfileDisplayMode

/** Toolbar order: matches `encodeGradientProfileCompute` (ingest → … → tag decode). */
export const GRADIENT_PROFILE_DISPLAY_MODES: ReadonlyArray<{
  mode: GradientProfileDisplayMode
  label: string
  title?: string
}> = [
  { mode: 'grayscale', label: 'Gray' },
  { mode: 'undistort', label: 'Undistort' },
  { mode: 'edges', label: 'Edges' },
  { mode: 'nms', label: 'NMS' },
  { mode: 'labels', label: 'Labels' },
  { mode: 'edgeLabels', label: 'Edge labels' },
  { mode: 'lineRejects', label: 'Line rejects' },
  { mode: 'fittedLines', label: 'Lines' },
  { mode: 'quadReject', label: 'Quad reject', title: 'Labels colored by quad registration outcome' },
  { mode: 'quads', label: 'Quads' },
  { mode: 'quadGrid', label: 'Quad grid' },
]

function destroyGpuTexture(tex: GPUTexture | undefined) {
  tex?.destroy()
}

export function createGradientProfilePipeline(
  root: TgpuRoot,
  cameraCanvas: HTMLCanvasElement,
  orientHistCanvas: HTMLCanvasElement | undefined,
  profileCanvas: HTMLCanvasElement,
  histCanvas: HTMLCanvasElement | undefined,
  tagHistCanvas: HTMLCanvasElement | undefined,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const cameraContext = root.configureContext({ canvas: cameraCanvas, alphaMode: 'premultiplied' })
  const profileContext = root.configureContext({ canvas: profileCanvas, alphaMode: 'premultiplied' })

  let cameraMsaaTex: GPUTexture | undefined
  function ensureCameraMsaa(w: number, h: number) {
    if (cameraMsaaTex && cameraMsaaTex.width === w && cameraMsaaTex.height === h) {
      return
    }
    destroyGpuTexture(cameraMsaaTex)
    cameraMsaaTex = root.device.createTexture({
      label: 'gradient-profile-camera-msaa',
      size: [w, h, 1],
      format: presentationFormat,
      sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
  }
  ensureCameraMsaa(width, height)

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
  const orientHistViz = orientHistCanvas
    ? createOrientHistVizStage(
        root,
        edgeHistogram.labelClusters,
        edgeHistogram.quadSourceLabelId,
        edgeHistogram.quadCount,
        presentationFormat,
      )
    : undefined
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
  const hostQuadReadback = createHostQuadReadbackStage(root, grid.quadCornersBuffer)
  const publishQuadCount = createQuadCountPublishStage(
    root,
    edgeHistogram.quadCount,
    tagDecode.activeQuadCountBuf,
    grid.drawIndirectBuf,
  )
  const tagHistContext = tagHistCanvas
    ? root.configureContext({ canvas: tagHistCanvas, alphaMode: 'premultiplied' })
    : undefined
  const tagHistogramDisplay = tagHistContext
    ? createTagHistogramDisplayStage(root, tagDecode.histBuf, tagDecode.thresholdBuf, presentationFormat)
    : undefined
  const profile = createEdgeProfileStage(
    root,
    width,
    height,
    MAX_FLAT_EDGES,
    gray.buffer,
    nms.filteredBuffer,
    edgeHistogram.packedEdgeLabels,
    edgeHistogram.labelToQuadId,
    lineFit.lineOut,
  )
  const profilePlot = createEdgeProfilePlotStage(root, presentationFormat)
  const plotBindGroup = profilePlot.createPlotBindGroup(lineFit.lineOut, profile.profileAvg, profile.profileBuckets)

  const edges = createEdgesPipeline(root, width, height, presentationFormat, {
    sobelBuffer: sobel.buffer,
    filteredBuffer: nms.filteredBuffer,
  })
  const labelViz = createLabelVizPipeline(root, width, height, presentationFormat)
  const quadsLabelViz = createQuadsLabelVizPipeline(root, width, height, presentationFormat)
  const quadRejectViz = createQuadRejectVizPipeline(root, width, height, presentationFormat)
  const labelVizBindGroup = root.createBindGroup(labelViz.layout, {
    labelBuffer: compact.compactLabelBuffer,
  })
  const quadsBindGroup = root.createBindGroup(quadsLabelViz.layout, {
    quadLabelBuffer: edgeHistogram.quadLabelBuffer,
  })
  const quadRejectBindGroup = root.createBindGroup(quadRejectViz.layout, {
    compactLabels: compact.compactLabelBuffer,
    labelToQuadId: edgeHistogram.labelToQuadId,
    labelQuadReject: edgeHistogram.labelQuadReject,
  })
  const edgeLabelsBindGroup = root.createBindGroup(labelViz.layout, {
    labelBuffer: edgeHistogram.packedEdgeLabels,
  })
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
  const fittedLineInstances = MAX_EXTENT_COMPONENTS * MAX_EDGES_PER_LABEL
  const fittedLines = createEdgeFittedLineOverlayStage(
    root,
    width,
    height,
    presentationFormat,
    edgeHistogram.labelLineOut,
    edgeHistogram.labelToQuadId,
    edgeHistogram.quadPeakEdge,
    fittedLineInstances,
  )
  const fittedLinesMsaa = createEdgeFittedLineOverlayStage(
    root,
    width,
    height,
    presentationFormat,
    edgeHistogram.labelLineOut,
    edgeHistogram.labelToQuadId,
    edgeHistogram.quadPeakEdge,
    fittedLineInstances,
    { sampleCount: RESULTS_MSAA_SAMPLE_COUNT },
  )
  const lineRejects = createLineFitDebugStage(
    root,
    width,
    height,
    presentationFormat,
    nms.filteredBuffer,
    compact.compactLabelBuffer,
    edgeHistogram.labelClusters,
    edgeHistogram.labelLineReduce,
    edgeHistogram.labelInlierStats,
    edgeHistogram.labelLineOut,
  )

  const orientHistContext = orientHistCanvas
    ? root.configureContext({ canvas: orientHistCanvas, alphaMode: 'premultiplied' })
    : undefined
  const histContext = histCanvas ? root.configureContext({ canvas: histCanvas }) : undefined

  let msaaColorTex: GPUTexture | undefined
  let profileWidth = 0
  let profileHeight = 0

  function resizeProfileTargets(w: number, h: number) {
    if (w === profileWidth && h === profileHeight && msaaColorTex) {
      return
    }
    profileWidth = w
    profileHeight = h
    destroyGpuTexture(msaaColorTex)
    msaaColorTex = root.device.createTexture({
      label: 'profile-plot-msaa',
      size: [w, h, 1],
      format: presentationFormat,
      sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
  }

  return {
    cameraContext,
    cameraCanvas,
    orientHistContext,
    profileContext,
    histContext,
    width,
    height,
    histWidth: HIST_WIDTH,
    histHeight: HIST_HEIGHT,
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
    orientHistViz,
    tagHistContext,
    tagHistogramDisplay,
    lineFit,
    quadHomography,
    tagDecode,
    hostQuadReadback,
    publishQuadCount,
    grid,
    profile,
    profilePlot,
    plotBindGroup,
    validEdgeCount: lineFit.validEdgeCount,
    lineRejects,
    resizeProfileTargets,
    get msaaColorTex() {
      return msaaColorTex
    },
    msaa: {
      grid: gridMsaa,
      fittedLines: fittedLinesMsaa,
      get cameraMsaaTex() {
        return cameraMsaaTex
      },
      ensureCameraMsaa,
    },
    undistortUniform,
    render: {
      edges,
      labelViz,
      quadsLabelViz,
      quadRejectViz,
      labelVizBindGroup,
      quadsBindGroup,
      quadRejectBindGroup,
      edgeLabelsBindGroup,
      grayscale,
      grayscaleMsaa,
      sobel: sobelRender,
      filtered,
      undistort,
      fittedLines,
    },
    destroyProfileTargets() {
      destroyGpuTexture(msaaColorTex)
      msaaColorTex = undefined
      profileWidth = 0
      profileHeight = 0
    },
  }
}

export type GradientProfilePipeline = ReturnType<typeof createGradientProfilePipeline>
