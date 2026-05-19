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
import { createEdgeProfileStage } from '@/gpu/pipelines/edgeProfilePipeline'
import { createEdgeProfilePlotStage } from '@/gpu/pipelines/edgeProfilePlotPipeline'
import { createEdgesPipeline } from '@/gpu/pipelines/edgesPipeline'
import { createFilteredRenderPipeline } from '@/gpu/pipelines/filteredRenderPipeline'
import { createGrayStage } from '@/gpu/pipelines/grayPipeline'
import { createGrayRenderPipeline, GrayRenderParams } from '@/gpu/pipelines/grayRenderPipeline'
import { createGridVizStage } from '@/gpu/pipelines/gridVizPipeline'
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
  const grayRenderParamsBuffer = root.createBuffer(GrayRenderParams).$usage('uniform')

  const ingest = createCopyIngest(root, width, height)
  const gray = createGrayStage(root, width, height, ingest.grayTex)
  const sobel = createSobelStage(root, width, height, gray.buffer)
  const nms = createEdgeFilterStage(root, width, height, sobel.buffer)
  const histogram = createHistogramStage(root, width, height, sobel.buffer, presentationFormat)
  const pointerJump = createPointerJumpLabeling(root, width, height, nms.filteredBuffer)
  const compact = createCompactLabelStage(root, width, height, MAX_EXTENT_COMPONENTS, pointerJump.pointerJumpBuffer0)
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
  const grayscale = createGrayRenderPipeline(root, width, height, presentationFormat, {
    grayBuffer: gray.buffer,
    params: grayRenderParamsBuffer,
  })
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
    edgeHistogram,
    orientHistViz,
    tagHistContext,
    tagHistogramDisplay,
    lineFit,
    quadHomography,
    tagDecode,
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
    undistortUniform,
    render: {
      edges,
      labelViz,
      quadsLabelViz,
      quadRejectViz,
      grayscale,
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
