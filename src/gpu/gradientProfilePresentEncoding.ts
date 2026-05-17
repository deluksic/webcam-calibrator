import type { ColorAttachment, TgpuRoot } from 'typegpu'

import type {
  GradientProfileNonGridDisplayMode,
  GradientProfilePipeline,
} from './gradientProfilePipeline'

export function encodeGradientProfileCameraPresent(
  enc: GPUCommandEncoder,
  root: TgpuRoot,
  pipeline: GradientProfilePipeline,
  displayMode: GradientProfileNonGridDisplayMode,
  timeSec: number,
): void {
  pipeline.grayRenderParamsBuffer.write({
    timeSec,
    grayScale: displayMode === 'fittedLines' ? 0.38 : 1,
  })
  const mainAttachment: ColorAttachment = { view: pipeline.cameraContext }

  if (displayMode === 'fittedLines') {
    pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)
    pipeline.render.fittedLines.encodeOverlay(enc, mainAttachment)
  } else if (displayMode === 'lineFitDebug') {
    pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 0.35 })
    pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)
    pipeline.lineFitDebug.encodeToCanvas(enc, mainAttachment)
    pipeline.render.fittedLines.encodeOverlay(enc, mainAttachment)
  } else if (displayMode === 'edges') {
    pipeline.render.sobel.encodeToCanvas(enc, mainAttachment)
  } else if (displayMode === 'nms') {
    pipeline.render.edges.encodeToCanvas(enc, mainAttachment)
  } else if (displayMode === 'labels') {
    const labelVizBindGroup = root.createBindGroup(pipeline.render.labelViz.layout, {
      labelBuffer: pipeline.compact.compactLabelBuffer,
    })
    pipeline.render.labelViz.encodeToCanvas(enc, mainAttachment, labelVizBindGroup)
  } else if (displayMode === 'quads') {
    const quadsBindGroup = root.createBindGroup(pipeline.render.quadsLabelViz.layout, {
      quadLabelBuffer: pipeline.edgeHistogram.quadLabelBuffer,
    })
    pipeline.render.quadsLabelViz.encodeToCanvas(enc, mainAttachment, quadsBindGroup)
  } else if (displayMode === 'edgeLabels') {
    const labelVizBindGroup = root.createBindGroup(pipeline.render.labelViz.layout, {
      labelBuffer: pipeline.edgeHistogram.packedEdgeLabels,
    })
    pipeline.render.labelViz.encodeToCanvas(enc, mainAttachment, labelVizBindGroup)
  } else {
    pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)
  }

  if (pipeline.histContext) {
    pipeline.histogram.encodeDisplay(enc, { view: pipeline.histContext })
  }
}

export function encodeOrientHistPresent(enc: GPUCommandEncoder, pipeline: GradientProfilePipeline): void {
  const ctx = pipeline.orientHistContext
  const viz = pipeline.orientHistViz
  if (!ctx || !viz) {
    return
  }

  viz.encodeDisplay(enc, { view: ctx, clearValue: { r: 0.1, g: 0.1, b: 0.14, a: 1 } })
}

export function encodeGradientProfilePlotPresent(
  enc: GPUCommandEncoder,
  pipeline: GradientProfilePipeline,
  profileCanvasWidth: number,
  profileCanvasHeight: number,
): void {
  pipeline.resizeProfileTargets(profileCanvasWidth, profileCanvasHeight)
  const msaa = pipeline.msaaColorTex
  if (!msaa) {
    return
  }

  const resolveTarget = pipeline.profileContext.getCurrentTexture().createView()
  const pass = enc.beginRenderPass({
    label: 'profile plot',
    colorAttachments: [
      {
        view: msaa.createView(),
        resolveTarget,
        clearValue: { r: 0.06, g: 0.07, b: 0.09, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  })
  pipeline.profilePlot.encodeClearAndDraw(
    pass,
    pipeline.plotBindGroup,
    profileCanvasWidth,
    profileCanvasHeight,
  )
  pass.end()
}
