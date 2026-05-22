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

  if (displayMode === 'quadGrid') {
    pipeline.msaa.ensureCameraMsaa(pipeline.cameraCanvas.width, pipeline.cameraCanvas.height)
    const msaaView = pipeline.msaa.cameraMsaaTex!.createView()
    const canvasView = pipeline.cameraContext.getCurrentTexture().createView()
    // Pass 1: Base → MSAA (clear + store; grid pass resolves)
    const baseAttach: ColorAttachment = {
      view: msaaView,
      loadOp: 'clear',
      storeOp: 'store',
    }
    pipeline.render.grayscaleMsaa.encodeToCanvas(enc, baseAttach)

    // Pass 2: Grid overlay → MSAA (instance count from drawIndirect, published post-compute)
    const gridAttach: ColorAttachment = {
      view: msaaView,
      loadOp: 'load',
      storeOp: 'discard',
      resolveTarget: canvasView,
    }
    pipeline.msaa.grid.encodeToCanvas(enc, gridAttach)
  } else if (displayMode === 'fittedLines') {
    pipeline.msaa.ensureCameraMsaa(pipeline.cameraCanvas.width, pipeline.cameraCanvas.height)
    const msaaView = pipeline.msaa.cameraMsaaTex!.createView()
    const canvasView = pipeline.cameraContext.getCurrentTexture().createView()

    // Pass 1: Base → MSAA (clear + store, no resolve)
    const baseAttach: ColorAttachment = {
      view: msaaView,
      loadOp: 'clear',
      storeOp: 'store',
    }
    pipeline.render.grayscaleMsaa.encodeToCanvas(enc, baseAttach)

    // Pass 2: Fitted lines → MSAA (load + discard + resolve)
    const linesAttach: ColorAttachment = {
      view: msaaView,
      loadOp: 'load',
      storeOp: 'discard',
      resolveTarget: canvasView,
    }
    pipeline.msaa.fittedLines.encodeOverlay(enc, linesAttach)
  } else {
    const mainAttachment: ColorAttachment = { view: pipeline.cameraContext }
    if (displayMode === 'lineRejects') {
      pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 0.35 })
      pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)
      pipeline.lineRejects.encodeToCanvas(enc, mainAttachment)
    } else if (displayMode === 'edges') {
      pipeline.render.sobel.encodeToCanvas(enc, mainAttachment)
    } else if (displayMode === 'nms') {
      pipeline.render.edges.encodeToCanvas(enc, mainAttachment)
    } else if (displayMode === 'labels') {
    pipeline.render.labelViz.encodeToCanvas(enc, mainAttachment, pipeline.render.labelVizBindGroup)
  } else if (displayMode === 'quads') {
    pipeline.render.quadsLabelViz.encodeToCanvas(enc, mainAttachment, pipeline.render.quadsBindGroup)
  } else if (displayMode === 'quadReject') {
    pipeline.render.quadRejectViz.encodeToCanvas(enc, mainAttachment, pipeline.render.quadRejectBindGroup)
  } else if (displayMode === 'edgeLabels') {
    pipeline.render.labelViz.encodeToCanvas(enc, mainAttachment, pipeline.render.edgeLabelsBindGroup)
  } else if (displayMode === 'undistort') {
    pipeline.render.undistort.encodeToCanvas(enc, mainAttachment)
    } else if (displayMode === 'voteDebug') {
      pipeline.voteDebugDisplay.encodeDisplay(enc, mainAttachment)
    } else {
      pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)
    }
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
