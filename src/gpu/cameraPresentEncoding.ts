import type { ColorAttachment, TgpuRoot } from 'typegpu'

import type { CameraPipeline, NonGridDisplayMode } from './cameraPipeline'

/** Non-grid display: main canvas + optional histogram canvas. */
export function encodePresentNonGrid(
  enc: GPUCommandEncoder,
  root: TgpuRoot,
  pipeline: CameraPipeline,
  displayMode: NonGridDisplayMode,
  timeSec: number,
  onError?: (msg: string) => void,
): void {
  pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 1 })
  const mainAttachment: ColorAttachment = { view: pipeline.context }

  if (displayMode === 'edges') {
    try {
      pipeline.render.sobel.encodeToCanvas(enc, mainAttachment)
    } catch (e) {
      const msg = `[camera] sobelRender failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'nms') {
    try {
      pipeline.render.edges.encodeToCanvas(enc, mainAttachment)
    } catch (e) {
      const msg = `[camera] edgesPipeline (nms) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'labels' || displayMode === 'debug') {
    const labelVizBindGroup = root.createBindGroup(pipeline.render.labelViz.layout, {
      labelBuffer: pipeline.compact.compactLabelBuffer,
    })
    try {
      pipeline.render.labelViz.encodeToCanvas(enc, mainAttachment, labelVizBindGroup)
    } catch (e) {
      const msg = `[camera] labelVizPipeline (${displayMode}) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'undistort') {
    try {
      pipeline.render.undistort.encodeToCanvas(enc, mainAttachment)
    } catch (e) {
      const msg = `[camera] undistort render failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else {
    try {
      pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)
    } catch (e) {
      const msg = `[camera] grayRender fallback failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  }

  if (pipeline.histContext) {
    pipeline.histogram.encodeDisplay(enc, { view: pipeline.histContext })
  }
}

/**
 * Grid mode: live gray buffer + GPU quad overlay + reprojection + histogram.
 * Appends to `enc`; does not submit.
 *
 * All layers render to a shared 4x MSAA texture. Only the final pass resolves
 * to the canvas so the multisampled content is preserved across passes.
 */
export function encodeGridPresent(
  enc: GPUCommandEncoder,
  pipeline: CameraPipeline,
  timeSec: number,
  gridInstanceCount: number,
): void {
  pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 1 })

  pipeline.msaa.ensureMsaa(pipeline.canvas.width, pipeline.canvas.height)
  const msaaView = pipeline.msaa.msaaColorTex!.createView()
  const gpuCtx = pipeline.canvas.getContext('webgpu')!
  const canvasView = gpuCtx.getCurrentTexture().createView()

  const reprojN = pipeline.reproj.reprojOverlayDrawState.instanceCount
  const hasOverlays = gridInstanceCount > 0 || reprojN > 0
  const hasGrid = gridInstanceCount > 0

  // Pass 1: Base layer → MSAA (clear + store, no resolve)
  {
    const attach: ColorAttachment = {
      view: msaaView,
      loadOp: 'clear',
      storeOp: hasOverlays ? 'store' : 'discard',
      resolveTarget: hasOverlays ? undefined : canvasView,
    }
    pipeline.render.grayscaleMsaa.encodeToCanvas(enc, attach)
  }

  // Pass 2: Grid overlay → MSAA (load + store, no resolve)
  if (hasGrid) {
    const gridFinal = reprojN === 0
    const attach: ColorAttachment = {
      view: msaaView,
      loadOp: 'load',
      storeOp: gridFinal ? 'discard' : 'store',
      resolveTarget: gridFinal ? canvasView : undefined,
    }
    try {
      pipeline.msaa.grid.encodeToCanvas(enc, attach, gridInstanceCount, { hideNonDecoded: true })
    } catch (e) {
      console.error('[encodeGridPresent] gridViz failed:', e)
    }
  }

  // Pass 3: Reprojection overlay → MSAA (load + discard + resolve to canvas)
  if (reprojN > 0) {
    const attach: ColorAttachment = {
      view: msaaView,
      loadOp: 'load',
      storeOp: 'discard',
      resolveTarget: canvasView,
    }
    try {
      pipeline.msaa.reproj.encodeOverlayToCanvas(enc, attach, reprojN)
    } catch (e) {
      console.error('[encodeGridPresent] reprojection overlay failed:', e)
    }
  }

  if (pipeline.histContext) {
    pipeline.histogram.encodeDisplay(enc, { view: pipeline.histContext })
  }
}
