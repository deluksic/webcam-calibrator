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
    try {
      pipeline.render.labelViz.encodeToCanvas(enc, mainAttachment, pipeline.render.labelVizBindGroup)
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
export function encodeGridPresent(enc: GPUCommandEncoder, pipeline: CameraPipeline, timeSec: number): void {
  pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 1 })

  pipeline.msaa.ensureMsaa(pipeline.canvas.width, pipeline.canvas.height)
  const msaaView = pipeline.msaa.msaaColorTex!.createView()
  const canvasView = pipeline.context.getCurrentTexture().createView()

  const reprojN = pipeline.reproj.reprojOverlayDrawState.instanceCount

  // Pass 1: Base layer → MSAA (clear + store; grid or reproj pass resolves)
  {
    const attach: ColorAttachment = {
      view: msaaView,
      loadOp: 'clear',
      storeOp: 'store',
    }
    pipeline.render.grayscaleMsaa.encodeToCanvas(enc, attach)
  }

  // Pass 2: Grid overlay → MSAA (drawIndirect instance count from publish pass)
  {
    const gridFinal = reprojN === 0
    const attach: ColorAttachment = {
      view: msaaView,
      loadOp: 'load',
      storeOp: gridFinal ? 'discard' : 'store',
      resolveTarget: gridFinal ? canvasView : undefined,
    }
    try {
      pipeline.msaa.grid.encodeToCanvas(enc, attach, { hideNonDecoded: true })
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
