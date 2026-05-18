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
 */
export function encodeGridPresent(
  enc: GPUCommandEncoder,
  pipeline: CameraPipeline,
  timeSec: number,
  gridInstanceCount: number,
): void {
  pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 1 })

  const mainAttachment: ColorAttachment = { view: pipeline.context }
  pipeline.render.grayscale.encodeToCanvas(enc, mainAttachment)

  if (gridInstanceCount > 0) {
    const gridAttachment: ColorAttachment = {
      view: pipeline.context,
      loadOp: 'load',
      storeOp: 'store',
    }
    try {
      pipeline.grid.encodeToCanvas(enc, gridAttachment, gridInstanceCount)
    } catch (e) {
      console.error('[encodeGridPresent] gridViz failed:', e)
    }
  }

  const reprojN = pipeline.reproj.reprojOverlayDrawState.instanceCount
  if (reprojN > 0) {
    const reprojAttachment: ColorAttachment = {
      view: pipeline.context,
      loadOp: 'load',
      storeOp: 'store',
    }
    try {
      pipeline.reproj.encodeOverlayToCanvas(enc, reprojAttachment, reprojN)
    } catch (e) {
      console.error('[encodeGridPresent] reprojection overlay failed:', e)
    }
  }

  if (pipeline.histContext) {
    pipeline.histogram.encodeDisplay(enc, { view: pipeline.histContext })
  }
}
