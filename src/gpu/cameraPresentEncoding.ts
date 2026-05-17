import type { ColorAttachment, TgpuRoot } from 'typegpu'

import type { FrameSlot } from '@/gpu/frameSlotPool'

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
 * Grid mode: gray from slot, overlay, reprojection, histogram; submits the encoder.
 */
export function encodeAndSubmitGridPresent(
  root: TgpuRoot,
  pipeline: CameraPipeline,
  slot: FrameSlot,
  timeSec: number,
): void {
  const enc = root.device.createCommandEncoder({ label: 'grid frame present' })
  pipeline.grayRenderParamsBuffer.write({ timeSec, grayScale: 1 })

  const loadMain: ColorAttachment = { view: pipeline.context, loadOp: 'load', storeOp: 'store' }
  pipeline.render.grayscale.encodeToCanvas(enc, loadMain, slot.grayRenderBindGroup)

  try {
    pipeline.grid.encodeToCanvas(enc, loadMain)
  } catch (e) {
    console.error('[presentGridFrame] gridViz failed:', e)
  }

  const reprojN = pipeline.reproj.reprojOverlayDrawState.instanceCount
  if (reprojN > 0) {
    try {
      pipeline.reproj.encodeOverlayToCanvas(enc, loadMain, reprojN)
    } catch (e) {
      console.error('[presentGridFrame] reprojection overlay failed:', e)
    }
  }

  if (pipeline.histContext) {
    pipeline.histogram.encodeDisplay(enc, { view: pipeline.histContext })
  }

  root.device.queue.submit([enc.finish()])
}
