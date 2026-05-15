import type { TgpuRoot } from 'typegpu'

import {
  createDistortionFieldStage,
  type DistortionColoringMode,
  type DistortionUniformGpuBuffer,
} from '@/gpu/pipelines/distortionFieldPipeline'

export interface ResultsDistortionCanvasPipeline {
  root: TgpuRoot
  context: GPUCanvasContext
  format: GPUTextureFormat
  distortionUniform: DistortionUniformGpuBuffer
  encodeDistortionFrame: () => void
  destroyTargets: () => void
}

/** Second results canvas: fullscreen distortion field (no MSAA). */
export function createResultsDistortionCanvasPipeline(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  format: GPUTextureFormat,
  opts?: { coloring?: DistortionColoringMode },
): ResultsDistortionCanvasPipeline {
  root.configureContext({ canvas, format, alphaMode: 'opaque' })
  const ctx = canvas.getContext('webgpu')
  if (!ctx) {
    throw new Error('Could not create WebGPU context.')
  }
  const context: GPUCanvasContext = ctx

  const stage = createDistortionFieldStage(root, format, { coloring: opts?.coloring })

  function encodeDistortionFrame() {
    const enc = root.device.createCommandEncoder({ label: 'distortion field' })
    const view = context.getCurrentTexture().createView()
    stage.encodeToCanvas(enc, {
      view,
      clearValue: [0.02, 0.02, 0.05, 1],
      loadOp: 'clear',
      storeOp: 'store',
    })
    root.device.queue.submit([enc.finish()])
  }

  return {
    root,
    context,
    format,
    distortionUniform: stage.uniform,
    encodeDistortionFrame,
    destroyTargets: () => {},
  }
}
