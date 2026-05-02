import type { TgpuRoot } from 'typegpu'

import { createAxesResultsStage } from '@/gpu/pipelines/resultsAxesPipeline'
import type { AxisPassUniformGpuBuffer } from '@/gpu/pipelines/resultsAxesPipeline'
import { createMarkerResultsStage } from '@/gpu/pipelines/resultsMarkerPipeline'
import type { MarkerPassUniformGpuBuffer, MarkersCentersBuffer } from '@/gpu/pipelines/resultsMarkerPipeline'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { createTagQuadsResultsStage } from '@/gpu/pipelines/resultsTagQuadsPipeline'
import type { TagQuadsResultsBuffer } from '@/gpu/pipelines/resultsTagQuadsPipeline'
import {
  allocResultsCameraUniform,
  resultsCameraBindLayout,
  type ResultsCameraUniformGpuBuffer,
} from '@/gpu/pipelines/resultsCameraTransform'

function destroyGpuTexture(tex: GPUTexture | undefined) {
  tex?.destroy()
}

export interface ResultsCanvasPipeline {
  root: TgpuRoot
  context: GPUCanvasContext
  format: GPUTextureFormat
  cameraUniform: ResultsCameraUniformGpuBuffer
  markerUniform: MarkerPassUniformGpuBuffer
  axisUniform: AxisPassUniformGpuBuffer
  centersBuf: MarkersCentersBuffer
  tagQuadsBuf: TagQuadsResultsBuffer
  resize: (width: number, height: number) => void
  clearAttachments: () => void
  encodeScene: (markerInstances: number, tagCount: number) => void
  destroyTargets: () => void
}

/**
 * Results orbit canvas: shared camera bind group + three stages (each owns pipeline, pass bind group, encode).
 * Same responsibility split as {@link createCameraPipeline} vs {@link LiveCameraPipeline}.
 */
export function createResultsCanvasPipeline(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  format: GPUTextureFormat,
): ResultsCanvasPipeline {
  root.configureContext({ canvas, format, alphaMode: 'opaque' })
  const ctx = canvas.getContext('webgpu')
  if (!ctx) {
    throw new Error('Could not create WebGPU context.')
  }
  const context: GPUCanvasContext = ctx

  const cameraUniform = allocResultsCameraUniform(root)
  const cameraBg = root.createBindGroup(resultsCameraBindLayout, { transform: cameraUniform })

  const markerStage = createMarkerResultsStage(root, format)
  const axesStage = createAxesResultsStage(root, format)
  const tagStage = createTagQuadsResultsStage(root, format)

  let msaaColorTex: GPUTexture | undefined
  let depthTex: GPUTexture | undefined

  function resize(width: number, height: number) {
    destroyGpuTexture(msaaColorTex)
    destroyGpuTexture(depthTex)
    msaaColorTex = root.device.createTexture({
      label: 'results-msaa',
      size: [width, height, 1],
      format,
      sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    depthTex = root.device.createTexture({
      label: 'results-depth',
      size: [width, height],
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      format: 'depth24plus',
      sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
    })
  }

  function clearAttachments() {
    if (!msaaColorTex || !depthTex) {
      return
    }
    const enc = root.device.createCommandEncoder({ label: 'results clear' })
    const resolveTarget = context.getCurrentTexture().createView()
    const rp = enc.beginRenderPass({
      label: 'results clear pass',
      colorAttachments: [
        {
          view: msaaColorTex.createView(),
          resolveTarget,
          clearValue: [0.1, 0.1, 0.2, 1],
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })
    rp.end()
    root.device.queue.submit([enc.finish()])
  }

  function encodeScene(markerInstances: number, tagCount: number) {
    if (!msaaColorTex || !depthTex) {
      return
    }
    const enc = root.device.createCommandEncoder({ label: 'results frame' })
    const resolveTarget = context.getCurrentTexture().createView()
    const pass = enc.beginRenderPass({
      label: 'results scene',
      colorAttachments: [
        {
          view: msaaColorTex.createView(),
          resolveTarget,
          clearValue: [0.1, 0.1, 0.2, 1],
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: depthTex.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    })

    axesStage.encodeToPass(pass, cameraBg)
    tagStage.encodeToPass(pass, cameraBg, tagCount)
    markerStage.encodeToPass(pass, cameraBg, markerInstances)

    pass.end()
    root.device.queue.submit([enc.finish()])
  }

  function destroyTargets() {
    destroyGpuTexture(msaaColorTex)
    destroyGpuTexture(depthTex)
    msaaColorTex = undefined
    depthTex = undefined
  }

  return {
    root,
    context,
    format,
    cameraUniform,
    markerUniform: markerStage.markerUniform,
    axisUniform: axesStage.axisUniform,
    centersBuf: markerStage.centersBuf,
    tagQuadsBuf: tagStage.tagQuadsBuf,
    resize,
    clearAttachments,
    encodeScene,
    destroyTargets,
  }
}
