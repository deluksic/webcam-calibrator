import * as ort from 'onnxruntime-web/webgpu'
import { common, d, std, tgpu } from 'typegpu'
import type { TgpuRoot } from 'typegpu'
import { atomicAdd, atomicStore, intdiv } from 'typegpu/std'

import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { createRoiBackgroundStage } from '@/gpu/roiLab/roiBackgroundPipeline'
import { createRoiCircleStage } from '@/gpu/roiLab/roiCirclePipeline'
import { copyBindGroupLayout, createCopyPipeline } from '@/gpu/pipelines/copyPipeline'
import { createRoiComponentsStage, type RoiComponentPoint } from '@/gpu/roiLab/roiComponentsPipeline'
import { createRoiGraphStage, type RoiGraphData } from '@/gpu/roiLab/roiGraphPipeline'
import {
  analyzeRoiHistogram,
  ROI_HISTOGRAM_BINS,
  type RoiHistogramAnalysis,
} from '@/lib/roiHistogram'

const { performance } = globalThis

/**
 * Browser-side ROI segmentation.
 *
 * ONNX graph: normalized grayscale 1×H×192 -> mask probability 1×H×192.
 * The browser computes per-image mean/std from a small canvas readback and
 * feeds `0.4 + 0.2 * z` (z = (luma - mean) / std); the graph's fixed
 * normalization recovers z. This matches the per-image standardization used
 * for training without adding reduction ops to the ONNX graph.
 *
 * The ONNX Runtime WebGPU EP shares the TypeGPU device and queue. The compute
 * pass fills the input buffer, `session.run` reads/writes GPU buffers, then a
 * second compute pass turns the output buffer into a sampled mask texture.
 */

export const ROI_MODEL_WIDTH = 256

/**
 * Histogram readbacks are informational: the segmentation shader always uses
 * the last thresholds already on the GPU. Sampling a new histogram every few
 * frames keeps the threshold fresh without putting a mapAsync in the render
 * path (which would make the blob overlay lag the camera).
 */
const ROI_HISTOGRAM_FRAME_INTERVAL = 4
/** Extra tightening of the black/white cuts, as a fraction of each gray-side gap. */
const ROI_CUT_MARGIN_FRACTION = 0.2

const prepareParams = d.struct({
  size: d.vec2f,
  stats: d.vec2f,
})

const roiHistogramSchema = d.arrayOf(d.atomic(d.u32), ROI_HISTOGRAM_BINS)

const histogramResetLayout = tgpu.bindGroupLayout({
  histogram: { storage: roiHistogramSchema, access: 'mutable' },
})

const histogramParams = d.struct({
  /** Gain applied to (luma - local background) before binning. */
  residualGain: d.f32,
  _pad: d.vec3f,
})

const histogramAccumulateLayout = tgpu.bindGroupLayout({
  cameraTexture: { texture: d.texture2d() },
  maskTexture: { texture: d.texture2d() },
  backgroundTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
  histogram: { storage: roiHistogramSchema, access: 'mutable' },
  params: { uniform: histogramParams },
})

const resetRoiHistogram = (bin: number) => {
  'use gpu'
  atomicStore(histogramResetLayout.$.histogram[d.u32(bin)]!, d.u32(0))
}

/**
 * Full-resolution flattened-luma histogram, restricted to the ROI mask. The
 * camera texture and illumination background are the same ones used by blob
 * classification, so histogram cuts and components see the same signal.
 */
const accumulateRoiHistogram = (x: number, y: number) => {
  'use gpu'
  const px = d.u32(x)
  const py = d.u32(y)
  const size = std.textureDimensions(histogramAccumulateLayout.$.cameraTexture)
  const maskSize = std.textureDimensions(histogramAccumulateLayout.$.maskTexture)
  const maskX = intdiv(px * maskSize.x, size.x)
  const maskY = intdiv(py * maskSize.y, size.y)
  const mask = std.textureLoad(
    histogramAccumulateLayout.$.maskTexture,
    d.vec2u(maskX, maskY),
    d.i32(0),
  ).r
  if (mask < d.f32(0.5)) {
    return
  }
  const color = std.textureLoad(
    histogramAccumulateLayout.$.cameraTexture,
    d.vec2u(px, py),
    d.i32(0),
  )
  const uv = (d.vec2f(px, py) + d.vec2f(0.5, 0.5)).div(d.vec2f(size.x, size.y))
  const background = std.textureSampleLevel(
    histogramAccumulateLayout.$.backgroundTexture,
    histogramAccumulateLayout.$.sampler,
    uv,
    d.f32(0),
  ).r
  const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  const flattened = std.clamp(
    d.f32(0.5) + (luma - background) * histogramAccumulateLayout.$.params.residualGain,
    d.f32(0),
    d.f32(1),
  )
  const bin = d.u32(flattened * d.f32(255) + d.f32(0.5))
  atomicAdd(histogramAccumulateLayout.$.histogram[bin]!, d.u32(1))
}

const prepareLayout = tgpu.bindGroupLayout({
  inputTexture: { externalTexture: d.textureExternal() },
  outputBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  sampler: { sampler: 'filtering' },
  params: { uniform: prepareParams },
})

const prepareAccess = tgpu.accessor(prepareParams)

const prepareModelInput = (x: number, y: number) => {
  'use gpu'
  const params = prepareAccess.$
  const uv = (d.vec2f(x, y) + d.vec2f(0.5, 0.5)).div(params.size)
  const color = std.textureSampleBaseClampToEdge(
    prepareLayout.$.inputTexture,
    prepareLayout.$.sampler,
    uv,
  )
  const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  prepareLayout.$.outputBuffer[y * d.u32(params.size.x) + x] =
    (luma - params.stats.x) / params.stats.y
}

const maskToTextureLayout = tgpu.bindGroupLayout({
  outputBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  maskTexture: {
    storageTexture: d.textureStorage2d('rgba8unorm', 'write-only'),
  },
})

const overlayLayout = tgpu.bindGroupLayout({
  cameraTexture: { texture: d.texture2d() },
  maskTexture: { texture: d.texture2d() },
  backgroundTexture: { texture: d.texture2d() },
  modelInputTexture: { texture: d.texture2d() },
  labelTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
})

const overlayParams = d.struct({
  mode: d.u32,
  /** Bins below this luma cut are black blobs; 0 disables the class. */
  blackCut: d.u32,
  /** Bins at or above this luma cut are white blobs; 256 disables the class. */
  whiteCut: d.u32,
  valid: d.u32,
})

const overlayAccess = tgpu.accessor(overlayParams)

const viewParams = d.struct({
  zoom: d.f32,
  panX: d.f32,
  panY: d.f32,
  frameWidth: d.f32,
  frameHeight: d.f32,
})

const viewAccess = tgpu.accessor(viewParams)

const debugInputLayout = tgpu.bindGroupLayout({
  inputBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  inputTexture: {
    storageTexture: d.textureStorage2d('rgba8unorm', 'write-only'),
  },
})

const debugInputToTexture = (x: number, y: number) => {
  'use gpu'
  const width = std.textureDimensions(debugInputLayout.$.inputTexture).x
  const value = debugInputLayout.$.inputBuffer[y * width + x]!
  const display = std.smoothstep(d.f32(-2.5), d.f32(2.5), value)
  std.textureStore(
    debugInputLayout.$.inputTexture,
    d.vec2u(x, y),
    d.vec4f(display, display, display, 1),
  )
}

const maskToTexture = (x: number, y: number) => {
  'use gpu'
  const width = std.textureDimensions(maskToTextureLayout.$.maskTexture).x
  const value = maskToTextureLayout.$.outputBuffer[y * width + x]!
  std.textureStore(
    maskToTextureLayout.$.maskTexture,
    d.vec2u(x, y),
    d.vec4f(value, value, value, 1),
  )
}

const overlayFragment = ({ uv }: { uv: d.v2f }) => {
  'use gpu'
  const view = viewAccess.$
  const srcUv = d.vec2f(
    (uv.x - view.panX / view.frameWidth) / view.zoom,
    (uv.y - view.panY / view.frameHeight) / view.zoom,
  )
  if (srcUv.x < d.f32(0) || srcUv.x > d.f32(1) || srcUv.y < d.f32(0) || srcUv.y > d.f32(1)) {
    return d.vec4f(d.f32(0.03), d.f32(0.03), d.f32(0.04), d.f32(1))
  }
  if (overlayAccess.$.mode === d.u32(7)) {
    const px = d.u32(std.clamp(srcUv.x * view.frameWidth, d.f32(0), view.frameWidth - d.f32(1)))
    const py = d.u32(std.clamp(srcUv.y * view.frameHeight, d.f32(0), view.frameHeight - d.f32(1)))
    return std.textureLoad(overlayLayout.$.labelTexture, d.vec2u(px, py), d.i32(0))
  }
  const video = std.textureSampleLevel(
    overlayLayout.$.cameraTexture,
    overlayLayout.$.sampler,
    srcUv,
    d.f32(0),
  )
  if (overlayAccess.$.mode === d.u32(3) || overlayAccess.$.mode === d.u32(9)) {
    return video
  }
  const mask = std.textureSampleLevel(
    overlayLayout.$.maskTexture,
    overlayLayout.$.sampler,
    srcUv,
    d.f32(0),
  ).r
  if (overlayAccess.$.mode === d.u32(1)) {
    return d.vec4f(mask, mask, mask, 1)
  }
  if (overlayAccess.$.mode === d.u32(2)) {
    const modelInput = std.textureSampleLevel(
      overlayLayout.$.modelInputTexture,
      overlayLayout.$.sampler,
      srcUv,
      d.f32(0),
    ).r
    return d.vec4f(modelInput, modelInput, modelInput, 1)
  }
  if (
    overlayAccess.$.mode === d.u32(5) ||
    overlayAccess.$.mode === d.u32(6) ||
    overlayAccess.$.mode === d.u32(8)
  ) {
    const luma = std.dot(video.rgb, d.vec3f(0.299, 0.587, 0.114))
    const dim = mask >= d.f32(0.5) ? luma : luma * d.f32(0.3)
    return d.vec4f(dim, dim, dim, d.f32(1))
  }
  if (overlayAccess.$.mode === d.u32(4)) {
    const luma = std.dot(video.rgb, d.vec3f(0.299, 0.587, 0.114))
    if (mask < d.f32(0.5)) {
      const dim = luma * d.f32(0.3)
      return d.vec4f(dim, dim, dim, 1)
    }
    const background = std.textureSampleLevel(
      overlayLayout.$.backgroundTexture,
      overlayLayout.$.sampler,
      srcUv,
      d.f32(0),
    ).r
    const flattened = std.clamp(d.f32(0.5) + (luma - background), d.f32(0), d.f32(1))
    const bin = d.u32(flattened * d.f32(255) + d.f32(0.5))
    if (overlayAccess.$.valid !== d.u32(0) && bin < overlayAccess.$.blackCut) {
      return d.vec4f(d.f32(0.16), d.f32(0.46), d.f32(1.0), d.f32(1))
    }
    if (overlayAccess.$.valid !== d.u32(0) && bin >= overlayAccess.$.whiteCut) {
      return d.vec4f(d.f32(1.0), d.f32(0.68), d.f32(0.12), d.f32(1))
    }
    return d.vec4f(luma, luma, luma, 1)
  }
  return video
}

export type RoiHistogramUpdate = {
  counts: Uint32Array
  analysis: RoiHistogramAnalysis
}

export type RoiStepTimings = {
  imageStatsMs: number
  statsAgeFrames: number
  statsRefreshed: boolean
  preprocessMs: number
  queueWaitMs: number
  /** `await session.run(...)` only. */
  inferenceRunMs: number
  /** Queue drain immediately after ORT inference. */
  inferenceDrainMs: number
  /** Backward-compatible total = run + drain. */
  inferenceMs: number
  cameraCopyMs: number
  debugInputMs: number
  maskMs: number
  backgroundMs: number
  componentsEncodeMs: number
  readbackMs: number
  readbackComponents: number
  readbackBytes: number
  readbackTruncated: boolean
  identityMs: number
  renderMs: number
  /** Wait for all frame GPU work before histogram submission. */
  frameDrainMs: number
  /** Histogram reset/accumulate submission (and every 4th frame analysis scheduling). */
  histogramMs: number
  /** Sum of the non-overlapping step intervals (should be close to totalMs). */
  accountedMs: number
  /** totalMs - accountedMs, usually performance.now overhead. */
  overheadMs: number
  totalMs: number
  frameId: number
  /** GPU pass timings from TypeGPU's timestamp-query profiling; may lag one frame. */
  gpuMs: Readonly<Record<string, number>>
}

export type RoiLabPipeline = {
  processFrame: (video: HTMLVideoElement) => Promise<{
    inferenceMs: number
    totalMs: number
    readbackMs: number
    identityMs: number
    frameId: number
    steps: RoiStepTimings
  }>
  resize: (width: number, height: number) => void
  setViewMode: (mode: 1 | 2 | 3 | 4 | 6 | 7 | 8 | 9) => void
  setViewport: (zoom: number, panX: number, panY: number) => void
  setPaused: (paused: boolean) => void
  dumpStats: () => Promise<{
    input: { min: number; mean: number; max: number }
    output: { min: number; mean: number; max: number }
  }>
  destroy: () => void
}

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_resolve, reject) => {
      setTimeout(() => reject(new Error(message)), ms)
    }),
  ])
}

// ORT WebGPU allows only one session to be created at a time. Rapid camera
// resolution changes can otherwise overlap session creation and fail with
// "another WebGPU EP inference session is being created".
let sessionSlot: Promise<void> = Promise.resolve()

async function takeSessionSlot(): Promise<() => void> {
  const previous = sessionSlot
  let freeSlot!: () => void
  sessionSlot = new Promise<void>((resolve) => {
    freeSlot = resolve
  })
  await previous
  return freeSlot
}

export async function createRoiLabPipeline(
  root: TgpuRoot,
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  frameWidth: number,
  frameHeight: number,
  modelHeight: number,
  modelUrl: string,
  onStatus?: (message: string) => void,
  onHistogram?: (update: RoiHistogramUpdate) => void,
  onIdentity?: (points: readonly RoiComponentPoint[], frameId: number) => RoiGraphData | undefined,
): Promise<RoiLabPipeline> {
  const sampler = root.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  })

  const modelWidth = ROI_MODEL_WIDTH
  const maskWidth = Math.max(1, Math.floor(modelWidth / 2))
  const maskHeight = Math.max(1, Math.floor(modelHeight / 2))
  const cameraWidth = Math.max(1, Math.floor(frameWidth))
  const cameraHeight = Math.max(1, Math.floor(frameHeight))

  let overlayMode = 7
  let blackCut = 0
  let whiteCut = ROI_HISTOGRAM_BINS
  let thresholdValid = 0
  let paused = false
  let destroyed = false
  let hasFrame = false
  let frameSequence = 0
  const gpuMs: Record<string, number> = {}
  const gpuTimingEnabled = device.features.has('timestamp-query')
  const withGpuTiming = <T extends {
    withPerformanceCallback(callback: (start: bigint, end: bigint) => void): T
  }>(
    pipeline: T,
    name: string,
  ): T =>
    gpuTimingEnabled
      ? pipeline.withPerformanceCallback((start, end) => {
          gpuMs[name] = Number(end - start) / 1e6
        })
      : pipeline
  const presentationFormat = globalThis.navigator.gpu.getPreferredCanvasFormat()
  const overlayUniform = root.createUniform(overlayParams, {
    mode: overlayMode,
    blackCut,
    whiteCut,
    valid: thresholdValid,
  })
  const writeOverlayParams = () => {
    overlayUniform.write({
      mode: overlayMode,
      blackCut,
      whiteCut,
      valid: thresholdValid,
    })
  }

  const inputBuffer = root
    .createBuffer(d.arrayOf(d.f32, modelWidth * modelHeight))
    .$usage('storage')
  const outputBuffer = root
    .createBuffer(d.arrayOf(d.f32, maskWidth * maskHeight))
    .$usage('storage')
  const prepareUniform = root.createUniform(prepareParams, {
    size: d.vec2f(modelWidth, modelHeight),
    stats: d.vec2f(0, 1),
  })
  const histogramBuffer = root.createBuffer(roiHistogramSchema).$usage('storage')
  const histogramResetPipeline = withGpuTiming(
    root.createGuardedComputePipeline(resetRoiHistogram),
    'histogramReset',
  )
  const histogramAccumulatePipeline = withGpuTiming(
    root.createGuardedComputePipeline(accumulateRoiHistogram),
    'histogramAccumulate',
  )
  const histogramResetBindGroup = root.createBindGroup(histogramResetLayout, {
    histogram: histogramBuffer,
  })
  const maskTexture = root
    .createTexture({
      size: [maskWidth, maskHeight],
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'render', 'storage')
  const backgroundStage = createRoiBackgroundStage(
    root,
    modelWidth,
    modelHeight,
    inputBuffer,
    maskTexture,
    sampler,
    (name, ms) => {
      gpuMs[name] = ms
    },
  )
  const histogramParamsBuffer = root.createUniform(histogramParams, {
    residualGain: 1,
    _pad: d.vec3f(0, 0, 0),
  })
  const modelInputTexture = root
    .createTexture({
      size: [modelWidth, modelHeight],
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'render', 'storage')
  // Cache each camera frame in a regular texture. The external texture used
  // for ingest is short-lived, so the overlay uses this copy; that also lets
  // paused frames re-render when the view mode changes.
  const cameraTexture = root
    .createTexture({
      size: [cameraWidth, cameraHeight],
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'render')
  const cameraView = cameraTexture.createView()
  const cameraCopyPipeline = withGpuTiming(createCopyPipeline(root), 'cameraCopy')
  const histogramAccumulateBindGroup = root.createBindGroup(histogramAccumulateLayout, {
    cameraTexture,
    maskTexture,
    backgroundTexture: backgroundStage.backgroundTexture,
    sampler,
    histogram: histogramBuffer,
    params: histogramParamsBuffer,
  })

  const pixelRatio = Math.max(1, canvas.width / Math.max(1, canvas.clientWidth || canvas.width))
  const componentsStage = createRoiComponentsStage(
    root,
    cameraWidth,
    cameraHeight,
    cameraTexture,
    maskTexture,
    backgroundStage.backgroundTexture,
    sampler,
  )
  const componentCircleStage = createRoiCircleStage(
    root,
    presentationFormat,
    canvas.width,
    canvas.height,
    cameraWidth,
    cameraHeight,
    pixelRatio,
    componentsStage.circleBuffer,
    componentsStage.drawIndirectBuffer,
  )
  const graphStage = createRoiGraphStage(
    root,
    presentationFormat,
    canvas.width,
    canvas.height,
    cameraWidth,
    cameraHeight,
    pixelRatio,
  )
  componentsStage.setThresholds(blackCut, whiteCut, false)

  const maskPipeline = withGpuTiming(root.createGuardedComputePipeline(maskToTexture), 'mask')
  const debugInputPipeline = withGpuTiming(
    root.createGuardedComputePipeline(debugInputToTexture),
    'debugInput',
  )
  const debugInputBindGroup = root.createBindGroup(debugInputLayout, {
    inputBuffer,
    inputTexture: modelInputTexture,
  })
  const overlayBindGroup = root.createBindGroup(overlayLayout, {
    cameraTexture,
    maskTexture,
    backgroundTexture: backgroundStage.backgroundTexture,
    modelInputTexture,
    labelTexture: componentsStage.labelTexture as never,
    sampler,
  })
  const viewUniform = root.createUniform(viewParams, {
    zoom: 1,
    panX: 0,
    panY: 0,
    frameWidth: cameraWidth,
    frameHeight: cameraHeight,
  })
  let viewZoom = 1
  let viewPanX = 0
  let viewPanY = 0
  const writeViewParams = () => {
    viewUniform.write({
      zoom: viewZoom,
      panX: viewPanX,
      panY: viewPanY,
      frameWidth: cameraWidth,
      frameHeight: cameraHeight,
    })
  }
  const overlayPipeline = withGpuTiming(
    root
      .with(overlayAccess, overlayUniform)
      .with(viewAccess, viewUniform)
      .createRenderPipeline({
        vertex: common.fullScreenTriangle,
        fragment: overlayFragment,
        targets: { format: presentationFormat },
        multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
      }),
    'render',
  )

  onStatus?.('fetching model')
  const response = await fetch(modelUrl, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`Failed to fetch ROI model: ${response.status} ${response.statusText}`)
  }
  const modelData = await response.arrayBuffer()
  onStatus?.(`model ${(modelData.byteLength / 1024).toFixed(0)} KiB; creating session`)
  // ORT WebGPU still bootstraps its wasm runtime even though inference runs
  // on the WebGPU EP. Vite copies the package runtime assets into public/ort
  // at dev/build time; that directory is generated and ignored.
  ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`
  onStatus?.('creating WebGPU session')
  const freeSessionSlot = await takeSessionSlot()
  let session: Awaited<ReturnType<typeof ort.InferenceSession.create>>
  try {
    session = await withTimeout(
      ort.InferenceSession.create(modelData, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      }),
      45_000,
      'ONNX Runtime WebGPU session creation timed out.',
    )
  } catch (cause) {
    freeSessionSlot()
    throw cause
  }
  onStatus?.('running')
  const inputGpuBuffer = root.unwrap(inputBuffer)
  const outputGpuBuffer = root.unwrap(outputBuffer)
  const inputGpuTensor = ort.Tensor.fromGpuBuffer(inputGpuBuffer, {
    dataType: 'float32',
    dims: [1, 1, modelHeight, modelWidth],
  })
  const outputGpuTensor = ort.Tensor.fromGpuBuffer(outputGpuBuffer, {
    dataType: 'float32',
    dims: [1, 1, maskHeight, maskWidth],
  })
  // Configure the canvas only after the session slot is held. The previous
  // pipeline unconfigures this same canvas as part of destroy(); reconfiguring
  // before that completes leaves the swap chain at the old size (black frame).
  const context = root.configureContext({
    canvas,
    format: presentationFormat,
    alphaMode: 'opaque',
  })
  const msaaTexture = device.createTexture({
    label: 'roi-lab-msaa',
    size: [Math.max(1, canvas.width), Math.max(1, canvas.height), 1],
    format: presentationFormat,
    sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })
  const msaaView = msaaTexture.createView()

  const renderScene = () => {
    const encoder = device.createCommandEncoder({ label: 'roi-lab-scene' })
    const pass = encoder.beginRenderPass({
      label: 'roi-lab-scene',
      colorAttachments: [
        {
          view: msaaView,
          resolveTarget: context.getCurrentTexture().createView(),
          clearValue: [0, 0, 0, 1],
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
    })
    overlayPipeline.with(pass).with(overlayBindGroup).draw(3)
    if (overlayMode === 6) {
      componentCircleStage.encode(pass)
    } else if (overlayMode === 8 || overlayMode === 9) {
      graphStage.encode(pass)
    }
    pass.end()
    device.queue.submit([encoder.finish()])
    hasFrame = true
  }

  let readingStats = false
  const readStats = async (source: GPUBuffer, count: number) => {
    const size = count * 4
    const readback = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(source, 0, readback, 0, size)
    device.queue.submit([encoder.finish()])
    await readback.mapAsync(GPUMapMode.READ)
    const values = new Float32Array(readback.getMappedRange()).slice()
    readback.unmap()
    readback.destroy()
    let min = Number.POSITIVE_INFINITY
    let max = Number.NEGATIVE_INFINITY
    let sum = 0
    for (const value of values) {
      min = Math.min(min, value)
      max = Math.max(max, value)
      sum += value
    }
    return {
      min: Number.isFinite(min) ? min : 0,
      mean: values.length > 0 ? sum / values.length : 0,
      max: Number.isFinite(max) ? max : 0,
    }
  }

  const statsCanvas = document.createElement('canvas')
  statsCanvas.width = 64
  statsCanvas.height = 36
  const statsContext = statsCanvas.getContext('2d', { willReadFrequently: true })
  const preparePipeline = withGpuTiming(
    root.with(prepareAccess, prepareUniform).createGuardedComputePipeline(prepareModelInput),
    'preprocess',
  )

  const histogramReadback = device.createBuffer({
    size: ROI_HISTOGRAM_BINS * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: 'roi-histogram-readback',
  })
  let frameSerial = 0
  let readbackBusy = false
  let readbackPendingDestroy = false

  /**
   * Queue a histogram copy and let it resolve in the background. The frame
   * loop keeps rendering with the thresholds already in `overlayUniform`, so a
   * slow map never delays the camera overlay.
   */
  const scheduleHistogramReadback = () => {
    if (readbackBusy || paused) {
      return
    }
    readbackBusy = true
    const encoder = device.createCommandEncoder({ label: 'roi-histogram-readback' })
    encoder.copyBufferToBuffer(
      root.unwrap(histogramBuffer),
      0,
      histogramReadback,
      0,
      ROI_HISTOGRAM_BINS * 4,
    )
    device.queue.submit([encoder.finish()])
    void histogramReadback
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const mapped = histogramReadback.getMappedRange()
        const counts = new Uint32Array(mapped).slice()
        histogramReadback.unmap()
        if (destroyed || paused) {
          return
        }
        const analysis = analyzeRoiHistogram(counts)
        const blackMargin =
          analysis.blackCut > 0
            ? Math.max(1, Math.round((analysis.backgroundBin - analysis.blackCut) * ROI_CUT_MARGIN_FRACTION))
            : 0
        const whiteMargin =
          analysis.whiteCut < ROI_HISTOGRAM_BINS
            ? Math.max(1, Math.round((analysis.whiteCut - analysis.backgroundBin) * ROI_CUT_MARGIN_FRACTION))
            : 0
        const adjustedBlackCut = Math.max(0, analysis.blackCut - blackMargin)
        const adjustedWhiteCut = Math.min(ROI_HISTOGRAM_BINS, analysis.whiteCut + whiteMargin)
        const sumBins = (start: number, end: number) => {
          let total = 0
          for (let bin = start; bin < end; bin += 1) {
            total += counts[bin] ?? 0
          }
          return total
        }
        const blackPixels = sumBins(0, adjustedBlackCut)
        const whitePixels = sumBins(adjustedWhiteCut, ROI_HISTOGRAM_BINS)
        const adjustedAnalysis = {
          ...analysis,
          blackCut: adjustedBlackCut,
          whiteCut: adjustedWhiteCut,
          blackPixels,
          whitePixels,
          grayPixels: Math.max(0, analysis.maskedPixels - blackPixels - whitePixels),
        }
        blackCut = adjustedBlackCut
        whiteCut = adjustedWhiteCut
        thresholdValid = analysis.valid ? 1 : 0
        writeOverlayParams()
        componentsStage.setThresholds(blackCut, whiteCut, analysis.valid)
        onHistogram?.({ counts, analysis: adjustedAnalysis })
      })
      .catch((cause: unknown) => {
        if (!destroyed) {
          console.error('[roi-lab] histogram readback failed', cause)
        }
      })
      .finally(() => {
        if (histogramReadback.mapState === 'mapped') {
          histogramReadback.unmap()
        }
        readbackBusy = false
        if (readbackPendingDestroy) {
          histogramBuffer.destroy()
          histogramReadback.destroy()
        }
      })
  }

  const imageStats = (video: HTMLVideoElement) => {
    if (!statsContext || video.videoWidth < 2 || video.videoHeight < 2) {
      return { mean: 0, std: 1 }
    }
    statsContext.imageSmoothingEnabled = true
    statsContext.imageSmoothingQuality = 'high'
    statsContext.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 64, 36)
    const data = statsContext.getImageData(0, 0, 64, 36).data
    let sum = 0
    let sumSquares = 0
    for (let index = 0; index < data.length; index += 4) {
      const luma =
        (0.299 * (data[index] ?? 0) +
          0.587 * (data[index + 1] ?? 0) +
          0.114 * (data[index + 2] ?? 0)) /
        255
      sum += luma
      sumSquares += luma * luma
    }
    const count = data.length / 4
    const mean = sum / count
    const variance = Math.max(1e-6, sumSquares / count - mean * mean)
    return { mean, std: Math.max(0.05, Math.sqrt(variance)) }
  }

  // Global normalization stats change slowly with exposure/lighting. Reading
  // them back from a canvas forces a synchronous GPU readback every frame, so
  // refresh occasionally and reuse between reads.
  const IMAGE_STATS_INTERVAL = 8
  let cachedImageStats: { mean: number; std: number } | undefined
  let imageStatsAge = 0

  const preprocessFrame = (video: HTMLVideoElement) => {
    const refreshStats = cachedImageStats === undefined || imageStatsAge >= IMAGE_STATS_INTERVAL
    const imageStatsStart = performance.now()
    if (refreshStats) {
      cachedImageStats = imageStats(video)
      imageStatsAge = 0
    } else {
      imageStatsAge += 1
    }
    const stats = cachedImageStats!
    const imageStatsMs = refreshStats ? performance.now() - imageStatsStart : 0
    prepareUniform.write({
      size: d.vec2f(modelWidth, modelHeight),
      stats: d.vec2f(stats.mean, stats.std),
    })
    preparePipeline
      .with(
        root.createBindGroup(prepareLayout, {
          inputTexture: device.importExternalTexture({ source: video }),
          outputBuffer: inputBuffer,
          sampler,
          params: prepareUniform,
        }),
      )
      .dispatchThreads(modelWidth, modelHeight)
    return {
      stats,
      imageStatsMs,
      statsAgeFrames: imageStatsAge,
      statsRefreshed: refreshStats,
    }
  }

  const processFrame = async (video: HTMLVideoElement) => {
    const start = performance.now()
    const preprocessStart = performance.now()
    const { stats, imageStatsMs, statsAgeFrames, statsRefreshed } = preprocessFrame(video)
    const preprocessMs = performance.now() - preprocessStart

    // CPU preprocessing is followed by a queue write; wait for it before ORT
    // reads the same GPUBuffer, then wait before sampling the mask.
    const queueWaitStart = performance.now()
    await device.queue.onSubmittedWorkDone()
    const queueWaitMs = performance.now() - queueWaitStart
    const inferenceStart = performance.now()
    await session.run({ input: inputGpuTensor }, { mask: outputGpuTensor })
    const inferenceRunMs = performance.now() - inferenceStart
    const inferenceDrainStart = performance.now()
    await device.queue.onSubmittedWorkDone()
    const inferenceDrainMs = performance.now() - inferenceDrainStart
    const inferenceMs = inferenceRunMs + inferenceDrainMs

    // Copy the current frame into the persistent camera texture. The overlay
    // samples that copy instead of the short-lived external texture, so a
    // paused frame can be re-rendered with a different view mode.
    const cameraCopyBindGroup = root.createBindGroup(copyBindGroupLayout, {
      cameraTex: device.importExternalTexture({ source: video }),
    })
    const cameraCopyStart = performance.now()
    cameraCopyPipeline
      .withColorAttachment({ view: cameraView })
      .with(cameraCopyBindGroup)
      .draw(3)
    const cameraCopyMs = performance.now() - cameraCopyStart

    const debugInputStart = performance.now()
    debugInputPipeline
      .with(debugInputBindGroup)
      .dispatchThreads(modelWidth, modelHeight)
    const debugInputMs = performance.now() - debugInputStart

    const maskStart = performance.now()
    maskPipeline
      .with(
        root.createBindGroup(maskToTextureLayout, {
          outputBuffer,
          maskTexture,
        }),
      )
      .dispatchThreads(maskWidth, maskHeight)
    const maskMs = performance.now() - maskStart

    const backgroundStart = performance.now()
    backgroundStage.setStats(stats.mean, stats.std)
    backgroundStage.encode()
    const backgroundMs = performance.now() - backgroundStart

    frameSerial++
    const frameId = frameSequence++
    const componentsStart = performance.now()
    componentsStage.encodeComponents()
    const componentsEncodeMs = performance.now() - componentsStart

    const readbackStart = performance.now()
    const readback =
      onIdentity !== undefined
        ? await componentsStage.readComponents()
        : { points: [], count: 0, bytesCopied: 0, truncated: false }
    const componentPoints = readback.points
    const readbackMs = performance.now() - readbackStart
    const identityStart = performance.now()
    let graph: RoiGraphData | undefined
    if (onIdentity !== undefined) {
      try {
        graph = onIdentity(componentPoints, frameId)
      } catch (cause: unknown) {
        console.error('[roi-lab] identity callback failed', cause)
      }
    }
    graphStage.setGraph(graph ?? { points: [], edges: [] })
    const identityMs = performance.now() - identityStart
    const renderStart = performance.now()
    renderScene()
    const renderMs = performance.now() - renderStart

    // Let the queued readbacks and final render finish before a caller can
    // destroy this pipeline during a resolution/stream change.
    const frameDrainStart = performance.now()
    await device.queue.onSubmittedWorkDone()
    const frameDrainMs = performance.now() - frameDrainStart

    // Histogram work is queued after the overlay; only its CPU analysis is
    // asynchronous, and it feeds the next frame's thresholds.
    const histogramStart = performance.now()
    if (frameSerial % ROI_HISTOGRAM_FRAME_INTERVAL === 0) {
      histogramResetPipeline
        .with(histogramResetBindGroup)
        .dispatchThreads(ROI_HISTOGRAM_BINS)
      histogramAccumulatePipeline
        .with(histogramAccumulateBindGroup)
        .dispatchThreads(cameraWidth, cameraHeight)
      scheduleHistogramReadback()
    }
    const histogramMs = performance.now() - histogramStart

    const totalMs = performance.now() - start
    const accountedMs =
      preprocessMs +
      queueWaitMs +
      inferenceRunMs +
      inferenceDrainMs +
      cameraCopyMs +
      debugInputMs +
      maskMs +
      backgroundMs +
      componentsEncodeMs +
      readbackMs +
      identityMs +
      renderMs +
      frameDrainMs +
      histogramMs
    return {
      inferenceMs,
      totalMs,
      readbackMs,
      identityMs,
      frameId,
      steps: {
        imageStatsMs,
        statsAgeFrames,
        statsRefreshed,
        preprocessMs,
        queueWaitMs,
        inferenceRunMs,
        inferenceDrainMs,
        inferenceMs,
        cameraCopyMs,
        debugInputMs,
        maskMs,
        backgroundMs,
        componentsEncodeMs,
        readbackMs,
        readbackComponents: readback.count,
        readbackBytes: readback.bytesCopied,
        readbackTruncated: readback.truncated,
        identityMs,
        renderMs,
        frameDrainMs,
        histogramMs,
        accountedMs,
        overheadMs: totalMs - accountedMs,
        totalMs,
        frameId,
        gpuMs: { ...gpuMs },
      },
    }
  }

  return {
    processFrame,
    resize(width, height) {
      canvas.width = Math.max(1, Math.floor(width))
      canvas.height = Math.max(1, Math.floor(height))
    },
    setViewMode(mode) {
      overlayMode = mode
      writeOverlayParams()
      if (hasFrame) {
        // Re-render the cached frame so view switches still work while paused.
        renderScene()
      }
    },
    setViewport(zoom, panX, panY) {
      viewZoom = zoom
      viewPanX = panX
      viewPanY = panY
      writeViewParams()
      componentCircleStage.setViewport(zoom, panX, panY)
      graphStage.setViewport(zoom, panX, panY)
      if (paused && hasFrame) {
        renderScene()
      }
    },
    setPaused(value) {
      paused = value
    },
    async dumpStats() {
      if (readingStats) {
        throw new Error('A buffer readback is already in progress.')
      }
      readingStats = true
      try {
        return {
          input: await readStats(root.unwrap(inputBuffer), modelWidth * modelHeight),
          output: await readStats(root.unwrap(outputBuffer), maskWidth * maskHeight),
        }
      } finally {
        readingStats = false
      }
    },
    destroy() {
      if (destroyed) {
        return
      }
      destroyed = true
      context.unconfigure()
      void session.release().finally(freeSessionSlot)
      if (readbackBusy) {
        // The queued copy still references both buffers; let mapAsync settle
        // before destroying them.
        readbackPendingDestroy = true
      } else {
        if (histogramReadback.mapState === 'mapped') {
          histogramReadback.unmap()
        }
        histogramBuffer.destroy()
        histogramReadback.destroy()
      }
      inputBuffer.destroy()
      outputBuffer.destroy()
      maskTexture.destroy()
      modelInputTexture.destroy()
      cameraTexture.destroy()
      backgroundStage.destroy()
      componentsStage.destroy()
      graphStage.destroy()
      msaaTexture.destroy()
    },
  }
}
