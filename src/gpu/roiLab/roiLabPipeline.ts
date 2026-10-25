import * as ort from 'onnxruntime-web/webgpu'
import { common, d, std, tgpu } from 'typegpu'
import type { TgpuRoot } from 'typegpu'
import { atomicAdd, atomicStore } from 'typegpu/std'

import { copyBindGroupLayout, createCopyPipeline } from '@/gpu/pipelines/copyPipeline'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { createRoiBackgroundStage } from '@/gpu/roiLab/roiBackgroundPipeline'
import { createRoiCircleStage } from '@/gpu/roiLab/roiCirclePipeline'
import { createRoiGraphStage, type RoiGraphData } from '@/gpu/roiLab/roiGraphPipeline'
import type { RoiDetectedPoint } from '@/gpu/roiLab/roiPeakTypes'
import { createRoiRefinedPeakStage } from '@/gpu/roiLab/roiRefinedPeakPipeline'
import { analyzeRoiHistogram, ROI_HISTOGRAM_BINS, type RoiHistogramAnalysis } from '@/lib/roiHistogram'
import { type RoiNeighbourFilterResult, type RoiRefinedPeak } from '@/lib/roiNeighbourFilter'
import { disposeIdentityWorker, filterRefinedPeaksInWorker, warmIdentityWorker } from '@/workers/identityClient'

const { performance } = globalThis

/**
 * Browser-side ROI segmentation.
 *
 * ONNX graph: normalized grayscale 1×H×192 -> mask probability 1×H×192.
 * Per-image mean/std are reduced on the GPU every frame and consumed on the
 * GPU: `0.4 + 0.2 * z` (z = (luma - mean) / std) is written from a shader that
 * reads the stats buffer directly, so nothing is read back to the CPU. This
 * matches the per-image standardization used for training without adding
 * reduction ops to the ONNX graph.
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

const prepareParams = d.struct({
  size: d.vec2f,
})

/**
 * Frame statistics are reduced on the GPU every frame and consumed there:
 * the normaliser and the background stage both read this buffer, so the CPU
 * never sees mean/std and no readback is on the frame path.
 */
const IMAGE_STATS_COLUMNS = 64
const IMAGE_STATS_ROWS = 36
const IMAGE_STATS_SAMPLES = IMAGE_STATS_COLUMNS * IMAGE_STATS_ROWS

const imageStatsSampleLayout = tgpu.bindGroupLayout({
  inputTexture: { externalTexture: d.textureExternal() },
  sampler: { sampler: 'filtering' },
  samples: { storage: d.arrayOf(d.f32), access: 'mutable' },
})

const imageStatsReduceLayout = tgpu.bindGroupLayout({
  samples: { storage: d.arrayOf(d.f32), access: 'readonly' },
  stats: { storage: d.arrayOf(d.f32), access: 'mutable' },
})

const sampleImageLuma = (x: number, y: number) => {
  'use gpu'
  const uv = (d.vec2f(x, y) + d.vec2f(0.5, 0.5)).div(d.vec2f(IMAGE_STATS_COLUMNS, IMAGE_STATS_ROWS))
  const color = std.textureSampleBaseClampToEdge(
    imageStatsSampleLayout.$.inputTexture,
    imageStatsSampleLayout.$.sampler,
    uv,
  )
  const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  imageStatsSampleLayout.$.samples[d.u32(y) * d.u32(IMAGE_STATS_COLUMNS) + d.u32(x)] = luma
}

/**
 * Single-thread reduction over the 2304 luma samples. A serial loop beats a
 * parallel one here because f32 atomics are not available in WGSL; 2304 reads
 * from a 9 KiB storage buffer cost a few microseconds.
 */
const reduceImageStats = () => {
  'use gpu'
  let sum = d.f32(0)
  let sumSquares = d.f32(0)
  for (let index = 0; index < IMAGE_STATS_SAMPLES; index += 1) {
    const value = imageStatsReduceLayout.$.samples[d.u32(index)]!
    sum += value
    sumSquares += value * value
  }
  const count = d.f32(IMAGE_STATS_SAMPLES)
  const mean = sum / count
  const variance = std.max(sumSquares / count - mean * mean, d.f32(1e-6))
  imageStatsReduceLayout.$.stats[d.u32(0)] = mean
  imageStatsReduceLayout.$.stats[d.u32(1)] = std.max(std.sqrt(variance), d.f32(0.05))
}

const roiHistogramSchema = d.arrayOf(d.atomic(d.u32), ROI_HISTOGRAM_BINS)

const histogramResetLayout = tgpu.bindGroupLayout({
  histogram: { storage: roiHistogramSchema, access: 'mutable' },
})

const histogramParams = d.struct({
  size: d.vec2f,
})

const histogramAccumulateLayout = tgpu.bindGroupLayout({
  blurred: { storage: d.arrayOf(d.f32), access: 'readonly' },
  maskTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
  histogram: { storage: roiHistogramSchema, access: 'mutable' },
  params: { uniform: histogramParams },
})

const resetRoiHistogram = (bin: number) => {
  'use gpu'
  atomicStore(histogramResetLayout.$.histogram[d.u32(bin)]!, d.u32(0))
}

/**
 * Full-resolution histogram of the same blurred signal used by the peak stage,
 * restricted to the bilinearly sampled ROI mask. This matches the Python
 * `histogram_cuts(blurred, mask)` step.
 */
const accumulateRoiHistogram = (x: number, y: number) => {
  'use gpu'
  const px = d.u32(x)
  const py = d.u32(y)
  const size = histogramAccumulateLayout.$.params.size
  const uv = (d.vec2f(d.f32(px), d.f32(py)) + d.vec2f(0.5, 0.5)).div(size)
  const mask = std.textureSampleLevel(
    histogramAccumulateLayout.$.maskTexture,
    histogramAccumulateLayout.$.sampler,
    uv,
    d.f32(0),
  ).r
  if (mask < d.f32(0.5)) {
    return
  }
  const value = histogramAccumulateLayout.$.blurred[py * d.u32(size.x) + px]!
  const bin = d.u32(std.clamp(value, d.f32(0), d.f32(1)) * d.f32(255) + d.f32(0.5))
  atomicAdd(histogramAccumulateLayout.$.histogram[bin]!, d.u32(1))
}

/**
 * Empty pass used only to place a timestamp on the GPU timeline.
 *
 * GPU timestamps are recorded by the device, so the delta between two markers
 * is the exact device time of everything queued between them — including passes
 * we do not own and cannot instrument, such as ONNX Runtime's inference.
 */
const timestampMarker = (index: number) => {
  'use gpu'
  if (index >= 1) {
    return
  }
}

const prepareLayout = tgpu.bindGroupLayout({
  inputTexture: { externalTexture: d.textureExternal() },
  outputBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  stats: { storage: d.arrayOf(d.f32), access: 'readonly' },
  sampler: { sampler: 'filtering' },
  params: { uniform: prepareParams },
})

const prepareAccess = tgpu.accessor(prepareParams)

const prepareModelInput = (x: number, y: number) => {
  'use gpu'
  const params = prepareAccess.$
  const uv = (d.vec2f(x, y) + d.vec2f(0.5, 0.5)).div(params.size)
  const color = std.textureSampleBaseClampToEdge(prepareLayout.$.inputTexture, prepareLayout.$.sampler, uv)
  const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  prepareLayout.$.outputBuffer[y * d.u32(params.size.x) + x] =
    (luma - prepareLayout.$.stats[d.u32(0)]!) / prepareLayout.$.stats[d.u32(1)]!
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
  std.textureStore(debugInputLayout.$.inputTexture, d.vec2u(x, y), d.vec4f(display, display, display, 1))
}

const maskToTexture = (x: number, y: number) => {
  'use gpu'
  const width = std.textureDimensions(maskToTextureLayout.$.maskTexture).x
  const value = maskToTextureLayout.$.outputBuffer[y * width + x]!
  std.textureStore(maskToTextureLayout.$.maskTexture, d.vec2u(x, y), d.vec4f(value, value, value, 1))
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
  const video = std.textureSampleLevel(overlayLayout.$.cameraTexture, overlayLayout.$.sampler, srcUv, d.f32(0))
  if (overlayAccess.$.mode === d.u32(3) || overlayAccess.$.mode === d.u32(9)) {
    return video
  }
  const mask = std.textureSampleLevel(overlayLayout.$.maskTexture, overlayLayout.$.sampler, srcUv, d.f32(0)).r
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
  if (overlayAccess.$.mode === d.u32(5) || overlayAccess.$.mode === d.u32(6) || overlayAccess.$.mode === d.u32(8)) {
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

export type RoiPeakFrameAnalysis = {
  frameId: number
  peaks: readonly RoiRefinedPeak[]
  filter: RoiNeighbourFilterResult
}

export type RoiStepTimings = {
  /** Cost of encoding the GPU mean/std reduction passes. Nothing is read back. */
  statsMs: number
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
  detectionEncodeMs: number
  readbackMs: number
  readbackCandidates: number
  readbackRejected: number
  readbackBytes: number
  readbackTruncated: boolean
  neighbourFilterMs: number
  identityMs: number
  /** Frames between the one being encoded and the one just presented. */
  identityLagFrames: number
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
  setViewMode: (mode: 1 | 2 | 3 | 4 | 6 | 8 | 9) => void
  setViewport: (zoom: number, panX: number, panY: number) => void
  setSelectedPeak: (peak: { x: number; y: number } | undefined) => void
  setPaused: (paused: boolean) => void
  /** Peaks and filter decisions for the frame currently on screen. */
  getPresentedPeakAnalysis: () => RoiPeakFrameAnalysis | undefined
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
  onIdentity?: (
    points: readonly RoiDetectedPoint[],
    frameId: number,
  ) => RoiGraphData | undefined | Promise<RoiGraphData | undefined>,
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

  let overlayMode = 9
  let blackCut = 0
  let whiteCut = ROI_HISTOGRAM_BINS
  let thresholdValid = 0
  let paused = false
  let destroyed = false
  let hasFrame = false
  let frameSequence = 0
  /**
   * GPU timings come from one timestamp query set with an explicit index pair
   * per stage.
   *
   * TypeGPU's `withPerformanceCallback` hands every pipeline the same query
   * indices (0 and 1) and drops a read whenever another timed pass is submitted
   * before the previous read resolves. With a dozen timed passes per frame that
   * made most numbers stale. Resolving the whole set once per frame, with
   * distinct indices, gives every stage a fresh and correctly paired timestamp.
   *
   * A "pass" stage writes both boundaries itself; a "bracket" stage uses two
   * no-op marker passes around a region we do not own (ORT inference, the
   * multi-pass detection stage, the render pass).
   */
  const GPU_STAGES = ['preprocess', 'inference', 'detection', 'background', 'cameraCopy', 'render'] as const
  type GpuStage = (typeof GPU_STAGES)[number]
  const GPU_TIMESTAMP_COUNT = GPU_STAGES.length * 2
  const gpuMs: Record<string, number> = {}
  const gpuTimingEnabled = device.features.has('timestamp-query')
  const timingQuerySet = gpuTimingEnabled ? root.createQuerySet('timestamp', GPU_TIMESTAMP_COUNT) : undefined
  const timingResolveBuffer = gpuTimingEnabled
    ? device.createBuffer({
        label: 'roi-lab-timestamps-resolve',
        size: GPU_TIMESTAMP_COUNT * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : undefined
  const timingReadBuffer = gpuTimingEnabled
    ? device.createBuffer({
        label: 'roi-lab-timestamps-read',
        size: GPU_TIMESTAMP_COUNT * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      })
    : undefined

  const stageIndex = (stage: GpuStage): number => GPU_STAGES.indexOf(stage) * 2

  /** Records both boundaries of a single pass. */
  const timedPass = <
    T extends {
      withTimestampWrites(options: {
        querySet: NonNullable<typeof timingQuerySet>
        beginningOfPassWriteIndex?: number
        endOfPassWriteIndex?: number
      }): T
    },
  >(
    pipeline: T,
    stage: GpuStage,
  ): T =>
    timingQuerySet === undefined
      ? pipeline
      : pipeline.withTimestampWrites({
          querySet: timingQuerySet,
          beginningOfPassWriteIndex: stageIndex(stage),
          endOfPassWriteIndex: stageIndex(stage) + 1,
        })

  /** Records one boundary at each end of a region we do not encode ourselves. */
  const createGpuBracket = (stage: GpuStage): { begin: () => void; end: () => void } => {
    if (timingQuerySet === undefined) {
      return { begin: () => {}, end: () => {} }
    }
    const beginPipeline = root
      .createGuardedComputePipeline(timestampMarker)
      .withTimestampWrites({ querySet: timingQuerySet, beginningOfPassWriteIndex: stageIndex(stage) })
    const endPipeline = root.createGuardedComputePipeline(timestampMarker).withTimestampWrites({
      querySet: timingQuerySet,
      beginningOfPassWriteIndex: stageIndex(stage) + 1,
    })
    return {
      begin: () => beginPipeline.dispatchThreads(1),
      end: () => endPipeline.dispatchThreads(1),
    }
  }

  /** Device and CPU timings are averaged over this many frames before display. */
  /** The panel republishes its numbers once per window instead of every frame. */
  const TIMING_WINDOW = 5
  let timingBatch: RoiStepTimings[] = []
  let publishedTimings: { ms: Map<string, number>; gpu: Map<string, number> } | undefined

  const averageBatch = (
    batch: readonly RoiStepTimings[],
  ): {
    ms: Map<string, number>
    gpu: Map<string, number>
  } => {
    const ms = new Map<string, number>()
    const gpu = new Map<string, number>()
    const latest = batch[batch.length - 1]
    if (latest === undefined) {
      return { ms, gpu }
    }
    for (const key of Object.keys(latest)) {
      if (key === 'gpuMs' || !key.endsWith('Ms')) {
        continue
      }
      let total = 0
      let count = 0
      for (const sample of batch) {
        const value = (sample as unknown as Record<string, unknown>)[key]
        if (typeof value === 'number' && Number.isFinite(value)) {
          total += value
          count += 1
        }
      }
      if (count > 0) {
        ms.set(key, total / count)
      }
    }
    const stages = new Set<string>()
    for (const sample of batch) {
      for (const stage of Object.keys(sample.gpuMs)) {
        stages.add(stage)
      }
    }
    for (const stage of stages) {
      let total = 0
      let count = 0
      for (const sample of batch) {
        const value = sample.gpuMs[stage]
        if (typeof value === 'number' && Number.isFinite(value)) {
          total += value
          count += 1
        }
      }
      if (count > 0) {
        gpu.set(stage, total / count)
      }
    }
    return { ms, gpu }
  }

  /**
   * Publishes averaged timings once per `TIMING_WINDOW` frames, and holds them
   * frozen in between.
   *
   * A rolling average still changed every frame, which made the panel hard to
   * read; this samples instead. Only timing fields are held — `frameId`,
   * peak counts and flags always reflect the current frame. `accountedMs`,
   * `overheadMs` and `totalMs` are linear combinations of the averaged fields,
   * so the panel keeps adding up.
   */
  const smoothTimings = (steps: RoiStepTimings): RoiStepTimings => {
    timingBatch.push(steps)
    if (timingBatch.length >= TIMING_WINDOW || publishedTimings === undefined) {
      publishedTimings = averageBatch(timingBatch)
      if (timingBatch.length >= TIMING_WINDOW) {
        timingBatch = []
      }
    }
    const result = { ...steps } as unknown as Record<string, unknown>
    for (const [key, value] of publishedTimings!.ms) {
      result[key] = value
    }
    const gpuStages: Record<string, number> = { ...steps.gpuMs }
    for (const [stage, value] of publishedTimings!.gpu) {
      gpuStages[stage] = value
    }
    result.gpuMs = gpuStages
    return result as unknown as RoiStepTimings
  }

  let timingReadBusy = false
  /** Resolves the frame's timestamps and publishes every stage. Fire and forget. */
  const readGpuTimings = (): void => {
    if (
      timingQuerySet === undefined ||
      timingResolveBuffer === undefined ||
      timingReadBuffer === undefined ||
      timingReadBusy
    ) {
      return
    }
    timingReadBusy = true
    const encoder = device.createCommandEncoder({ label: 'roi-lab-timestamps' })
    encoder.resolveQuerySet(root.unwrap(timingQuerySet), 0, GPU_TIMESTAMP_COUNT, timingResolveBuffer, 0)
    encoder.copyBufferToBuffer(timingResolveBuffer, 0, timingReadBuffer, 0, GPU_TIMESTAMP_COUNT * 8)
    device.queue.submit([encoder.finish()])
    void timingReadBuffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const values = new BigUint64Array(timingReadBuffer.getMappedRange().slice())
        timingReadBuffer.unmap()
        for (let stage = 0; stage < GPU_STAGES.length; stage += 1) {
          const begin = values[stage * 2]
          const finish = values[stage * 2 + 1]
          // Unwritten query slots read as zero; a real GPU timestamp never is.
          if (begin === undefined || finish === undefined || begin === 0n || finish === 0n) {
            continue
          }
          if (finish < begin) {
            continue
          }
          gpuMs[GPU_STAGES[stage]!] = Number(finish - begin) / 1e6
        }
      })
      .catch((cause: unknown) => {
        if (!destroyed) {
          console.error('[roi-lab] timestamp readback failed', cause)
        }
      })
      .finally(() => {
        timingReadBusy = false
      })
  }

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

  const inputBuffer = root.createBuffer(d.arrayOf(d.f32, modelWidth * modelHeight)).$usage('storage')
  const outputBuffer = root.createBuffer(d.arrayOf(d.f32, maskWidth * maskHeight)).$usage('storage')
  const prepareUniform = root.createUniform(prepareParams, {
    size: d.vec2f(modelWidth, modelHeight),
  })
  const imageStatsSamples = root.createBuffer(d.arrayOf(d.f32, IMAGE_STATS_SAMPLES)).$usage('storage')
  const imageStatsBuffer = root.createBuffer(d.arrayOf(d.f32, 2)).$usage('storage')
  const histogramBuffer = root.createBuffer(roiHistogramSchema).$usage('storage')
  const histogramResetPipeline = root.createGuardedComputePipeline(resetRoiHistogram)
  const histogramAccumulatePipeline = root.createGuardedComputePipeline(accumulateRoiHistogram)
  const histogramResetBindGroup = root.createBindGroup(histogramResetLayout, {
    histogram: histogramBuffer,
  })
  const histogramParamsBuffer = root.createUniform(histogramParams, {
    size: d.vec2f(cameraWidth, cameraHeight),
  })
  const pixelRatio = Math.max(1, canvas.width / Math.max(1, canvas.clientWidth || canvas.width))
  const cameraCopyPipeline = timedPass(createCopyPipeline(root), 'cameraCopy')
  const preprocessBracket = createGpuBracket('preprocess')
  const inferenceBracket = createGpuBracket('inference')
  const backgroundBracket = createGpuBracket('background')
  const detectionBracket = createGpuBracket('detection')
  const renderBracket = createGpuBracket('render')
  const imageStatsSamplePipeline = root.createGuardedComputePipeline(sampleImageLuma)
  const imageStatsReducePipeline = root.createGuardedComputePipeline(reduceImageStats)
  const imageStatsReduceBindGroup = root.createBindGroup(imageStatsReduceLayout, {
    samples: imageStatsSamples,
    stats: imageStatsBuffer,
  })
  const maskPipeline = root.createGuardedComputePipeline(maskToTexture)
  const debugInputPipeline = root.createGuardedComputePipeline(debugInputToTexture)

  /**
   * Everything one frame writes and the overlay samples.
   *
   * Two slots are alive at once: frame N is presented from its own slot while
   * frame N+1 is encoded into the other. Keeping the pixels per slot is what
   * lets the render of N happen after N+1 has been captured, without ever
   * drawing N's identity over N+1's image.
   */
  const createFrameSlot = () => {
    const maskTexture = root
      .createTexture({
        size: [maskWidth, maskHeight],
        format: 'rgba8unorm',
      })
      .$usage('sampled', 'render', 'storage')
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
    const backgroundStage = createRoiBackgroundStage(
      root,
      modelWidth,
      modelHeight,
      inputBuffer,
      maskTexture,
      sampler,
      imageStatsBuffer,
    )
    const peakStage = createRoiRefinedPeakStage(
      root,
      cameraWidth,
      cameraHeight,
      cameraTexture,
      maskTexture,
      backgroundStage.backgroundTexture,
      sampler,
    )
    peakStage.setThresholds(blackCut, whiteCut, thresholdValid === 1)
    return {
      maskTexture,
      modelInputTexture,
      cameraTexture,
      cameraView: cameraTexture.createView(),
      backgroundStage,
      peakStage,
      peakCircleStage: createRoiCircleStage(
        root,
        presentationFormat,
        canvas.width,
        canvas.height,
        cameraWidth,
        cameraHeight,
        pixelRatio,
        peakStage.circleBuffer,
        peakStage.drawIndirectBuffer,
        peakStage.rejectBuffer,
        peakStage.rejectDrawIndirectBuffer,
      ),
      maskBindGroup: root.createBindGroup(maskToTextureLayout, {
        outputBuffer,
        maskTexture,
      }),
      debugInputBindGroup: root.createBindGroup(debugInputLayout, {
        inputBuffer,
        inputTexture: modelInputTexture,
      }),
      histogramAccumulateBindGroup: root.createBindGroup(histogramAccumulateLayout, {
        blurred: peakStage.blurredBuffer,
        maskTexture,
        sampler,
        histogram: histogramBuffer,
        params: histogramParamsBuffer,
      }),
      overlayBindGroup: root.createBindGroup(overlayLayout, {
        cameraTexture,
        maskTexture,
        backgroundTexture: backgroundStage.backgroundTexture,
        modelInputTexture,
        sampler,
      }),
    }
  }

  type FrameSlot = ReturnType<typeof createFrameSlot>

  const slots: [FrameSlot, FrameSlot] = [createFrameSlot(), createFrameSlot()]
  let nextSlot = 0

  const graphStage = createRoiGraphStage(
    root,
    presentationFormat,
    canvas.width,
    canvas.height,
    cameraWidth,
    cameraHeight,
    pixelRatio,
  )
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
  // The overlay is drawn into a caller-owned pass (see `renderScene`), where
  // pipeline-level timestamp writes are ignored by WebGPU. Render cost is
  // reported as CPU submit time (`renderMs`) instead.
  const overlayPipeline = root
    .with(overlayAccess, overlayUniform)
    .with(viewAccess, viewUniform)
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: overlayFragment,
      targets: { format: presentationFormat },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })

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
  warmIdentityWorker()
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

  let lastPresented: { slot: FrameSlot; graph: RoiGraphData | undefined } | undefined
  let presentedPeakAnalysis: RoiPeakFrameAnalysis | undefined

  const renderScene = (slot: FrameSlot) => {
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
    overlayPipeline.with(pass).with(slot.overlayBindGroup).draw(3)
    if (overlayMode === 6) {
      slot.peakCircleStage.encode(pass)
    } else if (overlayMode === 8 || overlayMode === 9) {
      graphStage.encode(pass)
    }
    pass.end()
    device.queue.submit([encoder.finish()])
    hasFrame = true
  }

  // The graph buffer is written immediately before its own draw is submitted,
  // so it stays queue-ordered with the frame it belongs to and needs no slot.
  const presentFrame = (slot: FrameSlot, graph: RoiGraphData | undefined) => {
    graphStage.setGraph(graph ?? { points: [], edges: [] })
    lastPresented = { slot, graph }
    renderScene(slot)
  }

  const redrawLastFrame = () => {
    if (hasFrame && lastPresented !== undefined) {
      presentFrame(lastPresented.slot, lastPresented.graph)
    }
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

  const preparePipeline = root.with(prepareAccess, prepareUniform).createGuardedComputePipeline(prepareModelInput)

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
    encoder.copyBufferToBuffer(root.unwrap(histogramBuffer), 0, histogramReadback, 0, ROI_HISTOGRAM_BINS * 4)
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
        blackCut = analysis.blackCut
        whiteCut = analysis.whiteCut
        thresholdValid = analysis.valid ? 1 : 0
        writeOverlayParams()
        for (const slot of slots) {
          slot.peakStage.setThresholds(blackCut, whiteCut, analysis.valid)
        }
        onHistogram?.({ counts, analysis })
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

  const preprocessFrame = (video: HTMLVideoElement) => {
    preprocessBracket.begin()
    const statsStart = performance.now()
    const inputTexture = device.importExternalTexture({ source: video })
    imageStatsSamplePipeline
      .with(
        root.createBindGroup(imageStatsSampleLayout, {
          inputTexture,
          sampler,
          samples: imageStatsSamples,
        }),
      )
      .dispatchThreads(IMAGE_STATS_COLUMNS, IMAGE_STATS_ROWS)
    imageStatsReducePipeline.with(imageStatsReduceBindGroup).dispatchThreads()
    const imageStatsMs = performance.now() - statsStart
    preparePipeline
      .with(
        root.createBindGroup(prepareLayout, {
          inputTexture,
          outputBuffer: inputBuffer,
          stats: imageStatsBuffer,
          sampler,
          params: prepareUniform,
        }),
      )
      .dispatchThreads(modelWidth, modelHeight)
    preprocessBracket.end()
    return { statsMs: imageStatsMs }
  }

  let pendingFrame:
    | {
        slot: FrameSlot
        frameId: number
        identity: Promise<RoiGraphData | undefined>
        analysis: RoiPeakFrameAnalysis
      }
    | undefined

  const processFrame = async (video: HTMLVideoElement) => {
    const start = performance.now()
    const slot = slots[nextSlot]!
    nextSlot = nextSlot === 0 ? 1 : 0
    const preprocessStart = performance.now()
    const { statsMs } = preprocessFrame(video)
    const preprocessMs = performance.now() - preprocessStart

    // No queue flushes around inference: ORT shares this device and queue, so
    // our prepare dispatch (writing `inputBuffer`) and our mask dispatch
    // (reading `outputBuffer`) are already ordered around ORT's work. The CPU
    // never reads either buffer here, so nothing needs a completion signal.
    const inferenceStart = performance.now()
    inferenceBracket.begin()
    await session.run({ input: inputGpuTensor }, { mask: outputGpuTensor })
    inferenceBracket.end()
    const inferenceRunMs = performance.now() - inferenceStart
    const queueWaitMs = 0
    const inferenceDrainMs = 0
    const inferenceMs = inferenceRunMs

    // Copy the current frame into the persistent camera texture. The overlay
    // samples that copy instead of the short-lived external texture, so a
    // paused frame can be re-rendered with a different view mode.
    const cameraCopyBindGroup = root.createBindGroup(copyBindGroupLayout, {
      cameraTex: device.importExternalTexture({ source: video }),
    })
    const cameraCopyStart = performance.now()
    cameraCopyPipeline.withColorAttachment({ view: slot.cameraView }).with(cameraCopyBindGroup).draw(3)
    const cameraCopyMs = performance.now() - cameraCopyStart

    const debugInputStart = performance.now()
    debugInputPipeline.with(slot.debugInputBindGroup).dispatchThreads(modelWidth, modelHeight)
    const debugInputMs = performance.now() - debugInputStart

    const maskStart = performance.now()
    maskPipeline.with(slot.maskBindGroup).dispatchThreads(maskWidth, maskHeight)
    const maskMs = performance.now() - maskStart

    const backgroundStart = performance.now()
    backgroundBracket.begin()
    slot.backgroundStage.encode()
    backgroundBracket.end()
    const backgroundMs = performance.now() - backgroundStart

    frameSerial++
    const frameId = frameSequence++
    const detectionStart = performance.now()
    detectionBracket.begin()
    slot.peakStage.encode()
    detectionBracket.end()
    const detectionEncodeMs = performance.now() - detectionStart

    const readbackStart = performance.now()
    const readback =
      onIdentity !== undefined
        ? await slot.peakStage.readPeaks()
        : {
            peaks: [],
            count: 0,
            candidateCount: 0,
            rejectedCount: 0,
            bytesCopied: 0,
            truncated: false,
          }
    const readbackMs = performance.now() - readbackStart

    // The 4 sigma filter is CPU-only; the identity pipeline receives the
    // surviving points, exactly like the Python model.
    const neighbourStart = performance.now()
    const filterResult: RoiNeighbourFilterResult =
      onIdentity !== undefined
        ? await filterRefinedPeaksInWorker(readback.peaks)
        : {
            keep: [],
            kept: [],
            zScores: new Float64Array(0),
            trimmedCounts: new Uint8Array(0),
          }
    const detectedPoints = filterResult.kept
    const neighbourFilterMs = performance.now() - neighbourStart
    const peakAnalysis: RoiPeakFrameAnalysis = {
      frameId,
      peaks: readback.peaks,
      filter: filterResult,
    }

    // Start identity now. It runs in a worker while the previous frame is
    // presented, and while the next frame is encoded.
    let identity: Promise<RoiGraphData | undefined> | undefined
    if (onIdentity !== undefined) {
      try {
        identity = Promise.resolve(onIdentity(detectedPoints, frameId)).catch((cause: unknown) => {
          console.error('[roi-lab] identity callback failed', cause)
          return undefined
        })
      } catch (cause: unknown) {
        console.error('[roi-lab] identity callback failed', cause)
      }
    }

    // Present the frame whose identity is ready. Its slot still holds that
    // frame's pixels, so the overlay and the image always belong together.
    let identityMs = 0
    let renderMs = 0
    let presentedFrameId: number | undefined
    if (pendingFrame !== undefined) {
      const waitStart = performance.now()
      const graph = await pendingFrame.identity
      identityMs = performance.now() - waitStart
      const renderStart = performance.now()
      renderBracket.begin()
      if (!destroyed) {
        presentFrame(pendingFrame.slot, graph)
        presentedPeakAnalysis = pendingFrame.analysis
        presentedFrameId = pendingFrame.frameId
      }
      renderBracket.end()
      renderMs = performance.now() - renderStart
    } else if (identity === undefined) {
      // No identity stage at all: present this frame directly.
      if (!destroyed) {
        presentFrame(slot, undefined)
        presentedPeakAnalysis = peakAnalysis
      }
    }
    pendingFrame = identity === undefined ? undefined : { slot, frameId, identity, analysis: peakAnalysis }

    // Let the queued readbacks and final render finish before a caller can
    // destroy this pipeline during a resolution/stream change.
    const frameDrainStart = performance.now()
    await device.queue.onSubmittedWorkDone()
    const frameDrainMs = performance.now() - frameDrainStart

    // Histogram work is queued after the overlay; only its CPU analysis is
    // asynchronous, and it feeds the next frame's thresholds.
    const histogramStart = performance.now()
    if (!destroyed && frameSerial % ROI_HISTOGRAM_FRAME_INTERVAL === 0) {
      histogramResetPipeline.with(histogramResetBindGroup).dispatchThreads(ROI_HISTOGRAM_BINS)
      histogramAccumulatePipeline.with(slot.histogramAccumulateBindGroup).dispatchThreads(cameraWidth, cameraHeight)
      scheduleHistogramReadback()
    }
    const histogramMs = performance.now() - histogramStart
    readGpuTimings()

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
      detectionEncodeMs +
      readbackMs +
      neighbourFilterMs +
      identityMs +
      renderMs +
      frameDrainMs +
      histogramMs
    const steps = smoothTimings({
      statsMs,
      preprocessMs,
      queueWaitMs,
      inferenceRunMs,
      inferenceDrainMs,
      inferenceMs,
      cameraCopyMs,
      debugInputMs,
      maskMs,
      backgroundMs,
      detectionEncodeMs,
      readbackMs,
      readbackCandidates: readback.candidateCount,
      readbackRejected: readback.rejectedCount,
      readbackBytes: readback.bytesCopied,
      readbackTruncated: readback.truncated,
      neighbourFilterMs,
      identityMs,
      identityLagFrames: presentedFrameId === undefined ? 0 : frameId - presentedFrameId,
      renderMs,
      frameDrainMs,
      histogramMs,
      accountedMs,
      overheadMs: totalMs - accountedMs,
      totalMs,
      frameId,
      gpuMs: { ...gpuMs },
    })
    return {
      inferenceMs: steps.inferenceMs,
      totalMs: steps.totalMs,
      readbackMs: steps.readbackMs,
      identityMs: steps.identityMs,
      frameId,
      steps,
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
      // Re-render the cached frame so view switches still work while paused.
      redrawLastFrame()
    },
    setViewport(zoom, panX, panY) {
      viewZoom = zoom
      viewPanX = panX
      viewPanY = panY
      writeViewParams()
      for (const slot of slots) {
        slot.peakCircleStage.setViewport(zoom, panX, panY)
      }
      graphStage.setViewport(zoom, panX, panY)
      if (paused) {
        redrawLastFrame()
      }
    },
    setSelectedPeak(peak) {
      for (const slot of slots) {
        slot.peakCircleStage.setSelectedPeak(peak)
      }
      if (paused) {
        redrawLastFrame()
      }
    },
    setPaused(value) {
      paused = value
    },
    getPresentedPeakAnalysis() {
      return presentedPeakAnalysis
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
      imageStatsSamples.destroy()
      imageStatsBuffer.destroy()
      timingQuerySet?.destroy()
      timingResolveBuffer?.destroy()
      timingReadBuffer?.destroy()
      for (const slot of slots) {
        slot.maskTexture.destroy()
        slot.modelInputTexture.destroy()
        slot.cameraTexture.destroy()
        slot.backgroundStage.destroy()
        slot.peakStage.destroy()
      }
      graphStage.destroy()
      msaaTexture.destroy()
      disposeIdentityWorker()
    },
  }
}
