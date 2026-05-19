import type { TgpuRoot } from 'typegpu'

/** Set true and reload. Logs pass GPU times every 30 frames (console.table: ms/frame + per-pass). */
export const GPU_PROFILING = false

const NS_TO_MS = 1e-6
const LOG_EVERY_FRAMES = 30
const MAX_PASS_SAMPLES = 32

type TimingBucket = { totalMs: number; count: number }
type PassSample = { name: string; beginIdx: number }

const buckets = new Map<string, TimingBucket>()
let framesSinceLog = 0
let warnedNoTimestamp = false
let loggedReady = false

let querySet: GPUQuerySet | undefined
let frameSamples: PassSample[] = []
let resolveChain: Promise<void> = Promise.resolve()

type TypegpuMeta = WeakMap<object, { name?: string }>

function pipelineName(pipeline: unknown, fallback?: string): string {
  const meta = (globalThis as { __TYPEGPU_META__?: TypegpuMeta }).__TYPEGPU_META__
  if (pipeline && typeof pipeline === 'object' && meta) {
    const name = meta.get(pipeline as object)?.name
    if (name) {
      return name
    }
  }
  return fallback ?? 'unnamed'
}

function resolvePassName(nameOrPipeline: string | unknown, descriptor?: { label?: string }): string {
  if (typeof nameOrPipeline === 'string') {
    return nameOrPipeline
  }
  return pipelineName(nameOrPipeline, descriptor?.label)
}

/** Call before encoding GPU work for the frame (creates query set when profiling is on). */
export function prepareGpuProfiling(root: TgpuRoot): void {
  ensureQuerySet(root)
}

function ensureQuerySet(root: TgpuRoot): boolean {
  if (!GPU_PROFILING) {
    return false
  }
  if (!root.enabledFeatures.has('timestamp-query')) {
    if (!warnedNoTimestamp) {
      warnedNoTimestamp = true
      console.warn('[gpu profile] timestamp-query not available — no timings')
    }
    return false
  }
  if (!querySet) {
    querySet = root.device.createQuerySet({ type: 'timestamp', count: MAX_PASS_SAMPLES * 2 })
    if (!loggedReady) {
      loggedReady = true
      console.info('[gpu profile] on — console.table every', LOG_EVERY_FRAMES, 'frames')
    }
  }
  return true
}

function recordSample(name: string, start: bigint, end: bigint): void {
  const ms = Number(end - start) * NS_TO_MS
  const prev = buckets.get(name)
  if (prev) {
    prev.totalMs += ms
    prev.count += 1
  } else {
    buckets.set(name, { totalMs: ms, count: 1 })
  }
}

/** One compute pass for a logical stage (e.g. all pointer-jump iterations). */
export function runComputeStage(
  enc: GPUCommandEncoder,
  nameOrPipeline: string | unknown,
  encode: (pass: GPUComputePassEncoder) => void,
  descriptor: GPUComputePassDescriptor = {},
): void {
  const pass = profileComputePass(enc, nameOrPipeline, descriptor)
  encode(pass)
  pass.end()
}

/** Begin a compute pass; second arg is a label string or a `.$name()`'d pipeline. */
export function profileComputePass(
  enc: GPUCommandEncoder,
  nameOrPipeline: string | unknown,
  descriptor: GPUComputePassDescriptor = {},
): GPUComputePassEncoder {
  const name = resolvePassName(nameOrPipeline, descriptor)
  if (!querySet) {
    return enc.beginComputePass(descriptor)
  }
  const beginIdx = frameSamples.length * 2
  if (beginIdx + 1 >= MAX_PASS_SAMPLES * 2) {
    return enc.beginComputePass(descriptor)
  }
  frameSamples.push({ name, beginIdx })
  return enc.beginComputePass({
    ...descriptor,
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: beginIdx,
      endOfPassWriteIndex: beginIdx + 1,
    },
  })
}

/** Begin a render pass; second arg is a label string or a `.$name()`'d pipeline. */
export function profileRenderPass(
  enc: GPUCommandEncoder,
  nameOrPipeline: string | unknown,
  descriptor: GPURenderPassDescriptor,
): GPURenderPassEncoder {
  const name = resolvePassName(nameOrPipeline, descriptor)
  if (!querySet) {
    return enc.beginRenderPass(descriptor)
  }
  const beginIdx = frameSamples.length * 2
  if (beginIdx + 1 >= MAX_PASS_SAMPLES * 2) {
    return enc.beginRenderPass(descriptor)
  }
  frameSamples.push({ name, beginIdx })
  return enc.beginRenderPass({
    ...descriptor,
    timestampWrites: {
      querySet,
      beginningOfPassWriteIndex: beginIdx,
      endOfPassWriteIndex: beginIdx + 1,
    },
  })
}

/** Call once per submitted frame after `queue.submit`. */
export function noteGpuProfileFrame(root: TgpuRoot): void {
  if (!GPU_PROFILING) {
    return
  }

  const profiling = ensureQuerySet(root)
  const samples = frameSamples
  frameSamples = []

  if (!profiling || samples.length === 0) {
    return
  }

  const device = root.device
  const queryCount = samples.length * 2
  resolveChain = resolveChain.then(async () => {
    const byteSize = queryCount * 8
    const resolveBuf = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    const readBuf = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    const enc = device.createCommandEncoder()
    enc.resolveQuerySet(querySet!, 0, queryCount, resolveBuf, 0)
    enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, byteSize)
    device.queue.submit([enc.finish()])
    await readBuf.mapAsync(GPUMapMode.READ)
    const times = new BigUint64Array(readBuf.getMappedRange())
    for (const { name, beginIdx } of samples) {
      recordSample(name, times[beginIdx]!, times[beginIdx + 1]!)
    }
    readBuf.unmap()
    resolveBuf.destroy()
    readBuf.destroy()

    framesSinceLog += 1
    if (framesSinceLog < LOG_EVERY_FRAMES || buckets.size === 0) {
      return
    }
    const frameCount = framesSinceLog
    framesSinceLog = 0

    const rows = [...buckets.entries()]
      .map(([name, { totalMs, count }]) => ({
        name,
        msPerFrame: totalMs / frameCount,
        msPerPass: count > 0 ? totalMs / count : 0,
        samples: count,
      }))
      .sort((a, b) => b.msPerFrame - a.msPerFrame)

    const totalMsPerFrame = rows.reduce((sum, row) => sum + row.msPerFrame, 0)
    const formatted = rows.map((row) => ({
      name: row.name,
      'ms/frame': row.msPerFrame.toFixed(3),
      'per pass': row.msPerPass.toFixed(3),
      samples: row.samples,
    }))
    formatted.push({
      name: 'TOTAL',
      'ms/frame': totalMsPerFrame.toFixed(3),
      'per pass': '',
      samples: frameCount,
    })

    console.table(formatted)
    buckets.clear()
  })
}
