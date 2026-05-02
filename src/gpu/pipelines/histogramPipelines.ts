// Histogram pipelines: reset and accumulate
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { atomicAdd, atomicLoad, atomicStore, length, log2 } from 'typegpu/std'

import type { RenderColorAttachment } from '@/gpu/renderEncodeTypes'

/** Bin count for magnitude histogram and bar-chart vertex instances. */
export const HISTOGRAM_BINS = 256

/** Histogram chart dimensions in pixels (bar display shader). */
export const HIST_WIDTH = 512
export const HIST_HEIGHT = 120

/** Full-frame compute tile; keep in sync with other camera passes using [16,16,1] and `computeDispatch2d` in cameraFrame. */
const FULL_FRAME_WG = 16

export const histogramStorageSchema = d.arrayOf(d.atomic(d.u32), HISTOGRAM_BINS)

export const histogramComputeLayout = tgpu.bindGroupLayout({
  sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  histogram: { storage: histogramStorageSchema, access: 'mutable' },
})

export const histogramResetLayout = tgpu.bindGroupLayout({
  histogram: { storage: histogramStorageSchema, access: 'mutable' },
})

export const histogramDisplayLayout = tgpu.bindGroupLayout({
  histogram: { storage: histogramStorageSchema, access: 'mutable' },
  thresholdBin: { uniform: d.u32 },
})

export type HistogramResetBindResources = ExtractBindGroupInputFromLayout<typeof histogramResetLayout.entries>
export type HistogramComputeBindResources = ExtractBindGroupInputFromLayout<typeof histogramComputeLayout.entries>
export type HistogramDisplayBindResources = ExtractBindGroupInputFromLayout<typeof histogramDisplayLayout.entries>

export function createHistogramResetPipeline(root: TgpuRoot) {
  const histogramResetKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu'
    const binIdx = input.gid.x
    if (d.i32(binIdx) >= d.i32(HISTOGRAM_BINS)) {
      return
    }
    atomicStore(histogramResetLayout.$.histogram[binIdx]!, d.u32(0))
  })

  return root.createComputePipeline({ compute: histogramResetKernel })
}

export function createHistogramAccumulatePipeline(root: TgpuRoot, width: number, height: number) {
  const histogramKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
  })((input) => {
    'use gpu'
    const zero = d.u32(0)
    const tileWidth = d.u32(16)
    const tileHeight = d.u32(16)
    const numBinsI = d.i32(HISTOGRAM_BINS)

    const startX = input.gid.x * tileWidth
    const startY = input.gid.y * tileHeight

    for (let dy = zero; dy < tileHeight; dy = dy + d.u32(1)) {
      for (let dx = zero; dx < tileWidth; dx = dx + d.u32(1)) {
        const px = startX + dx
        const py = startY + dy

        if (d.i32(px) >= d.i32(width) || d.i32(py) >= d.i32(height)) {
          continue
        }

        const idx = d.u32(d.i32(py) * d.i32(width) + d.i32(px))
        // Clamp magnitude to [0, 1] to prevent overflow into last bucket
        let mag = length(histogramComputeLayout.$.sobelBuffer[idx]!)
        if (mag > d.f32(1.0)) {
          mag = d.f32(1.0)
        }
        const bin = d.u32(mag * d.f32(HISTOGRAM_BINS))
        let clampedBin = bin
        if (d.i32(bin) >= numBinsI) {
          clampedBin = d.u32(numBinsI - d.i32(1))
        }

        atomicAdd(histogramComputeLayout.$.histogram[clampedBin]!, d.u32(1))
      }
    }
  })

  return root.createComputePipeline({ compute: histogramKernel })
}

const { floor } = Math

/** Histogram bar chart (instance draw into hist canvas). */
export function createHistogramRenderPipeline(
  root: TgpuRoot,
  presentationFormat: GPUTextureFormat,
  totalPixels: number,
) {
  const maxCount = floor(totalPixels * 0.1)

  const histogramVert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      uv: d.vec2f,
      barIndex: d.location(1, d.f32),
      position: d.builtin.position,
    },
  })((i) => {
    'use gpu'
    const idx = i.vertexIndex

    const uvs = [
      d.vec2f(0.0, 0.0),
      d.vec2f(1.0, 0.0),
      d.vec2f(1.0, 1.0),
      d.vec2f(0.0, 0.0),
      d.vec2f(1.0, 1.0),
      d.vec2f(0.0, 1.0),
    ]
    const uv = uvs[idx]!
    const histW = d.f32(HIST_WIDTH)
    const histH = d.f32(HIST_HEIGHT)
    const numBars = d.f32(HISTOGRAM_BINS)
    const barW = histW / numBars

    const barPxX = d.f32(i.instanceIndex) * barW + uv.x * barW
    const barPxY = uv.y * histH
    const clipX = (barPxX / histW) * d.f32(2.0) - d.f32(1.0)
    const clipY = (barPxY / histH) * d.f32(2.0) - d.f32(1.0)

    return {
      uv,
      barIndex: d.f32(i.instanceIndex),
      position: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
    }
  })

  const histogramFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f), barIndex: d.location(1, d.f32) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const bin = d.u32(i.barIndex)
    const countU32 = atomicLoad(histogramDisplayLayout.$.histogram[bin]!)

    const countF = d.f32(countU32)
    const logCountPlus1 = log2(countF + d.f32(1.0))
    const logMax = log2(d.f32(maxCount) + d.f32(1.0))
    const normalizedHeight = logCountPlus1 / logMax

    const isThreshold = bin >= histogramDisplayLayout.$.thresholdBin

    if (i.uv.y > normalizedHeight) {
      if (isThreshold) {
        return d.vec4f(d.f32(1.0), d.f32(0.0), d.f32(0.0), d.f32(1))
      }
      return d.vec4f(d.f32(0.1), d.f32(0.1), d.f32(0.15), d.f32(0))
    }

    if (isThreshold) {
      return d.vec4f(d.f32(1.0), d.f32(0.0), d.f32(0.0), d.f32(1))
    }
    return d.vec4f(d.f32(0.29), d.f32(0.62), d.f32(1.0), d.f32(1))
  })

  return root.createRenderPipeline({
    vertex: histogramVert,
    fragment: histogramFrag,
    targets: { format: presentationFormat },
  })
}

/** Allocates histogram storage, compute + bar-chart render pipelines, and bind groups; reads `sobelBuffer` (upstream). */
export function createHistogramStage(
  root: TgpuRoot,
  width: number,
  height: number,
  sobelBuffer: HistogramComputeBindResources['sobelBuffer'],
  presentationFormat: GPUTextureFormat,
) {
  const buffer = root.createBuffer(histogramStorageSchema).$usage('storage')
  const thresholdBinBuffer = root.createBuffer(d.u32).$usage('uniform')
  const resetPipeline = createHistogramResetPipeline(root)
  const resetBindGroup = root.createBindGroup(histogramResetLayout, { histogram: buffer })
  const computePipeline = createHistogramAccumulatePipeline(root, width, height)
  const computeBindGroup = root.createBindGroup(histogramComputeLayout, {
    sobelBuffer,
    histogram: buffer,
  })
  const displayBindGroup = root.createBindGroup(histogramDisplayLayout, {
    histogram: buffer,
    thresholdBin: thresholdBinBuffer,
  })
  const displayPipeline = createHistogramRenderPipeline(root, presentationFormat, width * height)
  const wgX = Math.ceil(width / FULL_FRAME_WG)
  const wgY = Math.ceil(height / FULL_FRAME_WG)
  const encodeAccumulateCompute = (pass: GPUComputePassEncoder) => {
    resetPipeline.with(pass).with(resetBindGroup).dispatchWorkgroups(HISTOGRAM_BINS)
    computePipeline.with(pass).with(computeBindGroup).dispatchWorkgroups(wgX, wgY)
  }
  const encodeDisplay = (enc: GPUCommandEncoder, colorAttachment: RenderColorAttachment) => {
    displayPipeline.with(enc).withColorAttachment(colorAttachment).with(displayBindGroup).draw(6, HISTOGRAM_BINS)
  }
  return {
    buffer,
    thresholdBinBuffer,
    encodeAccumulateCompute,
    encodeDisplay,
  }
}

/** Default percentile for adaptive edge threshold from CPU-side histogram readback. */
export const THRESHOLD_PERCENTILE = 0.95

export function computeThreshold(histogramData: number[], percentile: number = THRESHOLD_PERCENTILE): number {
  const totalPixels = histogramData.reduce((a, b) => a + b, 0)
  const targetCount = totalPixels * percentile

  let cumulative = 0
  for (let i = 0; i < histogramData.length; i++) {
    cumulative += histogramData[i]!
    if (cumulative >= targetCount) {
      return i / 255.0
    }
  }
  return 0.5
}
