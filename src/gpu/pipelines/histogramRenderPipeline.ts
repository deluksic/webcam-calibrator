// Histogram render pipeline: histogramBuffer → histogram bars
import { tgpu, d } from 'typegpu'
import { atomicLoad, log2 } from 'typegpu/std'

import { HISTOGRAM_BINS, HIST_WIDTH, HIST_HEIGHT } from '@/gpu/pipelines/constants'

const { floor } = Math

export function createHistogramRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  histogramDisplayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  presentationFormat: GPUTextureFormat,
  totalPixels: number,
) {
  // Log-scale histogram with max referring to fraction of total pixels
  // Pre-compute the divisor: log2(maxCount + 1)
  const maxCount = floor(totalPixels * 0.1)

  // ── Pass: render histogram bars using instanceIndex ───────────────────
  // 256 instances, each bar renders as a vertical rectangle
  // No vertex buffer needed - instance index directly accesses the histogram buffer
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
    // Vertices: 0,1,2 (first triangle) + 3,4,5 (second triangle)
    // Mapping to quad: (0,0), (1,0), (1,1), (0,0), (1,1), (0,1)
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
    // Clip space: -1 (bottom) to 1 (top)
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
    const countU32 = atomicLoad(histogramDisplayLayout.$.histogram[bin])

    // Log-scale normalization
    const countF = d.f32(countU32)
    const logCountPlus1 = log2(countF + d.f32(1.0))
    const logMax = log2(d.f32(maxCount) + d.f32(1.0))
    const normalizedHeight = logCountPlus1 / logMax

    // Mark threshold bar red
    const isThreshold = bin >= histogramDisplayLayout.$.thresholdBin

    // Clip bars above their height (make them empty/transparent)
    if (i.uv.y > normalizedHeight) {
      if (isThreshold) {
        return d.vec4f(d.f32(1.0), d.f32(0.0), d.f32(0.0), d.f32(1))
      }
      return d.vec4f(d.f32(0.1), d.f32(0.1), d.f32(0.15), d.f32(0))
    }

    // Blue bars for histogram below threshold, red for threshold and above
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
