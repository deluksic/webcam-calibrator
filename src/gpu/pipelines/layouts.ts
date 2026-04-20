// Pipeline bind group layouts
import { tgpu, d } from 'typegpu'

export function createLayouts(histogramSchema: ReturnType<typeof d.arrayOf>) {
  // Copy layout (render: external → grayTex)
  const copyLayout = tgpu.bindGroupLayout({
    cameraTex: { externalTexture: d.textureExternal() },
    sampler: { sampler: 'filtering' },
  })

  // Gray tex → buffer layout
  const grayTexToBufferLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.texture2d(d.f32), access: 'readonly' },
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  })

  // Sobel layout (compute: buffer → buffer) — stores (gx, gy) per pixel; magnitude = length(vec2).
  const sobelLayout = tgpu.bindGroupLayout({
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
  })

  // Histogram layout (compute: buffer → atomic buffer)
  const histogramLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    histogram: { storage: histogramSchema, access: 'mutable' },
  })

  // Histogram reset layout
  const histogramResetLayout = tgpu.bindGroupLayout({
    histogram: { storage: histogramSchema, access: 'mutable' },
  })

  // Display layout (edges) — HSV colorized by gradient direction.
  // Both fields are vec2f: sobelBuffer provides gradient for magnitude, filteredBuffer is
  // also vec2f so it can carry gradients (e.g. from dilated edge pipeline).
  const edgesLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    filteredBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  })

  // Histogram display layout
  const histogramDisplayLayout = tgpu.bindGroupLayout({
    histogram: { storage: histogramSchema, access: 'mutable' },
    thresholdBin: { uniform: d.u32 },
  })

  // Edge filter layout (compute: threshold filter)
  const edgeFilterLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    threshold: { uniform: d.f32 },
    filteredBuffer: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
  })

  // Tangent-only max merge: bridges gaps along the edge without thickening normal to the gradient.
  // dst outputs gradient (vec2f) so it can be used directly for corner detection.
  const edgeDilateLayout = tgpu.bindGroupLayout({
    src: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    grad: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    threshold: { uniform: d.f32 },
    dst: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
  })

  // Label visualization layout
  const labelVizLayout = tgpu.bindGroupLayout({
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
  })

  // Grayscale render layout
  const grayRenderLayout = tgpu.bindGroupLayout({
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  })

  // Sobel render layout
  const sobelRenderLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  })

  // Filtered render layout (debug: show edge filter output)
  const filteredRenderLayout = tgpu.bindGroupLayout({
    filteredBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  })

  return {
    copyLayout,
    grayTexToBufferLayout,
    sobelLayout,
    histogramLayout,
    histogramResetLayout,
    edgesLayout,
    histogramDisplayLayout,
    edgeFilterLayout,
    edgeDilateLayout,
    labelVizLayout,
    grayRenderLayout,
    sobelRenderLayout,
    filteredRenderLayout,
  }
}
