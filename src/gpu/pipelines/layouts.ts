// Pipeline bind group layouts
import { tgpu, d } from 'typegpu';
import { HISTOGRAM_BINS } from './constants';

export function createLayouts(root: Awaited<ReturnType<typeof tgpu.init>>, histogramSchema: ReturnType<typeof d.arrayOf>) {
  // Copy layout (render: external → grayTex)
  const copyLayout = tgpu.bindGroupLayout({
    cameraTex: { externalTexture: d.textureExternal() },
    sampler: { sampler: 'filtering' },
  });

  // Gray tex → buffer layout
  const grayTexToBufferLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.texture2d(d.f32), access: 'readonly' },
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  });

  // Sobel layout (compute: buffer → buffer)
  const sobelLayout = tgpu.bindGroupLayout({
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    sobelBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  });

  // Histogram layout (compute: buffer → atomic buffer)
  const histogramLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    histogram: { storage: histogramSchema, access: 'mutable' },
  });

  // Histogram reset layout
  const histogramResetLayout = tgpu.bindGroupLayout({
    histogram: { storage: histogramSchema, access: 'mutable' },
  });

  // Display layout (edges)
  const edgesLayout = tgpu.bindGroupLayout({
    filteredBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  });

  // Histogram display layout
  const histogramDisplayLayout = tgpu.bindGroupLayout({
    histogram: { storage: histogramSchema, access: 'mutable' },
    thresholdBin: { uniform: d.u32 },
  });

  // Edge filter layout (compute: threshold filter)
  const edgeFilterLayout = tgpu.bindGroupLayout({
    sobelBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    threshold: { uniform: d.f32 },
    filteredBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
  });

  return {
    copyLayout,
    grayTexToBufferLayout,
    sobelLayout,
    histogramLayout,
    histogramResetLayout,
    edgesLayout,
    histogramDisplayLayout,
    edgeFilterLayout,
  };
}
