// Histogram pipelines: reset and accumulate
import { tgpu, d } from 'typegpu';
import { atomicAdd, atomicStore } from 'typegpu/std';
import { HISTOGRAM_BINS } from './constants';

export function createHistogramResetPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  histogramResetLayout: ReturnType<typeof tgpu.bindGroupLayout>,
) {
  const histogramResetKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [1, 1, 1],
  })((input) => {
    'use gpu';
    const binIdx = input.gid.x;
    if (binIdx >= d.u32(HISTOGRAM_BINS)) { return; }
    atomicStore(histogramResetLayout.$.histogram[binIdx], d.u32(0));
  });

  return root.createComputePipeline({ compute: histogramResetKernel });
}

export function createHistogramAccumulatePipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  histogramLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const histogramKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    const zero = d.u32(0);
    const tileWidth = d.u32(16);
    const tileHeight = d.u32(16);
    const numBins = d.u32(HISTOGRAM_BINS);

    const startX = input.gid.x * tileWidth;
    const startY = input.gid.y * tileHeight;

    for (let dy = zero; dy < tileHeight; dy = dy + d.u32(1)) {
      for (let dx = zero; dx < tileWidth; dx = dx + d.u32(1)) {
        const px = startX + dx;
        const py = startY + dy;

        if (px >= d.u32(width) || py >= d.u32(height)) { continue; }

        const idx = py * d.u32(width) + px;
        // Clamp magnitude to [0, 1] to prevent overflow into last bucket
        let mag = histogramLayout.$.sobelBuffer[idx];
        if (mag > d.f32(1.0)) { mag = d.f32(1.0); }
        const bin = d.u32(mag * d.f32(HISTOGRAM_BINS));
        let clampedBin = bin;
        if (bin >= numBins) { clampedBin = numBins - d.u32(1); }

        atomicAdd(histogramLayout.$.histogram[clampedBin], d.u32(1));
      }
    }
  });

  return root.createComputePipeline({ compute: histogramKernel });
}
