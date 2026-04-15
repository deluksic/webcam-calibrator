// Histogram render pipeline: histogramBuffer → histogram bars
import { tgpu, d } from 'typegpu';
import { atomicLoad } from 'typegpu/std';
import { HISTOGRAM_BINS, HIST_WIDTH, HIST_HEIGHT } from './constants';

export function createHistogramRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  histogramDisplayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  presentationFormat: GPUTextureFormat,
  totalPixels: number,
) {
  // Log-scale histogram with max referring to fraction of total pixels
  const maxCount = Math.floor(totalPixels * 0.1);
  const logMax = Math.log2(maxCount + 1);

  // ── Pass: render histogram bars using instanceIndex ───────────────────
  // 256 instances, each bar renders as a vertical rectangle
  // No vertex buffer needed - instance index directly accesses the histogram buffer
  const histogramVert = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      uv: d.vec2f,
      barIndex: d.location(1, d.f32),
      position: d.builtin.position,
    },
  })((i) => {
    'use gpu';
    const vertInBar = i.vertexIndex % d.u32(6);
    const localU = d.f32(vertInBar % d.u32(2));
    const localV = d.f32(vertInBar / d.u32(2));

    const histW = d.f32(HIST_WIDTH);
    const histH = d.f32(HIST_HEIGHT);
    const numBars = d.f32(HISTOGRAM_BINS);
    const barW = histW / numBars;

    const barPxX = (d.f32(i.instanceIndex) * barW) + (localU * barW);
    // Flip vertical: barPxY = 0 at bottom, histH at top
    // localV = 0 → bottom of bar rectangle, localV = 1 → top of bar rectangle
    const barPxY = (d.f32(1) - localV) * histH;
    const clipX = (barPxX / histW) * d.f32(2.0) - d.f32(1.0);
    const clipY = d.f32(1.0) - (barPxY / histH) * d.f32(2.0);
    // UV.y: 0 at bottom (bar bottom), 1 at top (bar top)

    return {
      uv: d.vec2f(localU, localV),
      barIndex: d.f32(i.instanceIndex),
      position: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
    };
  });

  const histogramFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f), barIndex: d.location(1, d.f32) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const bin = d.u32(i.barIndex);
    const count = atomicLoad(histogramDisplayLayout.$.histogram[bin]);

    // Log-scale normalization
    const normalizedHeight = d.f32(Math.log2(count + 1)) / d.f32(logMax);

    // Clip bars above their height (make them empty/transparent)
    if (i.uv.y > normalizedHeight) {
      return d.vec4f(d.f32(0.1), d.f32(0.1), d.f32(0.15), d.f32(0));
    }

    // Blue bars for histogram
    return d.vec4f(d.f32(0.29), d.f32(0.62), d.f32(1.0), d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: histogramVert,
    fragment: histogramFrag,
    targets: { format: presentationFormat },
  });
}
