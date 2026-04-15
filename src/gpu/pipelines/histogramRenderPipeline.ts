// Histogram render pipeline: histogramBuffer → histogram bars
import { tgpu, d } from 'typegpu';
import { atomicLoad, log2 } from 'typegpu/std';
import { HISTOGRAM_BINS, HIST_WIDTH, HIST_HEIGHT } from './constants';

export function createHistogramRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  histogramDisplayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  presentationFormat: GPUTextureFormat,
  totalPixels: number,
) {
  // Log-scale histogram with max referring to fraction of total pixels
  // Pre-compute the divisor: log2(maxCount + 1)
  const maxCount = Math.floor(totalPixels * 0.1);
  const logMaxDivisor = Math.log2(maxCount + 1);

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
    // Vertices: 0,1,2 (first triangle) + 3,4,5 (second triangle)
    // Mapping to quad: (0,0), (1,0), (1,1), (0,0), (1,1), (0,1)
    // Pattern tables indexed by vertexIndex
    const idx = i.vertexIndex;
    const two = d.u32(2);
    const six = d.u32(6);

    // localU pattern: [0, 1, 1, 0, 1, 0] for idx 0,1,2,3,4,5
    // Simpler: U = 0 for idx=0,3; U = 1 for idx=1,2,4; U = anything for idx=5
    // Actually: idx=0->0, idx=1->1, idx=2->1, idx=3->0, idx=4->1, idx=5->0
    // This is: U = (idx < 3) ? (idx % 2 == 0 ? 0 : 1) : (idx == 3 ? 0 : (idx == 4 ? 1 : 0))
    // Simpler: U = 1 - (idx % 2), then fix idx=4,5
    // Let's just use explicit checks
    let localU = d.f32(0.0);
    let localV = d.f32(0.0);

    if (idx == d.u32(0)) { localU = d.f32(0.0); localV = d.f32(0.0); }
    else if (idx == d.u32(1)) { localU = d.f32(1.0); localV = d.f32(0.0); }
    else if (idx == d.u32(2)) { localU = d.f32(1.0); localV = d.f32(1.0); }
    else if (idx == d.u32(3)) { localU = d.f32(0.0); localV = d.f32(0.0); }
    else if (idx == d.u32(4)) { localU = d.f32(1.0); localV = d.f32(1.0); }
    else { localU = d.f32(0.0); localV = d.f32(1.0); }

    const histW = d.f32(HIST_WIDTH);
    const histH = d.f32(HIST_HEIGHT);
    const numBars = d.f32(HISTOGRAM_BINS);
    const barW = histW / numBars;

    const barPxX = (d.f32(i.instanceIndex) * barW) + (localU * barW);
    const barPxY = localV * histH;
    const clipX = (barPxX / histW) * d.f32(2.0) - d.f32(1.0);
    const clipY = d.f32(1.0) - (barPxY / histH) * d.f32(2.0);

    return {
      uv: d.vec2f(localU, localV),
      barIndex: d.f32(i.instanceIndex),
      position: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
    };
  });

  // Pre-bake the divisor into the shader (no runtime log2 needed)
  const fragLogMax = d.f32(logMaxDivisor);

  const histogramFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f), barIndex: d.location(1, d.f32) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const bin = d.u32(i.barIndex);
    const countU32 = atomicLoad(histogramDisplayLayout.$.histogram[bin]);

    // Log-scale normalization with explicit u32→f32 conversion
    // log2(count + 1) / logMax where count is u32
    const countF = d.f32(countU32);
    const logCountPlus1 = log2(countF + d.f32(1.0));
    const normalizedHeight = logCountPlus1 / fragLogMax;

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
