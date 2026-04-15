// Debug visualization pipeline: label buffer → stable pseudo-random color per component
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';
import { clamp, floor } from 'typegpu/std';
import { COMPONENT_LABEL_INVALID } from '../contour';

/** 32→32 mix (Murmur3 finalizer style); same label → same color. */
function hashU32(x: ReturnType<typeof d.u32>) {
  'use gpu';
  const h0 = x ^ (x >> d.u32(16));
  const h1 = h0 * d.u32(0x7feb352d);
  const h2 = h1 ^ (h1 >> d.u32(15));
  const h3 = h2 * d.u32(0x846ca68b);
  return h3 ^ (h3 >> d.u32(16));
}

export function createLabelVizPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  labelVizLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const labelVizFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    // i32 for last-index: (u32)width - 1 wraps at 0; clamp max uses signed last index → f32.
    const wi = d.i32(width);
    const hi = d.i32(height);
    const maxPx = d.f32(wi - d.i32(1));
    const maxPy = d.f32(hi - d.i32(1));
    // Plain u32(uv * size) is unstable at column/row boundaries (float error → wrong neighbor sample → 1px streaks).
    const px = d.u32(
      floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)),
    );
    const py = d.u32(
      floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)),
    );
    const idx = py * d.u32(wi) + px;
    const label = labelVizLayout.$.labelBuffer[idx];

    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.14), d.f32(1));
    }

    const h = hashU32(label);
    const r = d.f32(h & d.u32(255)) / d.f32(255);
    const g = d.f32((h >> d.u32(8)) & d.u32(255)) / d.f32(255);
    const b = d.f32((h >> d.u32(16)) & d.u32(255)) / d.f32(255);
    return d.vec4f(r, g, b, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
