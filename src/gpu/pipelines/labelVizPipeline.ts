// Debug visualization pipeline: label buffer → colored overlay
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';

export function createLabelVizPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  labelVizLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  // Pseudocolor mapping for labels (hash-based color assignment)
  const labelVizFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const label = labelVizLayout.$.labelBuffer[idx];

    // INVALID label = dark gray (visible background for non-edge pixels)
    if (label === d.u32(0xFFFFFFFF)) {
      return d.vec4f(d.f32(0.05), d.f32(0.05), d.f32(0.08), d.f32(1));
    }

    // Hash-based pseudocolor for each unique label
    // Use explicit f32 arithmetic to avoid implicit conversion warnings
    const labelF = d.f32(label);
    const r = (labelF / d.f32(7.0)) % d.f32(7.0) / d.f32(7.0);
    const g = (labelF / d.f32(49.0)) % d.f32(7.0) / d.f32(7.0);
    const b = (labelF / d.f32(343.0)) % d.f32(7.0) / d.f32(7.0);

    // Boost saturation
    const boost = d.f32(1.5);
    return d.vec4f(
      r * boost,
      g * boost,
      b * boost,
      d.f32(0.8),
    );
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}