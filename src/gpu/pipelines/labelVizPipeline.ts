// Debug visualization pipeline: label buffer → colored overlay
import { tgpu, d, std } from 'typegpu';
import { common } from 'typegpu';

export function createLabelVizPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  labelVizLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  // Binary visualization: white = edge, dark = background
  const labelVizFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const label = labelVizLayout.$.labelBuffer[idx];

    // Pseudocolor based on label (shows connected components)
    const isValid = label !== d.u32(0xFFFFFFFF);
    const labelF = d.f32(label);
    const r = std.select(d.f32(0), (labelF / d.f32(7.0)) % d.f32(7.0) / d.f32(7.0), isValid);
    const g = std.select(d.f32(0), (labelF / d.f32(49.0)) % d.f32(7.0) / d.f32(7.0), isValid);
    const b = std.select(d.f32(0), (labelF / d.f32(343.0)) % d.f32(7.0) / d.f32(7.0), isValid);
    return d.vec4f(r, g, b, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
