// Debug render pipeline: filtered edge buffer → canvas
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';

export function createFilteredRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  filteredLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const filteredFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const val = filteredLayout.$.filteredBuffer[idx];
    // Show: white = edge pixel (1.0), black = background (0.0)
    return d.vec4f(val, val, val, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: filteredFrag,
    targets: { format: presentationFormat },
  });
}