// Edge render pipeline: filteredBuffer → edges canvas
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';
import { clamp, floor, select } from 'typegpu/std';

export function createEdgesPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgesLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
  binaryMask: boolean,
) {
  const edgesFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const wi = d.i32(width);
    const hi = d.i32(height);
    const maxPx = d.f32(wi - d.i32(1));
    const maxPy = d.f32(hi - d.i32(1));
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)));
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)));
    const idx = py * d.u32(wi) + px;
    const mag = edgesLayout.$.filteredBuffer[idx];
    const v = binaryMask
      ? select(d.f32(0), d.f32(1), mag > d.f32(0))
      : mag;
    return d.vec4f(v, v, v, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: edgesFrag,
    targets: { format: presentationFormat },
  });
}
