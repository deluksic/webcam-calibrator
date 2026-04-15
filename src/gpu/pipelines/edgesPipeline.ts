// Edges pipeline: sobelBuffer → edges canvas
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';

export function createEdgesPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgesLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const edgesFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    // Compute pixel position from UV
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const mag = edgesLayout.$.sobelBuffer[idx];
    return d.vec4f(mag, mag, mag, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: edgesFrag,
    targets: { format: presentationFormat },
  });
}
