// Sobel buffer render pipeline: sobelBuffer → canvas
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';

export function createSobelRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  sobelLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const sobelFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const mag = sobelLayout.$.sobelBuffer[idx];
    return d.vec4f(mag, mag, mag, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: sobelFrag,
    targets: { format: presentationFormat },
  });
}
