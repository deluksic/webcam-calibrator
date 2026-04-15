// Grayscale render pipeline: grayBuffer → canvas
import { tgpu, d } from 'typegpu';
import { common } from 'typegpu';

export function createGrayRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  grayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const grayFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    const px = d.u32(i.uv.x * d.f32(width));
    const py = d.u32(i.uv.y * d.f32(height));
    const idx = py * d.u32(width) + px;
    const gray = grayLayout.$.grayBuffer[idx];
    return d.vec4f(gray, gray, gray, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: grayFrag,
    targets: { format: presentationFormat },
  });
}
