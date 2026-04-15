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

    // Debug: show label value as grayscale
    // 1 = edge (white), INVALID = mid-gray, propagated = scaled
    const isInvalid = label === d.u32(0xFFFFFFFF);
    const isEdge = label === d.u32(1);

    // Debug colors: red=INVALID, white=edge(1), blue=ANY propagated
    if (isInvalid) {
      return d.vec4f(d.f32(1), d.f32(0), d.f32(0), d.f32(1)); // red
    }
    if (isEdge) {
      return d.vec4f(d.f32(1), d.f32(1), d.f32(1), d.f32(1)); // white
    }
    // Any other label = propagated
    return d.vec4f(d.f32(0), d.f32(0), d.f32(1), d.f32(1)); // blue
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
