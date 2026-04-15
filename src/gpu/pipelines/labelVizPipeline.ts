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

    // Show: white = edge, dark = propagated label, gray = invalid
    // Use label value directly (mod 256) for visible variation
    if (isInvalid) {
      return d.vec4f(d.f32(0.5), d.f32(0.5), d.f32(0.5), d.f32(1));
    }
    if (isEdge) {
      return d.vec4f(d.f32(1), d.f32(1), d.f32(1), d.f32(1));
    }
    // Propagated: show as dark gray scaled by label (labels 1-10 become 0.1-1.0)
    const gray = d.f32(label) / d.f32(10.0);
    return d.vec4f(gray, gray, gray, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
