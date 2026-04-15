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
    const labelF = d.f32(label);
    const isInvalid = label === d.u32(0xFFFFFFFF);
    const isEdge = label === d.u32(1);
    const isOther = !isInvalid && !isEdge;
    const normalized = labelF / d.f32(100.0);
    const gray = std.select(std.select(normalized, d.f32(1), isEdge), d.f32(0.5), isInvalid);
    return d.vec4f(gray, gray, gray, d.f32(1));
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
