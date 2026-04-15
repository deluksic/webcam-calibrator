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

    // Hash label to ~9 distinct color buckets
    // Connected components will have labels from the same edge seed cluster
    const isValid = label !== d.u32(0xFFFFFFFF);
    const labelHash = (label / d.u32(10000)) % d.u32(9);
    const c0 = d.f32(0.2);
    const c1 = d.f32(0.8);
    // 3x3 color matrix for 9 buckets
    const r = std.select(c0, c1, labelHash < d.u32(3));
    const g = std.select(c0, c1, (labelHash / d.u32(3)) % d.u32(3) !== d.u32(0));
    const b = std.select(c0, c1, labelHash % d.u32(3) !== d.u32(0));
    return d.vec4f(
      std.select(d.f32(0.02), r, isValid),
      std.select(d.f32(0.02), g, isValid),
      std.select(d.f32(0.02), b, isValid),
      d.f32(1)
    );
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
