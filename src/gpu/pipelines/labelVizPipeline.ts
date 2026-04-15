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

    // Debug: show actual label values
    // label=1 (edge) should be white, INVALID=gray, propagated labels vary
    const isValid = label !== d.u32(0xFFFFFFFF);
    const isEdge = label === d.u32(1);
    const isInvalid = label === d.u32(0xFFFFFFFF);
    const white = d.f32(1);
    const midGray = d.f32(0.5);
    const dark = d.f32(0.1);
    return d.vec4f(
      std.select(std.select(dark, midGray, isInvalid), white, isEdge),
      std.select(std.select(dark, midGray, isInvalid), white, isEdge),
      std.select(std.select(dark, midGray, isInvalid), white, isEdge),
      d.f32(1)
    );
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  });
}
