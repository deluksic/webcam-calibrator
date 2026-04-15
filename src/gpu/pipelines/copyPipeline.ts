// Copy pipeline: external texture → grayTex
import { tgpu, d, std, common } from 'typegpu';

export function createCopyPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  copyLayout: ReturnType<typeof tgpu.bindGroupLayout>,
) {
  const copyFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    return std.textureSampleBaseClampToEdge(copyLayout.$.cameraTex, copyLayout.$.sampler, i.uv);
  });

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: copyFrag,
    targets: { format: 'rgba8unorm' },
  });
}
