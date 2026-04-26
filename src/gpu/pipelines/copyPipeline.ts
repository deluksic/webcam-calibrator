// Copy pipeline: external texture → grayTex
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std, common } from 'typegpu'

export const copyBindGroupLayout = tgpu.bindGroupLayout({
  cameraTex: { externalTexture: d.textureExternal() },
})

export function createCopyBindGroup(root: TgpuRoot, video: HTMLVideoElement) {
  return root.createBindGroup(copyBindGroupLayout, {
    cameraTex: root.device.importExternalTexture({ source: video }),
  })
}

/** Camera ingest: intermediate texture written by the external-texture copy pass. */
export function createCopyIngest(root: TgpuRoot, width: number, height: number) {
  const grayTex = root
    .createTexture({
      size: [width, height],
      format: 'rgba8unorm',
      dimension: '2d',
    })
    .$usage('storage', 'sampled', 'render')
  const copyPipeline = createCopyPipeline(root)
  return { grayTex, copyPipeline }
}

export function createCopyPipeline(root: TgpuRoot) {
  const copyFrag = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    return std.textureLoad(copyBindGroupLayout.$.cameraTex, d.vec2i(i.pos.xy))
  })

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: copyFrag,
    targets: { format: 'rgba8unorm' },
  })
}
