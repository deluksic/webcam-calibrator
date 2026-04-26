// Grayscale render pipeline: grayBuffer → canvas
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'

export const grayRenderLayout = tgpu.bindGroupLayout({
  grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
})

export type GrayRenderBindResources = ExtractBindGroupInputFromLayout<typeof grayRenderLayout.entries>

export function createGrayRenderPipeline(
  root: TgpuRoot,
  width: number,
  _height: number,
  presentationFormat: GPUTextureFormat,
  resources: GrayRenderBindResources,
) {
  const grayFrag = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const pos = d.vec2i(i.pos.xy)
    const idx = pos.y * width + pos.x
    const gray = grayRenderLayout.$.grayBuffer[idx]!
    return d.vec4f(gray, gray, gray, 1)
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: grayFrag,
    targets: { format: presentationFormat },
  })
  const bindGroup = root.createBindGroup(grayRenderLayout, resources)
  return { pipeline, bindGroup, layout: grayRenderLayout }
}
