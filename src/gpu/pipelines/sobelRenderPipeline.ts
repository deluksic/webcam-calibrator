// Sobel buffer render pipeline: sobelBuffer → canvas
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'
import { clamp, floor, length } from 'typegpu/std'

import type { RenderColorAttachment } from '@/gpu/renderEncodeTypes'

export const sobelRenderLayout = tgpu.bindGroupLayout({
  sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
})

export type SobelRenderBindResources = ExtractBindGroupInputFromLayout<typeof sobelRenderLayout.entries>

export function createSobelRenderPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
  resources: SobelRenderBindResources,
) {
  const sobelFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const wi = d.i32(width)
    const hi = d.i32(height)
    const maxPx = d.f32(wi - d.i32(1))
    const maxPy = d.f32(hi - d.i32(1))
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)))
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)))
    const idx = py * d.u32(wi) + px
    const mag = length(sobelRenderLayout.$.sobelBuffer[idx]!)
    return d.vec4f(mag, mag, mag, d.f32(1))
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: sobelFrag,
    targets: { format: presentationFormat },
  })
  const bindGroup = root.createBindGroup(sobelRenderLayout, resources)
  const encodeToCanvas = (enc: GPUCommandEncoder, colorAttachment: RenderColorAttachment) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }
  return { encodeToCanvas, layout: sobelRenderLayout }
}
