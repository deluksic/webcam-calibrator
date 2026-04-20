// Sobel buffer render pipeline: sobelBuffer → canvas
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'
import { clamp, floor, length } from 'typegpu/std'

export function createSobelRenderPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  sobelLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
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
    const mag = length(sobelLayout.$.sobelBuffer[idx])
    return d.vec4f(mag, mag, mag, d.f32(1))
  })

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: sobelFrag,
    targets: { format: presentationFormat },
  })
}
