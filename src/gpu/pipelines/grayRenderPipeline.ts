// Grayscale render pipeline: grayBuffer → canvas
import type {
  ColorAttachment,
  ExtractBindGroupInputFromLayout,
  TgpuBindGroup,
  TgpuRoot,
} from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { common } from 'typegpu'

export const grayRenderLayout = tgpu.bindGroupLayout({
  grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  timeSec: { uniform: d.f32 },
})

export type GrayRenderBindResources = ExtractBindGroupInputFromLayout<typeof grayRenderLayout.entries>

export type GrayRenderBindGroup = TgpuBindGroup<typeof grayRenderLayout.entries>

export function createGrayRenderPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
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
    const t = grayRenderLayout.$.timeSec
    const uv = d.vec2f(i.pos.x / width, i.pos.y / height)
    const stripePhase = std.sin((uv.x + uv.y) * 300 + t * 6) + 0.5
    const stripeMask = std.clamp(stripePhase / std.fwidth(stripePhase), 0, 1)
    const saturationMask = std.smoothstep(0.92, 0.98, gray)
    const hit = saturationMask * stripeMask
    const r = std.clamp(gray + hit * 0.52, 0, 1)
    const g = std.clamp(gray - hit * 0.14, 0, 1)
    const b = std.clamp(gray - hit * 0.42, 0, 1)
    return d.vec4f(r, g, b, 1)
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: grayFrag,
    targets: { format: presentationFormat },
  })
  const bindGroup = root.createBindGroup(grayRenderLayout, resources)
  const encodeToCanvas = (
    enc: GPUCommandEncoder,
    colorAttachment: ColorAttachment,
    bg: GrayRenderBindGroup = bindGroup,
  ) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bg).draw(3)
  }
  return { bindGroup, encodeToCanvas, layout: grayRenderLayout }
}
