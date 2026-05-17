import type { ColorAttachment, TgpuBindGroup, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'
import { clamp, floor } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import { stableHashToRgb01 } from '@/lib/hashStableColor'

export const labelVizLayout = tgpu.bindGroupLayout({
  labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
})

export const quadsLabelVizLayout = tgpu.bindGroupLayout({
  quadLabelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
})

export type LabelVizBindGroup = TgpuBindGroup<typeof labelVizLayout.entries>
export type QuadsLabelVizBindGroup = TgpuBindGroup<typeof quadsLabelVizLayout.entries>

function createPlainLabelVizFragment(width: number, height: number) {
  return tgpu.fragmentFn({
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
    const label = labelVizLayout.$.labelBuffer[idx]!

    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.14), d.f32(1))
    }

    const rgb = stableHashToRgb01(label)
    return d.vec4f(rgb, d.f32(1))
  })
}

function createQuadsLabelVizFragment(width: number, height: number) {
  return tgpu.fragmentFn({
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
    const quadId = quadsLabelVizLayout.$.quadLabelBuffer[idx]!

    if (quadId === d.u32(COMPONENT_LABEL_INVALID)) {
      return d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.14), d.f32(1))
    }

    const rgb = stableHashToRgb01(quadId)
    return d.vec4f(rgb, d.f32(1))
  })
}

export function createLabelVizPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const labelVizFrag = createPlainLabelVizFragment(width, height)

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: labelVizFrag,
    targets: { format: presentationFormat },
  })
  const encodeToCanvas = (enc: GPUCommandEncoder, colorAttachment: ColorAttachment, bindGroup: LabelVizBindGroup) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }
  return { encodeToCanvas, layout: labelVizLayout }
}

export function createQuadsLabelVizPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const frag = createQuadsLabelVizFragment(width, height)

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })
  const encodeToCanvas = (
    enc: GPUCommandEncoder,
    colorAttachment: ColorAttachment,
    bindGroup: QuadsLabelVizBindGroup,
  ) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }
  return { encodeToCanvas, layout: quadsLabelVizLayout }
}
