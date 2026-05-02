import type { ColorAttachment, TgpuBindGroup, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'
import { clamp, floor } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import { stableHashToRgb01 } from '@/lib/hashStableColor'

export const labelVizLayout = tgpu.bindGroupLayout({
  labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
})

export type LabelVizBindGroup = TgpuBindGroup<typeof labelVizLayout.entries>

export function createLabelVizPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const labelVizFrag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    // i32 for last-index: (u32)width - 1 wraps at 0; clamp max uses signed last index → f32.
    const wi = d.i32(width)
    const hi = d.i32(height)
    const maxPx = d.f32(wi - d.i32(1))
    const maxPy = d.f32(hi - d.i32(1))
    // Plain u32(uv * size) is unstable at column/row boundaries (float error → wrong neighbor sample → 1px streaks).
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)))
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)))
    const idx = py * d.u32(wi) + px
    const label = labelVizLayout.$.labelBuffer[idx]!

    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.14), d.f32(1))
    }

    const rgb = stableHashToRgb01(label)
    return d.vec4f(rgb, 1)
  })

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
