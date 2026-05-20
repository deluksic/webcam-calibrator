import type { ColorAttachment, TgpuBindGroup, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'
import { clamp, floor } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import { LabelQuadRejectCode } from '@/gpu/labelQuadReject'
import { stableHashToRgb01 } from '@/lib/hashStableColor'

export const labelVizLayout = tgpu.bindGroupLayout({
  labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
}).$name('label-viz-bgl')

export const quadsLabelVizLayout = tgpu.bindGroupLayout({
  quadLabelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
}).$name('quads-label-viz-bgl')

export const quadRejectVizLayout = tgpu.bindGroupLayout({
  compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labelToQuadId: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labelQuadReject: { storage: d.arrayOf(d.u32), access: 'readonly' },
}).$name('quad-reject-viz-bgl')

export type LabelVizBindGroup = TgpuBindGroup<typeof labelVizLayout.entries>
export type QuadsLabelVizBindGroup = TgpuBindGroup<typeof quadsLabelVizLayout.entries>
export type QuadRejectVizBindGroup = TgpuBindGroup<typeof quadRejectVizLayout.entries>

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
  }).$name('label-viz-render')
  const encodeToCanvas = (enc: GPUCommandEncoder, colorAttachment: ColorAttachment, bindGroup: LabelVizBindGroup) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }
  return { encodeToCanvas, layout: labelVizLayout }
}

const rejectReasonRgb = tgpu.fn([d.u32], d.vec3f)((code) => {
  'use gpu'
  if (code === d.u32(LabelQuadRejectCode.peaks)) {
    return d.vec3f(0.67, 0.33, 1)
  }
  if (code === d.u32(LabelQuadRejectCode.lineFit)) {
    return d.vec3f(0.27, 0.53, 1)
  }
  if (code === d.u32(LabelQuadRejectCode.parallel)) {
    return d.vec3f(1, 0.6, 0.2)
  }
  if (code === d.u32(LabelQuadRejectCode.cap)) {
    return d.vec3f(1, 0.27, 0.27)
  }
  return d.vec3f(0.45, 0.45, 0.5)
})

function createQuadRejectVizFragment(width: number, height: number) {
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
    const labelId = quadRejectVizLayout.$.compactLabels[idx]!

    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      return d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.14), d.f32(1))
    }

    const quadId = quadRejectVizLayout.$.labelToQuadId[labelId]!
    if (quadId !== d.u32(COMPONENT_LABEL_INVALID)) {
      return d.vec4f(d.f32(0.22), d.f32(0.22), d.f32(0.24), d.f32(1))
    }

    const reason = quadRejectVizLayout.$.labelQuadReject[labelId]!
    const rgb = rejectReasonRgb(reason)
    return d.vec4f(rgb, d.f32(1))
  })
}

export function createQuadRejectVizPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const frag = createQuadRejectVizFragment(width, height)

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  }).$name('quad-reject-viz-render')
  const encodeToCanvas = (
    enc: GPUCommandEncoder,
    colorAttachment: ColorAttachment,
    bindGroup: QuadRejectVizBindGroup,
  ) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }
  return { encodeToCanvas, layout: quadRejectVizLayout }
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
  }).$name('quads-label-viz-render')
  const encodeToCanvas = (
    enc: GPUCommandEncoder,
    colorAttachment: ColorAttachment,
    bindGroup: QuadsLabelVizBindGroup,
  ) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }
  return { encodeToCanvas, layout: quadsLabelVizLayout }
}
