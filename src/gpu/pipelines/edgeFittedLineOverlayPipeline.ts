// Draw per-label×edgeId fitted segments (p0 → p1) from labelLineOut.
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { abs, mul, select } from 'typegpu/std'

import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import type { LabelLineOutBuffer } from '@/gpu/pipelines/labelLineFitPipeline'

const premultipliedAlphaBlend: GPUBlendState = {
  color: {
    operation: 'add',
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
  },
  alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
}

const fittedLineLayout = tgpu.bindGroupLayout({
  lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
})

function imagePxToClip(p: { x: number; y: number }, width: number, height: number) {
  'use gpu'
  const clipX = (d.f32(2) * p.x) / d.f32(width) - d.f32(1)
  const clipY = d.f32(1) - (d.f32(2) * p.y) / d.f32(height)
  return d.vec4f(clipX, clipY, d.f32(0), d.f32(1))
}

export function createEdgeFittedLineOverlayStage(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
  lineOut: LabelLineOutBuffer,
  lineInstanceCount: number,
) {
  const vert = tgpu
    .vertexFn({
      in: {
        vertexIndex: d.builtin.vertexIndex,
        instanceIndex: d.builtin.instanceIndex,
      },
      out: { clipPos: d.builtin.position },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      const line = fittedLineLayout.$.lineOut[instanceIndex]!
      const p0 = d.vec2f(line.p0x, line.p0y)
      const p1 = d.vec2f(line.p1x, line.p1y)
      const span = abs(p1.x - p0.x) + abs(p1.y - p0.y)
      const draw = line.count > d.u32(0) && span >= d.f32(0.5)
      const atEnd = vertexIndex === d.u32(1)
      const pLine = select(p0, p1, atEnd)
      const clipOn = imagePxToClip(pLine, width, height)
      const clipOff = d.vec4f(d.f32(-2), d.f32(-2), d.f32(0), d.f32(1))
      return { clipPos: select(clipOff, clipOn, draw) }
    })
    .$uses({ lineOut: fittedLineLayout })

  const frag = tgpu.fragmentFn({
    in: { clipPos: d.builtin.position },
    out: d.vec4f,
  })(() => {
    'use gpu'
    const rgb = d.vec3f(0.2, 0.95, 0.45)
    const a = d.f32(0.92)
    return d.vec4f(mul(rgb, a), a)
  })

  const pipeline = root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: { format: presentationFormat, blend: premultipliedAlphaBlend },
    primitive: { topology: 'line-list' },
  })

  const bindGroup = root.createBindGroup(fittedLineLayout, { lineOut })

  return {
    pipeline,
    bindGroup,
    encodeOverlay(enc: GPUCommandEncoder, colorAttachment: ColorAttachment): void {
      pipeline
        .with(enc)
        .withColorAttachment({ ...colorAttachment, loadOp: 'load', storeOp: 'store' })
        .with(bindGroup)
        .draw(2, lineInstanceCount)
    },
  }
}
