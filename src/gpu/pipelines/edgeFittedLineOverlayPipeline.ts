// Draw per-label fitted edge segments in image pixel space (tSampleMin → tSampleMax along tangent).
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { max, mul, select, sqrt } from 'typegpu/std'

import { EdgeLineEntry, type EdgeLineOutBuffer } from '@/gpu/pipelines/edgeLineFitPipeline'
import { MAX_FLAT_EDGES } from '@/gpu/pipelines/edgeHistogramClusterPipeline'

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
  lineOut: EdgeLineOutBuffer,
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
      const gLen = sqrt(line.sumGx * line.sumGx + line.sumGy * line.sumGy)
      const valid = line.valid !== d.u32(0) && gLen >= d.f32(1e-8)
      const invGLen = d.f32(1) / max(gLen, d.f32(1e-8))
      const nx = line.sumGx * invGLen
      const ny = line.sumGy * invGLen
      const tx = -ny
      const ty = nx
      const p0 = d.vec2f(
        line.tSampleMin * tx + line.nDotMean * nx,
        line.tSampleMin * ty + line.nDotMean * ny,
      )
      const p1 = d.vec2f(
        line.tSampleMax * tx + line.nDotMean * nx,
        line.tSampleMax * ty + line.nDotMean * ny,
      )
      const atEnd = vertexIndex === d.u32(1)
      const pLine = select(p0, p1, atEnd)
      const p = select(d.vec2f(0, 0), pLine, valid)
      return { clipPos: imagePxToClip(p, width, height) }
    })
    .$uses({ lineOut: fittedLineLayout })

  const frag = tgpu.fragmentFn({
    in: { clipPos: d.builtin.position },
    out: d.vec4f,
  })(() => {
    'use gpu'
    const rgb = d.vec3f(0.2, 0.95, 0.75)
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
        .draw(2, MAX_FLAT_EDGES)
    },
  }
}
