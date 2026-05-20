// Draw fitted segments only for edges of registered quads (labelToQuadId + quadPeakEdge).
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { abs, mul, select } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import { MAX_EDGES_PER_LABEL } from '@/gpu/lineFitThresholds'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import type { LabelLineOutBuffer } from '@/gpu/pipelines/labelLineFitPipeline'
import type { LabelToQuadIdBuffer, QuadPeakEdgeBuffer } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { PREMULTIPLIED_ALPHA_BLEND } from '@/gpu/pipelines/shared'

const fittedLineLayout = tgpu.bindGroupLayout({
  lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
  labelToQuadId: { storage: d.arrayOf(d.u32), access: 'readonly' },
  quadPeakEdge: { storage: d.arrayOf(d.u32), access: 'readonly' },
}).$name('fitted-lines-bgl')

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
  labelToQuadId: LabelToQuadIdBuffer,
  quadPeakEdge: QuadPeakEdgeBuffer,
  lineInstanceCount: number,
  options?: { sampleCount?: number },
) {
  const sampleCount = options?.sampleCount
  const vert = tgpu
    .vertexFn({
      in: {
        vertexIndex: d.builtin.vertexIndex,
        instanceIndex: d.builtin.instanceIndex,
      },
      out: { clipPos: d.builtin.position },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      const labelId = d.u32(instanceIndex / d.u32(MAX_EDGES_PER_LABEL))
      const edgeId = d.u32(instanceIndex % d.u32(MAX_EDGES_PER_LABEL))
      const quadId = fittedLineLayout.$.labelToQuadId[labelId]!
      const quadBase = quadId * d.u32(MAX_EDGES_PER_LABEL)
      const peakEdge = fittedLineLayout.$.quadPeakEdge[quadBase + edgeId]!
      const line = fittedLineLayout.$.lineOut[instanceIndex]!

      const p0 = d.vec2f(line.p0x, line.p0y)
      const p1 = d.vec2f(line.p1x, line.p1y)
      const span = abs(p1.x - p0.x) + abs(p1.y - p0.y)
      const inQuad =
        quadId !== d.u32(COMPONENT_LABEL_INVALID) && peakEdge !== d.u32(COMPONENT_LABEL_INVALID)
      const draw = inQuad && line.count > d.u32(0) && span >= d.f32(0.5)
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
    targets: { format: presentationFormat, blend: PREMULTIPLIED_ALPHA_BLEND },
    primitive: { topology: 'line-list' },
    ...(sampleCount !== undefined && sampleCount > 1 ? { multisample: { count: sampleCount } } : {}),
  }).$name('fitted-lines-render')

  const bindGroup = root.createBindGroup(fittedLineLayout, { lineOut, labelToQuadId, quadPeakEdge })

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
