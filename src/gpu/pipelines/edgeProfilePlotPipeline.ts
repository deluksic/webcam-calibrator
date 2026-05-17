// MSAA profile plot: one stroked segment per adjacent bucket pair per valid edge label.
import {
  caps,
  endCapSlot,
  joinSlot,
  joins,
  lineSegmentIndices,
  lineSegmentVariableWidth,
  LineControlPoint,
  startCapSlot,
} from '@typegpu/geometry'
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { arrayOf, u16 } from 'typegpu/data'
import { fwidth, max, mix, select, smoothstep } from 'typegpu/std'

import { MAX_FLAT_EDGES } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import type { EdgeLineOutBuffer } from '@/gpu/pipelines/edgeLineFitPipeline'
import { EdgeLineEntry, PROFILE_BUCKET_COUNT, PROFILE_NEIGHBORHOOD_HALF } from '@/gpu/pipelines/edgeLineFitPipeline'
import type { ProfileAvgBuffer, ProfileBucketsBuffer } from '@/gpu/pipelines/edgeProfilePipeline'
import { ProfileBucketGpu } from '@/gpu/pipelines/edgeProfilePipeline'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'

const PROFILE_JOIN_MAX = 2
const SEGMENTS_PER_LABEL = PROFILE_BUCKET_COUNT - 1
/** Float literal for storage indexing (must match {@link PROFILE_BUCKET_COUNT}). */
const PLOT_BUCKET_COUNT = 64
export const MAX_PROFILE_LINE_INSTANCES = MAX_FLAT_EDGES * SEGMENTS_PER_LABEL

const plotTriangleIndexU16 = new Uint16Array(lineSegmentIndices(PROFILE_JOIN_MAX))

const PlotUniformStruct = d.struct({
  viewportHalfWH: d.vec2f,
  /** NDC inset so polylines sit inside the canvas (0.92 → ~8% margin). */
  plotInset: d.f32,
})

export const plotBindLayout = tgpu.bindGroupLayout({
  plot: { uniform: PlotUniformStruct },
  lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
  profileAvg: { storage: d.arrayOf(d.f32), access: 'readonly' },
  /** Same buffer as compute; non-atomic view for vertex (readonly storage). */
  profileBuckets: { storage: d.arrayOf(ProfileBucketGpu), access: 'readonly' },
})

export function createEdgeProfilePlotStage(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const plotUniform = root.createBuffer(PlotUniformStruct).$usage('uniform')

  const vert = tgpu
    .vertexFn({
      in: {
        instanceIndex: d.builtin.instanceIndex,
        vertexIndex: d.builtin.vertexIndex,
      },
      out: {
        clipPos: d.builtin.position,
        uv: d.vec2f,
        discardStroke: d.interpolate('flat', d.u32),
      },
    })(({ instanceIndex, vertexIndex }) => {
      'use gpu'
      const labelId = d.u32(instanceIndex / d.u32(SEGMENTS_PER_LABEL))
      const segIdx = instanceIndex % d.u32(SEGMENTS_PER_LABEL)
      const b = segIdx

      const halfWH = plotBindLayout.$.plot.viewportHalfWH
      const inset = plotBindLayout.$.plot.plotInset

      const line = plotBindLayout.$.lineOut[labelId]!
      const bucketIdx0F = d.f32(labelId) * PLOT_BUCKET_COUNT + d.f32(b)
      const bucketIdx1F = bucketIdx0F + 1.0
      const slot0 = plotBindLayout.$.profileBuckets[bucketIdx0F]!
      const slot1 = plotBindLayout.$.profileBuckets[bucketIdx1F]!
      const c0 = slot0.count
      const c1 = slot1.count

      let discardStroke = d.u32(0)
      if (line.valid === d.u32(0) || (c0 === d.u32(0) && c1 === d.u32(0))) {
        discardStroke = d.u32(1)
      }

      let y0 = plotBindLayout.$.profileAvg[bucketIdx0F]!
      let y1 = plotBindLayout.$.profileAvg[bucketIdx1F]!
      if (c0 === d.u32(0)) {
        y0 = y1
      }
      if (c1 === d.u32(0)) {
        y1 = y0
      }

      const span = d.f32(2) * d.f32(PROFILE_NEIGHBORHOOD_HALF)
      const bF = d.f32(b)
      const s0 = -d.f32(PROFILE_NEIGHBORHOOD_HALF) + ((bF + 0.5) / PLOT_BUCKET_COUNT) * span
      const s1 = -d.f32(PROFILE_NEIGHBORHOOD_HALF) + ((bF + 1.5) / PLOT_BUCKET_COUNT) * span
      const xNdc0 = -inset + ((s0 + d.f32(PROFILE_NEIGHBORHOOD_HALF)) / span) * d.f32(2) * inset
      const yNdc0 = -inset + y0 * d.f32(2) * inset
      const xNdc1 = -inset + ((s1 + d.f32(PROFILE_NEIGHBORHOOD_HALF)) / span) * d.f32(2) * inset
      const yNdc1 = -inset + y1 * d.f32(2) * inset
      const p0 = d.vec2f(xNdc0 * halfWH.x, yNdc0 * halfWH.y)
      const p1 = d.vec2f(xNdc1 * halfWH.x, yNdc1 * halfWH.y)

      const rStroke = select(d.f32(0.85), d.f32(-1), discardStroke === d.u32(1))
      const A = LineControlPoint({ position: p0, radius: rStroke })
      const B = LineControlPoint({ position: p0, radius: rStroke })
      const C = LineControlPoint({ position: p1, radius: rStroke })
      const D = LineControlPoint({ position: p1, radius: rStroke })

      const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D, d.u32(PROFILE_JOIN_MAX))
      const w = result.w
      const pos = result.vertexPosition

      const ndcXY = d.vec2f(pos.x / halfWH.x, pos.y / halfWH.y)

      return {
        clipPos: d.vec4f(ndcXY.x * w, ndcXY.y * w, d.f32(0), w),
        uv: d.vec2f(d.f32(0), select(d.f32(0), d.f32(1), vertexIndex > d.u32(1))),
        discardStroke,
      }
    })
    .$uses({ plot: plotBindLayout })

  const frag = tgpu.fragmentFn({
    in: {
      uv: d.vec2f,
      discardStroke: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ uv, discardStroke }) => {
    'use gpu'
    const iy = uv.y
    const w = max(fwidth(iy), d.f32(0.001))
    const a = smoothstep(d.f32(1) - w * d.f32(1.5), d.f32(1), iy) * d.f32(0.22)
    const rgb = mix(d.vec3f(0.35, 0.75, 1), d.vec3f(0.15, 0.2, 0.28), a)
    const outA = select(a, d.f32(0), discardStroke === d.u32(1))
    const outRgb = select(rgb, d.vec3f(0), discardStroke === d.u32(1))
    return d.vec4f(outRgb, outA)
  })

  const alphaBlend: GPUBlendState = {
    color: {
      operation: 'add',
      srcFactor: 'src-alpha',
      dstFactor: 'one-minus-src-alpha',
    },
    alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  }

  const pipeline = root
    .with(joinSlot, joins.round)
    .with(startCapSlot, caps.round)
    .with(endCapSlot, caps.round)
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: {
        format: presentationFormat,
        blend: alphaBlend,
      },
      primitive: { topology: 'triangle-list' },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })
    .withIndexBuffer(root.createBuffer(arrayOf(u16, plotTriangleIndexU16.length), plotTriangleIndexU16).$usage('index'))

  const indexCount = plotTriangleIndexU16.length

  return {
    plotUniform,
    pipeline,
    indexCount,
    createPlotBindGroup(
      lineOut: EdgeLineOutBuffer,
      profileAvg: ProfileAvgBuffer,
      profileBuckets: ProfileBucketsBuffer,
    ) {
      return root.createBindGroup(plotBindLayout, {
        plot: plotUniform,
        lineOut,
        profileAvg,
        profileBuckets,
      })
    },
    encodeClearAndDraw: (
      pass: GPURenderPassEncoder,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- bind group from createPlotBindGroup
      plotBindGroup: any,
      width: number,
      height: number,
    ) => {
      plotUniform.write({
        viewportHalfWH: d.vec2f(width * 0.5, height * 0.5),
        plotInset: d.f32(0.92),
      })
      pipeline.with(pass).with(plotBindGroup).drawIndexed(indexCount, MAX_PROFILE_LINE_INSTANCES)
    },
  }
}

export type EdgeProfilePlotStage = ReturnType<typeof createEdgeProfilePlotStage>
