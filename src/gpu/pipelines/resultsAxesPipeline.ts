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
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { arrayOf, u16 } from 'typegpu/data'
import { clamp, fwidth, max, min, mix, mul, select, smoothstep, sqrt } from 'typegpu/std'

import { resultsCameraBindLayout, type ResultsCameraBindGroup } from '@/gpu/pipelines/resultsCameraTransform'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { WORLD_AXIS_HALF_LEN } from '@/lib/orbitOrthoMath'

/** Join triangle allowance for stroked-axis extrusion (@typegpu/geometry lines combo). */
const AXIS_JOIN_MAX = 3

const axisTriangleIndexU16 = new Uint16Array(lineSegmentIndices(AXIS_JOIN_MAX))

const AxisPassUniformStruct = d.struct({
  /** Half stroking width (~1.5 ⇒ ~3 px line) for axis arrows, in framebuffer pixels. */
  axisPolylineHalfStrokePixels: d.f32,
  /** Each RGB arm runs from `axisOriginBoard` ± this length along the corresponding axis (`WORLD_AXIS_HALF_LEN` on CPU). */
  worldAxisReachDistanceBoardUnits: d.f32,
  /** Origin where the RGB axes are planted (board space); fixed at **`(0,0,0)`** on CPU. */
  axisOriginBoard: d.vec3f,
})

export const axisBindLayout = tgpu
  .bindGroupLayout({
    axis: { uniform: AxisPassUniformStruct },
  })
  .$idx(1)

export type AxisGpuBindValues = ExtractBindGroupInputFromLayout<typeof axisBindLayout.entries>

function allocAxisPassUniform(root: TgpuRoot) {
  return root.createBuffer(AxisPassUniformStruct).$usage('uniform')
}

export type AxisPassUniformGpuBuffer = ReturnType<typeof allocAxisPassUniform>

/** Half stroke width in framebuffer pixels (full line ≈ 2× this). */
const axisPolylineHalfStrokePixels = 3.25 as const

export function writeAxisPassUniform(buf: AxisPassUniformGpuBuffer): void {
  buf.write({
    axisPolylineHalfStrokePixels: d.f32(axisPolylineHalfStrokePixels),
    worldAxisReachDistanceBoardUnits: d.f32(WORLD_AXIS_HALF_LEN),
    axisOriginBoard: d.vec3f(0, 0, 0),
  })
}

export function createAxesResultsStage(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const axisUniform = allocAxisPassUniform(root)
  const axisBg = root.createBindGroup(axisBindLayout, { axis: axisUniform })

  const vert = tgpu
    .vertexFn({
      in: {
        instanceIndex: d.builtin.instanceIndex,
        vertexIndex: d.builtin.vertexIndex,
      },
      out: {
        clipPos: d.builtin.position,
        instanceIndexFlat: d.interpolate('flat', d.u32),
        position: d.vec2f,
        uv: d.vec2f,
      },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      const cam = resultsCameraBindLayout.$.transform
      const u = axisBindLayout.$.axis
      const M = cam.clipHomogeneousMatrixFromBoard
      const reach = u.worldAxisReachDistanceBoardUnits
      const o = u.axisOriginBoard
      const ex = d.vec4f(o.x + reach, o.y, o.z, 1)
      const ey = d.vec4f(o.x, o.y + reach, o.z, 1)
      const ez = d.vec4f(o.x, o.y, o.z + reach, 1)
      const tipBoard = select(select(ex, ey, instanceIndex === d.u32(1)), ez, instanceIndex === d.u32(2))

      const clipOrigin = mul(M, d.vec4f(o.x, o.y, o.z, 1))
      const clipTip = mul(M, tipBoard)
      const invWo = d.f32(1) / clipOrigin.w
      const invWt = d.f32(1) / clipTip.w
      const ndcOrigin = clipOrigin.xy * invWo
      const ndcTip = clipTip.xy * invWt
      const halfWH = cam.viewportHalfSizePixels
      const originPixelOffsetFromFramebufferCenter = d.vec2f(ndcOrigin.x * halfWH.x, ndcOrigin.y * halfWH.y)
      const tipPixelOffsetFromFramebufferCenter = d.vec2f(ndcTip.x * halfWH.x, ndcTip.y * halfWH.y)

      const spine = tipPixelOffsetFromFramebufferCenter - originPixelOffsetFromFramebufferCenter
      const spineLenSq = spine.x * spine.x + spine.y * spine.y
      const spineLen = sqrt(max(spineLenSq, d.f32(0)))
      const rStroke = min(u.axisPolylineHalfStrokePixels, spineLen * d.f32(1 / 16))
      const A = LineControlPoint({ position: originPixelOffsetFromFramebufferCenter, radius: rStroke })
      const B = LineControlPoint({ position: originPixelOffsetFromFramebufferCenter, radius: rStroke })
      const C = LineControlPoint({ position: tipPixelOffsetFromFramebufferCenter, radius: rStroke })
      const D = LineControlPoint({ position: tipPixelOffsetFromFramebufferCenter, radius: rStroke })

      const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D, d.u32(AXIS_JOIN_MAX))

      const measured = result.vertexPosition
      const ndcExtrudedClipXY = d.vec2f(measured.x / halfWH.x, measured.y / halfWH.y)

      const ndcZ0 = clipOrigin.z * invWo
      const ndcZ1 = clipTip.z * invWt
      const fromO = d.vec2f(
        measured.x - originPixelOffsetFromFramebufferCenter.x,
        measured.y - originPixelOffsetFromFramebufferCenter.y,
      )
      const t = clamp((fromO.x * spine.x + fromO.y * spine.y) / max(spineLenSq, d.f32(1e-8)), d.f32(0), d.f32(1))
      const ndcZ = ndcZ0 * (d.f32(1) - t) + ndcZ1 * t

      // `lineSegmentVariableWidth`: vertex 0/1 = spine (B/C); ≥2 = extrusion + caps. Interpolation gives smooth uv.y across the stroke.
      const uvY = select(d.f32(0), d.f32(1), vertexIndex > d.u32(1))

      const w = result.w
      const clipPos = d.vec4f(ndcExtrudedClipXY.x * w, ndcExtrudedClipXY.y * w, ndcZ * w, w)

      return {
        clipPos,
        instanceIndexFlat: instanceIndex,
        position: ndcExtrudedClipXY,
        uv: d.vec2f(t, uvY),
      }
    })
    .$uses({ camera: resultsCameraBindLayout, axis: axisBindLayout })

  const frag = tgpu.fragmentFn({
    in: {
      instanceIndexFlat: d.interpolate('flat', d.u32),
      uv: d.vec2f,
    },
    out: d.vec4f,
  })(({ instanceIndexFlat, uv }) => {
    'use gpu'
    let rgb = d.vec3f(1, 0.12, 0.12)
    rgb = select(rgb, d.vec3f(0.14, 0.98, 0.22), instanceIndexFlat === d.u32(1))
    rgb = select(rgb, d.vec3f(0.24, 0.38, 1), instanceIndexFlat === d.u32(2))

    const iy = uv.y
    const w = max(fwidth(iy), d.f32(0.001))
    // Sharp rim at the outer edge (iy → 1); transition width ~1.5·fwidth for stable AA.
    const outlineMix = smoothstep(d.f32(1) - w * d.f32(1.5), d.f32(1), iy)
    const outlineRgb = d.vec3f(0.05, 0.05, 0.07)
    rgb = mix(rgb, outlineRgb, outlineMix)

    return d.vec4f(rgb, 1)
  })

  const pipeline = root
    .with(joinSlot, joins.round)
    .with(startCapSlot, caps.round)
    .with(endCapSlot, caps.arrow)
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: {
        format: presentationFormat,
      },
      primitive: {
        topology: 'triangle-list',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })
    .withIndexBuffer(root.createBuffer(arrayOf(u16, axisTriangleIndexU16.length), axisTriangleIndexU16).$usage('index'))

  const axisIndexCount = axisTriangleIndexU16.length
  const encodeToPass = (pass: GPURenderPassEncoder, cameraBg: ResultsCameraBindGroup) => {
    pipeline.with(pass).with(cameraBg).with(axisBg).drawIndexed(axisIndexCount, 3)
  }
  return { axisUniform, encodeToPass }
}
