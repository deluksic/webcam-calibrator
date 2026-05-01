import {
  caps,
  circle,
  circleVertexCount,
  endCapSlot,
  joinSlot,
  joins,
  lineSegmentIndices,
  lineSegmentVariableWidth,
  LineControlPoint,
  startCapSlot,
} from '@typegpu/geometry'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { arrayOf, u16 } from 'typegpu/data'
import { clamp, floor, max, mul, select } from 'typegpu/std'

import { MARKER_DISK_SUBDIV, axisBindLayout, markersBindLayout, tagQuadsBindLayout } from '@/gpu/resultsVizLayouts'

/**
 * When true: marker pass draws one bright clip-space triangle and ignores projection/circle math.
 * Confirms MSAA color + resolve + bind group wiring. Set to `false` after you see the triangle.
 */
export const RESULTS_MARKERS_CLIP_SPACE_DIAGNOSTIC_TRIANGLE = false as const

/** Join triangle allowance for stroked-axis extrusion (@typegpu/geometry lines combo). */
const AXIS_JOIN_MAX = 6

const axisTriangleIndexU16 = new Uint16Array(lineSegmentIndices(AXIS_JOIN_MAX))

/** Same as examples-from-typegpu/lines-combinations (MSAA x4). */
export const RESULTS_MSAA_SAMPLE_COUNT = 4 as const

/** NDC depth offset (× w) so marker disks win over coplanar tag quads; WebGPU nearer = smaller z. */
const RESULTS_MARKER_DEPTH_BIAS_NDC = 0.0005 as const

export function createMarkerResultsPipeline(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  if (RESULTS_MARKERS_CLIP_SPACE_DIAGNOSTIC_TRIANGLE) {
    const vertDiag = tgpu.vertexFn({
      in: {
        vertexIndex: d.builtin.vertexIndex,
        instanceIndex: d.builtin.instanceIndex,
      },
      out: { clipPos: d.builtin.position },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      if (instanceIndex > d.u32(0)) {
        return { clipPos: d.vec4f(0, 0, 2, 1) }
      }
      const vx = select(d.f32(-0.92), d.f32(0.92), vertexIndex === d.u32(2))
      const vy = select(d.f32(-0.92), d.f32(0.92), vertexIndex === d.u32(1))
      return { clipPos: d.vec4f(vx, vy, d.f32(0.5), d.f32(1)) }
    })

    const fragDiag = tgpu.fragmentFn({
      out: d.vec4f,
    })(() => {
      'use gpu'
      return d.vec4f(1, 0, 0, 1)
    })

    const pipeline = root.createRenderPipeline({
      vertex: vertDiag,
      fragment: fragDiag,
      targets: {
        format: presentationFormat,
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })

    return { pipeline, diskVerts: 3 as const }
  }

  const diskVerts = circleVertexCount(MARKER_DISK_SUBDIV)
  const vert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: { clipPos: d.builtin.position },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const u = markersBindLayout.$.markerUniform
    const row = markersBindLayout.$.centers[instanceIndex]!
    const pc = row.positionBoardUnits
    const centerClip = mul(u.clipHomogeneousMatrixFromBoard, d.vec4f(pc.x, -pc.y, pc.z, d.f32(1)))
    const unit = circle(vertexIndex)
    const half = u.viewportHalfSizePixels
    const rPx = u.markerDiskRadiusPixels
    const dNdcX = (unit.x * rPx) / half.x
    const dNdcY = (unit.y * rPx) / half.y
    const w = centerClip.w
    return {
      clipPos: d.vec4f(
        centerClip.x + dNdcX * w,
        centerClip.y + dNdcY * w,
        centerClip.z - d.f32(RESULTS_MARKER_DEPTH_BIAS_NDC) * w,
        centerClip.w,
      ),
    }
  })

  const frag = tgpu.fragmentFn({
    out: d.vec4f,
  })(() => {
    'use gpu'
    return d.vec4f(0.15, 0.92, 1.0, 1)
  })

  const pipeline = root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: {
      format: presentationFormat,
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'none',
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
    multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
  })

  return { pipeline, diskVerts }
}

function markerInstancesForEncode(requestedMarkers: number) {
  if (RESULTS_MARKERS_CLIP_SPACE_DIAGNOSTIC_TRIANGLE) {
    return 1 as const
  }
  return Math.max(requestedMarkers, 0)
}

export function createAxesResultsPipeline(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const vert = tgpu.vertexFn({
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
    const u = axisBindLayout.$.axisUniform
    const M = u.clipHomogeneousMatrixFromBoard
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
    const halfWH = u.viewportHalfSizePixels
    const originPixelOffsetFromFramebufferCenter = d.vec2f(ndcOrigin.x * halfWH.x, ndcOrigin.y * halfWH.y)
    const tipPixelOffsetFromFramebufferCenter = d.vec2f(ndcTip.x * halfWH.x, ndcTip.y * halfWH.y)

    const rStroke = u.axisPolylineHalfStrokePixels
    const A = LineControlPoint({ position: originPixelOffsetFromFramebufferCenter, radius: rStroke })
    const B = LineControlPoint({ position: originPixelOffsetFromFramebufferCenter, radius: rStroke })
    const C = LineControlPoint({ position: tipPixelOffsetFromFramebufferCenter, radius: rStroke })
    const D = LineControlPoint({ position: tipPixelOffsetFromFramebufferCenter, radius: rStroke })

    const result = lineSegmentVariableWidth(vertexIndex, A, B, C, D, d.u32(AXIS_JOIN_MAX))

    const measured = result.vertexPosition
    const ndcExtrudedClipXY = d.vec2f(measured.x / halfWH.x, measured.y / halfWH.y)

    const ndcZ0 = clipOrigin.z * invWo
    const ndcZ1 = clipTip.z * invWt
    const spine = tipPixelOffsetFromFramebufferCenter - originPixelOffsetFromFramebufferCenter
    const spineLenSq = spine.x * spine.x + spine.y * spine.y
    const fromO = d.vec2f(
      measured.x - originPixelOffsetFromFramebufferCenter.x,
      measured.y - originPixelOffsetFromFramebufferCenter.y,
    )
    const t = clamp((fromO.x * spine.x + fromO.y * spine.y) / max(spineLenSq, d.f32(1e-8)), d.f32(0), d.f32(1))
    const ndcZ = ndcZ0 * (d.f32(1) - t) + ndcZ1 * t

    const w = result.w
    const clipPos = d.vec4f(ndcExtrudedClipXY.x * w, ndcExtrudedClipXY.y * w, ndcZ * w, w)

    return {
      clipPos,
      instanceIndexFlat: instanceIndex,
      position: ndcExtrudedClipXY,
      uv: d.vec2f(0, select(d.f32(0), d.f32(1), vertexIndex > 1)),
    }
  })

  const frag = tgpu.fragmentFn({
    in: {
      instanceIndexFlat: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ instanceIndexFlat }) => {
    'use gpu'
    let rgb = d.vec3f(1, 0.12, 0.12)
    rgb = select(rgb, d.vec3f(0.14, 0.98, 0.22), instanceIndexFlat === d.u32(1))
    rgb = select(rgb, d.vec3f(0.24, 0.38, 1), instanceIndexFlat === d.u32(2))

    return d.vec4f(rgb, 1)
  })

  const pipeline = root
    .with(joinSlot, joins.round)
    .with(startCapSlot, caps.butt)
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

  return {
    pipeline,
    axisIndexCount: axisTriangleIndexU16.length,
  }
}

/** Two triangles for a TL/TR/BL/BR quad: (TL, TR, BL), (BL, TR, BR). */
const tagQuadIndexU16 = new Uint16Array([0, 1, 2, 2, 1, 3])

export function createTagQuadsResultsPipeline(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const vert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      clipPos: d.builtin.position,
      uv: d.vec2f,
      instanceIndexFlat: d.interpolate('flat', d.u32),
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const u = tagQuadsBindLayout.$.tagQuadUniform
    const tag = tagQuadsBindLayout.$.tags[instanceIndex]!
    const p = tag.corners[vertexIndex]!
    const uvU = d.f32(vertexIndex & d.u32(1))
    const uvV = d.f32(vertexIndex >> d.u32(1))
    return {
      clipPos: mul(u.clipHomogeneousMatrixFromBoard, d.vec4f(p.x, -p.y, p.z, d.f32(1))),
      uv: d.vec2f(uvU, uvV),
      instanceIndexFlat: instanceIndex,
    }
  })

  const frag = tgpu.fragmentFn({
    in: {
      uv: d.vec2f,
      instanceIndexFlat: d.interpolate('flat', d.u32),
      frontFacing: d.builtin.frontFacing,
    },
    out: d.vec4f,
  })(({ uv, instanceIndexFlat, frontFacing }) => {
    'use gpu'
    if (frontFacing) {
      return d.vec4f(0, 0, 0, 1)
    }
    const cell = d.vec2i(floor(uv * d.f32(8)))
    const onBorder = cell.x <= d.i32(0) || cell.x >= d.i32(7) || cell.y <= d.i32(0) || cell.y >= d.i32(7)
    if (onBorder) {
      return d.vec4f(0, 0, 0, 1)
    }
    const row = d.u32(cell.y - d.i32(1))
    const col = d.u32(cell.x - d.i32(1))
    const bitIndex = row * d.u32(6) + col
    const tag = tagQuadsBindLayout.$.tags[instanceIndexFlat]!
    const word = select(tag.packedPattern.x, tag.packedPattern.y, bitIndex >= d.u32(32))
    const shift = select(bitIndex, bitIndex - d.u32(32), bitIndex >= d.u32(32))
    const bit = (word >> shift) & d.u32(1)
    const v = d.f32(bit)
    return d.vec4f(v, v, v, 1)
  })

  const pipeline = root
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: {
        format: presentationFormat,
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })
    .withIndexBuffer(root.createBuffer(arrayOf(u16, tagQuadIndexU16.length), tagQuadIndexU16).$usage('index'))

  return {
    pipeline,
    indexCount: tagQuadIndexU16.length,
  }
}

export function encodeResultsCanvasFrame(
  root: TgpuRoot,
  context: GPUCanvasContext,
  msaaColorView: GPUTextureView,
  depthView: GPUTextureView,
  markerGpu: ReturnType<typeof createMarkerResultsPipeline>,
  axesGpu: ReturnType<typeof createAxesResultsPipeline>,
  tagQuadsGpu: ReturnType<typeof createTagQuadsResultsPipeline>,
  markerBg: object,
  axisBg: object,
  tagQuadsBg: object,
  markerInstances: number,
  tagCount: number,
): void {
  const enc = root.device.createCommandEncoder({ label: 'results frame' })
  const resolveTarget = context.getCurrentTexture().createView()
  const pass = enc.beginRenderPass({
    label: 'results scene',
    colorAttachments: [
      {
        view: msaaColorView,
        resolveTarget,
        clearValue: [0.1, 0.1, 0.2, 1],
        loadOp: 'clear',
        storeOp: 'discard',
      },
    ],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 1,
      depthLoadOp: 'clear',
      depthStoreOp: 'discard',
    },
  })

  axesGpu.pipeline
    .with(pass)
    .with(axisBg as never)
    .drawIndexed(axesGpu.axisIndexCount, 3)
  if (tagCount > 0) {
    tagQuadsGpu.pipeline
      .with(pass)
      .with(tagQuadsBg as never)
      .drawIndexed(tagQuadsGpu.indexCount, tagCount)
  }
  markerGpu.pipeline
    .with(pass)
    .with(markerBg as never)
    .draw(circleVertexCount(3), markerInstancesForEncode(markerInstances))
  pass.end()
  root.device.queue.submit([enc.finish()])
}

export function destroyGpuTexture(tex: GPUTexture | undefined) {
  tex?.destroy()
}
