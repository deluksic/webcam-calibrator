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
import { mul, select } from 'typegpu/std'

import { MARKER_DISK_SUBDIV, axisBindLayout, markersBindLayout } from '@/gpu/resultsVizLayouts'

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
        depthWriteEnabled: false,
        depthCompare: 'always',
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
    const pc = row.positionCentroidRelativeBoardUnits
    const centerClip = mul(u.clipHomogeneousMatrixFromCentroidBoard, d.vec4f(pc.x, pc.y, pc.z, d.f32(1)))
    const unit = circle(vertexIndex)
    const half = u.viewportHalfSizePixels
    const rPx = u.markerDiskRadiusPixels
    const dNdcX = unit.x * rPx / half.x
    const dNdcY = unit.y * rPx / half.y
    const w = centerClip.w
    return {
      clipPos: d.vec4f(centerClip.x + dNdcX * w, centerClip.y + dNdcY * w, centerClip.z, centerClip.w),
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
      /** Circles example has no depth; depth test was hiding disks in the MSAA pass. */
      depthWriteEnabled: false,
      depthCompare: 'always',
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
    const M = u.clipHomogeneousMatrixFromCentroidBoard
    const reach = u.worldAxisReachDistanceCentroidBoardUnits
    const ex = d.vec4f(reach, 0, 0, 1)
    const ey = d.vec4f(0, reach, 0, 1)
    const ez = d.vec4f(0, 0, reach, 1)
    const tipBoard = select(select(ex, ey, instanceIndex === d.u32(1)), ez, instanceIndex === d.u32(2))

    const clipOrigin = mul(M, d.vec4f(0, 0, 0, 1))
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

    const w = result.w
    const clipPos = d.vec4f(ndcExtrudedClipXY.x * w, ndcExtrudedClipXY.y * w, d.f32(0.5) * w, w)

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
        depthWriteEnabled: false,
        depthCompare: 'always',
      },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })
    .withIndexBuffer(root.createBuffer(arrayOf(u16, axisTriangleIndexU16.length), axisTriangleIndexU16).$usage('index'))

  return {
    pipeline,
    axisIndexCount: axisTriangleIndexU16.length,
  }
}

export function encodeResultsCanvasFrame(
  root: TgpuRoot,
  context: GPUCanvasContext,
  msaaColorView: GPUTextureView,
  depthView: GPUTextureView,
  markerGpu: ReturnType<typeof createMarkerResultsPipeline>,
  axesGpu: ReturnType<typeof createAxesResultsPipeline>,
  markerBg: object,
  axisBg: object,
  markerInstances: number,
): void {
  const enc = root.device.createCommandEncoder({ label: 'results frame' })
  const resolveTarget = context.getCurrentTexture().createView()
  const pass = enc.beginRenderPass({
    label: 'results scene',
    colorAttachments: [
      {
        view: msaaColorView,
        resolveTarget,
        clearValue: [0.05, 0.05, 0.065, 1],
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
