import { circle, circleVertexCount } from '@typegpu/geometry'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { abs, fwidth, length, mix, select, smoothstep } from 'typegpu/std'

import { PREMULTIPLIED_ALPHA_BLEND } from '@/gpu/pipelines/shared'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { stableHashToRgb01 } from '@/lib/hashStableColor'

const MAX_GRAPH_POINTS = 4096
const MAX_GRAPH_EDGES = 8192
const NODE_RADIUS_CSS_PX = 4.4
const EDGE_HALF_WIDTH_CSS_PX = 0.65

export const ROI_IDENTITY_RESOLVED = 0
export const ROI_IDENTITY_AMBIGUOUS = 1
export const ROI_IDENTITY_UNRESOLVED = 2
export const ROI_IDENTITY_CONFLICT = 3

export type RoiGraphPoint = {
  x: number
  y: number
  designId: number
  status: number
}

export type RoiGraphEdge = {
  a: number
  b: number
}

export type RoiGraphData = {
  points: RoiGraphPoint[]
  edges: RoiGraphEdge[]
}

const graphPointSchema = d.struct({
  x: d.f32,
  y: d.f32,
  designId: d.u32,
  status: d.u32,
})
const graphPointArray = d.arrayOf(graphPointSchema, MAX_GRAPH_POINTS)

const graphEdgeSchema = d.struct({
  a: d.u32,
  b: d.u32,
})
const graphEdgeArray = d.arrayOf(graphEdgeSchema, MAX_GRAPH_EDGES)

const graphParams = d.struct({
  zoom: d.f32,
  panX: d.f32,
  panY: d.f32,
  frameWidth: d.f32,
  frameHeight: d.f32,
  canvasWidth: d.f32,
  canvasHeight: d.f32,
  nodeRadiusPx: d.f32,
  edgeHalfWidthPx: d.f32,
  pointCount: d.u32,
  edgeCount: d.u32,
})

const graphLayout = tgpu.bindGroupLayout({
  points: { storage: graphPointArray, access: 'readonly' },
  edges: { storage: graphEdgeArray, access: 'readonly' },
  params: { uniform: graphParams },
})

const CIRCLE_SUBDIV = 2
const CIRCLE_VERTEX_COUNT = circleVertexCount(CIRCLE_SUBDIV)

const EDGE_CORNERS = tgpu.const(
  d.arrayOf(d.vec2f, 6),
  [
    d.vec2f(-1, 0),
    d.vec2f(1, 0),
    d.vec2f(-1, 1),
    d.vec2f(1, 0),
    d.vec2f(1, 1),
    d.vec2f(-1, 1),
  ],
)

export type RoiGraphStage = {
  setGraph: (data: RoiGraphData) => void
  setViewport: (zoom: number, panX: number, panY: number) => void
  encode: (pass: GPURenderPassEncoder) => void
  destroy: () => void
}

export function createRoiGraphStage(
  root: TgpuRoot,
  presentationFormat: GPUTextureFormat,
  canvasWidth: number,
  canvasHeight: number,
  frameWidth: number,
  frameHeight: number,
  displayScale: number,
): RoiGraphStage {
  const pointsBuffer = root.createBuffer(graphPointArray).$usage('storage')
  const edgesBuffer = root.createBuffer(graphEdgeArray).$usage('storage')
  const paramsBuffer = root.createUniform(graphParams, {
    zoom: 1,
    panX: 0,
    panY: 0,
    frameWidth,
    frameHeight,
    canvasWidth,
    canvasHeight,
    nodeRadiusPx: NODE_RADIUS_CSS_PX * displayScale,
    edgeHalfWidthPx: EDGE_HALF_WIDTH_CSS_PX * displayScale,
    pointCount: 0,
    edgeCount: 0,
  })
  let zoom = 1
  let panX = 0
  let panY = 0
  let pointCount = 0
  let edgeCount = 0

  const writeParams = () => {
    paramsBuffer.write({
      zoom,
      panX,
      panY,
      frameWidth,
      frameHeight,
      canvasWidth,
      canvasHeight,
      nodeRadiusPx: NODE_RADIUS_CSS_PX * displayScale,
      edgeHalfWidthPx: EDGE_HALF_WIDTH_CSS_PX * displayScale,
      pointCount,
      edgeCount,
    })
  }

  const bindGroup = root.createBindGroup(graphLayout, {
    points: pointsBuffer,
    edges: edgesBuffer,
    params: paramsBuffer,
  })

  const nodeVertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      clipPosition: d.builtin.position,
      unit: d.vec2f,
      designId: d.interpolate('flat', d.u32),
      status: d.interpolate('flat', d.u32),
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const point = graphLayout.$.points[instanceIndex]!
    const params = graphLayout.$.params
    const unit = circle(vertexIndex)
    const screenX = point.x * params.zoom + params.panX
    const screenY = point.y * params.zoom + params.panY
    const centerX = (screenX / params.frameWidth) * d.f32(2) - d.f32(1)
    const centerY = d.f32(1) - (screenY / params.frameHeight) * d.f32(2)
    const offsetX = (unit.x * params.nodeRadiusPx * d.f32(2)) / params.canvasWidth
    const offsetY = (unit.y * params.nodeRadiusPx * d.f32(2)) / params.canvasHeight
    return {
      clipPosition: d.vec4f(centerX + offsetX, centerY + offsetY, d.f32(0), d.f32(1)),
      unit,
      designId: point.designId,
      status: point.status,
    }
  })

  const nodeFragment = tgpu.fragmentFn({
    in: {
      unit: d.vec2f,
      designId: d.interpolate('flat', d.u32),
      status: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ unit, designId, status }) => {
    'use gpu'
    const radius = length(unit)
    const aa = fwidth(radius)
    const coverage = d.f32(1) - smoothstep(d.f32(1) - aa * d.f32(1.5), d.f32(1), radius)
    if (status === d.u32(ROI_IDENTITY_RESOLVED)) {
      const fill = stableHashToRgb01(designId)
      const outline = smoothstep(d.f32(0.72) - aa, d.f32(0.72) + aa, radius)
      const color = mix(fill, d.vec3f(0.035, 0.04, 0.055), outline)
      return d.vec4f(color, coverage)
    }
    const ringOuter = smoothstep(d.f32(0.62) - aa, d.f32(0.62) + aa, radius) *
      (d.f32(1) - smoothstep(d.f32(0.94) - aa, d.f32(0.94) + aa, radius))
    let ring = ringOuter
    if (status === d.u32(ROI_IDENTITY_CONFLICT)) {
      const inner = smoothstep(d.f32(0.26) - aa, d.f32(0.26) + aa, radius) *
        (d.f32(1) - smoothstep(d.f32(0.44) - aa, d.f32(0.44) + aa, radius))
      ring = ringOuter + inner
    }
    const base = status === d.u32(ROI_IDENTITY_UNRESOLVED)
      ? d.vec3f(0.55, 0.58, 0.64)
      : d.vec3f(1, 1, 1)
    return d.vec4f(base, ring * coverage)
  })

  const nodePipeline = root.createRenderPipeline({
    vertex: nodeVertex,
    fragment: nodeFragment,
    targets: { format: presentationFormat, blend: PREMULTIPLIED_ALPHA_BLEND },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
  })

  const edgeVertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      clipPosition: d.builtin.position,
      side: d.f32,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const params = graphLayout.$.params
    const edge = graphLayout.$.edges[instanceIndex]!
    const first = graphLayout.$.points[edge.a]!
    const second = graphLayout.$.points[edge.b]!
    const ax = first.x * params.zoom + params.panX
    const ay = first.y * params.zoom + params.panY
    const bx = second.x * params.zoom + params.panX
    const by = second.y * params.zoom + params.panY
    const dx = bx - ax
    const dy = by - ay
    const len = length(d.vec2f(dx, dy))
    const nx = select(d.f32(0), -dy / len, len > d.f32(1e-6))
    const ny = select(d.f32(0), dx / len, len > d.f32(1e-6))
    const corner = EDGE_CORNERS.$[vertexIndex]!
    const px = ax + dx * d.f32(corner.y) + nx * d.f32(corner.x) * params.edgeHalfWidthPx
    const py = ay + dy * d.f32(corner.y) + ny * d.f32(corner.x) * params.edgeHalfWidthPx
    const ndcX = (px / params.frameWidth) * d.f32(2) - d.f32(1)
    const ndcY = d.f32(1) - (py / params.frameHeight) * d.f32(2)
    return {
      clipPosition: d.vec4f(ndcX, ndcY, d.f32(0), d.f32(1)),
      side: d.f32(corner.x),
    }
  })

  const edgeFragment = tgpu.fragmentFn({
    in: { side: d.f32 },
    out: d.vec4f,
  })(({ side }) => {
    'use gpu'
    const aa = fwidth(side)
    const coverage = d.f32(1) - smoothstep(d.f32(1) - aa, d.f32(1), abs(side))
    return d.vec4f(d.vec3f(0.65, 0.7, 0.8), coverage * d.f32(0.65))
  })

  const edgePipeline = root.createRenderPipeline({
    vertex: edgeVertex,
    fragment: edgeFragment,
    targets: { format: presentationFormat, blend: PREMULTIPLIED_ALPHA_BLEND },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
  })

  return {
    setGraph(data) {
      const points = data.points.slice(0, MAX_GRAPH_POINTS)
      const edges = data.edges
        .filter((edge) => edge.a >= 0 && edge.a < points.length && edge.b >= 0 && edge.b < points.length)
        .slice(0, MAX_GRAPH_EDGES)
      pointCount = points.length
      edgeCount = edges.length
      if (points.length > 0) {
        pointsBuffer.write(points)
      }
      if (edges.length > 0) {
        edgesBuffer.write(edges)
      }
      writeParams()
    },
    setViewport(nextZoom, nextPanX, nextPanY) {
      zoom = nextZoom
      panX = nextPanX
      panY = nextPanY
      writeParams()
    },
    encode(pass) {
      if (edgeCount > 0) {
        edgePipeline.with(pass).with(bindGroup).draw(6, edgeCount)
      }
      if (pointCount > 0) {
        nodePipeline.with(pass).with(bindGroup).draw(CIRCLE_VERTEX_COUNT, pointCount)
      }
    },
    destroy() {
      pointsBuffer.destroy()
      edgesBuffer.destroy()
    },
  }
}
