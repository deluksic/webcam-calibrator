import { circle } from '@typegpu/geometry'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { abs, length, min, mix, smoothstep } from 'typegpu/std'

import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { PREMULTIPLIED_ALPHA_BLEND } from '@/gpu/pipelines/shared'
import { ROI_CIRCLE_RADIUS_CSS_PX, ROI_CIRCLE_VERTEX_COUNT } from '@/gpu/roiLab/roiCircleConstants'
import {
  ROI_REJECT_FIT,
  ROI_REJECT_GATE,
  ROI_REJECT_INDEFINITE,
  ROI_REJECT_LOW_AREA,
  ROI_REJECT_LOW_RESPONSE,
  ROI_REJECT_NMS,
  RoiCircleSchema,
} from '@/gpu/roiLab/roiPeakTypes'

const circleParams = d.struct({
  radiusPx: d.f32,
  canvasWidth: d.f32,
  canvasHeight: d.f32,
  frameWidth: d.f32,
  frameHeight: d.f32,
  zoom: d.f32,
  panX: d.f32,
  panY: d.f32,
})

const circlesLayout = tgpu.bindGroupLayout({
  circles: { storage: RoiCircleSchema, access: 'readonly' },
  params: { uniform: circleParams },
})

const selectionParams = d.struct({
  radiusPx: d.f32,
  canvasWidth: d.f32,
  canvasHeight: d.f32,
  frameWidth: d.f32,
  frameHeight: d.f32,
  zoom: d.f32,
  panX: d.f32,
  panY: d.f32,
  x: d.f32,
  y: d.f32,
  enabled: d.u32,
  _pad: d.u32,
})

const selectionLayout = tgpu.bindGroupLayout({
  params: { uniform: selectionParams },
})

export type RoiCircleStage = {
  encode: (pass: GPURenderPassEncoder) => void
  setViewport: (zoom: number, panX: number, panY: number) => void
  setShowRejections: (show: boolean) => void
  setSelectedPeak: (peak: { x: number; y: number } | undefined) => void
}

export function createRoiCircleStage(
  root: TgpuRoot,
  presentationFormat: GPUTextureFormat,
  canvasWidth: number,
  canvasHeight: number,
  frameWidth: number,
  frameHeight: number,
  devicePixelRatio: number,
  circleBuffer: unknown,
  drawIndirectBuffer: unknown,
  rejectBuffer?: unknown,
  rejectDrawIndirectBuffer?: unknown,
): RoiCircleStage {
  const radiusPx = ROI_CIRCLE_RADIUS_CSS_PX * devicePixelRatio
  const uniform = root.createUniform(circleParams, {
    radiusPx,
    canvasWidth,
    canvasHeight,
    frameWidth,
    frameHeight,
    zoom: 1,
    panX: 0,
    panY: 0,
  })
  const bindGroup = root.createBindGroup(circlesLayout, {
    circles: circleBuffer as never,
    params: uniform,
  })
  const rejectBindGroup =
    rejectBuffer !== undefined && rejectDrawIndirectBuffer !== undefined
      ? root.createBindGroup(circlesLayout, {
          circles: rejectBuffer as never,
          params: uniform,
        })
      : undefined
  let showRejects = false
  let selectedPeak: { x: number; y: number } | undefined
  let viewZoom = 1
  let viewPanX = 0
  let viewPanY = 0
  const selectionUniform = root.createUniform(selectionParams, {
    radiusPx,
    canvasWidth,
    canvasHeight,
    frameWidth,
    frameHeight,
    zoom: 1,
    panX: 0,
    panY: 0,
    x: 0,
    y: 0,
    enabled: 0,
    _pad: 0,
  })
  const selectionBindGroup = root.createBindGroup(selectionLayout, {
    params: selectionUniform,
  })
  const writeSelection = () => {
    selectionUniform.write({
      radiusPx,
      canvasWidth,
      canvasHeight,
      frameWidth,
      frameHeight,
      zoom: viewZoom,
      panX: viewPanX,
      panY: viewPanY,
      x: selectedPeak?.x ?? 0,
      y: selectedPeak?.y ?? 0,
      enabled: selectedPeak === undefined ? 0 : 1,
      _pad: 0,
    })
  }

  const vertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      clipPosition: d.builtin.position,
      unit: d.vec2f,
      polarity: d.interpolate('flat', d.u32),
      reason: d.interpolate('flat', d.u32),
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const circleInstance = circlesLayout.$.circles[instanceIndex]!
    const params = circlesLayout.$.params
    const unit = circle(vertexIndex)
    const screenX = circleInstance.x * params.zoom + params.panX
    const screenY = circleInstance.y * params.zoom + params.panY
    const centerX = (screenX / params.frameWidth) * d.f32(2) - d.f32(1)
    const centerY = d.f32(1) - (screenY / params.frameHeight) * d.f32(2)
    const markerRadius = circleInstance.reason === d.u32(0) ? params.radiusPx : params.radiusPx * d.f32(1.6)
    const offsetX = (unit.x * markerRadius * d.f32(2)) / params.canvasWidth
    const offsetY = (unit.y * markerRadius * d.f32(2)) / params.canvasHeight
    return {
      clipPosition: d.vec4f(centerX + offsetX, centerY + offsetY, d.f32(0), d.f32(1)),
      unit,
      polarity: circleInstance.polarity,
      reason: circleInstance.reason,
    }
  })

  const fragment = tgpu.fragmentFn({
    in: {
      unit: d.vec2f,
      polarity: d.interpolate('flat', d.u32),
      reason: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ unit, polarity, reason }) => {
    'use gpu'
    const radius = length(unit)
    if (reason !== d.u32(0)) {
      const cross = min(abs(unit.x - unit.y), abs(unit.x + unit.y))
      const alpha =
        (d.f32(1) - smoothstep(d.f32(0.06), d.f32(0.16), cross)) *
        (d.f32(1) - smoothstep(d.f32(0.85), d.f32(1), radius))
      let color = d.vec3f(0.2, 0.9, 1.0)
      if (reason === d.u32(ROI_REJECT_LOW_RESPONSE)) {
        color = d.vec3f(0.2, 0.9, 1.0)
      } else if (reason === d.u32(ROI_REJECT_FIT)) {
        color = d.vec3f(1.0, 0.2, 0.15)
      } else if (reason === d.u32(ROI_REJECT_INDEFINITE)) {
        color = d.vec3f(1.0, 0.55, 0.1)
      } else if (reason === d.u32(ROI_REJECT_NMS)) {
        color = d.vec3f(1.0, 0.2, 0.8)
      } else if (reason === d.u32(ROI_REJECT_LOW_AREA)) {
        color = d.vec3f(1.0, 0.55, 0.1)
      } else if (reason === d.u32(ROI_REJECT_GATE)) {
        color = d.vec3f(1.0, 0.85, 0.1)
      }
      return d.vec4f(color, alpha)
    }
    const alpha = d.f32(1) - smoothstep(d.f32(0.9), d.f32(1), radius)
    const base = polarity === d.u32(0) ? d.vec3f(0.15, 0.46, 1.0) : d.vec3f(1.0, 0.68, 0.12)
    const outlined = mix(base, d.vec3f(0.03, 0.04, 0.06), smoothstep(d.f32(0.6), d.f32(0.9), radius))
    return d.vec4f(outlined, alpha)
  })

  const pipeline = root.createRenderPipeline({
    vertex,
    fragment,
    targets: { format: presentationFormat, blend: PREMULTIPLIED_ALPHA_BLEND },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
  })

  const selectionVertex = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex },
    out: {
      clipPosition: d.builtin.position,
      unit: d.vec2f,
    },
  })(({ vertexIndex }) => {
    'use gpu'
    const params = selectionLayout.$.params
    const unit = circle(vertexIndex)
    const screenX = params.x * params.zoom + params.panX
    const screenY = params.y * params.zoom + params.panY
    const centerX = (screenX / params.frameWidth) * d.f32(2) - d.f32(1)
    const centerY = d.f32(1) - (screenY / params.frameHeight) * d.f32(2)
    const markerRadius = params.radiusPx * d.f32(2.6)
    const offsetX = (unit.x * markerRadius * d.f32(2)) / params.canvasWidth
    const offsetY = (unit.y * markerRadius * d.f32(2)) / params.canvasHeight
    return {
      clipPosition: d.vec4f(centerX + offsetX, centerY + offsetY, d.f32(0), d.f32(1)),
      unit,
    }
  })

  const selectionFragment = tgpu.fragmentFn({
    in: { unit: d.vec2f },
    out: d.vec4f,
  })(({ unit }) => {
    'use gpu'
    const radius = length(unit)
    const ring = d.f32(1) - smoothstep(d.f32(0.025), d.f32(0.1), abs(radius - d.f32(0.82)))
    const halo = (d.f32(1) - smoothstep(d.f32(0.9), d.f32(1), radius)) * d.f32(0.18)
    return d.vec4f(d.vec3f(1.0, 0.92, 0.2), ring * d.f32(0.95) + halo)
  })

  const selectionPipeline = root.createRenderPipeline({
    vertex: selectionVertex,
    fragment: selectionFragment,
    targets: { format: presentationFormat, blend: PREMULTIPLIED_ALPHA_BLEND },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
  })

  return {
    encode(pass) {
      pipeline
        .with(pass)
        .with(bindGroup)
        .drawIndirect(drawIndirectBuffer as never)
      if (showRejects && rejectBindGroup !== undefined && rejectDrawIndirectBuffer !== undefined) {
        pipeline
          .with(pass)
          .with(rejectBindGroup)
          .drawIndirect(rejectDrawIndirectBuffer as never)
      }
      if (selectedPeak !== undefined) {
        selectionPipeline.with(pass).with(selectionBindGroup).draw(ROI_CIRCLE_VERTEX_COUNT)
      }
    },
    setShowRejections(show) {
      showRejects = show
    },
    setSelectedPeak(peak) {
      selectedPeak = peak
      writeSelection()
    },
    setViewport(zoom, panX, panY) {
      viewZoom = zoom
      viewPanX = panX
      viewPanY = panY
      uniform.write({
        radiusPx,
        canvasWidth,
        canvasHeight,
        frameWidth,
        frameHeight,
        zoom,
        panX,
        panY,
      })
      writeSelection()
    },
  }
}
