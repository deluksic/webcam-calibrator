import { circle } from '@typegpu/geometry'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { abs, length, min, mix, smoothstep } from 'typegpu/std'

import { PREMULTIPLIED_ALPHA_BLEND } from '@/gpu/pipelines/shared'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { ROI_CIRCLE_RADIUS_CSS_PX } from '@/gpu/roiLab/roiCircleConstants'
import {
  ROI_REJECT_FIT,
  ROI_REJECT_GATE,
  ROI_REJECT_LOW_AREA,
  ROI_REJECT_LOW_RESPONSE,
  ROI_REJECT_NMS,
  RoiCircleSchema,
} from '@/gpu/roiLab/roiPeakPipeline'


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

export type RoiCircleStage = {
  encode: (pass: GPURenderPassEncoder) => void
  setViewport: (zoom: number, panX: number, panY: number) => void
  setShowRejections: (show: boolean) => void
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
    const markerRadius = circleInstance.reason === d.u32(0)
      ? params.radiusPx
      : params.radiusPx * d.f32(1.6)
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

  return {
    encode(pass) {
      pipeline.with(pass).with(bindGroup).drawIndirect(drawIndirectBuffer as never)
      if (showRejects && rejectBindGroup !== undefined && rejectDrawIndirectBuffer !== undefined) {
        pipeline
          .with(pass)
          .with(rejectBindGroup)
          .drawIndirect(rejectDrawIndirectBuffer as never)
      }
    },
    setShowRejections(show) {
      showRejects = show
    },
    setViewport(zoom, panX, panY) {
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
    },
  }
}
