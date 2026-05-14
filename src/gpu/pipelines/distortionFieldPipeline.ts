import { oklabToRgb } from '@typegpu/color'
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { common, std } from 'typegpu'

import { PinholeIntrinsicsGpu, RationalDistortion8Gpu } from '@/gpu/schemas/cameraGpuUniforms'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'

const DistortionViewportGpu = d.struct({
  videoSize: d.vec2f,
  canvasSize: d.vec2f,
  scale: d.f32,
  _pad0: d.f32,
  _pad1: d.vec2f,
})

const DistortionUniformStruct = d.struct({
  intrinsics: PinholeIntrinsicsGpu,
  distortion: RationalDistortion8Gpu,
  viewport: DistortionViewportGpu,
})

const distortionLayout = tgpu.bindGroupLayout({
  dist: { uniform: DistortionUniformStruct },
})

function allocDistortionUniform(root: TgpuRoot) {
  return root.createBuffer(DistortionUniformStruct).$usage('uniform')
}

export type DistortionUniformGpuBuffer = ReturnType<typeof allocDistortionUniform>

export type DistortionFieldUniformParams = {
  K: CameraIntrinsics
  distortion: RationalDistortion8
  scale: number
  videoWidth: number
  videoHeight: number
  canvasWidth: number
  canvasHeight: number
}

export function updateDistortionUniform(buf: DistortionUniformGpuBuffer, params: DistortionFieldUniformParams): void {
  const [k1, k2, p1, p2, k3, k4, k5, k6] = params.distortion
  buf.write({
    intrinsics: params.K,
    distortion: { k1, k2, p1, p2, k3, k4, k5, k6 },
    viewport: {
      videoSize: d.vec2f(params.videoWidth, params.videoHeight),
      canvasSize: d.vec2f(params.canvasWidth, params.canvasHeight),
      scale: params.scale,
      _pad0: 0,
      _pad1: d.vec2f(0, 0),
    },
  })
}

/**
 * Fullscreen distortion field visualization. Exposes CPU uniform updates and pass encoding only;
 * bind group and pipeline stay internal (same pattern as {@link createMarkerResultsStage}).
 */
export function createDistortionFieldStage(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const frag = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
  })(({ pos }) => {
    'use gpu'
    const u = distortionLayout.$.dist
    const intr = u.intrinsics
    const dist = u.distortion
    const vp = u.viewport

    const px = pos.x
    const py = pos.y

    const canvasW = vp.canvasSize.x
    const canvasH = vp.canvasSize.y
    const videoW = vp.videoSize.x
    const videoH = vp.videoSize.y

    const canvasAspect = canvasW / canvasH
    const videoAspect = videoW / videoH

    let fitScale = canvasW / videoW
    if (canvasAspect > videoAspect) {
      fitScale = canvasH / videoH
    }

    const scaledW = videoW * fitScale
    const scaledH = videoH * fitScale
    const ox = (canvasW - scaledW) * d.f32(0.5)
    const oy = (canvasH - scaledH) * d.f32(0.5)

    if (px < ox || px >= ox + scaledW || py < oy || py >= oy + scaledH) {
      return d.vec4f(d.f32(0.02), d.f32(0.02), d.f32(0.05), d.f32(1))
    }

    const xnPx = ((px - ox) / scaledW) * videoW
    const ynPx = ((py - oy) / scaledH) * videoH

    const xn = (xnPx - intr.cx) / intr.fx
    const yn = (ynPx - intr.cy) / intr.fy

    const r2 = xn * xn + yn * yn
    const r4 = r2 * r2
    const r6 = r4 * r2

    const radialNum = d.f32(1) + dist.k1 * r2 + dist.k2 * r4 + dist.k3 * r6
    const radialDen = d.f32(1) + dist.k4 * r2 + dist.k5 * r4 + dist.k6 * r6

    const xd = (xn * radialNum) / radialDen + d.f32(2) * dist.p1 * xn * yn + dist.p2 * (r2 + d.f32(2) * xn * xn)
    const yd = (yn * radialNum) / radialDen + dist.p1 * (r2 + d.f32(2) * yn * yn) + d.f32(2) * dist.p2 * xn * yn

    const dxPx = (xd - xn) * intr.fx * vp.scale
    const dyPx = (yd - yn) * intr.fy * vp.scale

    const a = std.clamp(dxPx, d.f32(-1), d.f32(1))
    const b = std.clamp(dyPx, d.f32(-1), d.f32(1))

    const rgb = oklabToRgb(d.vec3f(d.f32(0.6), a, b))
    return d.vec4f(rgb, d.f32(1))
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })

  const uniform = allocDistortionUniform(root)
  const bindGroup = root.createBindGroup(distortionLayout, { dist: uniform })

  const encodeToCanvas = (enc: GPUCommandEncoder, colorAttachment: ColorAttachment) => {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }

  return { uniform, encodeToCanvas }
}
