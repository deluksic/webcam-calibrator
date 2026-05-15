import { oklabToRgb } from '@typegpu/color'
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { common, std } from 'typegpu'
import { abs, clamp, exp, fwidth, length, max, min, round, select } from 'typegpu/std'

import { PinholeIntrinsicsGpu, RationalDistortion8Gpu } from '@/gpu/schemas/cameraGpuUniforms'
import { forwardDistortNormalized } from '@/gpu/shaders/forwardRationalDistortion'
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
 * Single spacing of |Δ| isolines (step = 1/invStep px). Plateau + linear rim; returns mask in [0, 1].
 * `fw` must be `fwidth` of the **original** smooth scalar field (e.g. |Δ|), never of `abs`/`round` outputs.
 */
const isoLevelPlateauMask = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32],
  d.f32,
)((mag, invStep, coreK, edgeK, fw) => {
  'use gpu'
  const dist = abs(mag * invStep - round(mag * invStep)) / invStep
  const fwSafe = max(fw, d.f32(1e-4))
  const core = coreK * fwSafe
  const edge = edgeK * fwSafe
  return d.f32(1) - clamp((dist - core) / edge, d.f32(0), d.f32(1))
})

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

    const inside = px >= ox && px < ox + scaledW && py >= oy && py < oy + scaledH

    /** Clamped canvas coords so distortion math stays continuous at the letterbox edge (derivatives). */
    const eps = d.f32(1e-4)
    const pxC = clamp(px, ox + eps, ox + scaledW - eps)
    const pyC = clamp(py, oy + eps, oy + scaledH - eps)

    const xnPx = ((pxC - ox) / scaledW) * videoW
    const ynPx = ((pyC - oy) / scaledH) * videoH

    const xn = (xnPx - intr.cx) / intr.fx
    const yn = (ynPx - intr.cy) / intr.fy

    const xyD = forwardDistortNormalized(d.vec2f(xn, yn), dist)
    const xd = xyD.x
    const yd = xyD.y

    const dxPx = (xd - xn) * intr.fx * vp.scale
    const dyPx = (yd - yn) * intr.fy * vp.scale

    /** Soft-clamp chroma magnitude. Linear up to 0.15, asymptote at 0.2. */
    const chromaMag = length(d.vec2f(dxPx, dyPx))
    const knee = d.f32(0.15)
    const headroom = d.f32(0.05)
    const over = max(chromaMag - knee, d.f32(0))
    const clampedMag = min(chromaMag, knee + headroom * (d.f32(1) - exp(-over / headroom)))
    const scale = clampedMag / max(chromaMag, d.f32(1e-8))
    const a = dxPx * scale
    const b = dyPx * scale

    /** Geometric |Δ| in image pixels (independent of visualization scale). */
    const dispPhysX = (xd - xn) * intr.fx
    const dispPhysY = (yd - yn) * intr.fy
    const magPhys = length(d.vec2f(dispPhysX, dispPhysY))
    const fw = max(fwidth(magPhys), d.f32(1e-4))

    /** Two spacings: whole px vs 0.1 px; per-level mask [0,1] then brightness weights, then OkLab L scale. */
    const isoMajor = isoLevelPlateauMask(magPhys, 1, 0.5, 1, fw)
    const isoMinor = isoLevelPlateauMask(magPhys, 10, 0.1, 0.8, fw)
    const tick = max(isoMajor, isoMinor * 0.5)
    const labL = 0.6 - tick * 0.15
    const rgb = oklabToRgb(d.vec3f(labL, a, b))

    const bg = d.vec4f(0.02, 0.02, 0.05, 1)
    const fg = d.vec4f(rgb, 1)
    return select(bg, fg, inside)
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
