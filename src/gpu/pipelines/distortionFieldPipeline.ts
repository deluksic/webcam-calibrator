import { oklabToRgb } from '@typegpu/color'
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { common, std } from 'typegpu'
import { abs, clamp, exp, fwidth, length, max, min, mix, pow, round, select } from 'typegpu/std'
import { sdBox2d } from '@typegpu/sdf'

import { PinholeIntrinsicsGpu, RationalDistortion8Gpu } from '@/gpu/schemas/cameraGpuUniforms'
import { forwardDistortNormalized } from '@/gpu/shaders/forwardRationalDistortion'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'

const DistortionViewportGpu = d.struct({
  videoSize: d.vec2f,
  canvasSize: d.vec2f,
  scale: d.f32,
  zoomOutFactor: d.f32,
  _pad: d.vec2f,
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
  zoomOutFactor: number
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
      zoomOutFactor: params.zoomOutFactor,
      _pad: d.vec2f(0, 0),
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
 * Coloring slot — swappable at pipeline creation.
 * Signature: (labL, aRaw, bRaw, chromaMagRaw, dotToCenterRaw, magPhys) → vec3f RGB.
 *
 * `labL` is the isoline-darkened base lightness. `chromaMagRaw` is the raw (scaled)
 * displacement magnitude. `dotToCenterRaw` is the normalized dot product in [-1, 1].
 * `magPhys` is the geometric |Δ| in image pixels, independent of visualization scale.
 */

/** Hue+Chroma: soft-clamp magnitude, then OKLab a/b from displacement direction. */
const colorHueChroma = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
  d.vec3f,
)((labL, aRaw, bRaw, chromaMag) => {
  const knee = d.f32(0.15)
  const headroom = d.f32(0.05)
  const over = max(chromaMag - knee, d.f32(0))
  const clampedMag = min(chromaMag, knee + headroom * (d.f32(1) - exp(-over / headroom)))
  const s = clampedMag / max(chromaMag, d.f32(1e-8))
  return oklabToRgb(d.vec3f(labL, aRaw * s, bRaw * s))
})

/** Gray: scaled magnitude drives lightness (near-black → white). Isolines via labL. */
const colorGray = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
  d.vec3f,
)((labL, aRaw, bRaw, chromaMag) => {
  const t = std.saturate(chromaMag / d.f32(0.2))
  const L = std.saturate(mix(d.f32(0.02), d.f32(1), t) - (d.f32(0.6) - labL))
  return oklabToRgb(d.vec3f(L, d.f32(0), d.f32(0)))
})

/** Radial: distortion toward center → red, away → blue, dark gray at zero → white at high magnitude. */
const colorRadial = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
  d.vec3f,
)((labL, aRaw, bRaw, chromaMag, dotToCenter) => {
  const scaleFactor = std.saturate(chromaMag / d.f32(0.2))
  const signedStrength = dotToCenter * scaleFactor * d.f32(0.3)
  const aOut = max(d.f32(0), signedStrength)
  const bOut = min(d.f32(0), signedStrength)
  const L = std.saturate(mix(d.f32(0.05), labL, abs(signedStrength) * d.f32(5)))
  return oklabToRgb(d.vec3f(L, aOut, bOut))
})

export const distortionColoringSlot = tgpu.slot(colorHueChroma)

export type DistortionColoringMode = 'hueChroma' | 'gray' | 'radial'

function coloringFn(mode: DistortionColoringMode) {
  if (mode === 'gray') {
    return colorGray
  }
  if (mode === 'radial') {
    return colorRadial
  }
  return colorHueChroma
}

/**
 * Fullscreen distortion field visualization. Exposes CPU uniform updates and pass encoding only;
 * bind group and pipeline stay internal (same pattern as {@link createMarkerResultsStage}).
 */
export function createDistortionFieldStage(
  root: TgpuRoot,
  presentationFormat: GPUTextureFormat,
  opts?: { coloring?: DistortionColoringMode },
) {
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

    let baseFitScale = canvasW / videoW
    if (canvasAspect > videoAspect) {
      baseFitScale = canvasH / videoH
    }
    const fitScale = baseFitScale * vp.zoomOutFactor

    const scaledW = videoW * fitScale
    const scaledH = videoH * fitScale
    const ox = (canvasW - scaledW) * d.f32(0.5)
    const oy = (canvasH - scaledH) * d.f32(0.5)

    /** Map canvas pixel to video pixel coords (no letterbox clamping — extends beyond frame). */
    const xnPx = ((px - ox) / scaledW) * videoW
    const ynPx = ((py - oy) / scaledH) * videoH

    const xn = (xnPx - intr.cx) / intr.fx
    const yn = (ynPx - intr.cy) / intr.fy

    const xyD = forwardDistortNormalized(d.vec2f(xn, yn), dist)
    const xd = xyD.x
    const yd = xyD.y

    /** Raw displacement in OKLab chroma space (before any clamping). */
    const dxPx = (xd - xn) * intr.fx * vp.scale * d.f32(0.2)
    const dyPx = (yd - yn) * intr.fy * vp.scale * d.f32(0.2)
    const chromaMagRaw = length(d.vec2f(dxPx, dyPx))

    /** Geometric |Δ| in image pixels (independent of visualization scale). */
    const dispPhysX = (xd - xn) * intr.fx
    const dispPhysY = (yd - yn) * intr.fy
    const magPhys = length(d.vec2f(dispPhysX, dispPhysY))
    const fw = max(fwidth(magPhys), d.f32(1e-4))

    /** Raw normalized dot product: displacement direction vs toward-center direction, in [-1, 1]. */
    const toCenterX = intr.cx - xnPx
    const toCenterY = intr.cy - ynPx
    const toCenterLen = max(length(d.vec2f(toCenterX, toCenterY)), d.f32(1e-8))
    const dispLen = max(length(d.vec2f(dispPhysX, dispPhysY)), d.f32(1e-8))
    const dotToCenterRaw =
      (dispPhysX * toCenterX + dispPhysY * toCenterY) / (dispLen * toCenterLen)

    /** Two spacings: whole px vs 0.1 px; per-level mask [0,1] then brightness weights, then OkLab L scale. */
    const isoMajor = isoLevelPlateauMask(magPhys, 1, 0.5, 1, fw)
    const isoMinor = isoLevelPlateauMask(magPhys, 10, 0.1, 0.8, fw)
    const tick = max(isoMajor, isoMinor * 0.5)
    const labL = 0.6 - tick * 0.15

    const rgb = distortionColoringSlot.$(labL, dxPx, dyPx, chromaMagRaw, dotToCenterRaw, magPhys)

    /** White gradient fading outward from the zoomed video area boundary (rounded SDF, ease-out). */
    const relX = px - (ox + scaledW * d.f32(0.5))
    const relY = py - (oy + scaledH * d.f32(0.5))
    const halfW = scaledW * d.f32(0.5)
    const halfH = scaledH * d.f32(0.5)
    const sdf = sdBox2d(d.vec2f(relX, relY), d.vec2f(halfW, halfH))
    const t = std.saturate(sdf / (scaledH * d.f32(0.25)))
    const outlineMask = select(d.f32(0), d.f32(0.6) * pow(std.saturate(d.f32(1) - t), d.f32(3)), sdf >= d.f32(0))

    const fg = d.vec4f(rgb, 1)
    return mix(fg, d.vec4f(1, 1, 1, 1), d.vec4f(outlineMask, outlineMask, outlineMask, d.f32(0)))
  })

  const rootWithSlot = opts?.coloring ? root.with(distortionColoringSlot, coloringFn(opts.coloring)) : root

  const pipeline = rootWithSlot.createRenderPipeline({
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
