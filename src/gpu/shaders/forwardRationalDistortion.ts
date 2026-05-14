import { d, tgpu } from 'typegpu'

import { RationalDistortion8Gpu } from '@/gpu/schemas/cameraGpuUniforms'

/**
 * OpenCV rational distortion + Brown–Conrady tangential terms on **ideal** normalized
 * image coordinates (xn, yn). Returns distorted normalized (xd, yd).
 */
export const forwardDistortNormalized = tgpu.fn(
  [d.vec2f, RationalDistortion8Gpu],
  d.vec2f,
)((xy, dist) => {
  'use gpu'
  const xn = xy.x
  const yn = xy.y
  const r2 = xn * xn + yn * yn
  const r4 = r2 * r2
  const r6 = r4 * r2

  const radialNum = d.f32(1) + dist.k1 * r2 + dist.k2 * r4 + dist.k3 * r6
  const radialDen = d.f32(1) + dist.k4 * r2 + dist.k5 * r4 + dist.k6 * r6

  const xd = (xn * radialNum) / radialDen + d.f32(2) * dist.p1 * xn * yn + dist.p2 * (r2 + d.f32(2) * xn * xn)
  const yd = (yn * radialNum) / radialDen + dist.p1 * (r2 + d.f32(2) * yn * yn) + d.f32(2) * dist.p2 * xn * yn

  return d.vec2f(xd, yd)
})
