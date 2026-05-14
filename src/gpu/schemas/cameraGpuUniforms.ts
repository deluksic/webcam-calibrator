import { d } from 'typegpu'

/** Pinhole intrinsics in a uniform-friendly block (skew = 0). */
export const PinholeIntrinsicsGpu = d.struct({
  fx: d.f32,
  fy: d.f32,
  cx: d.f32,
  cy: d.f32,
})

/** OpenCV rational model: k1…k6, p1, p2 in coefficient order. */
export const RationalDistortion8Gpu = d.struct({
  k1: d.f32,
  k2: d.f32,
  p1: d.f32,
  p2: d.f32,
  k3: d.f32,
  k4: d.f32,
  k5: d.f32,
  k6: d.f32,
})
