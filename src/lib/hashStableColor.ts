// Stable pseudo-color from a u32 design id (ROI identity overlay).
import { d } from 'typegpu'

/** Label/edge viz background ≈ (0.12, 0.12, 0.14); hash RGB is remapped to [floor, 1]. */
export const HASH_RGB_FLOOR = 0.42

/** Murmur3-style 32→32 mix; same input → same hash. */
export function stableHashU32(x: number) {
  'use gpu'
  const h0 = x ^ (x >>> d.u32(16))
  const h1 = h0 * d.u32(0x7feb352d)
  const h2 = h1 ^ (h1 >>> d.u32(15))
  const h3 = h2 * d.u32(0x846ca68b)
  return h3 ^ (h3 >>> d.u32(16))
}

/** Linear RGB in [HASH_RGB_FLOOR, 1]³ from a u32 id (byte lanes of `stableHashU32`). */
export function stableHashToRgb01(x: number) {
  'use gpu'
  const h = stableHashU32(x)
  const floor = d.f32(HASH_RGB_FLOOR)
  const scale = d.f32(1) - floor
  return d.vec3f(
    floor + scale * (d.f32(h & d.u32(255)) / d.f32(255)),
    floor + scale * (d.f32((h >>> d.u32(8)) & d.u32(255)) / d.f32(255)),
    floor + scale * (d.f32((h >>> d.u32(16)) & d.u32(255)) / d.f32(255)),
  )
}
