// Shared stable pseudo-color from a u32 id (labels, tag viz) — GPU only.
import { d } from 'typegpu';

/** Murmur3-style 32→32 mix; same input → same hash (matches label/grid viz). */
export function stableHashU32(x: number) {
  'use gpu';
  const h0 = x ^ (x >> d.u32(16));
  const h1 = h0 * d.u32(0x7feb352d);
  const h2 = h1 ^ (h1 >> d.u32(15));
  const h3 = h2 * d.u32(0x846ca68b);
  return h3 ^ (h3 >> d.u32(16));
}

/** Linear RGB in [0,1]³ from a u32 id (byte lanes of `stableHashU32`). */
export function stableHashToRgb01(x: number) {
  'use gpu';
  const h = stableHashU32(x);
  return d.vec3f(
    d.f32(h & d.u32(255)) / d.f32(255),
    d.f32((h >> d.u32(8)) & d.u32(255)) / d.f32(255),
    d.f32((h >> d.u32(16)) & d.u32(255)) / d.f32(255),
  );
}
