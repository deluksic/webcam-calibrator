// Shared stable pseudo-color from a u32 id (labels, tag viz).
import { d } from "typegpu";

const GRID_VIZ_RGB_SCALE = 0.55;

/** u32 hash — same ops as `stableHashU32` (CPU / CSS). */
export function stableHashU32Cpu(x: number): number {
  x >>>= 0;
  const h0 = (x ^ (x >>> 16)) >>> 0;
  const h1 = Math.imul(h0, 0x7feb352d) >>> 0;
  const h2 = (h1 ^ (h1 >>> 15)) >>> 0;
  const h3 = Math.imul(h2, -2066404021) >>> 0; // 0x846ca68b
  return (h3 ^ (h3 >>> 16)) >>> 0;
}

/** `rgb(...)` matching `gridVizPipeline` fill (`stableHashToRgb01` × 0.55). */
export function gridVizFillRgbCss(tagId: number): string {
  const h = stableHashU32Cpu(tagId >>> 0);
  const k = GRID_VIZ_RGB_SCALE;
  const r = Math.round(((h & 255) / 255) * k * 255);
  const g = Math.round((((h >>> 8) & 255) / 255) * k * 255);
  const b = Math.round((((h >>> 16) & 255) / 255) * k * 255);
  return `rgb(${r},${g},${b})`;
}

/** Murmur3-style 32→32 mix; same input → same hash (matches label/grid viz). */
export function stableHashU32(x: number) {
  "use gpu";
  const h0 = x ^ (x >> d.u32(16));
  const h1 = h0 * d.u32(0x7feb352d);
  const h2 = h1 ^ (h1 >> d.u32(15));
  const h3 = h2 * d.u32(0x846ca68b);
  return h3 ^ (h3 >> d.u32(16));
}

/** Linear RGB in [0,1]³ from a u32 id (byte lanes of `stableHashU32`). */
export function stableHashToRgb01(x: number) {
  "use gpu";
  const h = stableHashU32(x);
  return d.vec3f(
    d.f32(h & d.u32(255)) / d.f32(255),
    d.f32((h >> d.u32(8)) & d.u32(255)) / d.f32(255),
    d.f32((h >> d.u32(16)) & d.u32(255)) / d.f32(255),
  );
}
