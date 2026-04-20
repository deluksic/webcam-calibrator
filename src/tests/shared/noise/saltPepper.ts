import { xorshift32State, xorshift32U01 } from "../rng";

/** With probability `rate` per pixel, set to 0 or 1 (deterministic). */
export function applySaltPepper01(intensity: Float32Array, rate: number, seed: number): void {
  if (rate <= 0) return;
  const st = xorshift32State(seed);
  for (let i = 0; i < intensity.length; i++) {
    if (xorshift32U01(st) < rate) {
      intensity[i] = xorshift32U01(st) < 0.5 ? 0 : 1;
    }
  }
}
