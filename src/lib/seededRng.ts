/**
 * Deterministic xorshift32 PRNG for test harnesses and speckle (same stream as historical decode stress).
 */
export interface Xorshift32State {
  s: number
}

/** In-place xorshift32 on `stateRef.s`; returns `[0,1)`. */
export function xorshift32U01(stateRef: Xorshift32State): number {
  let x = stateRef.s >>> 0
  x ^= x << 13
  x ^= x >>> 17
  x ^= x << 5
  stateRef.s = x >>> 0
  return stateRef.s / 0x1_0000_0000
}

export function xorshift32State(seed: number): Xorshift32State {
  return { s: seed >>> 0 || 1 }
}
