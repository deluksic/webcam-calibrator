import type { BinaryColor } from '@/lib/pointIdentity/types'

/**
 * Small deterministic PRNG (mulberry32). Pattern/color experiments must be
 * reproducible so a viewing-angle result can be shared and re-checked.
 */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296
  }
}

/** Draw one of two colors with equal probability. */
export function randomBinaryColors(count: number, seed: number): BinaryColor[] {
  const random = createRandom(seed)
  const colors: BinaryColor[] = []
  for (let index = 0; index < count; index += 1) {
    colors.push(random() < 0.5 ? 0 : 1)
  }
  return colors
}
