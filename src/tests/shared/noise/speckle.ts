import { xorshift32U01 } from '@/lib/seededRng'

const { max, min } = Math

/** Uniform ±amplitude on `[0,1]` intensity; clamped to `[0,1]`. */
export function applySpeckle01(intensity: Float32Array, amplitude: number, seed: number): void {
  if (amplitude <= 0) {
    return
  }
  const st = { s: seed >>> 0 || 1 }
  for (let i = 0; i < intensity.length; i++) {
    const n = (xorshift32U01(st) * 2 - 1) * amplitude
    intensity[i] = min(1, max(0, intensity[i]! + n))
  }
}
