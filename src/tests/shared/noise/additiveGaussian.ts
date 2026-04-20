import { xorshift32State, xorshift32U01 } from '@/tests/shared/rng'

/** Additive Gaussian noise with σ in [0,1] intensity units; clamp to [0,1]. Box-Muller from uniform. */
export function applyAdditiveGaussian01(intensity: Float32Array, sigma: number, seed: number): void {
  if (sigma <= 0) return
  const st = xorshift32State(seed)
  let spare: number | undefined = undefined
  const nextGaussian = (): number => {
    if (spare !== undefined) {
      const z = spare
      spare = undefined
      return z
    }
    const u1 = xorshift32U01(st)
    const u2 = xorshift32U01(st)
    const r = Math.sqrt(-2 * Math.log(Math.max(1e-15, u1)))
    const z0 = r * Math.cos(2 * Math.PI * u2)
    const z1 = r * Math.sin(2 * Math.PI * u2)
    spare = z1
    return z0
  }
  for (let i = 0; i < intensity.length; i++) {
    intensity[i] = Math.min(1, Math.max(0, intensity[i]! + nextGaussian() * sigma))
  }
}
