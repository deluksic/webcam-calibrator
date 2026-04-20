/**
 * Shared numeric search helpers for decode-stress limit scripts (`scripts/find-decode-stress-*`).
 */

/** `true` iff the suite still passes at parameter `t`. */
export type DecodeStressPassPredicate = (t: number) => boolean

/**
 * Largest `t` with `passes(t)` in `[loPass, hiFail]`, assuming `passes(loPass)` and `!passes(hiFail)`.
 * Monotone cliff in between.
 */
export function binarySearchMaxPassing(
  passes: DecodeStressPassPredicate,
  loPass: number,
  hiFail: number,
  iterations = 56,
): { maxPass: number; failHi: number } {
  let lo = loPass
  let hi = hiFail
  for (let i = 0; i < iterations; i++) {
    const mid = (lo + hi) * 0.5
    if (passes(mid)) lo = mid
    else hi = mid
  }
  return { maxPass: lo, failHi: hi }
}

/**
 * Largest `s` in `{0, step, 2*step, …, hi}` with `passes(s)` (handy when `passes` is not monotone,
 * so binary search on a single “first cliff” is unsafe). Also counts **recoveries** (`fail` then `pass`
 * at the next grid point).
 */
export function gridMaxPassing(
  passes: DecodeStressPassPredicate,
  opts: { hi: number; step: number },
): { best: number; recoveries: number } {
  const { hi, step } = opts
  if (step <= 0 || !Number.isFinite(step)) throw new Error('gridMaxPassing: step must be finite and > 0')
  if (!passes(0)) throw new Error('gridMaxPassing: passes(0) must be true')

  let best = 0
  let prev = true
  let recoveries = 0
  const n = Math.floor(hi / step + 1e-9)
  for (let k = 1; k <= n; k++) {
    const s = Math.round(k * step * 1_000_000) / 1_000_000
    const ok = passes(s)
    if (ok) best = s
    if (ok && !prev) recoveries++
    prev = ok
  }
  return { best, recoveries }
}
