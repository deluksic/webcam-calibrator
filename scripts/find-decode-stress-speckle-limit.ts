/**
 * Finds the largest speckle amplitude where the decode stress battery still passes (matched homography).
 *
 *   pnpm run find:decode-stress-speckle
 */
import { binarySearchMaxPassing } from '@/lib/decodeStressSearch'
import { decodeStressSuiteFailuresFromOptions } from '@/lib/decodeStressSuite'

function passes(amp: number): boolean {
  return (
    decodeStressSuiteFailuresFromOptions({
      speckleAmp: amp,
      homographyMismatchScale: 0,
    }).length === 0
  )
}

if (!passes(1e-12)) {
  console.error('Even tiny speckle fails; check harness.')
  process.exit(1)
}

/** Smallest `a` in `(0, maxScan]` with `!passes(a)`, or `undefined` if none found. */
function findFirstFailure(maxScan: number): number | undefined {
  let a = 0.01
  while (a <= maxScan) {
    if (!passes(a)) return a
    const na = Math.min(maxScan, a * 1.2)
    if (na <= a) return passes(maxScan) ? undefined : maxScan
    a = na
  }
  return undefined
}

const firstFail = findFirstFailure(1)
if (firstFail === undefined) {
  console.log(JSON.stringify({ note: 'No failure up to amplitude 1.0' }, null, 2))
  console.log('\nSuggested DECODE_STRESS_SPECKLE_AMP = 1 (no failure up to 1.0)')
  process.exit(0)
}

const { maxPass, failHi } = binarySearchMaxPassing(passes, 0, firstFail, 56)
console.log(
  JSON.stringify(
    {
      maxPassingAmplitude: maxPass,
      firstFailingAmplitudeBracket: [maxPass, failHi],
      failuresAtBracketHigh: decodeStressSuiteFailuresFromOptions({
        speckleAmp: failHi,
        homographyMismatchScale: 0,
      }),
    },
    null,
    2,
  ),
)
console.log(`\nSuggested DECODE_STRESS_SPECKLE_AMP in src/lib/decodeStressHarness.ts ≈ ${maxPass}`)
