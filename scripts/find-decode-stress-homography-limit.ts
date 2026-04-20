/**
 * Estimates the largest homography mismatch scale that still passes the decode stress battery
 * at fixed `DECODE_STRESS_SPECKLE_AMP`, using a **grid scan** (`passes(scale)` is not monotone in
 * scale in general, so a single “first failure + binary search” bracket can be wrong).
 *
 *   pnpm run find:decode-stress-homography
 */
import { DECODE_STRESS_SPECKLE_AMP } from '@/lib/decodeStressHarness'
import { gridMaxPassing } from '@/lib/decodeStressSearch'
import { decodeStressSuiteFailuresFromOptions } from '@/lib/decodeStressSuite'

function passes(scale: number): boolean {
  return (
    decodeStressSuiteFailuresFromOptions({
      speckleAmp: DECODE_STRESS_SPECKLE_AMP,
      homographyMismatchScale: scale,
    }).length === 0
  )
}

if (!passes(0)) {
  console.error('Scale 0 (matched homography) fails; check speckle / suite.')
  process.exit(1)
}

const hi = 8
const step = 0.05
const { best, recoveries } = gridMaxPassing(passes, { hi, step })

console.log(
  JSON.stringify(
    {
      speckleAmp: DECODE_STRESS_SPECKLE_AMP,
      gridStep: step,
      scanRange: [0, hi],
      maxPassingHomographyScale: best,
      nonMonotoneRecoveriesAfterFail: recoveries,
      note:
        recoveries > 0
          ? '`passes(scale)` was not monotone on this grid; `best` is the largest passing sampled scale.'
          : 'No fail→pass jump on this grid (locally monotone sampling).',
    },
    null,
    2,
  ),
)
console.log(
  `\nLargest passing scale on this grid: ${best}. If you tighten tests, set DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE ≤ this (and re-run \`pnpm test\`).`,
)
