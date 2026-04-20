import {
  DECODE_STRESS_SPECKLE_AMP,
  DECODE_STRESS_SUPERSAMPLE_DEFAULT,
  DECODE_STRESS_SIZES,
  decodeStressAxisStrip,
  decodeStressFitPerspectiveStrip,
  decodeStressStripWithHomographyMismatchOffsetsPx,
  decodeStressSyntheticWithHomographyMismatch,
} from '@/lib/decodeStressHarness'
/**
 * Single entry point for “does the decode stress battery pass?” — used by tests and limit scripts.
 */
import type { Point } from '@/lib/geometry'
import { TAG36H11_CODES, codeToPattern } from '@/lib/tag36h11'

export type DecodeStressSuiteOptions = {
  /** Speckle amplitude on `[0,1]` intensity before Sobel. */
  speckleAmp?: number
  /**
   * `0` = raster and decode quads match.
   * `>0` = decode corners are raster corners + `scale *` template offsets (homography recomputed).
   */
  homographyMismatchScale?: number
  supersample?: number
  tagId?: number
}

function cellErrorsVsTruth(decoded: (0 | 1 | -1 | -2)[], truth: (0 | 1 | -1 | -2)[]): number {
  let err = 0
  for (let i = 0; i < 36; i++) {
    const d = decoded[i]
    if (d === -1 || d === -2) continue
    if (d !== truth[i]) err++
  }
  return err
}

/**
 * Human-readable failures (same battery as `decodeStress.test.ts`).
 * Empty ⇒ pass.
 */
export function decodeStressSuiteFailuresFromOptions(opts: DecodeStressSuiteOptions = {}): string[] {
  const speckleAmp = opts.speckleAmp ?? DECODE_STRESS_SPECKLE_AMP
  const homographyMismatchScale = opts.homographyMismatchScale ?? 0
  const supersample = opts.supersample ?? DECODE_STRESS_SUPERSAMPLE_DEFAULT
  const tagId = opts.tagId ?? 0
  const truth = codeToPattern(TAG36H11_CODES[tagId])

  const failures: string[] = []

  const run = (label: string, w: number, h: number, rasterStrip: [Point, Point, Point, Point]) => {
    const decodeStrip =
      homographyMismatchScale === 0
        ? rasterStrip
        : decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip, homographyMismatchScale)
    const { rot, best, decodedPattern } = decodeStressSyntheticWithHomographyMismatch(
      w,
      h,
      rasterStrip,
      decodeStrip,
      tagId,
      supersample,
      speckleAmp,
    )
    if (!rot || rot.id !== tagId) failures.push(`${label}: rot id want ${tagId} got ${rot?.id}`)
    if (best.id !== tagId) failures.push(`${label}: best.id want ${tagId} got ${best.id}`)
    // Aligned with `decodeStress.test.ts` homography-mismatch slack (radial + speckle).
    const maxDist = 2
    const maxCellErr = 2
    const maxUnknowns = 2
    if (best.dist > maxDist) failures.push(`${label}: best.dist want ≤${maxDist} got ${best.dist}`)
    const ce = cellErrorsVsTruth(decodedPattern, truth)
    if (ce > maxCellErr) failures.push(`${label}: cellErr want ≤${maxCellErr} got ${ce}`)
    const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length
    if (unknowns > maxUnknowns) failures.push(`${label}: unknowns want ≤${maxUnknowns} got ${unknowns}`)
  }

  {
    const w = 48
    const h = 48
    const side = 32
    const strip = decodeStressAxisStrip(w, h, 4, side)
    run('axis48', w, h, strip)
  }

  for (const w of [120, 72] as const) {
    const h = w
    const strip = decodeStressFitPerspectiveStrip(w, h)
    run(`w${w}`, w, h, strip)
  }

  for (const wh of DECODE_STRESS_SIZES) {
    const strip = decodeStressFitPerspectiveStrip(wh, wh)
    run(`table wh=${wh}`, wh, wh, strip)
  }

  return failures
}

/** Matched homography only; `speckleAmp` is the free parameter (legacy script API). */
export function decodeStressSuiteFailures(
  speckleAmp: number,
  supersample: number = DECODE_STRESS_SUPERSAMPLE_DEFAULT,
): string[] {
  return decodeStressSuiteFailuresFromOptions({
    speckleAmp,
    supersample,
    homographyMismatchScale: 0,
  })
}
