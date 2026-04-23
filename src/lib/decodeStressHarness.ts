/**
 * Shared synthetic decode stress: perspective quad, optional speckle, FD Sobel.
 * Used by `decodeStress.test.ts` and `scripts/find-decode-stress-speckle-limit.ts`.
 */
import type { Corners } from '@/lib/geometry'
import { decodeTagPattern } from '@/lib/grid'
import { xorshift32U01 } from '@/lib/seededRng'
import {
  tag36h11Code,
  codeToPattern,
  decodeTag36h11AnyRotation,
  decodeTag36h11Best,
  type TagPattern,
} from '@/lib/tag36h11'
import { finiteDifferenceSobelFromIntensity, renderAprilTagIntensity } from '@/tests/utils/syntheticAprilTag'

const { abs, imul, max, min } = Math

export const DECODE_STRESS_SUPERSAMPLE_DEFAULT = 4
export const DECODE_STRESS_SIZES = [200, 160, 120, 96, 80, 72, 64, 56, 48, 40] as const

/**
 * Max ± uniform speckle on `[0,1]` intensity (before Sobel) used by decode stress tests / suite.
 * Re-tune with `pnpm run find:decode-stress-speckle` (`decodeStressSuite` allows Hamming / cell / unknown slack).
 */
export const DECODE_STRESS_SPECKLE_AMP = 0.492

/**
 * Normalized **image-space** Δpx pattern per corner (TL, TR, BL, BR). Decode quad is
 * `raster + scale * template`. Re-tune `DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE` with
 * `pnpm run find:decode-stress-homography` (grid scan — `passes(scale)` is not monotone).
 */
export const DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX: Corners = [
  { x: 0.35, y: -0.2 },
  { x: -0.25, y: 0.25 },
  { x: 0.2, y: 0.3 },
  { x: -0.3, y: -0.25 },
]

/**
 * Scalar applied to `DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX` for tests / defaults.
 * `0` disables mismatch; `1` matches the historical fixed-offset test.
 */
export const DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE = 1

/** @deprecated Use `DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX` + scale. */
export const DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSETS_PX = DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX

/** `max |Δx|, |Δy|` over the offset template (pixels before scale). */
export function decodeStressHomographyMismatchTemplateMaxAxisPx(): number {
  let m = 0
  for (const p of DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX) {
    m = max(m, abs(p.x), abs(p.y))
  }
  return m
}

/**
 * Scale so that `scale * template` has **max per-axis** corner shift ≈ `maxAxisPx` image pixels
 * (same scale for all corners).
 */
export function decodeStressHomographyMismatchScaleForMaxAxisPx(maxAxisPx: number): number {
  const denom = decodeStressHomographyMismatchTemplateMaxAxisPx()
  if (denom <= 0) {
    throw new Error('decodeStressHomographyMismatchScaleForMaxAxisPx: empty template')
  }
  return maxAxisPx / denom
}

/** Decode strip = raster strip + `scale *` template offsets (recompute H via `buildTagGrid`). */
export function decodeStressStripWithHomographyMismatchOffsetsPx(
  rasterStrip: Corners,
  scale: number = DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE,
): Corners {
  return rasterStrip.map((p, i) => ({
    x: p.x + DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX[i]!.x * scale,
    y: p.y + DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX[i]!.y * scale,
  })) as Corners
}

/** Strong-perspective quad in ~320px space (same family as aprilTagRaycast tests). */
const REF_STRIP: Corners = [
  { x: 20, y: 20 },
  { x: 280, y: 45 },
  { x: 35, y: 260 },
  { x: 275, y: 265 },
]

const REF_CX = (20 + 280 + 35 + 275) / 4
const REF_CY = (20 + 45 + 260 + 265) / 4

/** Deterministic speckle: uniform in `[-amplitude, amplitude]`, clamped to `[0,1]`. */
export function decodeStressAddSpeckle01(intensity: Float32Array, amplitude: number, seed: number): void {
  if (amplitude <= 0) {
    return
  }
  const st = { s: seed >>> 0 || 1 }
  for (let i = 0; i < intensity.length; i++) {
    const n = (xorshift32U01(st) * 2 - 1) * amplitude
    intensity[i] = min(1, max(0, intensity[i]! + n))
  }
}

export function decodeStressSpeckleSeed(w: number, h: number, tagId: number): number {
  return (imul(w, 1_664_525) ^ imul(h, 1_013_904_223) ^ imul(tagId, 747_796_405)) >>> 0
}

/** Scale + center REF_STRIP into `[margin, w-margin] × [margin, h-margin]`. */
export function decodeStressFitPerspectiveStrip(
  w: number,
  h: number,
  opts?: { margin?: number; perspectiveBoost?: number },
): Corners {
  const margin = opts?.margin ?? 6
  const boost = opts?.perspectiveBoost ?? 1
  const cw = w - 2 * margin
  const ch = h - 2 * margin
  const spanX =
    max(REF_STRIP[0].x, REF_STRIP[1].x, REF_STRIP[2].x, REF_STRIP[3].x) -
    min(REF_STRIP[0].x, REF_STRIP[1].x, REF_STRIP[2].x, REF_STRIP[3].x)
  const spanY =
    max(REF_STRIP[0].y, REF_STRIP[1].y, REF_STRIP[2].y, REF_STRIP[3].y) -
    min(REF_STRIP[0].y, REF_STRIP[1].y, REF_STRIP[2].y, REF_STRIP[3].y)
  const s = min(cw / spanX, ch / spanY) * boost
  const cx = margin + cw / 2
  const cy = margin + ch / 2
  return REF_STRIP.map((p) => ({
    x: cx + (p.x - REF_CX) * s,
    y: cy + (p.y - REF_CY) * s,
  })) as Corners
}

export function decodeStressAxisStrip(w: number, h: number, margin: number, side: number): Corners {
  const x0 = margin
  const y0 = margin
  return [
    { x: x0, y: y0 },
    { x: x0 + side, y: y0 },
    { x: x0, y: y0 + side },
    { x: x0 + side, y: y0 + side },
  ]
}

export function decodeStressRasterSobel(
  w: number,
  h: number,
  strip: Corners,
  pattern: TagPattern,
  supersample: number,
  tagId: number,
  speckleAmp: number,
): { intensity: Float32Array; sobel: Float32Array } {
  const intensity = renderAprilTagIntensity({
    width: w,
    height: h,
    corners: strip,
    pattern,
    supersample,
  })
  decodeStressAddSpeckle01(intensity, speckleAmp, decodeStressSpeckleSeed(w, h, tagId))
  const sobel = finiteDifferenceSobelFromIntensity(intensity, w, h, {
    gradientScale: 4,
  })
  return { intensity, sobel }
}

export function decodeStressSynthetic(
  w: number,
  h: number,
  strip: Corners,
  tagId: number,
  supersample: number,
  speckleAmp: number,
) {
  return decodeStressSyntheticWithHomographyMismatch(w, h, strip, strip, tagId, supersample, speckleAmp)
}

/**
 * Raster tag with `rasterStrip` homography; decode grid built from `decodeStrip` (recomputes H).
 * Use `decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip)` for “slightly wrong corners”.
 */
export function decodeStressSyntheticWithHomographyMismatch(
  w: number,
  h: number,
  rasterStrip: Corners,
  decodeStrip: Corners,
  tagId: number,
  supersample: number,
  speckleAmp: number,
) {
  const pattern = codeToPattern(tag36h11Code(tagId))
  const { sobel } = decodeStressRasterSobel(w, h, rasterStrip, pattern, supersample, tagId, speckleAmp)
  const decodedPattern = decodeTagPattern(decodeStrip, sobel, w, undefined, h)
  const best = decodeTag36h11Best(decodedPattern, 8)
  const rot = decodeTag36h11AnyRotation(decodedPattern, 8)
  return { best, rot, decodedPattern }
}
