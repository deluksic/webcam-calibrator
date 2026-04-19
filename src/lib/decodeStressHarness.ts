/**
 * Shared synthetic decode stress: perspective quad, optional speckle, FD Sobel.
 * Used by `decodeStress.test.ts` and `scripts/find-decode-stress-speckle-limit.ts`.
 */
import type { Point } from './geometry';
import { buildTagGrid, decodeTagPattern } from './grid';
import {
  finiteDifferenceSobelFromIntensity,
  renderAprilTagIntensity,
} from './aprilTagRaycast';
import {
  TAG36H11_CODES,
  codeToPattern,
  decodeTag36h11AnyRotation,
  decodeTag36h11Best,
  type TagPattern,
} from './tag36h11';

export const DECODE_STRESS_SUPERSAMPLE_DEFAULT = 4;
export const DECODE_STRESS_SIZES = [200, 160, 120, 96, 80, 72, 64, 56, 48, 40] as const;

/**
 * Max ± uniform speckle on `[0,1]` intensity (before Sobel) that still clears `decodeStressSuiteFailuresFromOptions`.
 * First failure in the current pipeline is just above this (≈0.4925 at `wh=160`). Re-tune with
 * `pnpm run find:decode-stress-speckle`.
 */
export const DECODE_STRESS_SPECKLE_AMP = 0.492;

/**
 * Normalized **image-space** Δpx pattern per corner (TL, TR, BL, BR). Decode quad is
 * `raster + scale * template`. Re-tune `DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE` with
 * `pnpm run find:decode-stress-homography` (grid scan — `passes(scale)` is not monotone).
 */
export const DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX: readonly [Point, Point, Point, Point] = [
  { x: 0.35, y: -0.2 },
  { x: -0.25, y: 0.25 },
  { x: 0.2, y: 0.3 },
  { x: -0.3, y: -0.25 },
];

/**
 * Scalar applied to `DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX` for tests / defaults.
 * `0` disables mismatch; `1` matches the historical fixed-offset test.
 */
export const DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE = 1;

/** @deprecated Use `DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX` + scale. */
export const DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSETS_PX = DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX;

/** `max |Δx|, |Δy|` over the offset template (pixels before scale). */
export function decodeStressHomographyMismatchTemplateMaxAxisPx(): number {
  let m = 0;
  for (const p of DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX) {
    m = Math.max(m, Math.abs(p.x), Math.abs(p.y));
  }
  return m;
}

/**
 * Scale so that `scale * template` has **max per-axis** corner shift ≈ `maxAxisPx` image pixels
 * (same scale for all corners).
 */
export function decodeStressHomographyMismatchScaleForMaxAxisPx(maxAxisPx: number): number {
  const denom = decodeStressHomographyMismatchTemplateMaxAxisPx();
  if (denom <= 0) throw new Error('decodeStressHomographyMismatchScaleForMaxAxisPx: empty template');
  return maxAxisPx / denom;
}

/** Decode strip = raster strip + `scale *` template offsets (recompute H via `buildTagGrid`). */
export function decodeStressStripWithHomographyMismatchOffsetsPx(
  rasterStrip: [Point, Point, Point, Point],
  scale: number = DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE,
): [Point, Point, Point, Point] {
  return rasterStrip.map((p, i) => ({
    x: p.x + DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX[i]!.x * scale,
    y: p.y + DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX[i]!.y * scale,
  })) as [Point, Point, Point, Point];
}

/** Strong-perspective quad in ~320px space (same family as aprilTagRaycast tests). */
const REF_STRIP: [Point, Point, Point, Point] = [
  { x: 20, y: 20 },
  { x: 280, y: 45 },
  { x: 35, y: 260 },
  { x: 275, y: 265 },
];

const REF_CX = (20 + 280 + 35 + 275) / 4;
const REF_CY = (20 + 45 + 260 + 265) / 4;

/** In-place xorshift32 on `stateRef.s`; returns `[0,1)`. */
function xorshift32U01(stateRef: { s: number }): number {
  let x = stateRef.s >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  stateRef.s = x >>> 0;
  return stateRef.s / 0x1_0000_0000;
}

/** Deterministic speckle: uniform in `[-amplitude, amplitude]`, clamped to `[0,1]`. */
export function decodeStressAddSpeckle01(
  intensity: Float32Array,
  amplitude: number,
  seed: number,
): void {
  if (amplitude <= 0) return;
  const st = { s: (seed >>> 0) || 1 };
  for (let i = 0; i < intensity.length; i++) {
    const n = (xorshift32U01(st) * 2 - 1) * amplitude;
    intensity[i] = Math.min(1, Math.max(0, intensity[i]! + n));
  }
}

export function decodeStressSpeckleSeed(w: number, h: number, tagId: number): number {
  return (Math.imul(w, 1_664_525) ^ Math.imul(h, 1_013_904_223) ^ Math.imul(tagId, 747_796_405)) >>> 0;
}

/** TL, TR, BR, BL for `buildTagGrid` from homography strip TL, TR, BL, BR. */
export function decodeStressCornersGridOrder(
  strip: [Point, Point, Point, Point],
): [Point, Point, Point, Point] {
  return [strip[0], strip[1], strip[3], strip[2]];
}

/** Scale + center REF_STRIP into `[margin, w-margin] × [margin, h-margin]`. */
export function decodeStressFitPerspectiveStrip(
  w: number,
  h: number,
  opts?: { margin?: number; perspectiveBoost?: number },
): [Point, Point, Point, Point] {
  const margin = opts?.margin ?? 6;
  const boost = opts?.perspectiveBoost ?? 1;
  const cw = w - 2 * margin;
  const ch = h - 2 * margin;
  const spanX =
    Math.max(REF_STRIP[0].x, REF_STRIP[1].x, REF_STRIP[2].x, REF_STRIP[3].x) -
    Math.min(REF_STRIP[0].x, REF_STRIP[1].x, REF_STRIP[2].x, REF_STRIP[3].x);
  const spanY =
    Math.max(REF_STRIP[0].y, REF_STRIP[1].y, REF_STRIP[2].y, REF_STRIP[3].y) -
    Math.min(REF_STRIP[0].y, REF_STRIP[1].y, REF_STRIP[2].y, REF_STRIP[3].y);
  const s = Math.min(cw / spanX, ch / spanY) * boost;
  const cx = margin + cw / 2;
  const cy = margin + ch / 2;
  return REF_STRIP.map((p) => ({
    x: cx + (p.x - REF_CX) * s,
    y: cy + (p.y - REF_CY) * s,
  })) as [Point, Point, Point, Point];
}

export function decodeStressAxisStrip(
  w: number,
  h: number,
  margin: number,
  side: number,
): [Point, Point, Point, Point] {
  const x0 = margin;
  const y0 = margin;
  return [
    { x: x0, y: y0 },
    { x: x0 + side, y: y0 },
    { x: x0, y: y0 + side },
    { x: x0 + side, y: y0 + side },
  ];
}

export function decodeStressRasterSobel(
  w: number,
  h: number,
  strip: [Point, Point, Point, Point],
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
  });
  decodeStressAddSpeckle01(intensity, speckleAmp, decodeStressSpeckleSeed(w, h, tagId));
  const sobel = finiteDifferenceSobelFromIntensity(intensity, w, h, { gradientScale: 4 });
  return { intensity, sobel };
}

export function decodeStressSynthetic(
  w: number,
  h: number,
  strip: [Point, Point, Point, Point],
  tagId: number,
  supersample: number,
  speckleAmp: number,
) {
  return decodeStressSyntheticWithHomographyMismatch(
    w,
    h,
    strip,
    strip,
    tagId,
    supersample,
    speckleAmp,
  );
}

/**
 * Raster tag with `rasterStrip` homography; decode grid built from `decodeStrip` (recomputes H).
 * Use `decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip)` for “slightly wrong corners”.
 */
export function decodeStressSyntheticWithHomographyMismatch(
  w: number,
  h: number,
  rasterStrip: [Point, Point, Point, Point],
  decodeStrip: [Point, Point, Point, Point],
  tagId: number,
  supersample: number,
  speckleAmp: number,
) {
  const pattern = codeToPattern(TAG36H11_CODES[tagId]);
  const { sobel } = decodeStressRasterSobel(w, h, rasterStrip, pattern, supersample, tagId, speckleAmp);
  const grid = buildTagGrid(decodeStressCornersGridOrder(decodeStrip), 6);
  const decodedPattern = decodeTagPattern(grid, sobel, w, undefined, h);
  if (!decodedPattern) {
    throw new Error('decodeTagPattern returned null (unexpected in stress harness)');
  }
  const best = decodeTag36h11Best(decodedPattern, 8);
  const rot = decodeTag36h11AnyRotation(decodedPattern, 8);
  return { best, rot, decodedPattern };
}

