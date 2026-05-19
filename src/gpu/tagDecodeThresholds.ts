/** Tag decode (luma vote + dictionary) thresholds — shared constants and CPU helpers for tests. */

import { MIN_PEAK_BIN_SEPARATION, ORIENT_HIST_BINS } from '@/gpu/lineFitThresholds'

export const TAG_DECODE_HIST_BINS = 32
export const TAG_DECODE_PEAK_GAP_FRAC = 0.25
export const TAG_DECODE_MAX_DICT_ERROR = 3
/** Max weak (-1) cells for GPU `2^u` wildcard unroll per codeword thread. */
export const TAG_DECODE_MAX_WEAK_WILDCARD = 6

/** Linear bin separation for white peak (scale from orient histogram at 64 bins). */
export const TAG_DECODE_MIN_PEAK_BIN_SEP = Math.max(
  4,
  Math.round((MIN_PEAK_BIN_SEPARATION * TAG_DECODE_HIST_BINS) / ORIENT_HIST_BINS),
)

/** Minimum peak luma span (in bins) to accept thresholds. */
export const TAG_DECODE_MIN_PEAK_LUMA_BINS = 3

export const DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE = 0.02

/** Linear luma deadband bounds from black/white peak lumas in [0, 1]. */
export function tagDecodeDeadbandBounds(
  blackPeakLuma: number,
  whitePeakLuma: number,
  gapFrac: number = TAG_DECODE_PEAK_GAP_FRAC,
): { blackBound: number; whiteBound: number } {
  const diff = whitePeakLuma - blackPeakLuma
  const halfGap = diff * gapFrac * 0.5
  return {
    blackBound: blackPeakLuma + halfGap,
    whiteBound: whitePeakLuma - halfGap,
  }
}

/** Shortest side of strip-order quad TL, TR, BL, BR (perimeter edges only). */
export function shortestStripEdgePx(
  corners: readonly { x: number; y: number }[],
): number {
  const tl = corners[0]!
  const tr = corners[1]!
  const bl = corners[2]!
  const br = corners[3]!
  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.hypot(a.x - b.x, a.y - b.y)
  return Math.min(dist(tl, tr), dist(tr, br), dist(br, bl), dist(bl, tl))
}

export function tagDecodeMinVoteTotal(shortestEdgePx: number): number {
  return Math.max(2, Math.round(DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE * shortestEdgePx))
}

/** Classify one 6×6 module from pixel vote tallies (matches GPU B0). */
export function classifyModuleFromVoteCounts(
  whiteCount: number,
  blackCount: number,
  minVoteTotal: number,
  thresholdsValid: boolean,
): -2 | -1 | 0 | 1 {
  const sum = whiteCount + blackCount
  if (!thresholdsValid || sum < minVoteTotal) {
    return -1
  }
  if (blackCount > whiteCount) {
    return 0
  }
  if (whiteCount > blackCount) {
    return 1
  }
  return -2
}
