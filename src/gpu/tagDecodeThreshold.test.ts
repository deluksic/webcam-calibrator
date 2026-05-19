import { describe, expect, it } from 'vitest'

import {
  TAG_DECODE_HIST_BINS,
  TAG_DECODE_MIN_PEAK_BIN_SEP,
  TAG_DECODE_PEAK_GAP_FRAC,
  classifyModuleFromVoteCounts,
  shortestStripEdgePx,
  tagDecodeDeadbandBounds,
  tagDecodeMinVoteTotal,
} from '@/gpu/tagDecodeThresholds'

/** Mirrors GPU peak-finding in `tagDecodePipeline.ts` (global max + separation). */
function findTagDecodeHistPeaks(
  hist: readonly number[],
  bins: number = TAG_DECODE_HIST_BINS,
  minSep: number = TAG_DECODE_MIN_PEAK_BIN_SEP,
) {
  // Find global maximum → peak1
  let peak1Bin = 0
  let peak1Val = 0
  for (let bu = 0; bu < bins; bu++) {
    const v = hist[bu] ?? 0
    if (v > peak1Val) {
      peak1Val = v
      peak1Bin = bu
    }
  }

  // Find global maximum ≥ minSep+1 away from peak1 → peak2
  const minDist = minSep + 1
  let peak2Bin = 0
  let peak2Val = 0
  for (let bu = 0; bu < bins; bu++) {
    const v = hist[bu] ?? 0
    const dist = Math.abs(bu - peak1Bin)
    if (dist >= minDist && v > peak2Val) {
      peak2Val = v
      peak2Bin = bu
    }
  }

  if (peak2Val > 0) {
    const blackPeak = Math.min(peak1Bin, peak2Bin)
    const whitePeak = Math.max(peak1Bin, peak2Bin)
    return {
      blackPeak,
      whitePeak,
      blackVal: hist[blackPeak] ?? 0,
      whiteVal: hist[whitePeak] ?? 0,
    }
  }

  return {
    blackPeak: peak1Bin,
    whitePeak: peak1Bin,
    blackVal: peak1Val,
    whiteVal: 0,
  }
}

describe('tagDecodeThresholds', () => {
  it('findTagDecodeHistPeaks treats saturated last bin as white, not black', () => {
    const hist = Array.from<number>({ length: TAG_DECODE_HIST_BINS }).fill(0)
    hist[5] = 80
    hist[31] = 1000
    const peaks = findTagDecodeHistPeaks(hist)
    expect(peaks.blackPeak).toBe(5)
    expect(peaks.whitePeak).toBe(31)
    expect(peaks.blackVal).toBe(80)
    expect(peaks.whiteVal).toBe(1000)
  })

  it('findTagDecodeHistPeaks finds separated interior peaks', () => {
    const hist = Array.from<number>({ length: TAG_DECODE_HIST_BINS }).fill(0)
    hist[3] = 40
    hist[4] = 120
    hist[5] = 50
    hist[27] = 90
    hist[28] = 220
    hist[29] = 100
    const peaks = findTagDecodeHistPeaks(hist)
    expect(peaks.blackPeak).toBe(4)
    expect(peaks.whitePeak).toBe(28)
    expect(peaks.whitePeak - peaks.blackPeak).toBeGreaterThanOrEqual(TAG_DECODE_MIN_PEAK_BIN_SEP)
  })

  it('deadband leaves middle 25% of peak luma span undecided', () => {
    const blackBin = 4
    const whiteBin = 28
    const blackLuma = (blackBin + 0.5) / TAG_DECODE_HIST_BINS
    const whiteLuma = (whiteBin + 0.5) / TAG_DECODE_HIST_BINS
    const { blackBound, whiteBound } = tagDecodeDeadbandBounds(blackLuma, whiteLuma)

    const diff = whiteLuma - blackLuma
    const halfGap = diff * TAG_DECODE_PEAK_GAP_FRAC * 0.5
    expect(blackBound).toBeCloseTo(blackLuma + halfGap)
    expect(whiteBound).toBeCloseTo(whiteLuma - halfGap)
    expect(whiteBound).toBeGreaterThan(blackBound)

    const mid = (blackLuma + whiteLuma) * 0.5
    expect(mid).toBeGreaterThan(blackBound)
    expect(mid).toBeLessThan(whiteBound)
  })

  it('classifyModuleFromVoteCounts matches -1 / -2 / 0 / 1 semantics', () => {
    expect(classifyModuleFromVoteCounts(0, 0, 2, true)).toBe(-1)
    expect(classifyModuleFromVoteCounts(3, 3, 6, true)).toBe(-2)
    expect(classifyModuleFromVoteCounts(1, 5, 6, true)).toBe(0)
    expect(classifyModuleFromVoteCounts(5, 1, 6, true)).toBe(1)
    expect(classifyModuleFromVoteCounts(5, 1, 6, false)).toBe(-1)
  })

  it('minVoteTotal scales with shortest edge', () => {
    expect(tagDecodeMinVoteTotal(100)).toBe(2)
    expect(tagDecodeMinVoteTotal(200)).toBe(4)
  })

  it('shortestStripEdgePx uses perimeter edges not diagonals', () => {
    const strip = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 0, y: 10 },
      { x: 100, y: 10 },
    ] as const
    expect(shortestStripEdgePx(strip)).toBeCloseTo(10)
    expect(shortestStripEdgePx(strip)).toBeLessThan(100)
  })
})
