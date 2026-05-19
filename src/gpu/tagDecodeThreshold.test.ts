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

/** Mirrors GPU `findTagDecodeHistPeaks` in `tagDecodeHistPeaks.ts` — test-only, not used at runtime. */
function findTagDecodeHistPeaks(
  hist: readonly number[],
  bins: number = TAG_DECODE_HIST_BINS,
  minSep: number = TAG_DECODE_MIN_PEAK_BIN_SEP,
) {
  const lastBin = bins - 1
  const blackSearchEnd = lastBin - minSep

  let blackPeak = 0
  let blackVal = 0
  for (let bu = 0; bu <= blackSearchEnd; bu++) {
    const v = hist[bu] ?? 0
    const prev = bu > 0 ? (hist[bu - 1] ?? 0) : 0
    const next = bu < blackSearchEnd ? (hist[bu + 1] ?? 0) : 0
    const isPeak = bu === 0 ? v >= next : bu === blackSearchEnd ? v >= prev : v >= prev && v >= next
    if (isPeak && v > blackVal) {
      blackVal = v
      blackPeak = bu
    }
  }
  if (blackVal === 0) {
    for (let bu = 0; bu <= blackSearchEnd; bu++) {
      const v = hist[bu] ?? 0
      if (v > blackVal) {
        blackVal = v
        blackPeak = bu
      }
    }
  }

  let whitePeak = 0
  let whiteVal = 0
  const whiteSearchStart = blackPeak + minSep + 1
  for (let bu = whiteSearchStart; bu <= lastBin; bu++) {
    const v = hist[bu] ?? 0
    const prev = hist[bu - 1] ?? 0
    const next = bu < lastBin ? (hist[bu + 1] ?? 0) : 0
    const isPeak = bu === lastBin ? v >= prev : v >= prev && v >= next
    if (isPeak && v > whiteVal) {
      whiteVal = v
      whitePeak = bu
    }
  }

  return { blackPeak, whitePeak, blackVal, whiteVal }
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
