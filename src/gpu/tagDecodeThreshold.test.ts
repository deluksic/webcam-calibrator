import { describe, expect, it } from 'vitest'

import {
  TAG_DECODE_HIST_BINS,
  TAG_DECODE_PEAK_GAP_FRAC,
  classifyModuleFromVoteCounts,
  shortestStripEdgePx,
  tagDecodeDeadbandBounds,
  tagDecodeMinVoteTotal,
} from '@/gpu/tagDecodeThresholds'

describe('tagDecodeThresholds', () => {
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
