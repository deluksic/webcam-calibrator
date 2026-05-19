import { d, std, tgpu } from 'typegpu'
import { max } from 'typegpu/std'

import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { TAG_DECODE_HIST_BINS, TAG_DECODE_MIN_PEAK_BIN_SEP } from '@/gpu/tagDecodeThresholds'

const PER_QUAD_HIST = MAX_QUADS * TAG_DECODE_HIST_BINS
const LAST_BIN = TAG_DECODE_HIST_BINS - 1

export const TagDecodeHistPeaks = d.struct({
  blackPeak: d.u32,
  whitePeak: d.u32,
  blackVal: d.u32,
  whiteVal: d.u32,
  histMaxCount: d.u32,
})

/** Black/white peaks in a per-quad 32-bin luma histogram (handles saturated right-tail white). */
export const findTagDecodeHistPeaks = tgpu.fn(
  [d.arrayOf(d.u32, PER_QUAD_HIST), d.u32],
  TagDecodeHistPeaks,
)((hist, base) => {
  'use gpu'
  const lastBin = d.u32(LAST_BIN)
  const minSep = d.u32(TAG_DECODE_MIN_PEAK_BIN_SEP)
  const blackSearchEnd = lastBin - minSep

  let blackPeak = d.u32(0)
  let blackVal = d.u32(0)
  let histMaxCount = d.u32(0)

  for (const b of std.range(0, TAG_DECODE_HIST_BINS)) {
    const bu = d.u32(b)
    const v = hist[base + bu]!
    histMaxCount = max(histMaxCount, v)

    if (bu <= blackSearchEnd) {
      let prev = d.u32(0)
      let next = d.u32(0)
      if (bu > d.u32(0)) {
        prev = hist[base + bu - d.u32(1)]!
      }
      if (bu < blackSearchEnd) {
        next = hist[base + bu + d.u32(1)]!
      }
      let isPeak = false
      if (bu === d.u32(0)) {
        isPeak = v >= next
      } else if (bu === blackSearchEnd) {
        isPeak = v >= prev
      } else {
        isPeak = v >= prev && v >= next
      }
      if (isPeak && v > blackVal) {
        blackVal = v
        blackPeak = bu
      }
    }
  }

  if (blackVal === d.u32(0)) {
    for (const b of std.range(0, TAG_DECODE_HIST_BINS)) {
      const bu = d.u32(b)
      if (bu <= blackSearchEnd) {
        const v = hist[base + bu]!
        if (v > blackVal) {
          blackVal = v
          blackPeak = bu
        }
      }
    }
  }

  let whitePeak = d.u32(0)
  let whiteVal = d.u32(0)
  const whiteSearchStart = blackPeak + minSep + d.u32(1)

  for (const b of std.range(0, TAG_DECODE_HIST_BINS)) {
    const bu = d.u32(b)
    if (bu >= whiteSearchStart && bu <= lastBin) {
      const v = hist[base + bu]!
      const prev = hist[base + bu - d.u32(1)]!
      let next = d.u32(0)
      if (bu < lastBin) {
        next = hist[base + bu + d.u32(1)]!
      }
      let isPeak = false
      if (bu === lastBin) {
        isPeak = v >= prev
      } else {
        isPeak = v >= prev && v >= next
      }
      if (isPeak && v > whiteVal) {
        whiteVal = v
        whitePeak = bu
      }
    }
  }

  return TagDecodeHistPeaks({ blackPeak, whitePeak, blackVal, whiteVal, histMaxCount })
})
