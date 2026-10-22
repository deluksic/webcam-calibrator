import { describe, expect, it } from 'vitest'

import {
  analyzeRoiHistogram,
  ROI_HISTOGRAM_BINS,
  type RoiHistogramAnalysis,
} from '@/lib/roiHistogram'

type Peak = { mean: number; stdDev: number; weight: number }

function syntheticHistogram(peaks: Peak[]): Uint32Array {
  const counts = new Uint32Array(ROI_HISTOGRAM_BINS)
  for (let bin = 0; bin < ROI_HISTOGRAM_BINS; bin++) {
    let value = 0
    for (const peak of peaks) {
      const z = (bin - peak.mean) / peak.stdDev
      value += (peak.weight * Math.exp(-0.5 * z * z)) / (peak.stdDev * Math.sqrt(2 * Math.PI))
    }
    counts[bin] = Math.round(value)
  }
  return counts
}

function expectBlobSide(
  analysis: RoiHistogramAnalysis,
  side: 'black' | 'white',
  backgroundBin: number,
): void {
  const cut = side === 'black' ? analysis.blackCut : analysis.whiteCut
  const pixels = side === 'black' ? analysis.blackPixels : analysis.whitePixels
  expect(analysis.valid).toBe(true)
  expect(pixels).toBeGreaterThan(0)
  if (side === 'black') {
    expect(cut).toBeGreaterThan(30)
    expect(cut).toBeLessThan(backgroundBin - 5)
  } else {
    expect(cut).toBeGreaterThan(backgroundBin + 5)
    expect(cut).toBeLessThan(250)
  }
}

describe('analyzeRoiHistogram', () => {
  it('splits a three-level gray target into black, gray, and white', () => {
    const counts = syntheticHistogram([
      { mean: 70, stdDev: 9, weight: 900 },
      { mean: 180, stdDev: 3.5, weight: 24000 },
      { mean: 240, stdDev: 8, weight: 700 },
    ])
    const analysis = analyzeRoiHistogram(counts)
    expect(analysis.valid).toBe(true)
    expect(analysis.backgroundBin).toBe(180)
    expectBlobSide(analysis, 'black', analysis.backgroundBin)
    expectBlobSide(analysis, 'white', analysis.backgroundBin)
    expect(analysis.blackPixels + analysis.grayPixels + analysis.whitePixels).toBe(
      analysis.maskedPixels,
    )
    expect(analysis.grayPixels).toBeGreaterThan(analysis.maskedPixels * 0.8)
  })

  it('does not invent blobs on a flat gray target', () => {
    const counts = syntheticHistogram([{ mean: 180, stdDev: 4, weight: 24000 }])
    const analysis = analyzeRoiHistogram(counts)
    expect(analysis.valid).toBe(false)
    expect(analysis.blackCut).toBe(0)
    expect(analysis.whiteCut).toBe(ROI_HISTOGRAM_BINS)
  })

  it('finds black markers when there are no white markers', () => {
    const counts = syntheticHistogram([
      { mean: 60, stdDev: 8, weight: 700 },
      { mean: 180, stdDev: 3.5, weight: 24000 },
    ])
    const analysis = analyzeRoiHistogram(counts)
    expect(analysis.valid).toBe(true)
    expectBlobSide(analysis, 'black', analysis.backgroundBin)
    expect(analysis.whiteCut).toBe(ROI_HISTOGRAM_BINS)
    expect(analysis.whitePixels).toBe(0)
  })

  it('finds white markers when there are no black markers', () => {
    const counts = syntheticHistogram([
      { mean: 180, stdDev: 3.5, weight: 24000 },
      { mean: 245, stdDev: 6, weight: 650 },
    ])
    const analysis = analyzeRoiHistogram(counts)
    expect(analysis.valid).toBe(true)
    expectBlobSide(analysis, 'white', analysis.backgroundBin)
    expect(analysis.blackCut).toBe(0)
    expect(analysis.blackPixels).toBe(0)
  })

  it('returns a disabled analysis for an empty mask', () => {
    const analysis = analyzeRoiHistogram(new Uint32Array(ROI_HISTOGRAM_BINS))
    expect(analysis.valid).toBe(false)
    expect(analysis.maskedPixels).toBe(0)
    expect(analysis.blackCut).toBe(0)
    expect(analysis.whiteCut).toBe(ROI_HISTOGRAM_BINS)
  })
})
