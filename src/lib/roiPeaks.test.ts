import { describe, expect, it } from 'vitest'

import { finalizeRoiPeaks, fitQuadraticMaximum, type RoiPeakCandidate } from '@/lib/roiPeaks'

function parabolicCandidate(
  x: number,
  y: number,
  polarity: 0 | 1,
  peak: number,
  curvature = 0.2,
): RoiPeakCandidate {
  const values: number[] = []
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      values.push(peak - curvature * (dx * dx + dy * dy))
    }
  }
  return { x, y, polarity, values }
}

describe('fitQuadraticMaximum', () => {
  it('recovers a subpixel maximum', () => {
    const values: number[] = []
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        values.push(0.5 - 0.1 * (dx - 0.2) ** 2 - 0.2 * (dy + 0.1) ** 2)
      }
    }
    const fitted = fitQuadraticMaximum({ x: 10, y: 20, polarity: 0, values })
    expect(fitted).not.toBeNull()
    expect(fitted!.x).toBeCloseTo(10.2, 3)
    expect(fitted!.y).toBeCloseTo(19.9, 3)
    expect(fitted!.peak).toBeCloseTo(0.5, 3)
  })

  it('rejects a saddle', () => {
    const values: number[] = []
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        values.push(dx * dx - dy * dy)
      }
    }
    expect(fitQuadraticMaximum({ x: 5, y: 5, polarity: 0, values })).toBeNull()
  })
})

describe('finalizeRoiPeaks', () => {
  it('suppresses nearby duplicates and applies a per-color height gate', () => {
    const candidates: RoiPeakCandidate[] = [
      parabolicCandidate(10, 10, 0, 0.5),
      parabolicCandidate(11, 10, 0, 0.4),
      parabolicCandidate(30, 10, 0, 0.2),
      parabolicCandidate(50, 10, 0, 0.03),
      parabolicCandidate(10, 40, 1, 0.6),
      parabolicCandidate(30, 40, 1, 0.3),
    ]
    const analysis = finalizeRoiPeaks(candidates)
    expect(analysis.blackCount).toBe(2)
    expect(analysis.whiteCount).toBe(2)
    const blackPeaks = analysis.peaks
      .filter((peak) => peak.polarity === 0)
      .map((peak) => peak.peak)
    expect(blackPeaks).toHaveLength(2)
    expect(blackPeaks[0]).toBeCloseTo(0.5, 5)
    expect(blackPeaks[1]).toBeCloseTo(0.2, 5)
    expect(analysis.blackThreshold).toBeCloseTo(0.07, 3)
  })
})
