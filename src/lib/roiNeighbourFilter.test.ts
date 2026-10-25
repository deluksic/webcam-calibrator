import { describe, expect, it } from 'vitest'

import {
  describeRefinedPeak,
  filterRefinedPeaks,
  ROI_CONTRAST_FLOOR,
  type RoiRefinedPeak,
} from '@/lib/roiNeighbourFilter'

function peak(x: number, y: number, polarity: 0 | 1, feature: number): RoiRefinedPeak {
  return {
    x,
    y,
    polarity,
    peak: polarity === 0 ? 0.2 + feature * 0.002 : 0.8 + feature * 0.002,
    eigenvalue1: polarity === 0 ? 0.5 + feature * 0.001 : -0.5 - feature * 0.001,
    eigenvalue2: polarity === 0 ? 0.6 + feature * 0.001 : -0.6 - feature * 0.001,
    hessianXY: polarity === 0 ? 0.01 + feature * 0.001 : -0.01 - feature * 0.001,
  }
}

function parityFixture(): RoiRefinedPeak[] {
  const points: RoiRefinedPeak[] = []
  for (let index = 0; index < 10; index += 1) {
    points.push(peak(10 + (index % 5) * 12, 10 + Math.floor(index / 5) * 12, 0, index))
  }
  for (let index = 0; index < 10; index += 1) {
    points.push(peak(120 + (index % 5) * 12, 10 + Math.floor(index / 5) * 12, 1, index))
  }
  points.push({
    x: 80,
    y: 80,
    polarity: 0,
    peak: 0.95,
    eigenvalue1: 0.9,
    eigenvalue2: 1,
    hessianXY: 0.02,
  })
  points.push({
    x: 200,
    y: 80,
    polarity: 1,
    peak: 0.2,
    eigenvalue1: -0.1,
    eigenvalue2: -0.2,
    hessianXY: -0.02,
  })
  return points
}

describe('filterRefinedPeaks', () => {
  it('matches the expected trim/recompute/accept decisions', () => {
    const result = filterRefinedPeaks(parityFixture(), {
      neighbours: 40,
      trimSigma: 2,
      acceptSigma: 4,
      normalizeHessian: false,
    })
    expect(result.keep).toEqual([
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      true,
      false,
      false,
    ])
    expect(result.kept).toHaveLength(20)
    expect(result.zScores[20 * 4]).toBeCloseTo(122.3721, 2)
    expect(result.zScores[20 * 4 + 1]).toBeCloseTo(130.6294, 2)
    expect(result.zScores[21 * 4]).toBeCloseTo(-100.573, 2)
    expect(result.zScores[21 * 4 + 1]).toBeCloseTo(133.602, 2)
  })

  it('keeps all points when a polarity has fewer than four candidates', () => {
    const points = [
      peak(0, 0, 0, 0),
      peak(10, 0, 0, 1),
      peak(20, 0, 0, 2),
      {
        x: 0,
        y: 20,
        polarity: 1 as const,
        peak: 0.9,
        eigenvalue1: -0.5,
        eigenvalue2: -0.6,
        hessianXY: 0,
      },
    ]
    const result = filterRefinedPeaks(points, { neighbours: 40 })
    expect(result.keep.every(Boolean)).toBe(true)
    expect(result.kept).toHaveLength(4)
  })

  it('keeps a point when there are fewer than three neighbours', () => {
    const points = [peak(0, 0, 0, 0), peak(10, 0, 0, 1), peak(20, 0, 0, 2), peak(30, 0, 0, 3)]
    const result = filterRefinedPeaks(points, { neighbours: 2 })
    expect(result.keep.every(Boolean)).toBe(true)
  })

  it('drops immediately when peak fails, even with all Hessian features passing', () => {
    const points = [0, 1, 2, 3].map((index) => ({
      x: index * 10,
      y: 0,
      polarity: 0 as const,
      peak: index === 0 ? 0.9 : 0.2,
      eigenvalue1: 0.5,
      eigenvalue2: 0.6,
      hessianXY: 0,
    }))
    const options = { neighbours: 40, normalizeHessian: false }
    const result = filterRefinedPeaks(points, options)
    expect(result.keep[0]).toBe(false)
    expect(describeRefinedPeak(points, 0, options)?.kept).toBe(false)
  })

  it('keeps a point when peak passes and two of three Hessian features pass', () => {
    const points = [0, 1, 2, 3].map((index) => ({
      x: index * 10,
      y: 0,
      polarity: 0 as const,
      peak: 0.2,
      eigenvalue1: 0.5,
      eigenvalue2: 0.6,
      hessianXY: index === 0 ? 0.9 : 0,
    }))
    const options = { neighbours: 40, normalizeHessian: false }
    const result = filterRefinedPeaks(points, options)
    expect(result.keep[0]).toBe(true)
    expect(describeRefinedPeak(points, 0, options)?.kept).toBe(true)
  })

  it('drops a point when peak passes and only one Hessian feature passes', () => {
    const points = [0, 1, 2, 3].map((index) => ({
      x: index * 10,
      y: 0,
      polarity: 0 as const,
      peak: 0.2,
      eigenvalue1: index === 0 ? 0.9 : 0.5,
      eigenvalue2: 0.6,
      hessianXY: index === 0 ? 0.9 : 0,
    }))
    const options = { neighbours: 40, normalizeHessian: false }
    const result = filterRefinedPeaks(points, options)
    expect(result.keep[0]).toBe(false)
    expect(describeRefinedPeak(points, 0, options)?.kept).toBe(false)
  })

  it('leaves a feature untrimmed when trimming would leave fewer than three inliers', () => {
    const points = [0, 1, 2, 1].map((value, index) => ({
      x: index * 10,
      y: 0,
      polarity: 0 as const,
      peak: value,
      eigenvalue1: 0.5,
      eigenvalue2: 0.6,
      hessianXY: 0,
    }))
    const result = filterRefinedPeaks(points, {
      neighbours: 40,
      trimSigma: 0.1,
      acceptSigma: 4,
      normalizeHessian: false,
    })
    expect(result.keep.every(Boolean)).toBe(true)
    expect([...result.trimmedCounts]).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  })

  it('does not let one polarity affect the other', () => {
    const points = [
      ...Array.from({ length: 6 }, (_, index) => peak(index * 10, 0, 0, index)),
      ...Array.from({ length: 6 }, (_, index) => peak(100 + index * 10, 0, 1, index)),
    ]
    points.push({
      x: 160,
      y: 0,
      polarity: 1,
      peak: 5,
      eigenvalue1: -0.1,
      eigenvalue2: -0.2,
      hessianXY: 0,
    })
    const result = filterRefinedPeaks(points, { neighbours: 40 })
    expect(result.keep.slice(0, 6)).toEqual([true, true, true, true, true, true])
    expect(result.keep[12]).toBe(false)
  })

  it('describes the same per-point decision as the filter', () => {
    const points = parityFixture()
    const options = { neighbours: 40, trimSigma: 2, acceptSigma: 4 }
    const result = filterRefinedPeaks(points, options)
    for (let index = 0; index < points.length; index += 1) {
      const detail = describeRefinedPeak(points, index, options)
      expect(detail).toBeDefined()
      expect(detail!.kept).toBe(result.keep[index])
      expect(detail!.features).toHaveLength(4)
      for (let feature = 0; feature < detail!.features.length; feature += 1) {
        expect(detail!.features[feature]!.zScore).toBeCloseTo(result.zScores[index * 4 + feature]!, 10)
      }
      const hxy = detail!.features.find((feature) => feature.key === 'hessianXY')
      const scale = Math.max(Math.abs(points[index]!.peak - 0.5), ROI_CONTRAST_FLOOR)
      expect(hxy?.value).toBeCloseTo(points[index]!.hessianXY / scale, 10)
    }
  })

  it('normalizes Hessian features by peak contrast', () => {
    const points = parityFixture()
    const point = points[0]!
    const detail = describeRefinedPeak(points, 0, { neighbours: 40, trimSigma: 2, acceptSigma: 4 })
    expect(detail).toBeDefined()
    const scale = Math.max(Math.abs(point.peak - 0.5), ROI_CONTRAST_FLOOR)
    expect(detail!.features[1]!.value).toBeCloseTo(point.eigenvalue1 / scale, 10)
    expect(detail!.features[2]!.value).toBeCloseTo(point.eigenvalue2 / scale, 10)
    expect(detail!.features[3]!.value).toBeCloseTo(point.hessianXY / scale, 10)
  })

  it('reports why a point was not tested', () => {
    const points = [peak(0, 0, 0, 0), peak(10, 0, 0, 1), peak(20, 0, 0, 2)]
    const detail = describeRefinedPeak(points, 0, { neighbours: 40 })
    expect(detail?.tested).toBe(false)
    expect(detail?.note).toBe('fewer-than-four-same-polarity')
    expect(detail?.kept).toBe(true)
  })
})
