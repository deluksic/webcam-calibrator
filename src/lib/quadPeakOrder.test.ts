import { describe, it, expect } from 'vitest'

import { ORIENT_HIST_BINS } from '@/gpu/lineFitThresholds'
import type { Corners, Point } from '@/lib/geometry'
import { lineFromPoints, lineIntersection } from '@/lib/geometry'

/** Mirrors GPU `orderCornersTLTRBLBR` / edgeHistogramCluster CCW permute — test-only. */
const QUAD_TL_PICK_EPS = 1e-4

function circularForwardDist(from: number, to: number, bins = ORIENT_HIST_BINS): number {
  return (to + bins - from) % bins
}

function sortPeakSlotIndicesCcw(peakBins: readonly number[]): number[] {
  const n = peakBins.length
  if (n < 2) {
    return [...peakBins.keys()]
  }

  const bins = ORIENT_HIST_BINS
  let startBin = peakBins[0]!
  for (let k = 1; k < n; k++) {
    const kb = peakBins[k]!
    if (kb < startBin) {
      startBin = kb
    }
  }

  const key = peakBins.map((b) => circularForwardDist(startBin, b, bins))
  const order = peakBins.map((_, i) => i)

  for (let pass = 0; pass < n; pass++) {
    for (let j = 0; j < n - 1; j++) {
      const ja = order[j]!
      const jb = order[j + 1]!
      if (key[ja]! > key[jb]!) {
        order[j] = jb
        order[j + 1] = ja
      }
    }
  }

  return order
}

function orderCornersTLTRBLBR(ring: readonly [Point, Point, Point, Point]): Corners {
  let tlIdx = 0
  let bestY = ring[0]!.y
  let bestX = ring[0]!.x
  for (let i = 1; i < 4; i++) {
    const p = ring[i]!
    const pick =
      p.y < bestY - QUAD_TL_PICK_EPS ||
      (Math.abs(p.y - bestY) <= QUAD_TL_PICK_EPS && p.x < bestX - QUAD_TL_PICK_EPS)
    if (pick) {
      tlIdx = i
      bestY = p.y
      bestX = p.x
    }
  }

  const tl = ring[tlIdx]!
  const tr = ring[(tlIdx + 1) % 4]!
  const br = ring[(tlIdx + 2) % 4]!
  const bl = ring[(tlIdx + 3) % 4]!
  return [tl, tr, bl, br]
}

describe('quadPeakOrder (GPU algorithm mirrors)', () => {
  describe('circularForwardDist', () => {
    it('wraps across bin 0', () => {
      expect(circularForwardDist(61, 1, 64)).toBe(4)
      expect(circularForwardDist(61, 30, 64)).toBe(33)
    })
  })

  describe('sortPeakSlotIndicesCcw', () => {
    it('orders bins that straddle zero circularly', () => {
      const bins = [61, 1, 15, 30] as const
      const order = sortPeakSlotIndicesCcw(bins)
      const sorted = order.map((i) => bins[i]!)
      expect(sorted).toEqual([1, 15, 30, 61])
    })

    it('does not use naive numeric sort when wrap is needed', () => {
      const bins = [61, 1, 15, 30] as const
      const naive = [...bins].sort((a, b) => a - b)
      expect(naive).toEqual([1, 15, 30, 61])
      const order = sortPeakSlotIndicesCcw(bins)
      expect(order.map((i) => bins[i])).toEqual(naive)
    })
  })

  describe('orderCornersTLTRBLBR', () => {
    it('labels axis-aligned CCW ring as TL, TR, BL, BR', () => {
      const ring: [Point, Point, Point, Point] = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 80 },
        { x: 0, y: 80 },
      ]
      const strip = orderCornersTLTRBLBR(ring)
      expect(strip[0]).toEqual({ x: 0, y: 0 })
      expect(strip[1]).toEqual({ x: 100, y: 0 })
      expect(strip[2]).toEqual({ x: 0, y: 80 })
      expect(strip[3]).toEqual({ x: 100, y: 80 })
    })
  })

  describe('fixed-slot line intersections', () => {
    it('intersects four axis-aligned sides in slot order', () => {
      const lines = [
        lineFromPoints({ x: 0, y: 0 }, { x: 100, y: 0 })!,
        lineFromPoints({ x: 100, y: 0 }, { x: 100, y: 80 })!,
        lineFromPoints({ x: 100, y: 80 }, { x: 0, y: 80 })!,
        lineFromPoints({ x: 0, y: 80 }, { x: 0, y: 0 })!,
      ]
      const corners: [Point, Point, Point, Point] = [
        lineIntersection(lines[0]!, lines[1]!)!,
        lineIntersection(lines[1]!, lines[2]!)!,
        lineIntersection(lines[2]!, lines[3]!)!,
        lineIntersection(lines[3]!, lines[0]!)!,
      ]
      const strip = orderCornersTLTRBLBR(corners)
      expect(strip[0].x).toBeCloseTo(0, 5)
      expect(strip[0].y).toBeCloseTo(0, 5)
      expect(strip[1].x).toBeCloseTo(100, 5)
      expect(strip[1].y).toBeCloseTo(0, 5)
    })
  })
})
