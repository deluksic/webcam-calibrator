import { describe, expect, it } from 'vitest'

import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'

import type { Corners } from '@/lib/geometry'

const unitCorners: Corners = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: 1, y: 1 },
]

function frame(
  frameId: number,
  tagScores: { tagId: number; score: number }[],
): CalibrationFrameObservation {
  return {
    frameId,
    tags: tagScores.map((x) => ({
      tagId: x.tagId,
      rotation: 0,
      corners: unitCorners,
      score: x.score,
    })),
  }
}

describe('mergeCalibrationFramesTopK', () => {
  it('evicts lowest aggregate score when over K', () => {
    const k = 3
    const a = mergeCalibrationFramesTopK(
      [],
      [frame(1, [{ tagId: 1, score: 0.5 }]), frame(1, [{ tagId: 2, score: 0.9 }]), frame(1, [{ tagId: 3, score: 0.7 }])],
      k,
    )
    expect(a.next.length).toBe(3)
    const b = mergeCalibrationFramesTopK(a.next, [frame(2, [{ tagId: 4, score: 0.2 }])], k)
    expect(b.evicted).toBe(1)
    expect(b.next.length).toBe(3)
    const tagIds = b.next.flatMap((f) => f.tags.map((t) => t.tagId))
    expect(tagIds).not.toContain(4)
    expect(tagIds).toContain(1)
  })

  it('respects DEFAULT_CALIBRATION_TOP_K export', () => {
    expect(DEFAULT_CALIBRATION_TOP_K).toBeGreaterThan(1000)
  })
})
