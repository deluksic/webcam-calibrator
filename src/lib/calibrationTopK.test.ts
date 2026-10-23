import { describe, expect, it } from 'vitest'

import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'

function frame(frameId: number, scores: number[]): CalibrationFrameObservation {
  return {
    frameId,
    points: scores.map((score, index) => ({ designId: index, x: index, y: index, score })),
  }
}

describe('mergeCalibrationFramesTopK', () => {
  it('evicts the lowest aggregate score when over K', () => {
    const k = 3
    const first = mergeCalibrationFramesTopK(
      [],
      [frame(1, [0.5]), frame(2, [0.9]), frame(3, [0.7])],
      k,
    )
    expect(first.next).toHaveLength(3)
    const second = mergeCalibrationFramesTopK(first.next, [frame(4, [0.2])], k)
    expect(second.evicted).toBe(1)
    expect(second.next).toHaveLength(3)
    const frameIds = second.next.map((item) => item.frameId)
    expect(frameIds).not.toContain(4)
    expect(frameIds).toContain(1)
  })

  it('respects the exported default cap', () => {
    expect(DEFAULT_CALIBRATION_TOP_K).toBeGreaterThan(1000)
  })
})
