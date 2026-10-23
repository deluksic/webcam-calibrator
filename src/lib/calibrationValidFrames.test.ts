import { describe, expect, it } from 'vitest'

import { CALIBRATION_MIN_POINTS_PER_FRAME } from '@/lib/calibrationInput'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import { countValidSolveFrames, filterFramesForSolve } from '@/lib/calibrationValidFrames'

function frame(frameId: number, count: number): CalibrationFrameObservation {
  return {
    frameId,
    points: Array.from({ length: count }, (_, index) => ({
      designId: index,
      x: index,
      y: index,
    })),
  }
}

describe('calibrationValidFrames', () => {
  it('keeps only frames with enough unique points', () => {
    const pool = [
      frame(1, CALIBRATION_MIN_POINTS_PER_FRAME - 1),
      frame(2, CALIBRATION_MIN_POINTS_PER_FRAME),
      frame(3, CALIBRATION_MIN_POINTS_PER_FRAME + 2),
    ]
    expect(filterFramesForSolve(pool).map((item) => item.frameId)).toEqual([2, 3])
    expect(countValidSolveFrames(pool)).toBe(2)
  })
})
