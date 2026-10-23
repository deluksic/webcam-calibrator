import { describe, expect, it } from 'vitest'

import { CALIBRATION_MIN_POINTS_PER_FRAME } from '@/lib/calibrationInput'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import { countValidSolveFrames, filterFramesForSolve } from '@/lib/calibrationValidFrames'

function frame(
  frameId: number,
  count: number,
  imageSize?: { width: number; height: number },
): CalibrationFrameObservation {
  return {
    frameId,
    points: Array.from({ length: count }, (_, index) => ({
      designId: index,
      x: index,
      y: index,
    })),
    imageWidth: imageSize?.width,
    imageHeight: imageSize?.height,
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

  it('drops frames captured at another camera resolution', () => {
    const pool = [
      frame(1, CALIBRATION_MIN_POINTS_PER_FRAME, { width: 640, height: 480 }),
      frame(2, CALIBRATION_MIN_POINTS_PER_FRAME, { width: 1280, height: 720 }),
      frame(3, CALIBRATION_MIN_POINTS_PER_FRAME, { width: 1280, height: 720 }),
    ]
    const current = { width: 1280, height: 720 }
    expect(filterFramesForSolve(pool, current).map((item) => item.frameId)).toEqual([2, 3])
    expect(countValidSolveFrames(pool, current)).toBe(2)
  })
})
