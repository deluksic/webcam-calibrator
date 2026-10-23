import { CALIBRATION_MIN_POINTS_PER_FRAME } from '@/lib/calibrationInput'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'

export type ImageSize = { width: number; height: number }

/**
 * Frames that contribute to the pooled solve.
 *
 * Passing `imageSize` also drops frames captured at another camera resolution:
 * point coordinates are in image pixels, so mixing resolutions in one solve
 * produces a meaningless principal point and focal length.
 */
export function filterFramesForSolve(
  framePool: readonly CalibrationFrameObservation[],
  imageSize?: ImageSize,
): CalibrationFrameObservation[] {
  return framePool.filter((frame) => {
    if (frame.points.length < CALIBRATION_MIN_POINTS_PER_FRAME) {
      return false
    }
    if (imageSize === undefined) {
      return true
    }
    return frame.imageWidth === imageSize.width && frame.imageHeight === imageSize.height
  })
}

/** Count of solver-ready frames (for Results gating and guidance). */
export function countValidSolveFrames(
  framePool: readonly CalibrationFrameObservation[],
  imageSize?: ImageSize,
): number {
  return filterFramesForSolve(framePool, imageSize).length
}
