import { CALIBRATION_MIN_POINTS_PER_FRAME } from '@/lib/calibrationInput'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'

/** Frames that contribute to the pooled solve. */
export function filterFramesForSolve(
  framePool: readonly CalibrationFrameObservation[],
): CalibrationFrameObservation[] {
  return framePool.filter((frame) => frame.points.length >= CALIBRATION_MIN_POINTS_PER_FRAME)
}

/** Count of solver-ready frames (for Results gating and guidance). */
export function countValidSolveFrames(framePool: readonly CalibrationFrameObservation[]): number {
  return filterFramesForSolve(framePool).length
}
