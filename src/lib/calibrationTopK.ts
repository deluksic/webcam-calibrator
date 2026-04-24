import type { CalibrationFrameObservation, FramePoint } from '@/lib/calibrationTypes'

export const DEFAULT_CALIBRATION_TOP_K = 10_000

function frameTotalScore(f: CalibrationFrameObservation): number {
  let s = 0
  for (const fp of f.framePoints) {
    s += fp.imagePoint.score ?? 0
  }
  return s
}

/** Lower total score evicted first; tie-break older `frameId` first, then more points first. */
function compareFrames(a: CalibrationFrameObservation, b: CalibrationFrameObservation): number {
  const as = frameTotalScore(a)
  const bs = frameTotalScore(b)
  if (as !== bs) {
    return as - bs
  }
  if (a.frameId !== b.frameId) {
    return a.frameId - b.frameId
  }
  return a.framePoints.length - b.framePoints.length
}

/**
 * Merge `incoming` into `pool`, keep at most `maxK` **frames** by ascending aggregate score.
 * Returns a **new** array (does not mutate `pool`).
 */
export function mergeCalibrationFramesTopK(
  pool: readonly CalibrationFrameObservation[],
  incoming: readonly CalibrationFrameObservation[] | undefined,
  maxK: number,
): { next: CalibrationFrameObservation[]; evicted: number } {
  const merged: CalibrationFrameObservation[] = [...pool]
  if (incoming) {
    merged.push(...incoming)
  }
  merged.sort(compareFrames)
  let evicted = 0
  while (merged.length > maxK) {
    merged.shift()
    evicted++
  }
  return { next: merged, evicted }
}
