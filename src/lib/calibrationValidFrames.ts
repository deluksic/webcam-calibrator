import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import type { TargetLayout } from '@/lib/targetLayout'

/** Frames that contribute to the pooled solve (same filter as CalibrationRunContext `updateCalib`). */
export function filterFramesForLayout(
  framePool: CalibrationFrameObservation[],
  lay: TargetLayout | undefined,
): CalibrationFrameObservation[] {
  if (!lay) {
    return []
  }
  const layoutTagIds = new Set<number>()
  for (const tagId of lay.keys()) {
    layoutTagIds.add(tagId)
  }
  const filteredPool: CalibrationFrameObservation[] = []
  for (const frame of framePool) {
    const filteredTags = frame.tags.filter((ft) => layoutTagIds.has(ft.tagId))
    const cornerCount = filteredTags.length * 4
    if (cornerCount >= 8) {
      filteredPool.push({ frameId: frame.frameId, tags: filteredTags })
    }
  }
  return filteredPool
}

/** Count of solver-ready frames (for Results gating and guidance). */
export function countValidSolveFrames(
  framePool: CalibrationFrameObservation[],
  lay: TargetLayout | undefined,
): number {
  return filterFramesForLayout(framePool, lay).length
}
