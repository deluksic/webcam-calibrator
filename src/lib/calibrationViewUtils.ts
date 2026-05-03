import type { CalibrationResult } from '@/workers/calibrationClient'

export type SnapshotFeedback = { kind: 'idle' } | { kind: 'fail'; message: string }

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) {
    return 0
  }
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))
  return sorted[idx]!
}

export function isProgressShapedError(c: CalibrationResult | undefined): boolean {
  return c?.kind === 'error' && c.reason === 'too-few-views'
}
