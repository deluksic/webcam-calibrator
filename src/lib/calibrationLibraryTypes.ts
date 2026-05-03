import type { CalibrationResult } from '@/workers/calibrationClient'

export type CalibrationLibraryMeta = {
  validSolveFrameCount: number
  videoWidth?: number
  videoHeight?: number
}

export type CalibrationLibraryEntry = {
  id: string
  createdAt: number
  label: string
  result: CalibrationResult
  meta: CalibrationLibraryMeta
}

export function cloneCalibrationResult(r: CalibrationResult): CalibrationResult {
  return structuredClone(r)
}

export function createCalibrationLibraryEntry(
  result: CalibrationResult,
  meta: CalibrationLibraryMeta,
  label = '',
): CalibrationLibraryEntry {
  return {
    id: globalThis.crypto.randomUUID(),
    createdAt: Date.now(),
    label,
    result: cloneCalibrationResult(result),
    meta: { ...meta },
  }
}
