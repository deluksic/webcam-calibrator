import type { CalibrationOk } from '@/workers/calibration.worker'

export function downloadCalibrationOkJson(c: CalibrationOk) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  const blob = new Blob([JSON.stringify(c, undefined, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `calibration-ok-${stamp}.json`
  a.click()
  queueMicrotask(() => URL.revokeObjectURL(url))
}
