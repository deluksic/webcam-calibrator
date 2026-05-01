import type { ParentProps } from 'solid-js'
import { createContext, createSignal, useContext } from 'solid-js'

import { DEBUG_SEED_RESULTS_CALIBRATION, debugSeedCalibrationOk } from '@/lib/debugResultsCalibrationSeed'
import type { CalibrationResult } from '@/workers/calibrationClient'

export type CalibrationLatestContextValue = {
  latestCalibration: () => CalibrationResult | undefined
  setLatestCalibration: (v: CalibrationResult | undefined) => void
}

const CalibrationLatestContext = createContext<CalibrationLatestContextValue>()

export function useCalibrationLatest(): CalibrationLatestContextValue {
  const v = useContext(CalibrationLatestContext)
  if (!v) {
    throw new Error('useCalibrationLatest must be used within CalibrationLatestProvider')
  }
  return v
}

export function CalibrationLatestProvider(props: ParentProps) {
  const [latestCalibration, setLatestCalibration] = createSignal<CalibrationResult | undefined>(
    DEBUG_SEED_RESULTS_CALIBRATION ? debugSeedCalibrationOk() : undefined,
  )

  const value: CalibrationLatestContextValue = {
    latestCalibration,
    setLatestCalibration,
  }

  return <CalibrationLatestContext value={value}>{props.children}</CalibrationLatestContext>
}
