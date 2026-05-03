import type { ParentProps } from 'solid-js'
import { createContext, createMemo, createStore, useContext } from 'solid-js'

import { createCalibrationLibraryEntry, type CalibrationLibraryEntry } from '@/lib/calibrationLibraryTypes'
import { createDemoCalibrationLibraryEntry } from '@/lib/demoCalibrationExample'
import type { CalibrationResult } from '@/workers/calibrationClient'

/** Max columns in the intrinsics comparison table (labels A–…). */
export const CALIBRATION_COMPARE_MAX = 8

type LibraryState = {
  entries: CalibrationLibraryEntry[]
  selectedId: string | undefined
  /** Ordered selection: column A, B, C, … */
  compareIds: string[]
}

export type CalibrationLibraryContextValue = {
  entries: () => CalibrationLibraryEntry[]
  selectedId: () => string | undefined
  compareIds: () => string[]
  setSelectedId: (id: string | undefined) => void
  toggleCompareId: (id: string) => void
  clearCompare: () => void
  addFromCurrentSolve: (args: {
    result: CalibrationResult
    validSolveFrameCount: number
    video?: { width: number; height: number }
    label?: string
  }) => string
  remove: (id: string) => void
  rename: (id: string, label: string) => void
}

const CalibrationLibraryContext = createContext<CalibrationLibraryContextValue>()

export function useCalibrationLibrary(): CalibrationLibraryContextValue {
  const v = useContext(CalibrationLibraryContext)
  if (!v) {
    throw new Error('useCalibrationLibrary must be used within CalibrationLibraryProvider')
  }
  return v
}

export function CalibrationLibraryProvider(props: ParentProps) {
  const demoEntry = createDemoCalibrationLibraryEntry()
  const [store, setStore] = createStore<LibraryState>({
    entries: [demoEntry],
    selectedId: demoEntry.id,
    compareIds: [],
  })

  const entries = createMemo(() => store.entries)
  const selectedId = createMemo(() => store.selectedId)
  const compareIds = createMemo(() => store.compareIds)

  const setSelectedId = (id: string | undefined) => {
    setStore((s) => {
      s.selectedId = id
    })
  }

  const toggleCompareId = (id: string) => {
    setStore((s) => {
      const entry = s.entries.find((e) => e.id === id)
      if (!entry || entry.result.kind !== 'ok') {
        return
      }
      const i = s.compareIds.indexOf(id)
      if (i >= 0) {
        s.compareIds = s.compareIds.filter((x) => x !== id)
      } else if (s.compareIds.length < CALIBRATION_COMPARE_MAX) {
        s.compareIds = [...s.compareIds, id]
      }
    })
  }

  const clearCompare = () => {
    setStore((s) => {
      s.compareIds = []
    })
  }

  const addFromCurrentSolve: CalibrationLibraryContextValue['addFromCurrentSolve'] = (args) => {
    const entry = createCalibrationLibraryEntry(args.result, {
      validSolveFrameCount: args.validSolveFrameCount,
      videoWidth: args.video?.width,
      videoHeight: args.video?.height,
    }, args.label ?? '')
    setStore((s) => {
      s.entries = [...s.entries, entry]
      s.selectedId = entry.id
    })
    return entry.id
  }

  const remove: CalibrationLibraryContextValue['remove'] = (id) => {
    setStore((s) => {
      const next = s.entries.filter((e) => e.id !== id)
      s.entries = next
      if (s.selectedId === id) {
        s.selectedId = next[0]?.id
      }
      s.compareIds = s.compareIds.filter((cid) => cid !== id && next.some((e) => e.id === cid))
    })
  }

  const rename: CalibrationLibraryContextValue['rename'] = (id, label) => {
    setStore((s) => {
      const i = s.entries.findIndex((e) => e.id === id)
      if (i >= 0) {
        const prev = s.entries[i]!
        s.entries[i] = { ...prev, label }
      }
    })
  }

  const value: CalibrationLibraryContextValue = {
    entries,
    selectedId,
    compareIds,
    setSelectedId,
    toggleCompareId,
    clearCompare,
    addFromCurrentSolve,
    remove,
    rename,
  }

  return <CalibrationLibraryContext value={value}>{props.children}</CalibrationLibraryContext>
}
