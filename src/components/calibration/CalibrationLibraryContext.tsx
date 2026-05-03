import type { ParentProps } from 'solid-js'
import { createContext, createMemo, createStore, useContext } from 'solid-js'

import { createCalibrationLibraryEntry, type CalibrationLibraryEntry } from '@/lib/calibrationLibraryTypes'
import { createDemoCalibrationLibraryEntry } from '@/lib/demoCalibrationExample'
import type { CalibrationResult } from '@/workers/calibrationClient'

type LibraryState = {
  entries: CalibrationLibraryEntry[]
  selectedId: string | undefined
  compareId: string | undefined
}

export type CalibrationLibraryContextValue = {
  entries: () => CalibrationLibraryEntry[]
  selectedId: () => string | undefined
  compareId: () => string | undefined
  setSelectedId: (id: string | undefined) => void
  setCompareId: (id: string | undefined) => void
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
    compareId: undefined,
  })

  const entries = createMemo(() => store.entries)
  const selectedId = createMemo(() => store.selectedId)
  const compareId = createMemo(() => store.compareId)

  const setSelectedId = (id: string | undefined) => {
    setStore((s) => {
      s.selectedId = id
    })
  }

  const setCompareId = (id: string | undefined) => {
    setStore((s) => {
      s.compareId = id
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
      if (s.compareId === id) {
        s.compareId = next.find((e) => e.id !== s.selectedId)?.id
      }
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
    compareId,
    setSelectedId,
    setCompareId,
    addFromCurrentSolve,
    remove,
    rename,
  }

  return <CalibrationLibraryContext value={value}>{props.children}</CalibrationLibraryContext>
}
