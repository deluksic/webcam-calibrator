import type { ParentProps } from 'solid-js'
import { createContext, createEffect, createMemo, createSignal, useContext } from 'solid-js'

import { countValidSolveFrames } from '@/lib/calibrationValidFrames'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import { layoutToObjectTags, type TargetLayout } from '@/lib/targetLayout'
import { calibApi, type CalibrationResult } from '@/workers/calibrationClient'

export type Collection = 'idle' | 'running'

export type CalibRunStats = {
  framesProcessed: number
  framesAccepted: number
  evictions: number
}

export type CalibRun = {
  collection: Collection
  framePool: CalibrationFrameObservation[]
  stats: CalibRunStats
}

export const initialCalibRunStats: CalibRunStats = {
  framesProcessed: 0,
  framesAccepted: 0,
  evictions: 0,
}

const initialRun: CalibRun = {
  collection: 'idle',
  framePool: [],
  stats: { ...initialCalibRunStats },
}

export type CalibrationLatestMeta = {
  validSolveFrameCount: number
  video?: { width: number; height: number }
}

export type CalibrationRunContextValue = {
  run: () => CalibRun
  setRun: (v: CalibRun | ((prev: CalibRun) => CalibRun)) => void
  layout: () => TargetLayout | undefined
  setLayout: (v: TargetLayout | undefined) => void
  calib: () => CalibrationResult | undefined
  setCalib: (v: CalibrationResult | undefined) => void
  latestCalibration: () => CalibrationResult | undefined
  latestCalibrationMeta: () => CalibrationLatestMeta | undefined
  videoFrameSize: () => { width: number; height: number } | undefined
  setVideoFrameSize: (v: { width: number; height: number } | undefined) => void
  isSolving: () => boolean
  calibrationSessionActive: () => boolean
  resetSession: () => void
  startSession: () => void
}

const CalibrationRunContext = createContext<CalibrationRunContextValue>()

export function useCalibrationRun(): CalibrationRunContextValue {
  const v = useContext(CalibrationRunContext)
  if (!v) {
    throw new Error('useCalibrationRun must be used within CalibrationRunProvider')
  }
  return v
}

export function CalibrationRunProvider(props: ParentProps) {
  const [run, setRun] = createSignal<CalibRun>({ ...initialRun, stats: { ...initialCalibRunStats } })
  const [layout, setLayout] = createSignal<TargetLayout>()
  const [videoFrameSize, setVideoFrameSize] = createSignal<{ width: number; height: number }>()
  const [solveInFlight, setSolveInFlight] = createSignal(0)
  const [calib, setCalibInner] = createSignal<CalibrationResult>()
  const [latestCalibration, writeLatestCalibration] = createSignal<CalibrationResult>()
  const [latestCalibrationMeta, writeLatestCalibrationMeta] = createSignal<CalibrationLatestMeta>()

  const setLatestCalibration = (v: CalibrationResult | undefined, meta?: CalibrationLatestMeta) => {
    writeLatestCalibration(v)
    writeLatestCalibrationMeta(v === undefined ? undefined : meta)
  }

  const setCalib = (r: CalibrationResult | undefined) => {
    setCalibInner(r)
    if (r === undefined) {
      setLatestCalibration(undefined, undefined)
    } else {
      setLatestCalibration(r, {
        validSolveFrameCount: countValidSolveFrames(run().framePool, layout()),
        video: videoFrameSize(),
      })
    }
  }

  let pendingVersion = 0
  let lastSolveKey: string | undefined

  const updateCalib = async (
    collection: Collection,
    lay: TargetLayout | undefined,
    framePool: CalibrationFrameObservation[],
    frameSize: { width: number; height: number },
  ) => {
    if (collection === 'idle' || !lay || framePool.length < 1) {
      lastSolveKey = undefined
      setCalib(undefined)
      return
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
    if (filteredPool.length < 3) {
      lastSolveKey = undefined
      setCalib({ kind: 'error', reason: 'too-few-views' })
      return
    }

    const solveKey = `${collection}|${lay.size}|${filteredPool.map((f) => f.frameId).join(',')}`
    if (solveKey === lastSolveKey) {
      return
    }
    lastSolveKey = solveKey

    const layoutTagsModel = layoutToObjectTags(lay)
    const { width: w, height: h } = frameSize
    const currentVersion = ++pendingVersion
    try {
      setSolveInFlight((n) => n + 1)
      const result = await calibApi.solveCalibration(layoutTagsModel, filteredPool, {
        width: w,
        height: h,
      })
      if (currentVersion === pendingVersion) {
        setCalib(result)
      }
    } catch (err) {
      console.error('[CalibrationRun] calibration failed', err)
      if (currentVersion === pendingVersion) {
        setCalib({ kind: 'error', reason: 'singular', details: String(err) })
      }
    } finally {
      setSolveInFlight((n) => Math.max(0, n - 1))
    }
  }

  createEffect(
    () => ({ r: run(), lay: layout(), frameSize: videoFrameSize() }),
    ({ r, lay, frameSize }) => {
      if (frameSize) {
        void updateCalib(r.collection, lay, r.framePool, frameSize)
      }
    },
  )

  const isSolving = createMemo(() => solveInFlight() > 0)
  const calibrationSessionActive = createMemo(
    () => run().collection === 'running' || run().framePool.length > 0,
  )

  const resetSession = () => {
    setLayout(undefined)
    setCalib(undefined)
    setRun({
      collection: 'idle',
      framePool: [],
      stats: { ...initialCalibRunStats },
    })
  }

  const startSession = () => {
    setLayout(undefined)
    setRun({
      collection: 'running',
      framePool: [],
      stats: { ...initialCalibRunStats },
    })
  }

  const value: CalibrationRunContextValue = {
    run,
    setRun,
    layout,
    setLayout,
    calib,
    setCalib,
    latestCalibration,
    latestCalibrationMeta,
    videoFrameSize,
    setVideoFrameSize,
    isSolving,
    calibrationSessionActive,
    resetSession,
    startSession,
  }

  return <CalibrationRunContext value={value}>{props.children}</CalibrationRunContext>
}
