import type { ParentProps } from 'solid-js'
import { createContext, createEffect, createMemo, createSignal, useContext } from 'solid-js'

import { CameraStreamContext } from '@/components/camera/CameraStreamContext'
import type { CalibrationFrameObservation } from '@/lib/calibrationTypes'
import { countValidSolveFrames, filterFramesForSolve } from '@/lib/calibrationValidFrames'
import { DEFAULT_TARGET_CONFIG, normalizeHexTargetConfig, type HexTargetConfig } from '@/lib/hexTarget/config'
import { createHexTargetDesign, type HexTargetModel } from '@/lib/hexTarget/target'
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

/** The subset of run state that can change a calibration solve. */
type CalibrationSolveInput = {
  collection: Collection
  framePool: CalibrationFrameObservation[]
  frameSize: { width: number; height: number } | undefined
  cameraId: string | undefined
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
  framePool?: CalibrationFrameObservation[]
}

export type CalibrationRunContextValue = {
  run: () => CalibRun
  setRun: (v: CalibRun | ((prev: CalibRun) => CalibRun)) => void
  calib: () => CalibrationResult | undefined
  setCalib: (v: CalibrationResult | undefined) => void
  latestCalibration: () => CalibrationResult | undefined
  latestCalibrationMeta: () => CalibrationLatestMeta | undefined
  videoFrameSize: () => { width: number; height: number } | undefined
  setVideoFrameSize: (v: { width: number; height: number } | undefined) => void
  targetConfig: () => HexTargetConfig
  targetModel: () => HexTargetModel
  setTargetConfig: (config: HexTargetConfig) => void
  isSolving: () => boolean
  calibrationSessionActive: () => boolean
  resetSession: () => void
  startSession: () => void
}

const CalibrationRunContext = createContext<CalibrationRunContextValue>()

export function useCalibrationRun(): CalibrationRunContextValue {
  const value = useContext(CalibrationRunContext)
  if (!value) {
    throw new Error('useCalibrationRun must be used within CalibrationRunProvider')
  }
  return value
}

export function CalibrationRunProvider(props: ParentProps) {
  const cameraStream = useContext(CameraStreamContext)
  const selectedCameraDeviceId = () => cameraStream?.selectedCameraDeviceId()
  const [run, setRun] = createSignal<CalibRun>({ ...initialRun, stats: { ...initialCalibRunStats } })
  const [videoFrameSize, setVideoFrameSize] = createSignal<{ width: number; height: number }>()
  const [targetConfig, setTargetConfigSignal] = createSignal<HexTargetConfig>(DEFAULT_TARGET_CONFIG)
  const targetModel = createMemo(() => createHexTargetDesign(targetConfig()))
  const [solveInFlight, setSolveInFlight] = createSignal(0)
  const [calib, setCalibInner] = createSignal<CalibrationResult>()
  const [latestCalibration, writeLatestCalibration] = createSignal<CalibrationResult>()
  const [latestCalibrationMeta, writeLatestCalibrationMeta] = createSignal<CalibrationLatestMeta>()

  const setLatestCalibration = (value: CalibrationResult | undefined, meta?: CalibrationLatestMeta) => {
    writeLatestCalibration(value)
    writeLatestCalibrationMeta(value === undefined ? undefined : meta)
  }

  const setCalib = (result: CalibrationResult | undefined) => {
    setCalibInner(result)
    if (result === undefined) {
      setLatestCalibration(undefined, undefined)
      return
    }
    setLatestCalibration(result, {
      validSolveFrameCount: countValidSolveFrames(run().framePool, videoFrameSize()),
      video: videoFrameSize(),
      framePool: run().framePool,
    })
  }

  let pendingVersion = 0
  let lastSolveKey: string | undefined

  /** Discard any solve already in flight; its result no longer describes this run. */
  const invalidatePendingSolve = () => {
    pendingVersion += 1
    lastSolveKey = undefined
  }

  const updateCalib = async (
    collection: Collection,
    framePool: CalibrationFrameObservation[],
    frameSize: { width: number; height: number },
    cameraId: string | undefined,
  ) => {
    if (collection === 'idle') {
      invalidatePendingSolve()
      setCalib(undefined)
      return
    }
    const usableFrames = filterFramesForSolve(framePool, frameSize)
    if (usableFrames.length < 3) {
      // The frame set changed (for example a camera-resolution switch filtered
      // the old frames out). Any solve still in flight describes different
      // input now, so it must not publish a result.
      invalidatePendingSolve()
      setCalib({ kind: 'error', reason: 'too-few-views' })
      return
    }
    if (!cameraId) {
      return
    }

    const config = targetConfig()
    const solveKey = `${collection}|${cameraId}|${frameSize.width}x${frameSize.height}|${JSON.stringify(config)}|${usableFrames
      .map((frame) => `${frame.frameId}:${frame.points.length}`)
      .join(',')}`
    if (solveKey === lastSolveKey) {
      return
    }
    lastSolveKey = solveKey

    const currentVersion = ++pendingVersion
    try {
      setSolveInFlight((count) => count + 1)
      // Do not clone the full-resolution gray frames into the worker: the
      // solver only reads point correspondences, and the frame pool can hold
      // megabytes of image data that stays on this thread for export.
      const solveFrames = usableFrames.map(({ frameId, points }) => ({ frameId, points }))
      const result = await calibApi.solveCalibration([...targetModel().objectPoints], solveFrames, frameSize, cameraId)
      if (currentVersion === pendingVersion) {
        setCalib(result.kind === 'ok' ? { ...result, targetConfig: config } : result)
      }
    } catch (err) {
      console.error('[CalibrationRun] calibration failed', err)
      if (currentVersion === pendingVersion) {
        setCalib({ kind: 'error', reason: 'singular', details: String(err) })
      }
    } finally {
      setSolveInFlight((count) => Math.max(0, count - 1))
    }
  }

  /**
   * `run()` is a new object for every per-frame stats tick, but stats cannot
   * change a solve. This memo suppresses those notifications so the solve key
   * is not rebuilt (and the frame list not rescanned) on every camera frame.
   */
  const solveInput = createMemo<CalibrationSolveInput>(
    () => {
      const current = run()
      return {
        collection: current.collection,
        framePool: current.framePool,
        frameSize: videoFrameSize(),
        cameraId: selectedCameraDeviceId(),
      }
    },
    {
      equals: (previous, next) =>
        previous.collection === next.collection &&
        previous.framePool === next.framePool &&
        previous.frameSize?.width === next.frameSize?.width &&
        previous.frameSize?.height === next.frameSize?.height &&
        previous.cameraId === next.cameraId,
    },
  )

  createEffect(
    () => solveInput(),
    ({ collection, framePool, frameSize, cameraId }) => {
      if (frameSize) {
        void updateCalib(collection, framePool, frameSize, cameraId)
      }
    },
  )

  const isSolving = createMemo(() => solveInFlight() > 0)
  const calibrationSessionActive = createMemo(() => run().collection === 'running' || run().framePool.length > 0)

  const resetSession = () => {
    invalidatePendingSolve()
    setCalib(undefined)
    setRun({
      collection: 'idle',
      framePool: [],
      stats: { ...initialCalibRunStats },
    })
  }

  const startSession = () => {
    invalidatePendingSolve()
    setCalib(undefined)
    setRun({
      collection: 'running',
      framePool: [],
      stats: { ...initialCalibRunStats },
    })
  }

  const setTargetConfig = (next: HexTargetConfig) => {
    const normalized = normalizeHexTargetConfig(next)
    if (JSON.stringify(normalized) === JSON.stringify(targetConfig())) {
      return
    }
    setTargetConfigSignal(normalized)
    resetSession()
  }

  const value: CalibrationRunContextValue = {
    run,
    setRun,
    calib,
    setCalib,
    latestCalibration,
    latestCalibrationMeta,
    videoFrameSize,
    setVideoFrameSize,
    targetConfig,
    targetModel,
    setTargetConfig,
    isSolving,
    calibrationSessionActive,
    resetSession,
    startSession,
  }

  return <CalibrationRunContext value={value}>{props.children}</CalibrationRunContext>
}
