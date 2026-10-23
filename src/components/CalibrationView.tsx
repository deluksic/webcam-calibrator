import { A, useSearchParams } from '@solidjs/router'
import { Errored, Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { CalibrateGuidancePanel, type GuidanceBand } from '@/components/calibration/CalibrateGuidancePanel'
import { CalibrationAdvancedMetrics, type CalibMetricsRow } from '@/components/calibration/CalibrationAdvancedMetrics'
import { useCalibrationLibrary } from '@/components/calibration/CalibrationLibraryContext'
import { useCalibrationRun } from '@/components/calibration/CalibrationRunContext'
import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { HexCameraPipeline, type HexDetectionMeta, type SnapshotCallback } from '@/components/camera/HexCameraPipeline'
import type { RoiStepTimings } from '@/gpu/roiLab/roiLabPipeline'
import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import type { CalibrationFrameObservation, CalibrationPointObservation } from '@/lib/calibrationTypes'
import { countValidSolveFrames } from '@/lib/calibrationValidFrames'
import { isProgressShapedError, percentile, type SnapshotFeedback } from '@/lib/calibrationViewUtils'
import { formatFixed } from '@/lib/formatFixed'
import { buildHexReprojection, type HexReprojectionResult } from '@/lib/hexReprojection'
import { hexTargetConfigFromSearchParams, TARGET_MIN_CAPTURE_POINTS } from '@/lib/hexTarget/config'
import type { HexObservedPoint, HexTargetMatch } from '@/lib/hexTarget/match'
import type { HexTargetModel } from '@/lib/hexTarget/target'
import { configureHexMatchWorker, disposeHexMatchWorker, matchHexTargetInWorker } from '@/workers/hexMatchClient'

import styles from '@/components/CalibrationView.module.css'

function uniqueDesignCount(points: readonly CalibrationPointObservation[]): number {
  const ids = new Set<number>()
  for (const point of points) {
    ids.add(point.designId)
  }
  return ids.size
}

function CalibrationView() {
  const cam = useCameraStream()
  const runCtx = useCalibrationRun()
  const calibrationLibrary = useCalibrationLibrary()
  const [searchParams] = useSearchParams<Record<string, string>>()
  const [snapshotFeedback, setSnapshotFeedback] = createSignal<SnapshotFeedback>({ kind: 'idle' })
  const [reproj, setReproj] = createSignal<HexReprojectionResult>()
  const [pipelineTimings, setPipelineTimings] = createSignal<RoiStepTimings>()
  const [currentMatch, setCurrentMatch] = createSignal<HexTargetMatch>()
  const [currentPoints, setCurrentPoints] = createSignal<CalibrationPointObservation[]>([])
  const [requestSnapshotFn, setRequestSnapshotFn] = createSignal<((callback: SnapshotCallback) => void) | undefined>()
  let autoFirstAttempted = false

  createEffect(
    () => {
      const params = new URLSearchParams()
      for (const [key, value] of Object.entries(searchParams)) {
        if (typeof value === 'string') {
          params.set(key, value)
        }
      }
      return params.toString()
    },
    (query) => {
      runCtx.setTargetConfig(hexTargetConfigFromSearchParams(new URLSearchParams(query)))
    },
  )

  createEffect(
    () => runCtx.targetConfig(),
    (config) => configureHexMatchWorker(config),
  )
  onCleanup(() => disposeHexMatchWorker())

  /**
   * Identity runs in a worker so the ROI pipeline can overlap it with the next
   * frame's GPU work. The resolved graph is then presented with the pixels it
   * was computed from instead of stalling the camera loop.
   */
  const matchTarget = (design: HexTargetModel, points: readonly HexObservedPoint[]) =>
    matchHexTargetInWorker(design.config, points, {
      depth: design.config.identityDepth,
      recovery: true,
    })

  const minimumCapturePoints = createMemo(() =>
    Math.min(TARGET_MIN_CAPTURE_POINTS, Math.max(8, runCtx.targetModel().pattern.points.length)),
  )

  const attemptAddFrame = (frame: CalibrationFrameObservation, source: 'manual' | 'autoFirst'): boolean => {
    const run = runCtx.run()
    if (run.collection !== 'running') {
      if (source === 'manual') {
        setSnapshotFeedback({ kind: 'fail', message: 'Press Start before capturing snapshots.' })
      }
      return false
    }
    // Point coordinates are image pixels. Stamp every frame with the frame size
    // so frames from a previous camera resolution cannot enter the same solve.
    const size = runCtx.videoFrameSize()
    const acceptedFrame: CalibrationFrameObservation = {
      ...frame,
      imageWidth: frame.imageWidth ?? size?.width,
      imageHeight: frame.imageHeight ?? size?.height,
    }
    if (uniqueDesignCount(acceptedFrame.points) < minimumCapturePoints()) {
      if (source === 'manual') {
        setSnapshotFeedback({
          kind: 'fail',
          message: `Need at least ${minimumCapturePoints()} identified target markers in this frame.`,
        })
      }
      return false
    }

    runCtx.setRun((prev) => {
      const { next, evicted } = mergeCalibrationFramesTopK(prev.framePool, [acceptedFrame], DEFAULT_CALIBRATION_TOP_K)
      return {
        ...prev,
        framePool: next,
        stats: {
          ...prev.stats,
          framesAccepted: prev.stats.framesAccepted + 1,
          evictions: prev.stats.evictions + evicted,
        },
      }
    })
    setSnapshotFeedback({ kind: 'idle' })
    return true
  }

  const onDetection = (match: HexTargetMatch, points: CalibrationPointObservation[], meta: HexDetectionMeta) => {
    setCurrentMatch(match)
    setCurrentPoints(points)
    runCtx.setRun((run) => ({
      ...run,
      stats: { ...run.stats, framesProcessed: run.stats.framesProcessed + 1 },
    }))

    const run = runCtx.run()
    if (
      run.collection === 'running' &&
      run.framePool.length < 1 &&
      !autoFirstAttempted &&
      uniqueDesignCount(points) >= minimumCapturePoints()
    ) {
      autoFirstAttempted = attemptAddFrame(
        { frameId: meta.frameId, points, imageWidth: meta.width, imageHeight: meta.height },
        'autoFirst',
      )
    }
  }

  const handleSnapshotClick = () => {
    const request = requestSnapshotFn()
    if (request) {
      request((frame) => {
        attemptAddFrame(frame, 'manual')
      })
      return
    }
    attemptAddFrame({ frameId: Date.now(), points: currentPoints() }, 'manual')
  }

  const currentMatchedCount = createMemo(() => uniqueDesignCount(currentPoints()))
  const canPressStart = createMemo(() => currentMatchedCount() >= minimumCapturePoints())

  // The run object changes on every stats tick; keep the pool identity
  // separate so expensive pool scans only rerun when frames are added.
  const framePool = createMemo(() => runCtx.run().framePool)

  const uniquePointCount = createMemo(() => {
    const ids = new Set<number>()
    for (const frame of framePool()) {
      for (const point of frame.points) {
        ids.add(point.designId)
      }
    }
    return ids.size
  })

  const validSolveCount = createMemo(() => countValidSolveFrames(framePool(), runCtx.videoFrameSize()))

  const canResetSession = createMemo(() => {
    const run = runCtx.run()
    return run.collection === 'running' || framePool().length > 0 || runCtx.calib() !== undefined
  })

  const canOpenResults = createMemo(() => {
    const calibration = runCtx.calib()
    return calibration?.kind === 'ok' && validSolveCount() >= 4
  })

  const resultsTooltip = createMemo(() => {
    if (canOpenResults()) {
      return 'View 3D calibration and export JSON'
    }
    const calibration = runCtx.calib()
    if (calibration?.kind === 'error' && !isProgressShapedError(calibration)) {
      return 'Fix the calibration or capture new views before opening Results'
    }
    const count = validSolveCount()
    if (calibration?.kind === 'ok' && count < 4) {
      return `${count} of 4 solver-ready views — add a few more snapshots`
    }
    return 'Results unlocks after calibration stabilizes with at least four usable views'
  })

  createEffect(
    () => ({ calibration: runCtx.calib(), points: currentPoints() }),
    ({ calibration, points }) => {
      if (calibration?.kind !== 'ok' || uniqueDesignCount(points) < 4) {
        setReproj(undefined)
        return
      }
      setReproj(buildHexReprojection(runCtx.targetModel(), calibration.K, calibration.distortion, points))
    },
  )

  const guidanceCore = createMemo((): { band: GuidanceBand; lines: string[] } => {
    const run = runCtx.run()
    const calibration = runCtx.calib()
    const solving = runCtx.isSolving()
    const nValid = validSolveCount()
    const poolN = run.framePool.length
    const matched = currentMatchedCount()

    if (solving) {
      return { band: 'progress', lines: ['Solving…', 'Keep capturing if you like—new frames queue for the next pass.'] }
    }

    if (run.collection === 'idle' && poolN < 1) {
      return {
        band: 'progress',
        lines: [
          `Point the camera at the hex target. Start when ${minimumCapturePoints()}+ markers are identified.`,
          `${matched} identified in this frame.`,
        ],
      }
    }

    if (run.collection === 'running' && poolN < 1) {
      return {
        band: 'progress',
        lines: [
          `First snapshot auto when ${minimumCapturePoints()}+ markers are identified in one frame.`,
          'Move between shots for varied viewpoints.',
        ],
      }
    }

    if (calibration?.kind === 'error' && isProgressShapedError(calibration)) {
      return {
        band: 'progress',
        lines: [
          `${nValid} solver-ready, ${poolN} in pool — need shared markers across views.`,
          'Add snapshots until the solve stabilizes (Results needs 4 good views when ok).',
        ],
      }
    }

    if (calibration?.kind === 'error') {
      const detail = calibration.details ? ` (${calibration.details})` : ''
      return {
        band: 'needs-attention',
        lines: [`Solve failed: ${calibration.reason}${detail}.`, 'Try new angles, check the target, or Reset.'],
      }
    }

    if (calibration?.kind === 'ok' && nValid < 4) {
      return {
        band: 'progress',
        lines: [
          `${nValid}/4 overlapping views for Results—add snapshots from new poses.`,
          'Then open Results when ready.',
        ],
      }
    }

    if (calibration?.kind === 'ok') {
      const rms = formatFixed(calibration.rmsPx, 3)
      return {
        band: 'progress',
        lines: [
          `RMS ${rms} px · ${calibration.extrinsics.length} view(s).`,
          'Open Results for the 3D view and JSON export.',
        ],
      }
    }

    return {
      band: 'progress',
      lines: [
        `Pool ${poolN}; ${uniquePointCount()} target points seen.`,
        'Frames missing shared markers stay in the pool but may not feed the solver yet.',
      ],
    }
  })

  const guidance = createMemo(() => {
    const inner = guidanceCore()
    const lines = [...inner.lines]
    const snap = snapshotFeedback()
    const match = currentMatch()
    if (match && match.summary.conflicts > 0) {
      lines.push(`${match.summary.conflicts} identity conflict(s) in this frame.`)
    }
    if (snap.kind === 'fail') {
      lines.push(snap.message)
    }
    return { band: inner.band, lines }
  })

  const calibBlock = createMemo((): CalibMetricsRow => {
    const calibration = runCtx.calib()
    const size = runCtx.videoFrameSize()
    if (calibration?.kind === 'error') {
      return {
        solverSummary: `Solver: ${calibration.reason}${calibration.details ? ` (${calibration.details})` : ''}`,
      }
    }
    if (!calibration || calibration.kind !== 'ok' || !size) {
      return { solverSummary: 'Solver: —' }
    }
    const { width } = size
    const fovX = (2 * Math.atan((0.5 * width) / calibration.K.fx) * 180) / Math.PI
    const values = calibration.perFrameRmsPx.map(([, value]) => value).sort((a, b) => a - b)
    const p50 = percentile(values, 0.5)
    const p95 = percentile(values, 0.95)
    return {
      solverSummary: `RMS ${formatFixed(calibration.rmsPx, 3)} px · med ${formatFixed(p50, 3)} · p95 ${formatFixed(p95, 3)} · ${calibration.extrinsics.length} view(s)`,
      fxfy: `${formatFixed(calibration.K.fx, 1)} / ${formatFixed(calibration.K.fy, 1)}`,
      cxyc: `${formatFixed(calibration.K.cx, 1)}, ${formatFixed(calibration.K.cy, 1)}`,
      fov: formatFixed(fovX, 1),
      ratio: (calibration.K.fy / calibration.K.fx).toFixed(3),
      off: `${formatFixed(calibration.K.cx - width / 2, 1)}, ${formatFixed(calibration.K.cy - size.height / 2, 1)}`,
    }
  })

  return (
    <div class={styles.root}>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <CameraStreamSelects />
          <HexCameraPipeline
            stream={cam.stream()}
            onDetection={onDetection}
            matchTarget={matchTarget}
            onFrameSize={runCtx.setVideoFrameSize}
            setRequestSnapshot={setRequestSnapshotFn}
            targetModel={runCtx.targetModel()}
            reprojection={reproj()}
            showFocusOverlay={runCtx.run().framePool.length < 1}
            onTimings={setPipelineTimings}
          />
        </div>
      </Errored>

      <div class={styles.controls}>
        <button
          type="button"
          class={[
            runCtx.run().collection === 'running' || (runCtx.run().collection === 'idle' && canPressStart())
              ? styles.btnActive
              : styles.btn,
            (runCtx.isSolving() || (runCtx.run().collection === 'idle' && !canPressStart())) && styles.btnDisabled,
          ]}
          disabled={runCtx.isSolving() || (runCtx.run().collection === 'idle' && !canPressStart())}
          title={
            runCtx.run().collection === 'running'
              ? 'Add the current frame to the pool'
              : canPressStart()
                ? 'Begin a new capture session'
                : `Need at least ${minimumCapturePoints()} identified target markers in the frame`
          }
          onClick={() => {
            if (runCtx.run().collection === 'running') {
              handleSnapshotClick()
              return
            }
            if (!canPressStart()) {
              return
            }
            setSnapshotFeedback({ kind: 'idle' })
            calibrationLibrary.setSelectedId(undefined)
            autoFirstAttempted = false
            runCtx.startSession()
            autoFirstAttempted = attemptAddFrame({ frameId: Date.now(), points: currentPoints() }, 'autoFirst')
          }}
        >
          {runCtx.run().collection === 'running' ? `Snapshot (${runCtx.run().stats.framesAccepted})` : 'Start'}
        </button>
        <Show
          when={canOpenResults() && !runCtx.isSolving()}
          fallback={
            <span
              class={[styles.btn, styles.btnDisabled, styles.resultsStub]}
              title={resultsTooltip()}
              aria-disabled="true"
            >
              Results
            </span>
          }
        >
          <A href="/results" class={[styles.btn, styles.resultsLink]} title={resultsTooltip()}>
            Results
          </A>
        </Show>
        <button
          type="button"
          class={[styles.btn, (runCtx.isSolving() || !canResetSession()) && styles.btnDisabled]}
          disabled={runCtx.isSolving() || !canResetSession()}
          title={canResetSession() ? 'Clear pool and latest Results data' : 'Nothing to reset yet'}
          onClick={() => {
            setReproj(undefined)
            setSnapshotFeedback({ kind: 'idle' })
            setCurrentPoints([])
            setCurrentMatch(undefined)
            autoFirstAttempted = false
            runCtx.resetSession()
          }}
        >
          Reset
        </button>
      </div>

      <div class={styles.guidanceSlot}>
        <CalibrateGuidancePanel band={guidance().band} lines={guidance().lines} />
      </div>

      <CalibrationAdvancedMetrics
        run={runCtx.run()}
        validSolveCount={validSolveCount()}
        uniqueTargetPointCount={uniquePointCount()}
        matchedPointCount={currentMatchedCount()}
        calibBlock={calibBlock()}
        reproj={reproj()}
        timings={pipelineTimings()}
      />
    </div>
  )
}

export { CalibrationView }
