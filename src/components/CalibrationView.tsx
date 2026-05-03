import { A } from '@solidjs/router'
import { Errored, Show, createMemo, createSignal } from 'solid-js'

import { CalibrateGuidancePanel, type GuidanceBand } from '@/components/calibration/CalibrateGuidancePanel'
import { CalibrationAdvancedMetrics, type CalibMetricsRow } from '@/components/calibration/CalibrationAdvancedMetrics'
import { useCalibrationLibrary } from '@/components/calibration/CalibrationLibraryContext'
import { useCalibrationRun } from '@/components/calibration/CalibrationRunContext'
import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { LiveCameraPipeline } from '@/components/camera/LiveCameraPipeline'
import type { DetectedQuad } from '@/gpu/contour'
import { calibrationQuadScore } from '@/lib/calibrationQuality'
import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import type { TagObservation, ImageTag } from '@/lib/calibrationTypes'
import type { Corners3 } from '@/lib/calibrationTypes'
import { countValidSolveFrames } from '@/lib/calibrationValidFrames'
import { isProgressShapedError, percentile, type SnapshotFeedback } from '@/lib/calibrationViewUtils'
import { formatFixed } from '@/lib/formatFixed'
import { learnLayoutFromFrame, type TargetLayout } from '@/lib/targetLayout'
import type { Mat3, Vec3 } from '@/workers/calibration.worker'

import styles from '@/components/CalibrationView.module.css'

function CalibrationView() {
  const cam = useCameraStream()
  const runCtx = useCalibrationRun()
  const calibrationLibrary = useCalibrationLibrary()
  const [snapshotFeedback, setSnapshotFeedback] = createSignal<SnapshotFeedback>({ kind: 'idle' })
  const [reproj, setReproj] = createSignal<{
    rms: number
    tagCount: number
    tiltDeg: number
    dist: number
  }>()
  const [currentTagged, setCurrentTagged] = createSignal<DetectedQuad[]>([])

  const calibratedExtrinsics = createMemo(() => {
    const c = runCtx.calib()
    if (!c || c.kind !== 'ok') {
      return undefined
    }
    const result: Map<number, { R: Mat3; t: Vec3 }> = new Map()
    for (const ext of c.extrinsics) {
      result.set(ext.frameId, { R: ext.R, t: ext.t })
    }
    return result
  })

  const calibratedLayout = createMemo<TargetLayout | undefined>(() => {
    const c = runCtx.calib()
    const lay = runCtx.layout()
    if (!c || c.kind !== 'ok' || !lay) {
      return lay
    }
    if (c.updatedTargets.length < 1) {
      return lay
    }
    const refined = new Map<number, Corners3>()
    for (const t of c.updatedTargets) {
      const { tagId, corners: objCorners } = t
      refined.set(tagId, [{ ...objCorners[0]! }, { ...objCorners[1]! }, { ...objCorners[2]! }, { ...objCorners[3]! }])
    }
    if (refined.size === 0) {
      return lay
    }
    return refined
  })

  const attemptAddPooledFrame = (tagged: DetectedQuad[], source: 'manual' | 'autoFirst'): boolean => {
    const r = runCtx.run()
    if (r.collection !== 'running') {
      if (source === 'manual') {
        setSnapshotFeedback({ kind: 'fail', message: 'Press Start before capturing snapshots.' })
      }
      return false
    }

    if (tagged.length < 1) {
      if (source === 'manual') {
        setSnapshotFeedback({
          kind: 'fail',
          message: 'No tag quads in this frame—check lighting, focus, and that the target fills the view.',
        })
      }
      return false
    }

    const tags: TagObservation[] = []
    for (const q of tagged) {
      if (typeof q.decodedTagId === 'number') {
        tags.push({
          tagId: q.decodedTagId,
          rotation: q.decodedRotation ?? 0,
          corners: q.corners,
          score: calibrationQuadScore(q),
        })
      }
    }

    if (tags.length < 1) {
      if (source === 'manual') {
        setSnapshotFeedback({
          kind: 'fail',
          message:
            'Need at least one decoded tag ID in view (not only ?). Improve lighting or center the board; ? marks are not used for layout yet.',
        })
      }
      return false
    }

    const tagsById = new Map<number, TagObservation>()
    for (const tag of tags) {
      const prev = tagsById.get(tag.tagId)
      if (!prev || tag.score > prev.score) {
        tagsById.set(tag.tagId, tag)
      }
    }
    const uniqueTags = [...tagsById.values()]

    if (!runCtx.layout()) {
      if (uniqueTags.length < 2) {
        if (source === 'manual') {
          setSnapshotFeedback({
            kind: 'fail',
            message:
              'Need two different tag IDs visible to learn the board layout before the first snapshot can be saved.',
          })
        }
        return false
      }
      const L = learnLayoutFromFrame(uniqueTags)
      if (!L) {
        if (source === 'manual') {
          setSnapshotFeedback({
            kind: 'fail',
            message: 'Could not learn layout from this frame—try a clearer view with two or more tags.',
          })
        }
        return false
      }
      runCtx.setLayout(L)
    }

    const frameTagsModel = uniqueTags.map(
      (t): ImageTag => ({
        tagId: t.tagId,
        corners: [
          { ...t.corners[0]!, score: t.score },
          { ...t.corners[1]!, score: t.score },
          { ...t.corners[2]!, score: t.score },
          { ...t.corners[3]!, score: t.score },
        ],
      }),
    )

    runCtx.setRun((prev) => {
      const { next, evicted } = mergeCalibrationFramesTopK(
        prev.framePool,
        [{ frameId: Date.now(), tags: frameTagsModel }],
        DEFAULT_CALIBRATION_TOP_K,
      )
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

  const onQuadDetection = (quads: DetectedQuad[]) => {
    runCtx.setRun((r) => ({
      ...r,
      stats: { ...r.stats, framesProcessed: r.stats.framesProcessed + 1 },
    }))
    setCurrentTagged(quads)

    const rNow = runCtx.run()
    if (rNow.collection === 'running' && rNow.framePool.length < 1) {
      attemptAddPooledFrame(quads, 'autoFirst')
    }
  }

  const handleSnapshotClick = () => {
    attemptAddPooledFrame(currentTagged(), 'manual')
  }

  /** Same notion as first snapshot: need two distinct decoded IDs to learn layout (not "?"). */
  const decodedUniqueTagCountOnFrame = createMemo(() => {
    const ids = new Set<number>()
    for (const q of currentTagged()) {
      if (typeof q.decodedTagId === 'number') {
        ids.add(q.decodedTagId)
      }
    }
    return ids.size
  })

  const canPressStart = createMemo(() => decodedUniqueTagCountOnFrame() >= 2)

  const uniqueTagCount = createMemo(() => {
    const s = new Set<number>()
    for (const f of runCtx.run().framePool) {
      for (const ft of f.tags) {
        s.add(ft.tagId)
      }
    }
    return s.size
  })

  const validSolveCount = createMemo(() => countValidSolveFrames(runCtx.run().framePool, runCtx.layout()))

  const canResetSession = createMemo(() => {
    const r = runCtx.run()
    if (r.collection === 'running') {
      return true
    }
    if (r.framePool.length > 0) {
      return true
    }
    if (runCtx.layout()) {
      return true
    }
    if (runCtx.calib()) {
      return true
    }
    return false
  })

  const canOpenResults = createMemo(() => {
    const c = runCtx.calib()
    return c?.kind === 'ok' && validSolveCount() >= 4
  })

  const resultsTooltip = createMemo(() => {
    if (canOpenResults()) {
      return 'View 3D calibration and export JSON'
    }
    const c = runCtx.calib()
    if (c?.kind === 'error' && !isProgressShapedError(c)) {
      return 'Fix the calibration or capture new views before opening Results'
    }
    const n = validSolveCount()
    if (c?.kind === 'ok' && n < 4) {
      return `${n} of 4 solver-ready views — add a few more snapshots`
    }
    return 'Results unlocks after calibration stabilizes with at least four usable views'
  })

  const guidanceCore = createMemo((): { band: GuidanceBand; lines: string[] } => {
    const r = runCtx.run()
    const lay = runCtx.layout()
    const c = runCtx.calib()
    const solving = runCtx.isSolving()
    const nValid = validSolveCount()
    const poolN = r.framePool.length

    if (solving) {
      return { band: 'progress', lines: ['Solving…', 'Keep capturing if you like—new frames queue for the next pass.'] }
    }

    if (r.collection === 'idle' && poolN < 1) {
      return {
        band: 'progress',
        lines: ['Sharp view of 2+ tags? Press Start. Vary pose between snapshots.'],
      }
    }

    if (r.collection === 'running' && poolN < 1) {
      return {
        band: 'progress',
        lines: ['First snapshot auto when 2+ tag IDs decode in one frame.', 'Move between shots for varied views.'],
      }
    }

    if (c?.kind === 'error' && isProgressShapedError(c)) {
      return {
        band: 'progress',
        lines: [
          `${nValid} solver-ready, ${poolN} in pool — need more overlap.`,
          'Add snapshots until the solve stabilizes (Results needs 4 good views when ok).',
        ],
      }
    }

    if (c?.kind === 'error') {
      const detail = c.details ? ` (${c.details})` : ''
      return {
        band: 'needs-attention',
        lines: [`Solve failed: ${c.reason}${detail}.`, 'Try new angles, check Target IDs, or Reset.'],
      }
    }

    if (c?.kind === 'ok' && nValid < 4) {
      return {
        band: 'progress',
        lines: [
          `${nValid}/4 overlapping views for Results—add snapshots from new poses.`,
          'Then open Results when ready.',
        ],
      }
    }

    if (c?.kind === 'ok') {
      const rms = formatFixed(c.rmsPx, 3)
      return {
        band: 'progress',
        lines: [`RMS ${rms} px · ${c.extrinsics.length} view(s).`, 'Open Results for the 3D view and JSON export.'],
      }
    }

    return {
      band: 'progress',
      lines: [
        `Pool ${poolN}; layout tags ${lay ? lay.size : '—'}.`,
        'Frames missing shared tags stay in the pool but may not feed the solver yet.',
      ],
    }
  })

  const guidance = createMemo((): { band: GuidanceBand; lines: string[] } => {
    const inner = guidanceCore()
    const lines = [...inner.lines]
    const snap = snapshotFeedback()
    if (snap.kind === 'fail') {
      lines.push(snap.message)
    }
    return { band: inner.band, lines }
  })

  const calibBlock = createMemo((): CalibMetricsRow => {
    const c = runCtx.calib()
    const res = runCtx.videoFrameSize()
    if (c?.kind === 'error') {
      return {
        solverSummary: `Solver: ${c.reason}${c.details ? ` (${c.details})` : ''}`,
      }
    }
    if (!c || c.kind !== 'ok' || !res) {
      return { solverSummary: 'Solver: —' }
    }
    const { width, height } = res
    const fovX = (2 * Math.atan((0.5 * width) / c.K.fx) * 180) / Math.PI
    const vals = c.perFrameRmsPx.map(([, v]) => v).sort((a, b) => a - b)
    const p50 = percentile(vals, 0.5)
    const p95 = percentile(vals, 0.95)
    return {
      solverSummary: `RMS ${formatFixed(c.rmsPx, 3)} px · med ${formatFixed(p50, 3)} · p95 ${formatFixed(p95, 3)} · ${c.extrinsics.length} view(s)`,
      fxfy: `${formatFixed(c.K.fx, 1)} / ${formatFixed(c.K.fy, 1)}`,
      cxyc: `${formatFixed(c.K.cx, 1)}, ${formatFixed(c.K.cy, 1)}`,
      fov: formatFixed(fovX, 1),
      ratio: (c.K.fy / c.K.fx).toFixed(3),
      off: `${formatFixed(c.K.cx - width / 2, 1)}, ${formatFixed(c.K.cy - height / 2, 1)}`,
    }
  })

  return (
    <div class={styles.root}>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <CameraStreamSelects />
          <LiveCameraPipeline
            displayMode="grid"
            showFallbacks={false}
            showHistogramCanvas={false}
            showFocusOverlay={runCtx.run().framePool.length < 1}
            focusBottomHint={() => {
              const r = runCtx.run()
              if (r.collection === 'idle' && r.framePool.length < 1) {
                return (
                  <>
                    Center the target in the guide, then press <b>Start</b>
                  </>
                )
              }
              return undefined
            }}
            stream={cam.stream()}
            onQuadDetection={onQuadDetection}
            onLog={console.log}
            liveCalibration={() => {
              const l = calibratedLayout()
              const c = runCtx.calib()
              if (!l || !c || c.kind !== 'ok') {
                return undefined
              }
              return { k: c.K, distortion: c.distortion, layout: l, extrinsics: calibratedExtrinsics() }
            }}
            onReprojectionFrame={(m) => setReproj(m)}
            onFrameSize={runCtx.setVideoFrameSize}
            onQuadSnapshotRequest={handleSnapshotClick}
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
              ? 'Add current frame to the pool'
              : canPressStart()
                ? 'Begin a new capture session'
                : 'Need at least two decoded tag IDs visible in the frame'
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
            runCtx.startSession()
            // Same frame as Start — do not wait for the next onQuadDetection tick.
            attemptAddPooledFrame(currentTagged(), 'autoFirst')
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
          title={canResetSession() ? 'Clear pool, layout, and latest Results data' : 'Nothing to reset yet'}
          onClick={() => {
            setReproj(undefined)
            setSnapshotFeedback({ kind: 'idle' })
            runCtx.resetSession()
          }}
        >
          Reset
        </button>
      </div>

      <div class={styles.guidanceSlot}>
        <CalibrateGuidancePanel band={() => guidance().band} lines={() => guidance().lines} />
      </div>

      <CalibrationAdvancedMetrics
        run={runCtx.run()}
        validSolveCount={validSolveCount()}
        uniqueTagCount={uniqueTagCount()}
        layoutSize={runCtx.layout()?.size}
        calibBlock={calibBlock()}
        reproj={reproj()}
      />
    </div>
  )
}

export { CalibrationView }
