import { Errored, For, createMemo, createSignal } from 'solid-js'
import type { DetectedQuad } from '@/gpu/contour'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { LiveCameraPipeline } from '@/components/camera/LiveCameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import {
  acceptQuadForCalibration,
  calibrationQuadScore,
  frameHasDuplicateDecodedTagIds,
} from '@/lib/calibrationQuality'
import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import { solveCalibration, type CalibrationResult } from '@/lib/calibrationSolve'
import type { TagObservation, LabeledPoint, CalibrationFrameObservation, FramePoint } from '@/lib/calibrationTypes'
import { learnLayoutFromFrame, layoutToLabeledPoints, type TargetLayout } from '@/lib/targetLayout'
import { RESOLUTION_LADDER, type Resolution } from './camera/cameraStreamAcquire'

import styles from '@/components/CalibrationView.module.css'
import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'

function deviceScore(d: MediaDeviceInfo): number {
  const label = d.label.toLowerCase()
  let score = 0
  if (label.includes('back') || label.includes('rear')) {
    score += 100
  }
  if (label.includes('wide')) {
    score += 50
  }
  if (label.includes('ultra')) {
    score += 30
  }
  if (label.includes('tele')) {
    score -= 20
  }
  if (label.includes('front') || label.includes('user')) {
    score -= 100
  }
  return score
}

type Collection = 'idle' | 'running' | 'paused'

type CalibRunStats = {
  framesProcessed: number
  framesAccepted: number
  frameRejections: number
  quadRejects: number
  evictions: number
}

type CalibRun = {
  collection: Collection
  framePool: CalibrationFrameObservation[]
  stats: CalibRunStats
}

const initialCalibRun: CalibRun = {
  collection: 'idle',
  framePool: [],
  stats: {
    framesProcessed: 0,
    framesAccepted: 0,
    frameRejections: 0,
    quadRejects: 0,
    evictions: 0,
  },
}

const RES = RESOLUTION_LADDER
function approxFrameSize(res: Resolution): { w: number; h: number } {
  return res === 'medium' ? { w: 1280, h: 720 } : { w: 640, h: 480 }
}

function fmt(n: number | undefined, d: number) {
  if (n === undefined || !Number.isFinite(n)) {
    return '—'
  }
  return n.toFixed(d)
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) {
    return 0
  }
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))
  return sorted[idx]!
}

function CalibrationView() {
  const cam = useCameraStream()
  const [layout, setLayout] = createSignal<TargetLayout | undefined>(undefined)
  const [reproj, setReproj] = createSignal<{
    rms: number
    tagCount: number
    tiltDeg: number
    dist: number
  } | null>(null)
  const [run, setRun] = createSignal<CalibRun>({
    collection: 'idle',
    framePool: [],
    stats: { ...initialCalibRun.stats },
  })

  const displayMode = createMemo<DisplayMode>(() => 'grid')
  const showGrid = () => true
  const showFallbacks = () => false

  const devicesSorted = createMemo(() => {
    const list = cam.devices()
    return [...list].sort((a, b) => deviceScore(b) - deviceScore(a))
  })

  const frameSizeApprox = createMemo(() => approxFrameSize(cam.selectedResolution() as Resolution))

  // DAG: calib is a pure derivation of (collection, layout, framePool).
  const calib = createMemo<CalibrationResult | null>(() => {
    const r = run()
    const lay = layout()
    if (r.collection === 'idle' || !lay || r.framePool.length < 1) {
      return null
    }
    // Collect all tagIds present in the layout
    const layoutTagIds = new Set<number>()
    for (const tagId of lay.keys()) {
      layoutTagIds.add(tagId)
    }
    // Filter framePool to only include points from tags in the layout
    const filteredPool: CalibrationFrameObservation[] = []
    for (const frame of r.framePool) {
      const filteredPoints = frame.framePoints.filter((fp) => {
        const tagId = Math.floor(fp.pointId / 10000)
        return layoutTagIds.has(tagId)
      })
      if (filteredPoints.length >= 8) {
        filteredPool.push({ frameId: frame.frameId, framePoints: filteredPoints })
      }
    }
    if (filteredPool.length < 3) {
      return null
    }
    const labeledPoints = layoutToLabeledPoints(lay)
    return solveCalibration(lay, labeledPoints, filteredPool)
  })

  const onQuadDetection = (quads: DetectedQuad[], meta: { frameId: number }) => {
    setRun((r) => ({
      ...r,
      stats: { ...r.stats, framesProcessed: r.stats.framesProcessed + 1 },
    }))
    if (run().collection === 'idle' || run().collection === 'paused') {
      return
    }

    const decoded = quads.filter((q) => typeof q.decodedTagId === 'number')
    if (frameHasDuplicateDecodedTagIds(decoded)) {
      setRun((r) => ({
        ...r,
        stats: { ...r.stats, frameRejections: r.stats.frameRejections + 1 },
      }))
      return
    }

    const tags: TagObservation[] = []
    let quadRejectDelta = 0
    for (const q of quads) {
      if (!acceptQuadForCalibration(q)) {
        quadRejectDelta += 1
        continue
      }
      if (typeof q.decodedTagId !== 'number') {
        continue
      }
      tags.push({
        tagId: q.decodedTagId,
        rotation: q.decodedRotation ?? 0,
        corners: q.corners,
        score: calibrationQuadScore(q),
      })
    }
    if (quadRejectDelta > 0) {
      setRun((r) => ({
        ...r,
        stats: { ...r.stats, quadRejects: r.stats.quadRejects + quadRejectDelta },
      }))
    }

    if (tags.length < 1) {
      return
    }

    // Convert tags to labeled points for this frame
    const framePoints: FramePoint[] = []
    for (const t of tags) {
      const pointId = t.tagId * 10000
      for (let j = 0; j < 4; j++) {
        framePoints.push({
          pointId: pointId + j,
          imagePoint: t.corners[j]!,
        })
      }
    }

    if (!layout()) {
      if (tags.length < 2) {
        return
      }
      const L = learnLayoutFromFrame(tags)
      if (L) {
        setLayout(L)
      } else {
        return
      }
    }

    const frame: CalibrationFrameObservation = { frameId: meta.frameId, framePoints }
    setRun((r) => {
      const { next, evicted } = mergeCalibrationFramesTopK(r.framePool, [frame], DEFAULT_CALIBRATION_TOP_K)
      return {
        ...r,
        framePool: next,
        stats: {
          ...r.stats,
          framesAccepted: r.stats.framesAccepted + 1,
          evictions: r.stats.evictions + evicted,
        },
      }
    })
  }

  const uniqueTagCount = createMemo(() => {
    const s = new Set<number>()
    for (const f of run().framePool) {
      for (const fp of f.framePoints) {
        s.add(Math.floor(fp.pointId / 10000))
      }
    }
    return s.size
  })

  const calibBlock = createMemo(() => {
    const c = calib()
    if (!c || c.kind === 'error') {
      return { line1: c?.kind === 'error' ? `Solver: ${c.reason}` : 'Solver: —', k: null as null, rms: null as null }
    }
    const { w, h } = frameSizeApprox()
    const fovX = (2 * Math.atan(0.5 * w / c.K.fx) * 180) / Math.PI
    const vals = [...c.perFrameRmsPx.values()].sort((a, b) => a - b)
    const p50 = percentile(vals, 0.5)
    const p95 = percentile(vals, 0.95)
    return {
      line1: `ok  RMS ${fmt(c.rmsPx, 3)} px  med ${fmt(p50, 3)}  p95 ${fmt(p95, 3)}  views ${c.homographies.length}`,
      fov: fmt(fovX, 1),
      fxfy: `${fmt(c.K.fx, 1)} / ${fmt(c.K.fy, 1)}`,
      cxyc: `${fmt(c.K.cx, 1)}, ${fmt(c.K.cy, 1)}`,
      ratio: (c.K.fy / c.K.fx).toFixed(3),
      off: `${fmt(c.K.cx - w / 2, 1)}, ${fmt(c.K.cy - h / 2, 1)}`,
      rms: c.rmsPx,
    }
  })

  return (
    <div class={styles.root}>
      <p class={styles.hint}>
        Use valid AprilTags with <strong>unique</strong> IDs on a stiff, static target. Press <strong>Start</strong> with
        2+ tags visible; move the camera for varied views (3+ frames) to solve intrinsics.
      </p>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <select
            class={[pipelineStyles.cameraSelect, styles.calibrateCameraSelect]}
            value={cam.selectedCameraDeviceId() ?? ''}
            onChange={(e) => cam.setSelectedCameraDeviceId(e.currentTarget.value)}
          >
            <For each={devicesSorted()}>
              {(item) => (
                <option value={item().deviceId}>{item().label || `Camera ${item().deviceId.slice(0, 8)}`}</option>
              )}
            </For>
          </select>
          <select
            class={pipelineStyles.cameraSelect}
            value={cam.selectedResolution()}
            onChange={(e) => cam.setSelectedResolution(e.currentTarget.value as Resolution)}
          >
            <For each={Object.keys(RES)} keyed={false}>
              {(resolution) => <option value={resolution()}>{resolution()}</option>}
            </For>
          </select>
          <LiveCameraPipeline
            displayMode={displayMode()}
            showGrid={showGrid()}
            showFallbacks={showFallbacks()}
            showHistogramCanvas={false}
            stream={cam.stream()}
            onQuadDetection={onQuadDetection}
            onLog={console.log}
            liveCalibration={() => {
              const l = layout()
              const c = calib()
              if (!l || !c || c.kind !== 'ok') {
                return undefined
              }
              return { k: c.K, layout: l }
            }}
            onReprojectionFrame={(m) => setReproj(m)}
          />
        </div>
      </Errored>

      <div class={styles.controls}>
        <button
          type="button"
          class={run().collection === 'running' ? styles.btnActive : styles.btn}
          onClick={() => {
            setLayout(undefined)
            setRun((r) => {
              if (r.collection === 'idle' || r.collection === 'paused') {
                return {
                  collection: 'running',
                  framePool: [],
                  stats: { ...initialCalibRun.stats },
                }
              }
              return r
            })
          }}
        >
          Start
        </button>
        <button
          type="button"
          class={[styles.btn, run().collection !== 'running' && styles.btnDisabled]}
          disabled={run().collection !== 'running'}
          onClick={() =>
            setRun((r) => (r.collection === 'running' ? { ...r, collection: 'paused' } : r))
          }
        >
          Pause
        </button>
        <button
          type="button"
          class={styles.btn}
          onClick={() => {
            setLayout(undefined)
            setReproj(null)
            setRun({
              collection: 'idle',
              framePool: [],
              stats: { ...initialCalibRun.stats },
            })
          }}
        >
          Reset
        </button>
      </div>

      <div class={styles.stats}>
        <div>
          Pool: {run().framePool.length} / {DEFAULT_CALIBRATION_TOP_K} frames
        </div>
        <div>Layout tags: {layout() ? layout()!.size : '—'}</div>
        <div>Unique tag IDs in pool: {uniqueTagCount()}</div>
        <div>Frames processed: {run().stats.framesProcessed}</div>
        <div>Frames accepted: {run().stats.framesAccepted}</div>
        <div>Frame rejections: {run().stats.frameRejections}</div>
        <div>Quad rejects: {run().stats.quadRejects}</div>
        <div>Top-K evictions: {run().stats.evictions}</div>
        <div class={styles.statsSection}>Pooled solve</div>
        <div>
          <span style="color: var(--color-success)">ok</span>{' '}
          RMS <span>{fmt(calib() && calib()!.kind === 'ok' ? calib()!.rmsPx : undefined, 3)}</span>
          <span style="color: var(--color-text-muted)"> px</span>{' '}
          med <span>{calib() && calib()!.kind === 'ok' ? fmt(percentile([...calib()!.perFrameRmsPx.values()].sort((a, b) => a - b), 0.5), 3) : '—'}</span>
          {' p95 '}
          <span>{calib() && calib()!.kind === 'ok' ? fmt(percentile([...calib()!.perFrameRmsPx.values()].sort((a, b) => a - b), 0.95), 3) : '—'}</span>
          {' views '}
          <span>{calib() && calib()!.kind === 'ok' ? calib()!.homographies.length : '—'}</span>
        </div>
        <div>
          fx / fy: {calib() && calib()!.kind === 'ok' ? calibBlock().fxfy : '—'} px
        </div>
        <div>cx, cy: {calib() && calib()!.kind === 'ok' ? calibBlock().cxyc : '—'}</div>
        <div>H FOV (x): {calib() && calib()!.kind === 'ok' ? calibBlock().fov : '—'}°</div>
        <div>fy/fx: {calib() && calib()!.kind === 'ok' ? calibBlock().ratio : '—'}</div>
        <div>pp offset (cx-W/2, cy-H/2): {calib() && calib()!.kind === 'ok' ? calibBlock().off : '—'}</div>
        <div class={styles.statsSection}>Live frame</div>
        <div>
          RMS: {reproj() ? fmt(reproj()!.rms, 3) : '—'} px | tags: {reproj() ? reproj()!.tagCount : '—'}
        </div>
        <div>
          ‖t‖: {reproj() ? fmt(reproj()!.dist, 3) : '—'} (tag units) | tilt: {reproj() ? fmt(reproj()!.tiltDeg, 1) : '—'}°
        </div>
      </div>
    </div>
  )
}

export { CalibrationView }
