import { Errored, For, createMemo, createSignal, createEffect } from 'solid-js'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { LiveCameraPipeline } from '@/components/camera/LiveCameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import type { DetectedQuad } from '@/gpu/contour'
import { calibrationQuadScore } from '@/lib/calibrationQuality'
import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import type { TagObservation, CalibrationFrameObservation, FramePoint } from '@/lib/calibrationTypes'
import type { Point3 } from '@/lib/geometry'
import { learnLayoutFromFrame, layoutToLabeledPoints, type TargetLayout } from '@/lib/targetLayout'
import type { Mat3, Vec3 } from '@/workers/calibration.worker'
import { calibApi, type CalibrationResult } from '@/workers/calibrationClient'

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
  manualSnapshots: number
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
    manualSnapshots: 0,
  },
}

const RES = RESOLUTION_LADDER

function resolutionLabel(res: string): string {
  const ideal = (RES as Record<string, typeof RES['medium']>)[res]
  return `${ideal?.width.ideal}×${ideal?.height.ideal ?? 0}`
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
  const [currentTagged, setCurrentTagged] = createSignal<DetectedQuad[]>([])
  const [videoFrameSize, setVideoFrameSize] = createSignal<{ width: number; height: number }>()
  const [run, setRun] = createSignal<CalibRun>({
    collection: 'idle',
    framePool: [],
    stats: { ...initialCalibRun.stats },
  })
  const [solveInFlight, setSolveInFlight] = createSignal(0)

  const displayMode = createMemo<DisplayMode>(() => 'grid')
  const isSolving = createMemo(() => solveInFlight() > 0)

  const devicesSorted = createMemo(() => {
    const list = cam.devices()
    return [...list].sort((a, b) => deviceScore(b) - deviceScore(a))
  })

  // DAG: calib is a pure derivation of (collection, layout, framePool).
  const calibSignal = createSignal<CalibrationResult | null>(null)
  const [calib, setCalib] = calibSignal

  // Track pending promise to avoid showing stale results
  let pendingVersion = 0
  let lastSolveKey: string | undefined
  const updateCalib = async (
    collection: string,
    lay: TargetLayout | undefined,
    framePool: CalibrationFrameObservation[],
    frameSize: { width: number; height: number },
  ) => {
    if (collection === 'idle' || !lay || framePool.length < 1) {
      lastSolveKey = undefined
      setCalib(null)
      return
    }
    // Collect all tagIds present in the layout
    const layoutTagIds = new Set<number>()
    for (const tagId of lay.keys()) {
      layoutTagIds.add(tagId)
    }
    // Filter framePool to only include points from tags in the layout
    const filteredPool: CalibrationFrameObservation[] = []
    for (const frame of framePool) {
      const filteredPoints = frame.framePoints.filter((fp) => {
        const tagId = Math.floor(fp.pointId / 10000)
        return layoutTagIds.has(tagId)
      })
      if (filteredPoints.length >= 8) {
        filteredPool.push({ frameId: frame.frameId, framePoints: filteredPoints })
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

    const layoutPoints = layoutToLabeledPoints(lay)
    const size = frameSize
    if (!size) {
      setCalib({ kind: 'error', reason: 'too-few-views', details: 'waiting-for-video-size' })
      return
    }
    const { width: w, height: h } = size
    const currentVersion = ++pendingVersion
    try {
      setSolveInFlight((n) => n + 1)
      const result = await calibApi.solveCalibration(layoutPoints, filteredPool, {
        width: w,
        height: h,
      })
      if (currentVersion === pendingVersion) {
        setCalib(result)
      }
    } catch (err) {
      console.error('[CalibrationView] calibration failed', err)
      if (currentVersion === pendingVersion) {
        setCalib({ kind: 'error', reason: 'singular', details: String(err) })
      }
    } finally {
      setSolveInFlight((n) => Math.max(0, n - 1))
    }
  }

  // Re-run calibration when inputs change
  createEffect(
    () => ({ run: run(), layout: layout(), frameSize: videoFrameSize() }),
    ({ run, layout, frameSize }) => {
      if (frameSize) {
        updateCalib(run.collection, layout, run.framePool, frameSize)
      }
    },
  )

  const calibratedExtrinsics = createMemo(() => {
    const c = calib()
    if (!c || c.kind !== 'ok') {
      return null
    }
    const result: Map<number, { R: Mat3; t: Vec3 }> = new Map()
    for (const ext of c.extrinsics) {
      result.set(ext.frameId, { R: ext.R, t: ext.t })
    }
    return result
  })

  const calibratedLayout = createMemo<TargetLayout | undefined>(() => {
    const c = calib()
    if (!c || c.kind !== 'ok' || !layout()) {
      return layout()
    }
    if (c.updatedTargetPoints.length < 4) {
      return layout()
    }
    const refined = new Map<number, Point3[]>()
    for (const p of c.updatedTargetPoints) {
      const tagId = Math.floor(p.pointId / 10000)
      const cornerId = p.pointId % 10000
      if (cornerId < 0 || cornerId > 3) {
        continue
      }
      const corners = refined.get(tagId) ?? [
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
        { x: 0, y: 0, z: 0 },
      ]
      corners[cornerId] = { x: p.position.x, y: p.position.y, z: p.position.z }
      refined.set(tagId, corners)
    }
    if (refined.size === 0) {
      return layout()
    }
    return refined as TargetLayout
  })

  const onQuadDetection = (quads: DetectedQuad[]) => {
    setRun((r) => ({
      ...r,
      stats: { ...r.stats, framesProcessed: r.stats.framesProcessed + 1 },
    }))
    setCurrentTagged(quads)
  }

  const handleSnapshotClick = () => {
    const tagged = currentTagged()
    console.log(`[CalibrationView snapshot] tagged.length=${tagged.length}`)
    const r = run()
    if (r.collection !== 'running') {
      console.log(`[CalibrationView snapshot] not running, skipping`)
      return
    }

    if (tagged.length < 1) {
      console.log(`[CalibrationView snapshot] no tagged quads, skipping`)
      return
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
      return
    }

    // Keep one observation per tag id (highest score) to avoid duplicate-id collapse.
    const tagsById = new Map<number, TagObservation>()
    for (const tag of tags) {
      const prev = tagsById.get(tag.tagId)
      if (!prev || tag.score > prev.score) {
        tagsById.set(tag.tagId, tag)
      }
    }
    const uniqueTags = [...tagsById.values()]

    // Build layout from the first 2 tags if needed
    if (!layout()) {
      if (uniqueTags.length < 2) {
        console.log(`[CalibrationView snapshot] need at least 2 unique tag IDs for layout, skipping`)
        return
      }
      const L = learnLayoutFromFrame(uniqueTags)
      if (L) {
        setLayout(L)
      } else {
        console.log(`[CalibrationView snapshot] failed to learn layout, skipping`)
        return
      }
    }

    // Convert tags to labeled points for this frame
    const framePoints: FramePoint[] = []
    for (const t of uniqueTags) {
      const pointId = t.tagId * 10000
      for (let j = 0; j < 4; j++) {
        framePoints.push({
          pointId: pointId + j,
          imagePoint: t.corners[j]!,
        })
      }
    }

    setRun((r) => {
      const { next, evicted } = mergeCalibrationFramesTopK(
        r.framePool,
        [{ frameId: Date.now(), framePoints }],
        DEFAULT_CALIBRATION_TOP_K,
      )
      return {
        ...r,
        framePool: next,
        stats: {
          ...r.stats,
          manualSnapshots: r.stats.manualSnapshots + 1,
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
    const res = videoFrameSize()
    if (!c || c.kind === 'error' || !res) {
      return {
        line1: c?.kind === 'error' ? `Solver: ${c.reason}${c.details ? ` (${c.details})` : ''}` : 'Solver: —',
        k: null as null,
        rms: null as null,
      }
    }
    const { width, height } = res
    const fovX = (2 * Math.atan((0.5 * width) / c.K.fx) * 180) / Math.PI
    const vals = c.perFrameRmsPx.map(([, v]) => v).sort((a, b) => a - b)
    const p50 = percentile(vals, 0.5)
    const p95 = percentile(vals, 0.95)
    return {
      line1: `ok  RMS ${fmt(c.rmsPx, 3)} px  med ${fmt(p50, 3)}  p95 ${fmt(p95, 3)}  views ${c.extrinsics.length}`,
      fov: fmt(fovX, 1),
      fxfy: `${fmt(c.K.fx, 1)} / ${fmt(c.K.fy, 1)}`,
      cxyc: `${fmt(c.K.cx, 1)}, ${fmt(c.K.cy, 1)}`,
      ratio: (c.K.fy / c.K.fx).toFixed(3),
      off: `${fmt(c.K.cx - width / 2, 1)}, ${fmt(c.K.cy - height / 2, 1)}`,
      rms: c.rmsPx,
    }
  })

  return (
    <div class={styles.root}>
      <p class={styles.hint}>
        Use valid AprilTags with <strong>unique</strong> IDs on a stiff, static target. Press <strong>Start</strong>{' '}
        with 2+ tags visible; move the camera for varied views (3+ frames) to solve intrinsics.
      </p>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <div class={styles.cameraSelectsRow}>
            <select
              class={styles.cameraSelect}
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
              class={styles.cameraSelect}
              value={cam.selectedResolution()}
              onChange={(e) => cam.setSelectedResolution(e.currentTarget.value as Resolution)}
            >
              <For each={Object.keys(RES)} keyed={false}>
                {(resolution) => <option value={resolution()}>{resolutionLabel(resolution())}</option>}
              </For>
            </select>
          </div>
          <LiveCameraPipeline
            displayMode={displayMode()}
            showFallbacks={false}
            showHistogramCanvas={false}
            stream={cam.stream()}
            onQuadDetection={onQuadDetection}
            onLog={console.log}
            liveCalibration={() => {
              const l = calibratedLayout()
              const c = calib()
              if (!l || !c || c.kind !== 'ok') {
                return undefined
              }
              return { k: c.K, distortion: c.distortion, layout: l, extrinsics: calibratedExtrinsics() }
            }}
            onReprojectionFrame={(m) => setReproj(m)}
            onFrameSize={setVideoFrameSize}
            onQuadSnapshotRequest={handleSnapshotClick}
          />
        </div>
      </Errored>

      <div class={styles.controls}>
        <button
          type="button"
          class={run().collection === 'running' ? styles.btnActive : styles.btn}
          disabled={isSolving()}
          onClick={() => {
            if (run().collection === 'running') {
              handleSnapshotClick()
              return
            }
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
          {run().collection === 'running' ? `Snapshot (${run().stats.manualSnapshots})` : 'Start'}
        </button>
        <button
          type="button"
          class={[styles.btn, (run().collection !== 'running' || isSolving()) && styles.btnDisabled]}
          disabled={run().collection !== 'running' || isSolving()}
          onClick={() => setRun((r) => (r.collection === 'running' ? { ...r, collection: 'paused' } : r))}
        >
          Pause
        </button>
        <button
          type="button"
          class={[styles.btn, isSolving() && styles.btnDisabled]}
          disabled={isSolving()}
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
        <div>{calibBlock().line1}</div>
        <div>
          <span style="color: var(--color-success)">ok</span> RMS{' '}
          <span>
            {(() => {
              const c = calib()
              return c?.kind === 'ok' ? fmt(c!.rmsPx, 3) : '—'
            })()}
          </span>
          <span style="color: var(--color-text-muted)"> px</span> med{' '}
          <span>
            {(() => {
              const c = calib()
              return c?.kind === 'ok'
                ? fmt(
                    percentile(
                      c.perFrameRmsPx.map(([, v]) => v).sort((a, b) => a - b),
                      0.5,
                    ),
                    3,
                  )
                : '—'
            })()}
          </span>
          {' p95 '}
          <span>
            {(() => {
              const c = calib()
              return c?.kind === 'ok'
                ? fmt(
                    percentile(
                      c.perFrameRmsPx.map(([, v]) => v).sort((a, b) => a - b),
                      0.95,
                    ),
                    3,
                  )
                : '—'
            })()}
          </span>
          {' views '}
          <span>
            {(() => {
              const c = calib()
              return c?.kind === 'ok' ? c!.extrinsics.length : '—'
            })()}
          </span>
        </div>
        <div>
          fx / fy:{' '}
          {(() => {
            const c = calib()
            return c?.kind === 'ok' ? calibBlock().fxfy : '—'
          })()}{' '}
          px
        </div>
        <div>
          cx, cy:{' '}
          {(() => {
            const c = calib()
            return c?.kind === 'ok' ? calibBlock().cxyc : '—'
          })()}
        </div>
        <div>
          H FOV (x):{' '}
          {(() => {
            const c = calib()
            return c?.kind === 'ok' ? calibBlock().fov : '—'
          })()}
          °
        </div>
        <div>
          fy/fx:{' '}
          {(() => {
            const c = calib()
            return c?.kind === 'ok' ? calibBlock().ratio : '—'
          })()}
        </div>
        <div>
          pp offset (cx-W/2, cy-H/2):{' '}
          {(() => {
            const c = calib()
            return c?.kind === 'ok' ? calibBlock().off : '—'
          })()}
        </div>
        <div class={styles.statsSection}>Live frame</div>
        <div>
          RMS: {reproj() ? fmt(reproj()!.rms, 3) : '—'} px | tags: {reproj() ? reproj()!.tagCount : '—'}
        </div>
        <div>
          ‖t‖: {reproj() ? fmt(reproj()!.dist, 3) : '—'} (tag units) | tilt:{' '}
          {reproj() ? fmt(reproj()!.tiltDeg, 1) : '—'}°
        </div>
      </div>
    </div>
  )
}

export { CalibrationView }
