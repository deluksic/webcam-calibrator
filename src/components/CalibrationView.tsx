import { Errored, For, createMemo, createStore } from 'solid-js'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { LiveCameraPipeline } from '@/components/camera/LiveCameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import type { DetectedQuad } from '@/gpu/contour'
import {
  acceptQuadForCalibration,
  calibrationQuadScore,
  frameHasDuplicateDecodedTagIds,
} from '@/lib/calibrationQuality'
import { DEFAULT_CALIBRATION_TOP_K, mergeCalibrationSamplesTopK } from '@/lib/calibrationTopK'
import type { CalibrationSample } from '@/lib/calibrationTypes'

import type { Resolution } from './camera/cameraStreamAcquire'
import { RESOLUTION_LADDER } from './camera/cameraStreamAcquire'

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

function CalibrationView() {
  const cam = useCameraStream()

  const [store, setStore] = createStore({
    collection: 'idle' as Collection,
    samples: [] as CalibrationSample[],
    stats: {
      framesProcessed: 0,
      framesAccepted: 0,
      frameRejections: 0,
      quadRejects: 0,
      evictions: 0,
    },
  })

  const displayMode = createMemo<DisplayMode>(() => 'grid')
  const showGrid = () => true
  const showFallbacks = () => false

  const devicesSorted = createMemo(async () => {
    const list = cam.devices()
    return [...list].sort((a, b) => deviceScore(b) - deviceScore(a))
  })

  const onQuadDetection = (quads: DetectedQuad[], meta: { frameId: number }) => {
    setStore((s) => {
      s.stats.framesProcessed += 1
    })
    if (store.collection !== 'running') {
      return
    }

    const decoded = quads.filter((q) => typeof q.decodedTagId === 'number')
    if (frameHasDuplicateDecodedTagIds(decoded)) {
      setStore((s) => {
        s.stats.frameRejections += 1
      })
      return
    }

    const incoming: CalibrationSample[] = []
    for (const q of quads) {
      if (!acceptQuadForCalibration(q)) {
        setStore((s) => {
          s.stats.quadRejects += 1
        })
        continue
      }
      if (typeof q.decodedTagId !== 'number') {
        continue
      }
      const rot = q.decodedRotation ?? 0
      incoming.push({
        frameId: meta.frameId,
        tagId: q.decodedTagId,
        rotation: rot,
        score: calibrationQuadScore(q),
      })
    }

    if (incoming.length === 0) {
      return
    }

    setStore((s) => {
      s.stats.framesAccepted += 1
    })
    const { next, evicted } = mergeCalibrationSamplesTopK(store.samples, incoming, DEFAULT_CALIBRATION_TOP_K)
    if (evicted > 0) {
      setStore((s) => {
        s.stats.evictions += evicted
      })
    }
    setStore((s) => {
      s.samples = next
    })
  }

  const uniqueTagCount = createMemo(() => {
    const ids = new Set(store.samples.map((s) => s.tagId))
    return ids.size
  })

  return (
    <div class={styles.root}>
      <p class={styles.hint}>
        Use valid AprilTags with <strong>unique</strong> IDs on a stiff, static target.
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
            <For each={Object.keys(RESOLUTION_LADDER)} keyed={false}>
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
          />
        </div>
      </Errored>

      <div class={styles.controls}>
        <button
          type="button"
          class={store.collection === 'running' ? styles.btnActive : styles.btn}
          onClick={() =>
            setStore((s) => {
              if (s.collection === 'idle' || s.collection === 'paused') {
                s.collection = 'running'
              }
            })
          }
        >
          Start
        </button>
        <button
          type="button"
          class={[styles.btn, store.collection !== 'running' && styles.btnDisabled]}
          disabled={store.collection !== 'running'}
          onClick={() =>
            setStore((s) => {
              if (s.collection === 'running') {
                s.collection = 'paused'
              }
            })
          }
        >
          Pause
        </button>
        <button
          type="button"
          class={styles.btn}
          onClick={() => {
            setStore((s) => {
              s.collection = 'idle'
              s.samples = []
              s.stats.framesProcessed = 0
              s.stats.framesAccepted = 0
              s.stats.frameRejections = 0
              s.stats.quadRejects = 0
              s.stats.evictions = 0
            })
          }}
        >
          Reset
        </button>
      </div>

      <div class={styles.stats}>
        <div>
          Pool: {store.samples.length} / {DEFAULT_CALIBRATION_TOP_K}
        </div>
        <div>Unique tag IDs: {uniqueTagCount()}</div>
        <div>Frames processed: {store.stats.framesProcessed}</div>
        <div>Frames accepted: {store.stats.framesAccepted}</div>
        <div>Frame rejections: {store.stats.frameRejections}</div>
        <div>Quad rejects: {store.stats.quadRejects}</div>
        <div>Top-K evictions: {store.stats.evictions}</div>
      </div>
    </div>
  )
}

export { CalibrationView }
