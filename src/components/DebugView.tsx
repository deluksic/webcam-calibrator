import { Errored, For, createMemo, createSignal } from 'solid-js'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { LiveCameraPipeline } from '@/components/camera/LiveCameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'

import type { Resolution } from './camera/cameraStreamAcquire'
import { RESOLUTION_LADDER } from './camera/cameraStreamAcquire'

import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'
import styles from '@/components/DebugView.module.css'

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

export function DebugView() {
  const cam = useCameraStream()
  const [logs, setLogs] = createSignal<string[]>([])
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('grid')
  const [showFallbacks, setShowFallbacks] = createSignal(false)
  const showGrid = () => true

  const log = (msg: string) => {
    Promise.resolve().then(() => {
      setLogs((prev) => [...prev.slice(-8), `${new Date().toISOString().slice(11, 19)} ${msg}`])
      console.log(msg)
    })
  }

  const devicesSorted = createMemo(async () => {
    const list = cam.devices()
    return [...list].sort((a, b) => deviceScore(b) - deviceScore(a))
  })

  return (
    <div class={styles.root}>
      <p class={styles.hint}>Pipeline modes, histogram, and log tail for tuning detection.</p>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <div class={styles.debugToolbar}>
            <select
              class={[pipelineStyles.cameraSelect, styles.debugCameraSelect]}
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
            <div class={styles.debugModeRow}>
              <button
                type="button"
                class={displayMode() === 'grayscale' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                onClick={() => setDisplayMode('grayscale')}
              >
                Gray
              </button>
              <button
                type="button"
                class={displayMode() === 'edges' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                onClick={() => setDisplayMode('edges')}
              >
                Edges
              </button>
              <button
                type="button"
                class={displayMode() === 'nms' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                onClick={() => setDisplayMode('nms')}
              >
                NMS
              </button>
              <button
                type="button"
                class={displayMode() === 'labels' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                onClick={() => setDisplayMode('labels')}
              >
                Labels
              </button>
              <button
                type="button"
                class={displayMode() === 'grid' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                onClick={() => setDisplayMode('grid')}
              >
                Grid
              </button>
              <label class={pipelineStyles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={showFallbacks()}
                  onChange={(e) => setShowFallbacks(e.currentTarget.checked)}
                />
                Fallbk
              </label>
              <button
                type="button"
                class={displayMode() === 'debug' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                onClick={() => setDisplayMode('debug')}
              >
                Debug
              </button>
            </div>
          </div>
          <LiveCameraPipeline
            displayMode={displayMode()}
            showGrid={showGrid()}
            showFallbacks={showFallbacks()}
            showHistogramCanvas
            stream={cam.stream()}
            trackSize={cam.trackSize()}
            onLog={log}
          />
        </div>
      </Errored>
      <div class={styles.logTail}>
        <For each={logs()} keyed={false}>
          {(line) => <div>{line()}</div>}
        </For>
      </div>
    </div>
  )
}
