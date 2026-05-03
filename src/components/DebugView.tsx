import { Errored, For, createSignal } from 'solid-js'

import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { LiveCameraPipeline } from '@/components/camera/LiveCameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'
import styles from '@/components/DebugView.module.css'

export function DebugView() {
  const cam = useCameraStream()
  const [logs, setLogs] = createSignal<string[]>([])
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>('grid')
  const [showFallbacks, setShowFallbacks] = createSignal(false)

  const log = (msg: string) => {
    Promise.resolve().then(() => {
      setLogs((prev) => [...prev.slice(-8), `${new Date().toISOString().slice(11, 19)} ${msg}`])
      console.log(msg)
    })
  }

  return (
    <div class={styles.root}>
      <p class={styles.hint}>Pipeline modes, histogram, and log tail for tuning detection.</p>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <div class={styles.debugToolbar}>
            <CameraStreamSelects />
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
            showFallbacks={showFallbacks()}
            showHistogramCanvas
            stream={cam.stream()}
            onLog={log}
            liveCalibration={() => undefined}
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
