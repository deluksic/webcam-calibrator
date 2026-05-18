import { Errored, Show, createSignal } from 'solid-js'

import { useCalibrationRun } from '@/components/calibration/CalibrationRunContext'
import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { GradientProfilesPipeline } from '@/components/gradientProfiles/GradientProfilesPipeline'
import type { GradientProfileDisplayMode } from '@/gpu/gradientProfilePipeline'
import { LINE_FIT_DEBUG_LEGEND } from '@/gpu/pipelines/lineFitDebugPipeline'

import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'
import styles from '@/components/gradientProfiles/GradientProfilesView.module.css'

export function GradientProfilesView() {
  const cam = useCameraStream()
  const runCtx = useCalibrationRun()
  const [displayMode, setDisplayMode] = createSignal<GradientProfileDisplayMode>('nms')
  const [validEdgeCount, setValidEdgeCount] = createSignal(0)
  const [frozen, setFrozen] = createSignal(false)

  const log = (msg: string) => {
    console.log(msg)
  }

  return (
    <div class={styles.root}>
      <p class={styles.hint}>
        GPU detection pipeline tuning: display modes, edge threshold histogram, orientation and tag histograms, gradient
        profiles along fitted edges, and quad grid with tag decode. Undistort uses the latest successful calibration when
        available.
      </p>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <GradientProfilesPipeline
            displayMode={displayMode()}
            frozen={frozen()}
            showHistogramCanvas
            stream={cam.stream()}
            onLog={log}
            onValidEdgeCount={setValidEdgeCount}
            undistortParams={() => {
              const c = runCtx.calib()
              if (!c || c.kind !== 'ok') {
                return undefined
              }
              return { k: c.K, distortion: c.distortion }
            }}
            toolbar={
              <div class={styles.toolbar}>
                <CameraStreamSelects />
                <div class={styles.modeRow}>
                  <button
                    type="button"
                    class={frozen() ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                    onClick={() => setFrozen((f) => !f)}
                    title={frozen() ? 'Resume live processing' : 'Freeze frame for inspection'}
                  >
                    {frozen() ? 'Resume' : 'Pause'}
                  </button>
                </div>
                <div class={styles.modeRow}>
                  <button
                    type="button"
                    class={displayMode() === 'grayscale' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                    onClick={() => setDisplayMode('grayscale')}
                  >
                    Gray
                  </button>
                  <button
                    type="button"
                    class={displayMode() === 'undistort' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                    onClick={() => setDisplayMode('undistort')}
                  >
                    Undistort
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
                    class={displayMode() === 'quads' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                    onClick={() => setDisplayMode('quads')}
                  >
                    Quads
                  </button>
                  <button
                    type="button"
                    class={displayMode() === 'edgeLabels' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                    onClick={() => setDisplayMode('edgeLabels')}
                  >
                    Edge labels
                  </button>
                  <button
                    type="button"
                    class={
                      displayMode() === 'fittedLines' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton
                    }
                    onClick={() => setDisplayMode('fittedLines')}
                  >
                    Lines
                  </button>
                  <button
                    type="button"
                    class={displayMode() === 'quadGrid' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton}
                    onClick={() => setDisplayMode('quadGrid')}
                  >
                    Quad grid
                  </button>
                  <button
                    type="button"
                    class={
                      displayMode() === 'lineFitDebug' ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton
                    }
                    onClick={() => setDisplayMode('lineFitDebug')}
                  >
                    Line debug
                  </button>
                </div>
              </div>
            }
          />
          <p class={styles.status}>Valid edges (span ≥ 6px): {validEdgeCount()}</p>
          <Show when={displayMode() === 'lineFitDebug'}>
            <ul class={styles.debugLegend}>
              {LINE_FIT_DEBUG_LEGEND.map((item) => (
                <li class={styles.debugLegendItem}>
                  <span class={styles.debugSwatch} style={{ background: item.color }} />
                  {item.label}
                </li>
              ))}
            </ul>
          </Show>
        </div>
      </Errored>
    </div>
  )
}
