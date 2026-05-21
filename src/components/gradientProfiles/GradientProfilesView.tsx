import { Errored, For, Show, createSignal } from 'solid-js'

import { useCalibrationRun } from '@/components/calibration/CalibrationRunContext'
import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { GradientProfilesPipeline } from '@/components/gradientProfiles/GradientProfilesPipeline'
import { GRADIENT_PROFILE_DISPLAY_MODES, type GradientProfileDisplayMode } from '@/gpu/gradientProfilePipeline'
import { LABEL_QUAD_REJECT_LEGEND } from '@/gpu/labelQuadReject'
import { LINE_REJECT_LEGEND } from '@/gpu/pipelines/lineFitDebugPipeline'

import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'
import styles from '@/components/gradientProfiles/GradientProfilesView.module.css'

export function GradientProfilesView() {
  const cam = useCameraStream()
  const runCtx = useCalibrationRun()
  const [displayMode, setDisplayMode] = createSignal<GradientProfileDisplayMode>('nms')
  const [validEdgeCount, setValidEdgeCount] = createSignal(0)
  const [frozen, setFrozen] = createSignal(false)
  const [selectedQuadId, setSelectedQuadId] = createSignal<number | undefined>(undefined)

  const log = (msg: string) => {
    console.log(msg)
  }

  return (
    <div class={styles.root}>
      <p class={styles.hint}>
        GPU detection pipeline tuning: display modes, edge threshold histogram, orientation and tag histograms, gradient
        profiles along fitted edges, and quad grid with tag decode. Undistort uses the latest successful calibration
        when available.
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
            selectedQuadId={selectedQuadId()}
            onQuadSelect={setSelectedQuadId}
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
                  <For each={GRADIENT_PROFILE_DISPLAY_MODES}>
                    {(item) => (
                      <button
                        type="button"
                        class={
                          displayMode() === item().mode ? pipelineStyles.modeButtonActive : pipelineStyles.modeButton
                        }
                        onClick={() => setDisplayMode(item().mode)}
                        title={item().title}
                      >
                        {item().label}
                      </button>
                    )}
                  </For>
                </div>
              </div>
            }
          />
          <p class={styles.status}>Valid edges (span ≥ 6px): {validEdgeCount()}</p>
          <Show when={displayMode() === 'lineRejects'}>
            <ul class={styles.debugLegend}>
              {LINE_REJECT_LEGEND.map((item) => (
                <li class={styles.debugLegendItem}>
                  <span class={styles.debugSwatch} style={{ background: item.color }} />
                  {item.label}
                </li>
              ))}
            </ul>
          </Show>
          <Show when={displayMode() === 'quadReject'}>
            <p class={styles.status}>Registered quads are subdued gray. Rejected labels:</p>
            <ul class={styles.debugLegend}>
              <li class={styles.debugLegendItem}>
                <span class={styles.debugSwatch} style={{ background: '#38383d' }} />
                Registered quad
              </li>
              {LABEL_QUAD_REJECT_LEGEND.map((item) => (
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
