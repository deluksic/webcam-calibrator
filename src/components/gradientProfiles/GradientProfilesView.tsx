import { Errored, createSignal } from 'solid-js'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { GradientProfilesPipeline } from '@/components/gradientProfiles/GradientProfilesPipeline'
import type { GradientProfileDisplayMode } from '@/gpu/gradientProfilePipeline'

import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'
import styles from '@/components/gradientProfiles/GradientProfilesView.module.css'

export function GradientProfilesView() {
  const cam = useCameraStream()
  const [displayMode, setDisplayMode] = createSignal<GradientProfileDisplayMode>('nms')
  const [validEdgeCount, setValidEdgeCount] = createSignal(0)

  const log = (msg: string) => {
    console.log(msg)
  }

  return (
    <div class={styles.root}>
      <p class={styles.hint}>
        Live edge gradient profiles along each detected edge (normal = black → white). Move the board to compare blurry
        vs sharp edges.
      </p>
      <Errored fallback={(err) => <p class={styles.error}>Camera: {String(err)}</p>}>
        <div class={styles.cameraBlock}>
          <GradientProfilesPipeline
            displayMode={displayMode()}
            showHistogramCanvas
            stream={cam.stream()}
            onLog={log}
            onValidEdgeCount={setValidEdgeCount}
            toolbar={
              <div class={styles.toolbar}>
                <CameraStreamSelects />
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
                </div>
              </div>
            }
          />
          <p class={styles.status}>Valid edges (span ≥ 6px): {validEdgeCount()}</p>
        </div>
      </Errored>
    </div>
  )
}
