import { A } from '@solidjs/router'
import { Show, createEffect, createMemo, createSignal } from 'solid-js'

import { useCalibrationLibrary } from '@/components/calibration/CalibrationLibraryContext'
import { CalibrationLibraryPanel } from '@/components/calibration/CalibrationLibraryPanel'
import { useCalibrationRun } from '@/components/calibration/CalibrationRunContext'
import { ResultsDistortionCanvas } from '@/components/results/ResultsDistortionCanvas'
import { ResultsOrbitCanvas } from '@/components/results/ResultsOrbitCanvas'
import { downloadCalibrationOkJson } from '@/components/results/exportCalibrationJson'
import { buildResultsCalibrationScene } from '@/components/results/resultsCalibrationScene'
import { initGPU } from '@/gpu/init'
import { calibrationDefinedCornerCount } from '@/gpu/pipelines/resultsSceneCpu'
import type { TgpuRoot } from 'typegpu'

import type { CalibrationOk } from '@/workers/calibration.worker'
import type { CalibrationResult } from '@/workers/calibrationClient'

import styles from '@/components/results/ResultsView.module.css'

const { navigator } = globalThis

export function ResultsView() {
  const { latestCalibration, latestCalibrationMeta } = useCalibrationRun()
  const calibrationLibrary = useCalibrationLibrary()

  const displayCalibration = createMemo((): CalibrationResult | undefined => {
    const sid = calibrationLibrary.selectedId()
    if (sid) {
      const e = calibrationLibrary.entries().find((x) => x.id === sid)
      if (e?.result.kind === 'ok') {
        return e.result
      }
    }
    return latestCalibration()
  })

  const [gpuErr, setGpuErr] = createSignal('')
  const [gpuRoot, setGpuRoot] = createSignal<TgpuRoot>()
  const [canvasFormat, setCanvasFormat] = createSignal<GPUTextureFormat>('bgra8unorm')

  createEffect(
    () => null,
    () => {
      let canceled = false
      const webgpu = navigator.gpu
      if (!webgpu) {
        queueMicrotask(() => setGpuErr('WebGPU is not available in this browser.'))
        return () => {
          canceled = true
        }
      }
      void (async () => {
        try {
          setGpuErr('')
          const rt = await initGPU()
          if (canceled) {
            return
          }
          setCanvasFormat(webgpu.getPreferredCanvasFormat())
          setGpuRoot(rt)
        } catch (e) {
          if (!canceled) {
            setGpuErr(`${e}`)
          }
        }
      })()
      return () => {
        canceled = true
      }
    },
  )

  const calibrationScene = createMemo(() => {
    const c = displayCalibration()
    if (!c || c.kind !== 'ok') {
      return undefined
    }
    return buildResultsCalibrationScene(c)
  })

  const displayOk = createMemo((): CalibrationOk | undefined => {
    const c = displayCalibration()
    return c?.kind === 'ok' ? c : undefined
  })

  const statsLine = createMemo(() => {
    const c = displayCalibration()
    if (!c || c.kind !== 'ok') {
      return ''
    }
    return `RMS ${c.rmsPx.toFixed(3)} px • ${c.extrinsics.length} views • ${calibrationDefinedCornerCount(c)} points`
  })

  const hasOkDisplay = createMemo(() => displayCalibration()?.kind === 'ok')

  const videoSize = createMemo(() => {
    const c = displayCalibration()
    if (c?.kind === 'ok') {
      return c.imageSize
    }
    return undefined
  })

  const canExportCalibrationJson = createMemo(() => {
    const c = displayCalibration()
    return c !== undefined && c.kind === 'ok'
  })

  const canSaveToLibrary = createMemo(() => {
    if (calibrationLibrary.selectedId()) {
      return false
    }
    const c = latestCalibration()
    if (!c || c.kind !== 'ok') {
      return false
    }
    const nValid = latestCalibrationMeta()?.validSolveFrameCount ?? c.extrinsics.length
    return nValid >= 4
  })

  function saveCurrentSolveToLibrary() {
    if (!canSaveToLibrary()) {
      return
    }
    const c = latestCalibration()
    if (!c || c.kind !== 'ok') {
      return
    }
    const meta = latestCalibrationMeta()
    calibrationLibrary.addFromCurrentSolve({
      result: c,
      validSolveFrameCount: meta?.validSolveFrameCount ?? c.extrinsics.length,
      video: meta?.video,
    })
  }

  function exportCalibrationJson() {
    const c = displayCalibration()
    if (c === undefined || c.kind !== 'ok') {
      return
    }
    downloadCalibrationOkJson(c)
  }

  return (
    <div class={styles.root}>
      <Show when={!gpuErr()} fallback={<p class={styles.placeholderText}>{gpuErr() || 'Unavailable'}</p>}>
        <div class={styles.panel}>
          <div class={styles.toolbar}>
            <p class={[styles.meta, !statsLine() ? styles.metaHidden : false]}>{statsLine()}</p>
            <div class={styles.toolbarActions}>
              <Show when={!calibrationLibrary.selectedId()}>
                <A
                  href="/calibrate"
                  class={styles.continueCalibration}
                  title="Open the live camera capture and calibration flow"
                >
                  Continue Calibration
                </A>
              </Show>
              <Show when={calibrationLibrary.selectedId()}>
                <button
                  type="button"
                  class={styles.secondaryBtn}
                  title="Show the latest calibration from Calibrate when it is available"
                  onClick={() => calibrationLibrary.setSelectedId(undefined)}
                >
                  Use latest solve
                </button>
              </Show>
              <Show when={!calibrationLibrary.selectedId()}>
                <button
                  type="button"
                  class={styles.exportJsonBtn}
                  disabled={!canSaveToLibrary()}
                  title={
                    canSaveToLibrary()
                      ? 'Copy this calibration into Saved calibrations below'
                      : 'Need a finished calibration with at least four solver-ready views from Calibrate'
                  }
                  onClick={() => saveCurrentSolveToLibrary()}
                >
                  Save to library
                </button>
              </Show>
              <button
                type="button"
                class={styles.exportJsonBtn}
                disabled={!canExportCalibrationJson()}
                onClick={exportCalibrationJson}
              >
                Export JSON
              </button>
            </div>
          </div>
          <Show when={!hasOkDisplay()}>
            <p class={styles.hint}>
              When calibration data is available (from <strong>Calibrate</strong> or a saved entry below), this view
              shows refined tag corners and axes. Nothing is wrong yet if you are still setting up—run Calibrate, or
              pick a saved calibration.
            </p>
          </Show>
          <Show when={hasOkDisplay()}>
            <p class={styles.successLine}>
              Drag to orbit the board (touch: two-finger drag; pinch or scroll to zoom). The panel on the right shows
              the distortion field. Use <strong>Export JSON</strong> when you are done.
            </p>
          </Show>
          <Show when={gpuRoot()}>
            <div class={styles.resultsCanvasRow}>
              <ResultsOrbitCanvas root={gpuRoot()!} format={canvasFormat()} scene={calibrationScene()} />
              <ResultsDistortionCanvas
                root={gpuRoot()!}
                format={canvasFormat()}
                ok={displayOk()}
                imageSize={videoSize()}
              />
            </div>
          </Show>
          <CalibrationLibraryPanel />
        </div>
      </Show>
    </div>
  )
}
