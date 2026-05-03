import { A } from '@solidjs/router'
import { Show, createEffect, createMemo, createSignal } from 'solid-js'

import { useCalibrationRun } from '@/components/calibration/CalibrationRunContext'
import { CalibrationLibraryPanel } from '@/components/calibration/CalibrationLibraryPanel'
import { useCalibrationLibrary } from '@/components/calibration/CalibrationLibraryContext'
import { downloadCalibrationOkJson } from '@/components/results/exportCalibrationJson'
import { initGPU } from '@/gpu/init'
import { createResultsCanvasPipeline, type ResultsCanvasPipeline } from '@/gpu/resultsCanvasPipeline'
import { writeResultsCameraTransform } from '@/gpu/pipelines/resultsCameraTransform'
import {
  markerCenterWritesForGpu,
  MAX_RESULTS_MARKER_POINTS,
  writeMarkerPassUniform,
} from '@/gpu/pipelines/resultsMarkerPipeline'
import {
  calibrationDefinedCornerCount,
  orthoExtentYForPoints,
} from '@/gpu/pipelines/resultsSceneCpu'
import {
  tagQuadDrawCountForGpu,
  tagQuadWritesForGpu,
} from '@/gpu/pipelines/resultsTagQuadsPipeline'
import { writeAxisPassUniform } from '@/gpu/pipelines/resultsAxesPipeline'
import { applyOrbitPitchVerticalPlane, applyOrbitYawWorldY } from '@/lib/orbitOrthoMath'
import type { Vec3Arg } from 'wgpu-matrix'
import { createDragHandler } from '@/utils/createDragHandler'
import { createElementSize } from '@/utils/createElementSize'
import { createPinchHandler } from '@/utils/createPinchHandler'
import type { CalibrationOk } from '@/workers/calibration.worker'
import type { CalibrationResult } from '@/workers/calibrationClient'

import styles from '@/components/results/ResultsView.module.css'

const { navigator } = globalThis

/** Unit direction from board origin toward the camera (initial framing). */
const DEFAULT_ORBIT_EYE_DIR: Vec3Arg = [0.563, 0.247, 0.788]

/** Workspace for orbit drag: yaw step → pitch step (avoid per-move vec3 allocations). */
const orbitAfterYaw: Vec3Arg = [0, 0, 0]
const orbitAfterPitch: Vec3Arg = [0, 0, 0]

type CalibrationScene = {
  ok: CalibrationOk
  baseOrthoExtentY: number
  centerWrites: ReturnType<typeof markerCenterWritesForGpu>
  tagQuadWrites: ReturnType<typeof tagQuadWritesForGpu>
  tagQuadDrawCount: number
}

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
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>()
  const canvasSize = createElementSize(canvasEl)
  const [gpuPack, setGpuPack] = createSignal<ResultsCanvasPipeline>()
  const [orbitEyeDir, setOrbitEyeDir] = createSignal<Vec3Arg>([
    DEFAULT_ORBIT_EYE_DIR[0]!,
    DEFAULT_ORBIT_EYE_DIR[1]!,
    DEFAULT_ORBIT_EYE_DIR[2]!,
  ])
  const [orbitZoom, setOrbitZoom] = createSignal(1)

  const startPinch = createPinchHandler((initEv) => {
    let prevDist = initEv.distance
    return {
      onPinchMove(ev) {
        const factor = prevDist > 1e-4 ? ev.distance / prevDist : 1
        setOrbitZoom((z) => Math.min(22, Math.max(0.15, z * factor)))
        prevDist = ev.distance
      },
    }
  })

  function onWheelResults(ev: WheelEvent) {
    ev.preventDefault()
    setOrbitZoom((z) => Math.min(22, Math.max(0.15, z * Math.exp(ev.deltaY * 0.0016))))
  }

  function onTwoFingerTouch(e: TouchEvent) {
    const el = canvasEl()
    if (!el || e.touches.length !== 2 || e.target !== el) {
      return
    }
    e.preventDefault()
    startPinch(e)
  }

  const calibrationScene = createMemo((): CalibrationScene | undefined => {
    const c = displayCalibration()
    if (!c || c.kind !== 'ok') {
      return undefined
    }
    return {
      ok: c,
      baseOrthoExtentY: orthoExtentYForPoints(c),
      centerWrites: markerCenterWritesForGpu(c),
      tagQuadWrites: tagQuadWritesForGpu(c),
      tagQuadDrawCount: tagQuadDrawCountForGpu(c),
    }
  })

  /** Reset when GPU context is recreated so storage is re-uploaded before draw */
  let lastCentersUploadedFor: CalibrationScene | undefined
  let lastTagQuadsUploadedFor: CalibrationScene | undefined

  let rafId = 0
  const stopRaf = () => {
    cancelAnimationFrame(rafId)
  }

  function tick(): void {
    const pip = gpuPack()
    if (!pip) {
      return
    }

    rafId = requestAnimationFrame(tick)

    const el = canvasEl()
    if (!el || el.width < 8 || el.height < 8) {
      return
    }

    const scene = calibrationScene()

    if (!scene) {
      pip.clearAttachments()
      return
    }

    const pointCount = Math.min(calibrationDefinedCornerCount(scene.ok), MAX_RESULTS_MARKER_POINTS)
    const tagCount = scene.tagQuadDrawCount

    if (lastCentersUploadedFor !== scene) {
      try {
        pip.centersBuf.write(scene.centerWrites)
        lastCentersUploadedFor = scene
      } catch (e) {
        console.warn('[ResultsView] center upload failed', e)
      }
    }

    if (lastTagQuadsUploadedFor !== scene) {
      try {
        pip.tagQuadsBuf.write(scene.tagQuadWrites)
        lastTagQuadsUploadedFor = scene
      } catch (e) {
        console.warn('[ResultsView] tag quad upload failed', e)
      }
    }

    writeResultsCameraTransform({
      aspectWidthOverHeight: el.width / el.height,
      orbitEyeDirUnit: orbitEyeDir(),
      baseOrthoExtentY: scene.baseOrthoExtentY,
      orthoZoom: orbitZoom(),
      viewportWidthPx: el.width,
      viewportHeightPx: el.height,
      cameraUniform: pip.cameraUniform,
    })
    writeMarkerPassUniform(pip.markerUniform, pointCount)
    writeAxisPassUniform(pip.axisUniform)

    pip.encodeScene(pointCount, tagCount)
  }

  createEffect(
    () => canvasEl(),
    (el) => {
      if (!el) {
        return
      }
      const webgpu = navigator.gpu
      if (!webgpu) {
        queueMicrotask(() => setGpuErr('WebGPU is not available in this browser.'))
        return
      }

      let canceled = false
      let pip: ResultsCanvasPipeline | undefined

      void (async () => {
        try {
          setGpuErr('')
          const rt = await initGPU()
          if (canceled) {
            return
          }

          const format = webgpu.getPreferredCanvasFormat()

          pip = createResultsCanvasPipeline(rt, el, format)

          el.addEventListener('touchstart', onTwoFingerTouch, { passive: false })
          el.addEventListener('wheel', onWheelResults, { passive: false })

          setGpuPack(pip)

          stopRaf()
          rafId = requestAnimationFrame(tick)
        } catch (e) {
          if (!canceled) {
            setGpuErr(`${e}`)
          }
        }
      })()

      return () => {
        canceled = true
        lastCentersUploadedFor = undefined
        lastTagQuadsUploadedFor = undefined
        stopRaf()
        el.removeEventListener('touchstart', onTwoFingerTouch)
        el.removeEventListener('wheel', onWheelResults)
        pip?.destroyTargets()
        setGpuPack(undefined)
      }
    },
  )

  createEffect(
    () => ({ el: canvasEl(), pip: gpuPack(), s: canvasSize() }),
    ({ el, pip, s }) => {
      if (!el || !pip || !s) {
        return
      }
      const w = Math.max(1, Math.round(s.widthPX))
      const h = Math.max(1, Math.round(s.heightPX))
      if (el.width === w && el.height === h) {
        return
      }
      el.width = w
      el.height = h
      pip.resize(w, h)
    },
  )

  const statsLine = createMemo(() => {
    const c = displayCalibration()
    if (!c || c.kind !== 'ok') {
      return ''
    }
    return `RMS ${c.rmsPx.toFixed(3)} px • ${c.extrinsics.length} views • ${calibrationDefinedCornerCount(c)} points`
  })

  const hasOkDisplay = createMemo(() => displayCalibration()?.kind === 'ok')

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

  const onDragStartCanvas = createDragHandler((down) => {
    let lastX = down.clientX
    let lastY = down.clientY
    const sens = 0.005
    return {
      onPointerMove(move) {
        const dx = move.clientX - lastX
        const dy = move.clientY - lastY
        lastX = move.clientX
        lastY = move.clientY
        setOrbitEyeDir((dir) => {
          const v = applyOrbitPitchVerticalPlane(
            applyOrbitYawWorldY(dir, -dx * sens, orbitAfterYaw),
            -dy * sens,
            orbitAfterPitch,
          )
          return [v[0]!, v[1]!, v[2]!]
        })
      },
    }
  })

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
              shows refined tag corners and axes. Nothing is wrong yet if you are still setting up—run Calibrate, or pick
              a saved calibration.
            </p>
          </Show>
          <Show when={hasOkDisplay()}>
            <p class={styles.successLine}>
              Drag to orbit the board (touch: two-finger drag; pinch or scroll to zoom). Use <strong>Export JSON</strong>{' '}
              when you are done.
            </p>
          </Show>
          <div class={styles.canvasViewport}>
            <canvas
              class={styles.canvas}
              ref={setCanvasEl}
              tabindex={0}
              aria-label="Calibration result orbit view"
              onPointerDown={(e) => onDragStartCanvas(e)}
            />
          </div>
          <CalibrationLibraryPanel />
        </div>
      </Show>
    </div>
  )
}
