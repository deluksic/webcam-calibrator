import { Show, createEffect, createMemo, createSignal } from 'solid-js'

import { useCalibrationLatest } from '@/components/calibration/CalibrationLatestContext'
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
  MAX_RESULTS_TAG_QUADS,
  tagQuadWritesForGpu,
} from '@/gpu/pipelines/resultsTagQuadsPipeline'
import { writeAxisPassUniform } from '@/gpu/pipelines/resultsAxesPipeline'
import { applyOrbitPitchVerticalPlane, applyOrbitYawWorldY } from '@/lib/orbitOrthoMath'
import type { Vec3Arg } from 'wgpu-matrix'
import { createDragHandler } from '@/utils/createDragHandler'
import { createPinchHandler } from '@/utils/createPinchHandler'
import type { CalibrationOk } from '@/workers/calibration.worker'

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
}

export function ResultsView() {
  const { latestCalibration } = useCalibrationLatest()

  const [gpuErr, setGpuErr] = createSignal('')
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>()
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
    const c = latestCalibration()
    if (!c || c.kind !== 'ok') {
      return undefined
    }
    return {
      ok: c,
      baseOrthoExtentY: orthoExtentYForPoints(c),
      centerWrites: markerCenterWritesForGpu(c),
      tagQuadWrites: tagQuadWritesForGpu(c),
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
    const tagCount = Math.min(scene.tagQuadWrites.length, MAX_RESULTS_TAG_QUADS)

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
      let resizeObs: ResizeObserver | undefined
      let pip: ResultsCanvasPipeline | undefined

      void (async () => {
        try {
          setGpuErr('')
          const rt = await initGPU()
          if (canceled) {
            return
          }

          const format = webgpu.getPreferredCanvasFormat()

          const dpr = Math.min(2, globalThis.devicePixelRatio ?? 1)

          pip = createResultsCanvasPipeline(rt, el, format)

          const resize = () => {
            if (canceled || !pip) {
              return
            }
            const rect = el.getBoundingClientRect()
            el.width = Math.max(1, Math.round(rect.width * dpr))
            el.height = Math.max(1, Math.round(rect.height * dpr))
            pip.resize(el.width, el.height)
          }

          resize()
          resizeObs = new ResizeObserver(resize)
          resizeObs.observe(el)

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
        resizeObs?.disconnect()
        el.removeEventListener('touchstart', onTwoFingerTouch)
        el.removeEventListener('wheel', onWheelResults)
        pip?.destroyTargets()
        setGpuPack(undefined)
      }
    },
  )

  const statsLine = createMemo(() => {
    const c = latestCalibration()
    if (!c || c.kind !== 'ok') {
      return ''
    }
    return `RMS ${c.rmsPx.toFixed(3)} px • ${c.extrinsics.length} views • ${calibrationDefinedCornerCount(c)} points`
  })

  const showCalibrateHint = createMemo(() => latestCalibration()?.kind !== 'ok')

  const canExportCalibrationJson = createMemo(() => {
    const c = latestCalibration()
    return c !== undefined && c.kind === 'ok'
  })

  function exportCalibrationJson() {
    const c = latestCalibration()
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
            <button
              type="button"
              class={styles.exportJsonBtn}
              disabled={!canExportCalibrationJson()}
              onClick={exportCalibrationJson}
            >
              Export JSON
            </button>
          </div>
          <Show when={showCalibrateHint()}>
            <p class={styles.hint}>
              After Calibrate reports a pooled ok solve, this view shows refined object points and axes. Reset on
              Calibrate clears results.
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
        </div>
      </Show>
    </div>
  )
}
