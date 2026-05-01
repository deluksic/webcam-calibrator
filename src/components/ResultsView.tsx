import { Show, createEffect, createMemo, createSignal } from 'solid-js'
import type { TgpuRoot } from 'typegpu'

import { useCalibrationLatest } from '@/components/calibration/CalibrationLatestContext'
import { initGPU } from '@/gpu/init'
import { clearResultsAttachments } from '@/gpu/resultsViz_clear'
import {
  MAX_RESULTS_MARKER_POINTS,
  allocAxisUni,
  allocMarkerUni,
  allocMarkersCenters,
  axisBindLayout,
  markersBindLayout,
} from '@/gpu/resultsVizLayouts'
import {
  RESULTS_MSAA_SAMPLE_COUNT,
  createAxesResultsPipeline,
  createMarkerResultsPipeline,
  destroyGpuTexture,
  encodeResultsCanvasFrame,
} from '@/gpu/resultsVizPipelines'
import { writeResultsMarkerAndAxisUniforms } from '@/lib/resultsFrameUniforms'
import {
  centroidOfUpdatedPoints,
  markerCenterWritesForGpu,
  orthoExtentYForPoints,
} from '@/lib/resultsSceneCpu'
import { createDragHandler } from '@/utils/createDragHandler'
import { createPinchHandler } from '@/utils/createPinchHandler'
import type { CalibrationOk } from '@/workers/calibration.worker'

import styles from '@/components/ResultsView.module.css'

const { navigator } = globalThis

function clampPitch(p: number) {
  return Math.min(1.42, Math.max(-1.36, p))
}

type CalibrationScene = {
  ok: CalibrationOk
  centroid: Float32Array
  baseOrthoExtentY: number
  centerWrites: ReturnType<typeof markerCenterWritesForGpu>
}

type GpuPack = {
  root: TgpuRoot
  context: GPUCanvasContext
  format: GPUTextureFormat
  msaaColorTex: GPUTexture
  depthTex: GPUTexture
  markerGpu: ReturnType<typeof createMarkerResultsPipeline>
  axesGpu: ReturnType<typeof createAxesResultsPipeline>
  markersBg: object
  axisBg: object
  markerUni: ReturnType<typeof allocMarkerUni>
  axisUni: ReturnType<typeof allocAxisUni>
  centersBuf: ReturnType<typeof allocMarkersCenters>
}

export function ResultsView() {
  const { latestCalibration } = useCalibrationLatest()

  const [gpuErr, setGpuErr] = createSignal('')
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>()
  const [gpuPack, setGpuPack] = createSignal<GpuPack>()
  const [orbitYaw, setOrbitYaw] = createSignal(0.62)
  const [orbitPitch, setOrbitPitch] = createSignal(0.25)
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
    const centroid = centroidOfUpdatedPoints(c)
    return {
      ok: c,
      centroid,
      baseOrthoExtentY: orthoExtentYForPoints(c, centroid),
      centerWrites: markerCenterWritesForGpu(c, centroid),
    }
  })

  /** Reset when GPU context is recreated so storage is re-uploaded before draw */
  let lastCentersUploadedFor: CalibrationScene | undefined

  let rafId = 0
  const stopRaf = () => {
    cancelAnimationFrame(rafId)
  }

  function tick(): void {
    const gpu = gpuPack()
    if (!gpu) {
      return
    }

    rafId = requestAnimationFrame(tick)

    const el = canvasEl()
    if (!el || el.width < 8 || el.height < 8) {
      return
    }

    const depthView = gpu.depthTex.createView()
    const msaaColorView = gpu.msaaColorTex.createView()

    const scene = calibrationScene()

    if (!scene) {
      clearResultsAttachments(gpu.root, gpu.context, msaaColorView, depthView)
      return
    }

    const pointCount = Math.min(scene.ok.updatedTargetPoints.length, MAX_RESULTS_MARKER_POINTS)

    if (lastCentersUploadedFor !== scene) {
      try {
        gpu.centersBuf.write(scene.centerWrites)
        lastCentersUploadedFor = scene
      } catch (e) {
        console.warn('[ResultsView] center upload failed', e)
      }
    }

    writeResultsMarkerAndAxisUniforms({
      aspectWidthOverHeight: el.width / el.height,
      yawRad: orbitYaw(),
      pitchRad: orbitPitch(),
      baseOrthoExtentY: scene.baseOrthoExtentY,
      orthoZoom: orbitZoom(),
      viewportWidthPx: el.width,
      viewportHeightPx: el.height,
      pointCount,
      markerUni: gpu.markerUni,
      axisUni: gpu.axisUni,
    })

    encodeResultsCanvasFrame(
      gpu.root,
      gpu.context,
      msaaColorView,
      depthView,
      gpu.markerGpu,
      gpu.axesGpu,
      gpu.markersBg,
      gpu.axisBg,
      pointCount,
    )
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
      let depthTexture: GPUTexture | undefined
      let msaaColorTexture: GPUTexture | undefined

      void (async () => {
        try {
          setGpuErr('')
          const rt = await initGPU()
          if (canceled) {
            return
          }

          const format = webgpu.getPreferredCanvasFormat()
          const ctx = el.getContext('webgpu')
          if (!ctx) {
            setGpuErr('Could not create WebGPU context.')
            return
          }
          rt.configureContext({ canvas: el, format, alphaMode: 'opaque' })

          const dpr = Math.min(2, globalThis.devicePixelRatio ?? 1)

          const resize = () => {
            if (canceled) {
              return
            }
            const rect = el.getBoundingClientRect()
            el.width = Math.max(1, Math.round(rect.width * dpr))
            el.height = Math.max(1, Math.round(rect.height * dpr))

            destroyGpuTexture(depthTexture)
            destroyGpuTexture(msaaColorTexture)
            msaaColorTexture = rt.device.createTexture({
              label: 'results-msaa',
              size: [el.width, el.height, 1],
              format,
              sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
            })
            depthTexture = rt.device.createTexture({
              label: 'results-depth',
              size: [el.width, el.height],
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
              format: 'depth24plus',
              sampleCount: RESULTS_MSAA_SAMPLE_COUNT,
            })
            setGpuPack((prev) =>
              prev
                ? {
                    ...prev,
                    depthTex: depthTexture!,
                    msaaColorTex: msaaColorTexture!,
                  }
                : prev,
            )
          }

          resize()
          resizeObs = new ResizeObserver(resize)
          resizeObs.observe(el)

          const markerGpu = createMarkerResultsPipeline(rt, format)
          const axesGpu = createAxesResultsPipeline(rt, format)
          const markerUni = allocMarkerUni(rt)
          const axisUni = allocAxisUni(rt)
          const centersBuf = allocMarkersCenters(rt)

          const markersBg = rt.createBindGroup(markersBindLayout, {
            markerUniform: markerUni,
            centers: centersBuf,
          })
          const axisBg = rt.createBindGroup(axisBindLayout, {
            axisUniform: axisUni,
          })

          el.addEventListener('touchstart', onTwoFingerTouch, { passive: false })
          el.addEventListener('wheel', onWheelResults, { passive: false })

          if (!depthTexture || !msaaColorTexture) {
            return
          }

          setGpuPack({
            root: rt,
            context: ctx,
            format,
            msaaColorTex: msaaColorTexture,
            depthTex: depthTexture,
            markerGpu,
            axesGpu,
            markersBg,
            axisBg,
            markerUni,
            axisUni,
            centersBuf,
          })

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
        stopRaf()
        resizeObs?.disconnect()
        el.removeEventListener('touchstart', onTwoFingerTouch)
        el.removeEventListener('wheel', onWheelResults)
        destroyGpuTexture(depthTexture)
        destroyGpuTexture(msaaColorTexture)
        setGpuPack(undefined)
      }
    },
  )

  const statsLine = createMemo(() => {
    const c = latestCalibration()
    if (!c || c.kind !== 'ok') {
      return ''
    }
    return `RMS ${c.rmsPx.toFixed(3)} px • ${c.extrinsics.length} views • ${c.updatedTargetPoints.length} points`
  })

  const showCalibrateHint = createMemo(() => latestCalibration()?.kind !== 'ok')

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
        // Turntable: drag right → positive yaw; drag down → lower pitch (look more from above)
        setOrbitYaw((y) => y + dx * sens)
        setOrbitPitch((p) => clampPitch(p - dy * sens))
      },
    }
  })

  return (
    <div class={styles.root}>
      <Show when={!gpuErr()} fallback={<p class={styles.placeholderText}>{gpuErr() || 'Unavailable'}</p>}>
        <div class={styles.panel}>
          <p class={[styles.meta, !statsLine() ? styles.metaHidden : false]}>{statsLine()}</p>
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
