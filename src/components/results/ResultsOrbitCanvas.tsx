import type { Component } from 'solid-js'
import { createEffect, createSignal } from 'solid-js'
import type { Vec3Arg } from 'wgpu-matrix'

import type { ResultsCalibrationScene } from '@/components/results/resultsCalibrationScene'
import { writeAxisPassUniform } from '@/gpu/pipelines/resultsAxesPipeline'
import { writeResultsCameraTransform } from '@/gpu/pipelines/resultsCameraTransform'
import { MAX_RESULTS_MARKER_POINTS, writeMarkerPassUniform } from '@/gpu/pipelines/resultsMarkerPipeline'
import { calibrationDefinedCornerCount } from '@/gpu/pipelines/resultsSceneCpu'
import {
  createResultsOrbitCanvasPipeline,
  type ResultsOrbitCanvasPipeline,
} from '@/gpu/resultsOrbitCanvasPipeline'
import { applyOrbitPitchVerticalPlane, applyOrbitYawWorldY } from '@/lib/orbitOrthoMath'
import type { TgpuRoot } from 'typegpu'

import { createDragHandler } from '@/utils/createDragHandler'
import { createElementSize } from '@/utils/createElementSize'
import { createPinchHandler } from '@/utils/createPinchHandler'

import styles from '@/components/results/ResultsView.module.css'

const DEFAULT_ORBIT_EYE_DIR: Vec3Arg = [0.563, 0.247, 0.788]
const orbitAfterYaw: Vec3Arg = [0, 0, 0]
const orbitAfterPitch: Vec3Arg = [0, 0, 0]

export type ResultsOrbitCanvasProps = {
  root: TgpuRoot
  format: GPUTextureFormat
  scene: ResultsCalibrationScene | undefined
}

export const ResultsOrbitCanvas: Component<ResultsOrbitCanvasProps> = (props) => {
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>()
  const canvasSize = createElementSize(canvasEl)
  const [pip, setPip] = createSignal<ResultsOrbitCanvasPipeline>()
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

  let lastCentersUploadedFor: ResultsCalibrationScene | undefined
  let lastTagQuadsUploadedFor: ResultsCalibrationScene | undefined

  createEffect(
    () => ({
      root: props.root,
      format: props.format,
      el: canvasEl(),
    }),
    ({ root, format, el }) => {
      if (!root || !el) {
        return
      }

      let rafId = 0
      const stopRaf = () => {
        cancelAnimationFrame(rafId)
      }

      function tick(): void {
        const p = pip()
        if (!p) {
          return
        }
        rafId = requestAnimationFrame(tick)

        const elNow = canvasEl()
        if (!elNow || elNow.width < 8 || elNow.height < 8) {
          return
        }

        const scene = props.scene
        if (!scene) {
          p.clearAttachments()
          return
        }

        const pointCount = Math.min(calibrationDefinedCornerCount(scene.ok), MAX_RESULTS_MARKER_POINTS)
        const tagCount = scene.tagQuadDrawCount

        if (lastCentersUploadedFor !== scene) {
          try {
            p.centersBuf.write(scene.centerWrites)
            lastCentersUploadedFor = scene
          } catch (e) {
            console.warn('[ResultsOrbitCanvas] center upload failed', e)
          }
        }

        if (lastTagQuadsUploadedFor !== scene) {
          try {
            p.tagQuadsBuf.write(scene.tagQuadWrites)
            lastTagQuadsUploadedFor = scene
          } catch (e) {
            console.warn('[ResultsOrbitCanvas] tag quad upload failed', e)
          }
        }

        writeResultsCameraTransform({
          aspectWidthOverHeight: elNow.width / elNow.height,
          orbitEyeDirUnit: orbitEyeDir(),
          baseOrthoExtentY: scene.baseOrthoExtentY,
          orthoZoom: orbitZoom(),
          viewportWidthPx: elNow.width,
          viewportHeightPx: elNow.height,
          cameraUniform: p.cameraUniform,
        })
        writeMarkerPassUniform(p.markerUniform, pointCount)
        writeAxisPassUniform(p.axisUniform)

        p.encodeScene(pointCount, tagCount)
      }

      lastCentersUploadedFor = undefined
      lastTagQuadsUploadedFor = undefined

      const pipeline = createResultsOrbitCanvasPipeline(root, el, format)
      setPip(pipeline)

      el.addEventListener('touchstart', onTwoFingerTouch, { passive: false })
      el.addEventListener('wheel', onWheelResults, { passive: false })

      stopRaf()
      rafId = requestAnimationFrame(tick)

      return () => {
        stopRaf()
        el.removeEventListener('touchstart', onTwoFingerTouch)
        el.removeEventListener('wheel', onWheelResults)
        pipeline.destroyTargets()
        setPip(undefined)
      }
    },
  )

  createEffect(
    () => ({
      el: canvasEl(),
      p: pip(),
      s: canvasSize(),
    }),
    ({ el, p, s }) => {
      if (!el || !p || !s) {
        return
      }
      const w = Math.max(1, Math.round(s.widthPX))
      const h = Math.max(1, Math.round(s.heightPX))
      if (el.width === w && el.height === h) {
        return
      }
      el.width = w
      el.height = h
      p.resize(w, h)
    },
  )

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
    <div class={styles.orbitPanel}>
      <div class={styles.canvasViewport}>
        <canvas
          class={styles.canvas}
          ref={setCanvasEl}
          tabindex={0}
          aria-label="Calibration result orbit view"
          onPointerDown={onDragStartCanvas}
        />
      </div>
    </div>
  )
}
