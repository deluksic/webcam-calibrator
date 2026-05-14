import type { Component } from 'solid-js'
import { createEffect, createSignal } from 'solid-js'
import type { TgpuRoot } from 'typegpu'

import { updateDistortionUniform } from '@/gpu/pipelines/distortionFieldPipeline'
import {
  createResultsDistortionCanvasPipeline,
  type ResultsDistortionCanvasPipeline,
} from '@/gpu/resultsDistortionCanvasPipeline'
import { createElementSize } from '@/utils/createElementSize'
import type { CalibrationOk } from '@/workers/calibration.worker'

import styles from '@/components/results/ResultsView.module.css'

export type ResultsDistortionCanvasProps = {
  root: TgpuRoot
  format: GPUTextureFormat
  ok: CalibrationOk | undefined
  imageSize: { width: number; height: number } | undefined
}

export const ResultsDistortionCanvas: Component<ResultsDistortionCanvasProps> = (props) => {
  const [scaleExp, setScaleExp] = createSignal(0)
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>()
  const canvasSize = createElementSize(canvasEl)
  const [pip, setPip] = createSignal<ResultsDistortionCanvasPipeline>()

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

        const ok = props.ok
        const vs = props.imageSize
        const cs = canvasSize()
        if (!ok || !vs || !cs) {
          return
        }

        updateDistortionUniform(p.distortionUniform, {
          K: ok.K,
          distortion: ok.distortion,
          scale: 2 ** scaleExp(),
          videoWidth: vs.width,
          videoHeight: vs.height,
          canvasWidth: cs.widthPX,
          canvasHeight: cs.heightPX,
        })
        p.encodeDistortionFrame()
      }

      const pipeline = createResultsDistortionCanvasPipeline(root, el, format)
      setPip(pipeline)

      stopRaf()
      rafId = requestAnimationFrame(tick)

      return () => {
        stopRaf()
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
    },
  )

  return (
    <div class={styles.distortionPanel}>
      <div class={styles.canvasViewport}>
        <canvas class={styles.canvas} ref={setCanvasEl} tabindex={0} aria-label="Distortion field visualization" />
      </div>
      <label class={styles.scaleRow}>
        <span class={styles.distortionScaleLabel}>Distortion scale</span>
        <input
          type="range"
          class={styles.scaleSlider}
          min="-8"
          max="5"
          step="0.1"
          value={scaleExp()}
          onInput={(e) => setScaleExp(parseFloat(e.currentTarget.value))}
        />
        {(2 ** scaleExp()).toFixed(2)}x
      </label>
    </div>
  )
}
