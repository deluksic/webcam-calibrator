import type { JSX } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { encodeGradientProfileCompute } from '@/gpu/gradientProfileComputeEncoding'
import {
  encodeGradientProfileCameraPresent,
  encodeGradientProfilePlotPresent,
  encodeOrientHistPresent,
} from '@/gpu/gradientProfilePresentEncoding'
import {
  ORIENT_HIST_CANVAS_HEIGHT,
  ORIENT_HIST_CANVAS_WIDTH,
} from '@/gpu/pipelines/orientHistVizPipeline'
import { createGradientProfilePipeline } from '@/gpu/gradientProfilePipeline'
import type { GradientProfileDisplayMode } from '@/gpu/gradientProfilePipeline'
import { initGPU } from '@/gpu/init'
import { computeThreshold, THRESHOLD_PERCENTILE } from '@/gpu/pipelines/histogramPipelines'
import { createElementSize } from '@/utils/createElementSize'
import { createFrameLoop } from '@/utils/createFrameLoop'

import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'

const { navigator, performance } = globalThis

export type GradientProfilesPipelineProps = {
  displayMode: GradientProfileDisplayMode
  showHistogramCanvas: boolean
  stream: MediaStream | undefined
  onLog: (msg: string) => void
  onValidEdgeCount?: (count: number) => void
  toolbar?: JSX.Element
}

export function GradientProfilesPipeline(props: GradientProfilesPipelineProps) {
  const [cameraCanvas, setCameraCanvas] = createSignal<HTMLCanvasElement>()
  const [orientHistCanvas, setOrientHistCanvas] = createSignal<HTMLCanvasElement>()
  const [profileCanvas, setProfileCanvas] = createSignal<HTMLCanvasElement>()
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>()
  const [threshold, setThreshold] = createSignal(0, { ownedWrite: true })

  const videoElement = createMemo(async () => {
    const canvas = cameraCanvas()
    if (!canvas || !props.stream) {
      return undefined
    }
    const el = document.createElement('video')
    el.muted = true
    el.srcObject = props.stream
    el.play()
    onCleanup(() => {
      el.pause()
      el.srcObject = null
    })
    return el
  })

  const [frameSize, setFrameSize] = createSignal<{ width: number; height: number } | undefined>()

  createEffect(videoElement, (video) => {
    if (!video) {
      return
    }
    const onResize = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        setFrameSize({ width: video.videoWidth, height: video.videoHeight })
      }
    }
    video.addEventListener('resize', onResize)
    onResize()
    return () => video.removeEventListener('resize', onResize)
  })

  const profileCanvasSize = createElementSize(profileCanvas)

  const gpu = createMemo(async () => {
    try {
      return await initGPU()
    } catch (e) {
      props.onLog(`GPU init failed: ${e}`)
      return undefined
    }
  })

  createMemo(async () => {
    const video = videoElement()
    const size = frameSize()
    const camCanvas = cameraCanvas()
    const profCanvas = profileCanvas()
    const orientCanvas = orientHistCanvas()
    if (!video || !size || !camCanvas || !orientCanvas || !profCanvas) {
      return undefined
    }

    const g = gpu()
    if (!g) {
      return undefined
    }

    let disposed = false
    let frameLoop: ReturnType<typeof createFrameLoop> | undefined

    onCleanup(() => {
      disposed = true
      frameLoop?.dispose()
    })

    const { width, height } = size
    camCanvas.width = width
    camCanvas.height = height

    const format = navigator.gpu.getPreferredCanvasFormat()
    orientCanvas.width = ORIENT_HIST_CANVAS_WIDTH
    orientCanvas.height = ORIENT_HIST_CANVAS_HEIGHT

    const pip = createGradientProfilePipeline(
      g,
      camCanvas,
      orientCanvas,
      profCanvas,
      props.showHistogramCanvas ? histCanvasEl() : undefined,
      width,
      height,
      format,
    )
    props.onLog(`Gradient profile pipeline ${width}×${height}`)

    frameLoop = createFrameLoop({
      video,
      onFrame: () => {
        if (disposed) {
          return
        }
        const gNow = gpu()
        if (!gNow) {
          return
        }

        const profSize = profileCanvasSize()
        const profW = profSize?.width ?? profCanvas.clientWidth
        const profH = profSize?.height ?? profCanvas.clientHeight
        if (profW > 0 && profH > 0) {
          profCanvas.width = Math.floor(profW)
          profCanvas.height = Math.floor(profH)
        }

        const enc = gNow.device.createCommandEncoder({ label: 'gradient profiles frame' })
        encodeGradientProfileCompute(enc, gNow, pip, video, threshold())
        encodeGradientProfileCameraPresent(enc, gNow, pip, props.displayMode, performance.now() * 0.001)
        if (pip.orientHistViz) {
          encodeOrientHistPresent(enc, pip)
        }
        if (profCanvas.width > 0 && profCanvas.height > 0) {
          encodeGradientProfilePlotPresent(enc, pip, profCanvas.width, profCanvas.height)
        }
        gNow.device.queue.submit([enc.finish()])

        void pip.validEdgeCount.read().then((v) => {
          if (!disposed) {
            const n = Array.isArray(v) ? (v[0] ?? 0) : Number(v)
            props.onValidEdgeCount?.(n)
          }
        })

        void pip.histogram.buffer.read().then((bins) => {
          if (disposed) {
            return
          }
          setThreshold(computeThreshold([...bins], THRESHOLD_PERCENTILE))
        })
      },
    })

    onCleanup(() => {
      pip.destroyProfileTargets()
    })

    return pip
  })

  return (
    <div class={pipelineStyles.feedRow}>
      <div class={[pipelineStyles.feedPanel, pipelineStyles.feedPanelMain]}>
        <div class={pipelineStyles.feedHeader}>
          <span class={pipelineStyles.feedLabel}>
            Camera — {frameSize()?.width ?? '-'}×{frameSize()?.height ?? '-'}
          </span>
          {props.toolbar}
        </div>
        <div class={pipelineStyles.feedContainer}>
          <div class={pipelineStyles.feedCanvasWrap}>
            <canvas ref={setCameraCanvas} class={pipelineStyles.feedCanvas} />
          </div>
        </div>
      </div>

      <div class={[pipelineStyles.feedPanel, pipelineStyles.feedPanelMain]}>
        <span class={pipelineStyles.feedLabel}>Orientation histograms</span>
        <div class={[pipelineStyles.feedContainer, pipelineStyles.orientHistScroll]}>
          <div class={pipelineStyles.feedCanvasWrap}>
            <canvas
              ref={setOrientHistCanvas}
              class={pipelineStyles.feedCanvas}
              width={ORIENT_HIST_CANVAS_WIDTH}
              height={ORIENT_HIST_CANVAS_HEIGHT}
              style={{
                width: '100%',
                'max-width': `${ORIENT_HIST_CANVAS_WIDTH}px`,
                'image-rendering': 'pixelated',
              }}
            />
          </div>
        </div>
      </div>

      <div class={[pipelineStyles.feedPanel, pipelineStyles.feedPanelMain]}>
        <span class={pipelineStyles.feedLabel}>Edge gradient profiles</span>
        <div class={pipelineStyles.feedContainer}>
          <div class={pipelineStyles.feedCanvasWrap}>
            <canvas
              ref={setProfileCanvas}
              class={pipelineStyles.feedCanvas}
              style={{ width: '100%', 'max-width': '960px', height: '280px' }}
            />
          </div>
        </div>
      </div>

      <Show when={props.showHistogramCanvas}>
        <div class={[pipelineStyles.feedPanel, pipelineStyles.feedPanelSide]}>
          <span class={pipelineStyles.feedLabel}>Edge threshold</span>
          <canvas ref={setHistCanvasEl} class={pipelineStyles.histogramCanvas} width={512} height={120} />
          <div class={pipelineStyles.histogramInfo}>
            <span class={pipelineStyles.thresholdLabel}>
              {(THRESHOLD_PERCENTILE * 100).toFixed(0)}th percentile
            </span>
            <span class={pipelineStyles.thresholdValue}>{(threshold() * 255).toFixed(1)} / 255</span>
          </div>
        </div>
      </Show>
    </div>
  )
}
