import type { JSX } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { encodeGradientProfileCompute } from '@/gpu/gradientProfileComputeEncoding'
import {
  encodeGradientProfileCameraPresent,
  encodeGradientProfilePlotPresent,
  encodeOrientHistPresent,
  encodeTagHistPresent,
} from '@/gpu/gradientProfilePresentEncoding'
import {
  ORIENT_HIST_CANVAS_HEIGHT,
  ORIENT_HIST_CANVAS_WIDTH,
} from '@/gpu/pipelines/orientHistVizPipeline'
import { TAG_HIST_CANVAS_W, TAG_HIST_CANVAS_H } from '@/gpu/pipelines/tagDecodePipeline'
import { createGradientProfilePipeline } from '@/gpu/gradientProfilePipeline'
import type { GradientProfileDisplayMode } from '@/gpu/gradientProfilePipeline'
import { initGPU } from '@/gpu/init'
import { computeThreshold, THRESHOLD_PERCENTILE } from '@/gpu/pipelines/histogramPipelines'
import { writeUndistortUniform } from '@/gpu/pipelines/undistortPipeline'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import { createElementSize } from '@/utils/createElementSize'
import { createFrameLoop } from '@/utils/createFrameLoop'

import pipelineStyles from '@/components/camera/LiveCameraPipeline.module.css'

const { navigator, performance } = globalThis

export type GradientProfilesPipelineProps = {
  displayMode: GradientProfileDisplayMode
  /** When true, skip ingest/compute and keep the last processed frame for inspection. */
  frozen: boolean
  showHistogramCanvas: boolean
  stream: MediaStream | undefined
  onLog: (msg: string) => void
  onValidEdgeCount?: (count: number) => void
  /** Latest calibration intrinsics/distortion for undistort preview; identity when omitted. */
  undistortParams?: () => { k: CameraIntrinsics; distortion: RationalDistortion8 } | undefined
  toolbar?: JSX.Element
}

export function GradientProfilesPipeline(props: GradientProfilesPipelineProps) {
  const [cameraCanvas, setCameraCanvas] = createSignal<HTMLCanvasElement>()
  const [orientHistCanvas, setOrientHistCanvas] = createSignal<HTMLCanvasElement>()
  const [profileCanvas, setProfileCanvas] = createSignal<HTMLCanvasElement>()
  const [tagHistCanvas, setTagHistCanvas] = createSignal<HTMLCanvasElement>()
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>()
  const [threshold, setThreshold] = createSignal(0, { ownedWrite: true })
  let lastQuadCount = 0

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
    /** While paused, only recompute when display mode changes (labeling uses non-deterministic atomics). */
    let lastFrozenComputeMode: GradientProfileDisplayMode | undefined

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
      tagHistCanvas(),
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
        const profW = profSize?.widthPX ?? profSize?.width ?? profCanvas.clientWidth
        const profH = profSize?.heightPX ?? profSize?.height ?? profCanvas.clientHeight
        if (profW > 0 && profH > 0) {
          profCanvas.width = Math.floor(profW)
          profCanvas.height = Math.floor(profH)
        }

        const enc = gNow.device.createCommandEncoder({ label: 'gradient profiles frame' })
        const mode = props.displayMode
        const shouldRunCompute = !props.frozen || lastFrozenComputeMode !== mode
        if (shouldRunCompute) {
          encodeGradientProfileCompute(enc, gNow, pip, video, threshold(), {
            quadCount: lastQuadCount,
            displayMode: mode,
            skipIngest: props.frozen,
          })
          lastFrozenComputeMode = props.frozen ? mode : undefined
        }
        if (props.displayMode === 'undistort') {
          const ud = props.undistortParams?.()
          writeUndistortUniform(pip.undistortUniform, {
            K: ud?.k ?? { fx: 1, fy: 1, cx: 0, cy: 0 },
            distortion: ud?.distortion ?? [0, 0, 0, 0, 0, 0, 0, 0],
            width,
            height,
          })
        }
        encodeGradientProfileCameraPresent(
          enc,
          gNow,
          pip,
          props.displayMode,
          performance.now() * 0.001,
          lastQuadCount,
        )
        if (pip.orientHistViz) {
          encodeOrientHistPresent(enc, pip)
        }
        if (pip.tagHistogramDisplay) {
          encodeTagHistPresent(enc, pip)
        }
        if (profCanvas.width > 0 && profCanvas.height > 0) {
          encodeGradientProfilePlotPresent(enc, pip, profCanvas.width, profCanvas.height)
        }
        gNow.device.queue.submit([enc.finish()])

        if (props.frozen) {
          return
        }

        void pip.validEdgeCount.read().then((v) => {
          if (!disposed) {
            const n = Array.isArray(v) ? (v[0] ?? 0) : Number(v)
            props.onValidEdgeCount?.(n)
          }
        })

        void pip.edgeHistogram.quadCount.read().then((v) => {
          if (!disposed) {
            lastQuadCount = Array.isArray(v) ? (v[0] ?? 0) : Number(v)
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
            {props.frozen ? ' (paused)' : ''}
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
        <span class={pipelineStyles.feedLabel}>Tag grayscale histogram</span>
        <div class={[pipelineStyles.feedContainer, pipelineStyles.orientHistScroll]}>
          <div class={pipelineStyles.feedCanvasWrap}>
            <canvas
              ref={setTagHistCanvas}
              class={pipelineStyles.feedCanvas}
              width={TAG_HIST_CANVAS_W}
              height={TAG_HIST_CANVAS_H}
              style={{
                width: '100%',
                'max-width': `${TAG_HIST_CANVAS_W}px`,
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
