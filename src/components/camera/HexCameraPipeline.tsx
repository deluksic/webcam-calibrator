import type { Component } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, createTrackedEffect } from 'solid-js'

import { CalibrateFocusOverlay } from '@/components/calibration/CalibrateFocusOverlay'
import { initSharedGPUForOnnx } from '@/gpu/init'
import type { RoiComponentPoint } from '@/gpu/roiLab/roiComponentsPipeline'
import type { RoiGraphData } from '@/gpu/roiLab/roiGraphPipeline'
import {
  ROI_MODEL_WIDTH,
  createRoiLabPipeline,
  type RoiLabPipeline,
  type RoiStepTimings,
} from '@/gpu/roiLab/roiLabPipeline'
import type { CalibrationFrameObservation, CalibrationPointObservation } from '@/lib/calibrationTypes'
import type { HexReprojectionResult } from '@/lib/hexReprojection'
import { toCalibrationFramePoints } from '@/lib/hexTarget/calibrationPoints'
import { matchHexTargetPoints, type HexTargetMatch } from '@/lib/hexTarget/match'
import type { HexTargetModel } from '@/lib/hexTarget/target'
import { createElementSize } from '@/utils/createElementSize'
import { createFrameLoop } from '@/utils/createFrameLoop'

import styles from '@/components/camera/HexCameraPipeline.module.css'

const MODEL_URL = `${import.meta.env.BASE_URL}models/roi_tiny_no_concat.onnx`

export type SnapshotCallback = (frame: CalibrationFrameObservation) => void

export type HexDetectionMeta = {
  frameId: number
  width: number
  height: number
}

export type HexCameraPipelineProps = {
  stream: MediaStream | undefined
  targetModel: HexTargetModel
  onDetection: (match: HexTargetMatch, points: CalibrationPointObservation[], meta: HexDetectionMeta) => void
  /**
   * Async identity matcher (normally the worker). Returning a promise lets the
   * ROI pipeline overlap matching with the next frame's GPU work; the resolved
   * graph is presented with the pixels it was computed from.
   */
  matchTarget?: (design: HexTargetModel, points: readonly RoiComponentPoint[]) => Promise<HexTargetMatch>
  onFrameSize?: (size: { width: number; height: number }) => void
  onError?: (message: string | undefined) => void
  setRequestSnapshot?: (request: ((callback: SnapshotCallback) => void) | undefined) => void
  reprojection?: HexReprojectionResult
  showFocusOverlay?: boolean
  onTimings?: (timings: RoiStepTimings) => void
}

function captureGray(
  video: HTMLVideoElement,
  width: number,
  height: number,
): { data: Float32Array; width: number; height: number } | undefined {
  if (width < 2 || height < 2) {
    return undefined
  }
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    return undefined
  }
  context.drawImage(video, 0, 0, width, height)
  const pixels = context.getImageData(0, 0, width, height).data
  const data = new Float32Array(width * height)
  for (let index = 0; index < data.length; index += 1) {
    const offset = index * 4
    data[index] =
      (0.299 * (pixels[offset] ?? 0) + 0.587 * (pixels[offset + 1] ?? 0) + 0.114 * (pixels[offset + 2] ?? 0)) / 255
  }
  return { data, width, height }
}

function graphDataFor(match: HexTargetMatch): RoiGraphData {
  return {
    points: match.points.map((point) => ({
      x: point.x,
      y: point.y,
      designId: point.designId ?? 0,
      status: point.status,
    })),
    edges: match.edges.map(([a, b]) => ({ a, b })),
  }
}

export const HexCameraPipeline: Component<HexCameraPipelineProps> = (props) => {
  const [displayElement, setDisplayElement] = createSignal<HTMLDivElement>()
  const [videoElement, setVideoElement] = createSignal<HTMLVideoElement>()
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [overlayElement, setOverlayElement] = createSignal<HTMLCanvasElement>()
  const [frameSize, setFrameSize] = createSignal<{ width: number; height: number }>()
  const [error, setError] = createSignal<string>()
  const displaySize = createElementSize(displayElement)
  const [retryNonce, setRetryNonce] = createSignal(0)

  /** Serializes pipeline setup/teardown across resolution or stream changes. */
  let lifecycle: Promise<unknown> = Promise.resolve()

  let pendingSnapshot: SnapshotCallback | undefined
  let deviceLossRetries = 0

  const reportPipelineError = (cause: unknown, cancelled: () => boolean): void => {
    if (cancelled()) {
      return
    }
    const message = cause instanceof Error ? cause.message : String(cause)
    setError(message)
    if (/device.*lost|mapAsync/i.test(message) && deviceLossRetries < 3) {
      deviceLossRetries += 1
      setTimeout(() => setRetryNonce((value) => value + 1), 300)
    }
  }

  const requestSnapshot = (callback: SnapshotCallback): void => {
    pendingSnapshot = callback
  }
  createTrackedEffect(() => {
    props.setRequestSnapshot?.(requestSnapshot)
    return () => props.setRequestSnapshot?.(undefined)
  })

  createEffect(
    () => ({ video: videoElement(), stream: props.stream }),
    ({ video, stream }) => {
      if (!video) {
        return
      }
      if (stream) {
        video.srcObject = stream
        void video.play().catch((cause: unknown) => {
          setError(cause instanceof Error ? cause.message : String(cause))
        })
      } else {
        video.srcObject = null
      }
      return () => {
        video.srcObject = null
      }
    },
  )

  createEffect(
    () => videoElement(),
    (video) => {
      if (!video) {
        return
      }
      const update = (): void => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          const size = { width: video.videoWidth, height: video.videoHeight }
          setFrameSize(size)
          props.onFrameSize?.(size)
        }
      }
      update()
      video.addEventListener('loadedmetadata', update)
      video.addEventListener('resize', update)
      return () => {
        video.removeEventListener('loadedmetadata', update)
        video.removeEventListener('resize', update)
      }
    },
  )

  createEffect(
    () => ({
      video: videoElement(),
      canvas: canvasElement(),
      overlay: overlayElement(),
      size: frameSize(),
      retry: retryNonce(),
    }),
    ({ video, canvas, overlay, size }) => {
      if (!video || !canvas || !size || size.width < 8 || size.height < 8) {
        return
      }
      let cancelled = false
      let pipeline: RoiLabPipeline | undefined
      let loop: ReturnType<typeof createFrameLoop> | undefined
      let inFlight: Promise<void> | undefined
      let busy = false

      canvas.width = size.width
      canvas.height = size.height
      if (overlay) {
        overlay.width = size.width
        overlay.height = size.height
      }

      const setup = lifecycle.then(async () => {
        if (cancelled) {
          return
        }
        const shared = await initSharedGPUForOnnx()
        if (cancelled) {
          return
        }
        const rawModelHeight = Math.round((ROI_MODEL_WIDTH * size.height) / size.width)
        const modelHeight = Math.min(224, Math.max(96, Math.ceil(rawModelHeight / 16) * 16))
        const created = await createRoiLabPipeline(
          shared.root,
          shared.device,
          canvas,
          size.width,
          size.height,
          modelHeight,
          MODEL_URL,
          () => {
            if (!cancelled) {
              setError(undefined)
            }
          },
          undefined,
          async (points: readonly RoiComponentPoint[], frameId: number) => {
            if (cancelled) {
              return { points: [], edges: [] }
            }
            const design = props.targetModel
            const matchTarget = props.matchTarget

            // Capture a pending snapshot before awaiting identity. The worker
            // can take longer than a camera frame; grabbing the pixels first
            // keeps the saved image paired with the components we just read.
            const snapshotRequest = pendingSnapshot
            const canCapture =
              snapshotRequest !== undefined &&
              video.readyState >= 2 &&
              video.videoWidth === size.width &&
              video.videoHeight === size.height
            const gray = canCapture ? captureGray(video, size.width, size.height) : undefined

            let match: HexTargetMatch
            try {
              match = matchTarget ? await matchTarget(design, points) : matchHexTargetPoints(design, points)
            } catch (cause: unknown) {
              reportPipelineError(cause, () => cancelled)
              return { points: [], edges: [] }
            }
            if (cancelled) {
              return { points: [], edges: [] }
            }

            const conversion = toCalibrationFramePoints(match, design)
            const accepted = conversion.points
            // A newer snapshot request may have arrived while the matcher ran;
            // only satisfy the request this frame was captured for.
            if (canCapture && pendingSnapshot === snapshotRequest) {
              pendingSnapshot = undefined
              snapshotRequest({
                frameId,
                points: accepted,
                grayData: gray?.data,
                imageWidth: gray?.width,
                imageHeight: gray?.height,
              })
            }
            props.onDetection(match, accepted, {
              frameId,
              width: size.width,
              height: size.height,
            })
            return graphDataFor(match)
          },
        )
        if (cancelled) {
          created.destroy()
          return
        }
        pipeline = created
        setError(undefined)
        created.setViewMode(9)
        loop = createFrameLoop({
          video,
          onFrame: () => {
            if (
              busy ||
              cancelled ||
              !pipeline ||
              video.readyState < 2 ||
              video.videoWidth !== size.width ||
              video.videoHeight !== size.height
            ) {
              return
            }
            busy = true
            inFlight = pipeline
              .processFrame(video)
              .then((timings) => {
                props.onTimings?.(timings.steps)
                deviceLossRetries = 0
                return undefined
              })
              .catch((cause: unknown) => {
                reportPipelineError(cause, () => cancelled)
              })
              .finally(() => {
                busy = false
                inFlight = undefined
              })
          },
        })
      })
      lifecycle = setup.catch((cause: unknown) => {
        reportPipelineError(cause, () => cancelled)
      })

      return () => {
        cancelled = true
        loop?.dispose()
        const teardown = (async () => {
          if (inFlight) {
            await inFlight.catch(() => undefined)
          }
          const old = pipeline
          pipeline = undefined
          old?.destroy()
        })()
        lifecycle = lifecycle.then(() => teardown).catch(() => undefined)
        pendingSnapshot = undefined
      }
    },
  )

  createEffect(
    () => ({ canvas: overlayElement(), size: frameSize() }),
    ({ canvas, size }) => {
      if (!canvas || !size) {
        return
      }
      if (canvas.width !== size.width || canvas.height !== size.height) {
        canvas.width = size.width
        canvas.height = size.height
      }
    },
  )

  createEffect(
    () => ({ reprojection: props.reprojection, canvas: overlayElement(), size: frameSize() }),
    ({ reprojection, canvas, size }) => {
      if (!canvas || !size) {
        return
      }
      const context = canvas.getContext('2d')
      if (!context) {
        return
      }
      context.clearRect(0, 0, canvas.width, canvas.height)
      if (!reprojection) {
        return
      }
      context.strokeStyle = 'rgba(80, 220, 255, 0.9)'
      context.fillStyle = 'rgba(80, 220, 255, 0.95)'
      context.lineWidth = 1.2
      for (const pair of reprojection.pairs) {
        context.beginPath()
        context.moveTo(pair.original.x, pair.original.y)
        context.lineTo(pair.reprojected.x, pair.reprojected.y)
        context.stroke()
        context.beginPath()
        context.arc(pair.reprojected.x, pair.reprojected.y, 1.8, 0, Math.PI * 2)
        context.fill()
      }
    },
  )

  const stageStyle = createMemo(() => {
    const available = displaySize()
    const size = frameSize()
    if (!available || !size || available.width < 1 || available.height < 1) {
      return { width: '0px', height: '0px' }
    }
    const aspect = size.width / Math.max(1, size.height)
    let width = available.width
    let height = width / aspect
    if (height > available.height) {
      height = available.height
      width = height * aspect
    }
    return {
      width: `${Math.max(1, Math.floor(width))}px`,
      height: `${Math.max(1, Math.floor(height))}px`,
    }
  })

  return (
    <div class={styles.root}>
      <div class={styles.display} ref={setDisplayElement}>
        <div class={styles.stage} style={stageStyle()}>
          <video ref={setVideoElement} class={styles.video} muted playsinline />
          <canvas ref={setCanvasElement} class={styles.canvas} />
          <canvas ref={setOverlayElement} class={styles.overlay} />
          <Show when={props.showFocusOverlay}>
            <CalibrateFocusOverlay />
          </Show>
          <Show when={error()}>{(message) => <div class={styles.error}>{message()}</div>}</Show>
        </div>
      </div>
    </div>
  )
}
