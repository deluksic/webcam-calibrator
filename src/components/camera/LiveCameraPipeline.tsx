import type { JSX } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { CalibrateFocusOverlay } from '@/components/calibration/CalibrateFocusOverlay'
import { TagIdGridOverlay } from '@/components/camera/LiveCameraPipelineOverlays'
import { encodeCameraCompute } from '@/gpu/cameraComputeEncoding'
import { detectForSlot } from '@/gpu/cameraDetection'
import { updateReprojectionOverlayBuffer } from '@/gpu/cameraFrame'
import { createCameraPipeline } from '@/gpu/cameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import { encodeGridPresent, encodePresentNonGrid } from '@/gpu/cameraPresentEncoding'
import type { DetectedQuad } from '@/gpu/detectedQuad'
import type { FrameSlot } from '@/gpu/frameSlotPool'
import { noteGpuProfileFrame } from '@/gpu/gpuProfiling'
import { initGPU } from '@/gpu/init'
import { MAX_DETECTED_TAGS } from '@/gpu/pipelines/gridVizPipeline'
import { computeThreshold, THRESHOLD_PERCENTILE } from '@/gpu/pipelines/histogramPipelines'
import { writeUndistortUniform } from '@/gpu/pipelines/undistortPipeline'
import { acceptQuadForTagUse } from '@/lib/acceptQuadForTagUse'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { CustomTagOverlaySession } from '@/lib/customTagOverlaySession'
import { buildReprojectionOverlayPairs, cameraDistanceFromT, cameraTiltDegFromR } from '@/lib/reprojectionLive'
import type { TargetLayout } from '@/lib/targetLayout'
import { createElementSize } from '@/utils/createElementSize'
import { createFrameLoop } from '@/utils/createFrameLoop'
import type { Mat3, Vec3 } from '@/workers/calibration.worker'

import styles from '@/components/camera/LiveCameraPipeline.module.css'

const { navigator, performance } = globalThis

export type LiveCalibrationPayload = {
  k: CameraIntrinsics
  distortion?: RationalDistortion8
  layout?: TargetLayout
  extrinsics?: Map<number, { R: Mat3; t: Vec3 }>
}

export type LiveCameraPipelineProps = {
  displayMode: DisplayMode
  showFallbacks: boolean
  showHistogramCanvas: boolean
  stream: MediaStream | undefined
  onLog: (msg: string) => void
  onQuadDetection?: (quads: DetectedQuad[], meta: { frameId: number; grayData?: Float32Array }) => void
  /** When set, feed GPU reprojection overlay and report live metrics. */
  liveCalibration: LiveCalibrationPayload | undefined | (() => LiveCalibrationPayload | undefined)
  onReprojectionFrame?: (m: { rms: number; tagCount: number; tiltDeg: number; dist: number } | undefined) => void
  onFrameSize?: (size: { width: number; height: number }) => void
  /** Called when snapshot button is pressed - passes current tagged quads. */
  onQuadSnapshotRequest?: () => void
  /** Registers a function to capture the next frame's gray buffer alongside detection. */
  setRequestGraySnapshot?: (fn: (callback: (grayData: Float32Array) => void) => void) => void
  /** Extra controls (camera select, mode buttons, …). */
  toolbar?: JSX.Element
  /** Advisory 75%×75% framing guide over the **displayed** canvas (same box as tag overlays). */
  showFocusOverlay?: boolean
  /** Short line centered in the strip below the focus rect (inside canvas). */
  focusBottomHint?: () => JSX.Element | undefined
  /** Calibrate: `*` / `*0`… for custom (negative) tag ids while session is running; omit in Debug to show technical labels. */
  customTagOverlay?: () => CustomTagOverlaySession
}

export function LiveCameraPipeline(props: LiveCameraPipelineProps) {
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>()

  const [threshold, setThreshold] = createSignal(0, { ownedWrite: true })
  const [gridOverlayQuads, setGridOverlayQuads] = createSignal<DetectedQuad[]>([], {
    ownedWrite: true,
  })

  /** Props read in memos/JSX; also snapshotted for onFrame / async .then (non-tracking). */
  const focusBottomHintContent = createMemo(() => props.focusBottomHint?.())

  const pipelineInteraction = createMemo(() => {
    const lc = props.liveCalibration
    const liveCalibration = lc === undefined ? undefined : typeof lc === 'function' ? lc() : lc
    return {
      onLog: props.onLog,
      displayMode: props.displayMode,
      showFallbacks: props.showFallbacks,
      liveCalibration,
      onReprojectionFrame: props.onReprojectionFrame,
      onQuadDetection: props.onQuadDetection,
      onQuadSnapshotRequest: props.onQuadSnapshotRequest,
      customTagOverlay: props.customTagOverlay,
    }
  })

  const videoElement = createMemo(async () => {
    const canvas = canvasElement()
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

  const [frameSize, setFrameSize] = createSignal(() => {
    const video = videoElement()
    if (!video || video.videoWidth === 0 || video.videoHeight === 1) {
      return undefined
    }
    return { width: video.videoWidth, height: video.videoHeight }
  })

  createEffect(videoElement, (video) => {
    if (!video) {
      return
    }
    const onResize = () => {
      setFrameSize({ width: video.videoWidth, height: video.videoHeight })
    }
    video.addEventListener('resize', onResize)
    return () => {
      video.removeEventListener('resize', onResize)
    }
  })

  createEffect(frameSize, (size) => {
    if (!size) {
      return
    }
    props.onFrameSize?.(size)
  })

  const canvasSize = createElementSize(canvasElement)
  const scale = createMemo(() => {
    const video = videoElement()
    const canvasSize_ = canvasSize()
    if (!canvasSize_ || !video) {
      return { x: 0, y: 0 }
    }
    return { x: canvasSize_.width / video.videoWidth, y: canvasSize_.height / video.videoHeight }
  })

  const log = (msg: string) => {
    props.onLog?.(msg)
  }

  const gpu = createMemo(async () => {
    try {
      const g = await initGPU()
      log('GPU ready')
      return g
    } catch (e) {
      log(`GPU init failed: ${e}`)
      return undefined
    }
  })

  createMemo(async () => {
    const video = videoElement()
    const size = frameSize()
    if (!video || !size) {
      return undefined
    }

    const g = gpu()
    const canvas = canvasElement()
    const histCanvas = histCanvasEl()
    let disposed = false
    let frameLoop: ReturnType<typeof createFrameLoop> | undefined

    onCleanup(() => {
      disposed = true
      frameLoop?.dispose()
      log('Pipeline cleanup')
    })

    if (!g || !canvas) {
      log('Pipeline: missing deps')
      return undefined
    }

    if (disposed) {
      return undefined
    }

    log('Creating pipeline...')

    const { width, height } = size
    canvas.width = width
    canvas.height = height

    const pip = createCameraPipeline(g, canvas, histCanvas, width, height, navigator.gpu.getPreferredCanvasFormat())
    log(`Pipeline created ${width}x${height}`)

    let lastAppliedDetectionFrameId = -1

    let captureGrayOnNextDetection = false
    let onGraySnapshotCallback: ((grayData: Float32Array) => void) | null = null

    const requestGraySnapshot = (callback: (grayData: Float32Array) => void) => {
      captureGrayOnNextDetection = true
      onGraySnapshotCallback = callback
    }
    props.setRequestGraySnapshot?.(requestGraySnapshot)

    const scheduleQuadDetection = (slot: FrameSlot, sf: boolean) => {
      const gNow = gpu()
      if (!gNow) {
        pip.frameSlotPool.releaseSlot(slot)
        return
      }

      const readGray = captureGrayOnNextDetection
      captureGrayOnNextDetection = false

      void detectForSlot(gNow, pip, slot, readGray)
        .then((result) => {
          if (disposed) {
            return
          }
          if (slot.frameId < lastAppliedDetectionFrameId) {
            return
          }
          lastAppliedDetectionFrameId = slot.frameId

          const pi = pipelineInteraction()
          const liveCalib = pi.liveCalibration

          const { quads } = result
          quads.sort((a, b) => b.count - a.count)
          const top = quads.slice(0, MAX_DETECTED_TAGS)
          const tagged = top.map((q) => {
            const ok = q.hasCorners && q.cornerDebug && q.cornerDebug.failureCode === 0
            return {
              ...q,
              vizTagId: ok
                ? typeof q.decodedTagId === 'number'
                  ? q.decodedTagId
                  : q.vizTagId !== undefined
                    ? q.vizTagId
                    : undefined
                : undefined,
            }
          })

          if (liveCalib?.layout) {
            const accepted = tagged.filter((q) => acceptQuadForTagUse(q, sf))
            const built = buildReprojectionOverlayPairs(
              liveCalib.layout,
              liveCalib.k,
              liveCalib.distortion,
              accepted,
              width,
              height,
            )
            if (built) {
              if (updateReprojectionOverlayBuffer(pip, built.pairs, built.count)) {
                pi.onReprojectionFrame?.({
                  rms: built.rms,
                  tagCount: built.tagCount,
                  tiltDeg: cameraTiltDegFromR(built.R),
                  dist: cameraDistanceFromT(built.t),
                })
              }
            } else if (updateReprojectionOverlayBuffer(pip, [], 0)) {
              pi.onReprojectionFrame?.(undefined)
            }
          } else if (updateReprojectionOverlayBuffer(pip, [], 0)) {
            pi.onReprojectionFrame?.(undefined)
          }

          const overlayQuads = tagged.filter((q) => acceptQuadForTagUse(q, sf))
          setGridOverlayQuads(overlayQuads)

          const presentEnc = gNow.device.createCommandEncoder({ label: 'grid frame present' })
          encodeGridPresent(presentEnc, pip, performance.now() * 0.001, slot.frameId % 2)
          gNow.device.queue.submit([presentEnc.finish()])

          pi.onQuadDetection?.(tagged, {
            frameId: slot.frameId,
            grayData: result.grayData.length > 0 ? result.grayData : undefined,
          })

          const cb = onGraySnapshotCallback
          if (cb && result.grayData.length > 0) {
            onGraySnapshotCallback = null
            cb(result.grayData)
          }
        })
        .catch((e) => {
          if (!disposed) {
            log(`detectForSlot error: ${e}`)
          }
        })
        .finally(() => {
          pip.frameSlotPool.releaseSlot(slot)
        })
    }

    frameLoop = createFrameLoop({
      video,
      onFrame: () => {
        if (disposed) {
          return
        }

        const gpuNow = gpu()
        if (!gpuNow) {
          return
        }
        const timeSec = performance.now() * 0.001
        const pi = pipelineInteraction()
        const dm = pi.displayMode
        const enc = gpuNow.device.createCommandEncoder({ label: 'camera frame' })

        if (dm === 'grid') {
          const slot = pip.frameSlotPool.acquireFreeSlot()
          if (slot !== undefined) {
            encodeCameraCompute(enc, gpuNow, pip, video, threshold(), slot)
            gpuNow.device.queue.submit([enc.finish()])
            noteGpuProfileFrame(gpuNow)
            scheduleQuadDetection(slot, pi.showFallbacks)
          }
        } else {
          // Non-grid modes: compute + present synchronously as before.
          if (dm === 'undistort') {
            const lc = pi.liveCalibration
            const fs = frameSize()
            writeUndistortUniform(pip.undistortUniform, {
              K: lc?.k ?? { fx: 1, fy: 1, cx: 0, cy: 0 },
              distortion: lc?.distortion ?? [0, 0, 0, 0, 0, 0, 0, 0],
              width: fs?.width ?? 1,
              height: fs?.height ?? 1,
            })
          }
          encodeCameraCompute(enc, gpuNow, pip, video, threshold())
          encodePresentNonGrid(enc, gpuNow, pip, dm, timeSec, (_err) => {})
          gpuNow.device.queue.submit([enc.finish()])
          noteGpuProfileFrame(gpuNow)
        }

        if (pip.histogram.consumeThresholdReadbackDue()) {
          // TODO: read using Uint32Array directly
          void pip.histogram.buffer.read().then((bins) => {
            if (disposed) {
              return
            }
            const data = new Uint32Array(bins)
            setThreshold(computeThreshold([...data], THRESHOLD_PERCENTILE))
          })
        }
      },
    })
    log('rVFC loop started')

    return pip
  })

  return (
    <div class={styles.feedRow}>
      <div class={[styles.feedPanel, styles.feedPanelMain]}>
        <div class={styles.feedHeader}>
          <span class={styles.feedLabel}>
            Camera Feed — {frameSize()?.width ?? '-'}×{frameSize()?.height ?? '-'}
          </span>
          {props.toolbar}
        </div>
        <div class={styles.feedContainer}>
          <div class={styles.feedCanvasWrap}>
            <canvas ref={setCanvasElement} class={styles.feedCanvas} />
            <Show when={props.showFocusOverlay}>
              <CalibrateFocusOverlay />
            </Show>
            <Show when={focusBottomHintContent()}>
              <div class={styles.focusBottomHint} role="note">
                <span class={styles.focusBottomHintText}>{focusBottomHintContent()}</span>
              </div>
            </Show>
            <Show when={props.displayMode === 'grid'}>
              <TagIdGridOverlay quads={gridOverlayQuads()} scale={scale()} customTagOverlay={props.customTagOverlay} />
            </Show>
          </div>
        </div>
      </div>
      <div class={[styles.feedPanel, styles.feedPanelSide]}>
        <Show when={props.showHistogramCanvas}>
          <span class={styles.feedLabel}>Edge Detection</span>
          <canvas ref={setHistCanvasEl} class={styles.histogramCanvas} width={512} height={120} />
          <div class={styles.histogramInfo}>
            <span class={styles.thresholdLabel}>{(THRESHOLD_PERCENTILE * 100).toFixed(0)}th Percentile Threshold</span>
            <span class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255 <span>({(threshold() * 100).toFixed(1)}%)</span>
            </span>
          </div>
        </Show>
      </div>
    </div>
  )
}
