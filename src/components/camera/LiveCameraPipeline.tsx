import type { JSX } from 'solid-js'
import { For, Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { encodeCameraCompute } from '@/gpu/cameraComputeEncoding'
import { detectForSlot } from '@/gpu/cameraDetection'
import type { ExtentRow } from '@/gpu/cameraDetection'
import { updateQuadCornersBuffer, updateReprojectionOverlayBuffer } from '@/gpu/cameraFrame'
import { createCameraPipeline } from '@/gpu/cameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import { encodeAndSubmitGridPresent, encodePresentNonGrid } from '@/gpu/cameraPresentEncoding'
import type { DetectedQuad } from '@/gpu/contour'
import type { FrameSlot } from '@/gpu/frameSlotPool'
import { initGPU } from '@/gpu/init'
import { MAX_U32 } from '@/gpu/pipelines/extentTrackingPipeline'
import { MAX_DETECTED_TAGS } from '@/gpu/pipelines/gridVizPipeline'
import { computeThreshold, THRESHOLD_PERCENTILE } from '@/gpu/pipelines/histogramPipelines'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import { buildReprojectionOverlayPairs, cameraDistanceFromT, cameraTiltDegFromR } from '@/lib/reprojectionLive'
import type { TargetLayout } from '@/lib/targetLayout'
import { createElementSize } from '@/utils/createElementSize'
import { createFrameLoop } from '@/utils/createFrameLoop'

import styles from '@/components/camera/LiveCameraPipeline.module.css'

const { navigator, performance } = globalThis
const { max, abs } = Math

interface Bbox {
  minX: number
  minY: number
  maxX: number
  maxY: number
  area: number
}

function QuadCandidateOverlay(props: { bboxes: Bbox[]; scale: { x: number; y: number } }) {
  const candidates = createMemo(() => {
    const MIN_AREA = 40 * 40
    const MAX_AREA = 200000
    const MIN_AR = 0.3
    const MAX_AR = 3.5
    return props.bboxes.filter((b) => {
      const w = b.maxX - b.minX
      const h = b.maxY - b.minY
      if (w <= 0 || h <= 0) {
        return false
      }
      const area = w * h
      if (area < MIN_AREA || area > MAX_AREA) {
        return false
      }
      const ar = w / h
      if (ar < MIN_AR || ar > MAX_AR) {
        return false
      }
      return true
    })
  })

  return (
    <For each={candidates()} keyed={false}>
      {(box) => (
        <div
          class={styles.bbox}
          style={{
            '--bbox-x': `${box().minX * props.scale.x}px`,
            '--bbox-y': `${box().minY * props.scale.y}px`,
            '--bbox-w': `${(box().maxX - box().minX) * props.scale.x}px`,
            '--bbox-h': `${(box().maxY - box().minY) * props.scale.y}px`,
          }}
        />
      )}
    </For>
  )
}

function TagIdGridOverlay(props: { quads: DetectedQuad[]; scale: { x: number; y: number } }) {
  return (
    <For each={props.quads} keyed={false}>
      {(quad) => {
        const c = () => quad().corners
        const cx = () => (c()[0].x + c()[1].x + c()[2].x + c()[3].x) / 4
        const cy = () => (c()[0].y + c()[1].y + c()[2].y + c()[3].y) / 4
        const height = () =>
          max(abs(c()[0].y - c()[1].y), abs(c()[1].y - c()[2].y), abs(c()[2].y - c()[3].y), abs(c()[3].y - c()[0].y))

        const label = () => {
          const q = quad()
          if (typeof q.decodedTagId === 'number') {
            return String(q.decodedTagId)
          }
          return '?'
        }
        return (
          <div
            class={styles.tagIdOverlay}
            style={{
              '--tag-x': `${cx() * props.scale.x}px`,
              '--tag-y': `${cy() * props.scale.y}px`,
              '--tag-size': `${height() * props.scale.y}px`,
            }}
          >
            {label()}
          </div>
        )
      }}
    </For>
  )
}

export type LiveCameraPipelineProps = {
  displayMode: DisplayMode
  showFallbacks: boolean
  showHistogramCanvas: boolean
  stream: MediaStream | undefined
  onLog: (msg: string) => void
  onQuadDetection?: (quads: DetectedQuad[], meta: { frameId: number }) => void
  /** When set, feed GPU reprojection overlay and report live metrics. */
  liveCalibration?: () => { k: CameraIntrinsics; distortion?: RationalDistortion8; layout: TargetLayout } | undefined
  onReprojectionFrame?: (m: { rms: number; tagCount: number; tiltDeg: number; dist: number } | undefined) => void
  onFrameSize?: (size: { width: number; height: number }) => void
  /** Called when snapshot button is pressed - passes current tagged quads. */
  onQuadSnapshotRequest?: () => void
  /** Extra controls (camera select, mode buttons, …). */
  toolbar?: JSX.Element
}

export function LiveCameraPipeline(props: LiveCameraPipelineProps) {
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>()

  const [threshold, setThreshold] = createSignal(0, { ownedWrite: true })
  const [bboxes, setBboxes] = createSignal<Bbox[]>([], { ownedWrite: true })
  const [gridOverlayQuads, setGridOverlayQuads] = createSignal<DetectedQuad[]>([], {
    ownedWrite: true,
  })

  /** Props read in memos/JSX; also snapshotted for onFrame / async .then (non-tracking). */
  const pipelineInteraction = createMemo(() => ({
    onLog: props.onLog,
    displayMode: props.displayMode,
    showFallbacks: props.showFallbacks,
    liveCalibration: props.liveCalibration,
    onReprojectionFrame: props.onReprojectionFrame,
    onQuadDetection: props.onQuadDetection,
    onQuadSnapshotRequest: props.onQuadSnapshotRequest,
  }))

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

    let extentReadPending = false

    const scheduleExtentRead = () => {
      if (extentReadPending || disposed) {
        return
      }
      extentReadPending = true
      pip.extent.extentBuffer
        .read()
        .then((extentData: ExtentRow[]) => {
          if (disposed) {
            return
          }
          extentReadPending = false
          const boxes: Bbox[] = []
          for (const entry of extentData) {
            if (entry.minX === MAX_U32) {
              continue
            }
            const w = entry.maxX - entry.minX
            const h = entry.maxY - entry.minY
            if (w <= 0 || h <= 0) {
              continue
            }
            boxes.push({
              minX: entry.minX,
              minY: entry.minY,
              maxX: entry.maxX,
              maxY: entry.maxY,
              area: w * h,
            })
          }
          boxes.sort((a, b) => b.area - a.area)
          setBboxes(boxes.slice(0, 128))
        })
        .finally(() => {
          extentReadPending = false
        })
    }

    const scheduleQuadDetection = (slot: FrameSlot, sf: boolean) => {
      const gNow = gpu()
      if (!gNow) {
        pip.frameSlotPool.releaseSlot(slot)
        return
      }
      const pi = pipelineInteraction()
      const liveCalib = typeof pi.liveCalibration === 'function' ? pi.liveCalibration() : undefined

      detectForSlot(gNow, pip, slot)
        .then((result) => {
          if (disposed) {
            pip.frameSlotPool.releaseSlot(slot)
            return
          }
          const { quads } = result
          quads.sort((a, b) => b.count - a.count)
          const top = quads.slice(0, MAX_DETECTED_TAGS)
          const tagged = top.map((q) => {
            const ok = q.hasCorners && q.cornerDebug && q.cornerDebug.failureCode === 0
            return {
              ...q,
              vizTagId: ok && typeof q.decodedTagId === 'number' ? q.decodedTagId : undefined,
            }
          })

          // Write corners + render gray+grid+histogram in the same synchronous
          // block so the GPU overlay always matches slot.graySnapshot.
          updateQuadCornersBuffer(pip, tagged, sf)

          if (liveCalib) {
            const built = buildReprojectionOverlayPairs(
              liveCalib.layout,
              liveCalib.k,
              liveCalib.distortion,
              tagged,
              width,
              height,
            )
            if (built) {
              updateReprojectionOverlayBuffer(pip, built.pairs, built.count)
              if (typeof pi.onReprojectionFrame === 'function') {
                pi.onReprojectionFrame({
                  rms: built.rms,
                  tagCount: built.tagCount,
                  tiltDeg: cameraTiltDegFromR(built.R),
                  dist: cameraDistanceFromT(built.t),
                })
              }
            } else {
              updateReprojectionOverlayBuffer(pip, [], 0)
              if (typeof pi.onReprojectionFrame === 'function') {
                pi.onReprojectionFrame(undefined)
              }
            }
          } else {
            updateReprojectionOverlayBuffer(pip, [], 0)
            if (typeof pi.onReprojectionFrame === 'function') {
              pi.onReprojectionFrame(undefined)
            }
          }

          pip.frameSlotPool.swapDisplaySlot(slot)
          encodeAndSubmitGridPresent(gNow, pip, slot, performance.now() * 0.001)

          setGridOverlayQuads(
            tagged.filter((q) => {
              if (!q?.hasCorners || q.cornerDebug?.failureCode !== 0) {
                return false
              }
              if (sf) {
                return true
              }
              return typeof q.decodedTagId === 'number'
            }),
          )

          pi.onQuadDetection?.(tagged, { frameId: slot.frameId })
        })
        .catch((e) => {
          if (!disposed) {
            log(`detectForSlot error: ${e}`)
          }
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
          // Grid mode: acquire a slot, run compute with copies pinned into it,
          // submit, then kick off async detection. Canvas is NOT repainted here;
          // encodeAndSubmitGridPresent does that after detection resolves.
          const slot = pip.frameSlotPool.acquireFreeSlot()
          if (slot !== undefined) {
            encodeCameraCompute(enc, gpuNow, pip, video, threshold(), slot)
            gpuNow.device.queue.submit([enc.finish()])
            scheduleQuadDetection(slot, pi.showFallbacks)
          }
          // If no slot is free, skip this frame entirely (backpressure).
        } else {
          // Non-grid modes: compute + present synchronously as before.
          encodeCameraCompute(enc, gpuNow, pip, video, threshold())
          encodePresentNonGrid(enc, gpuNow, pip, dm, timeSec, (_err) => {})
          gpuNow.device.queue.submit([enc.finish()])
          if (dm === 'debug') {
            scheduleExtentRead()
          }
        }

        // TODO: read using Uint32Array directly
        void pip.histogram.buffer.read().then((bins) => {
          if (disposed) {
            return
          }
          const data = new Uint32Array(bins)
          setThreshold(computeThreshold([...data], THRESHOLD_PERCENTILE))
        })
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
          <canvas ref={setCanvasElement} class={styles.feedCanvas} />
          <Show when={props.displayMode === 'debug'}>
            <QuadCandidateOverlay bboxes={bboxes()} scale={scale()} />
          </Show>
          <Show when={props.displayMode === 'grid'}>
            <TagIdGridOverlay quads={gridOverlayQuads()} scale={scale()} />
          </Show>
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
