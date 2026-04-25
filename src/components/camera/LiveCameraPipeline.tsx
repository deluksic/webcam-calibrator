import type { JSX } from 'solid-js'
import { For, Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { detectForSlot } from '@/gpu/cameraDetection'
import type { ExtentRow } from '@/gpu/cameraDetection'
import { presentFrame, presentGridFrame, runCompute, updateQuadCornersBuffer } from '@/gpu/cameraFrame'
import type { NonGridDisplayMode } from '@/gpu/cameraFrame'
import { createCameraPipeline, MAX_U32, MAX_DETECTED_TAGS } from '@/gpu/cameraPipeline'
import type { DisplayMode } from '@/gpu/cameraPipeline'
import type { DetectedQuad } from '@/gpu/contour'
import type { FrameSlot } from '@/gpu/frameSlotPool'
import { initGPU } from '@/gpu/init'
import { computeThreshold, THRESHOLD_PERCENTILE } from '@/gpu/pipelines/constants'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import {
  buildReprojectionDrawOps,
  cameraDistanceFromT,
  cameraTiltDegFromR,
} from '@/lib/reprojectionLive'
import type { TargetLayout } from '@/lib/targetLayout'
import { createElementSize } from '@/utils/createElementSize'
import { createFrameLoop } from '@/utils/createFrameLoop'

import styles from '@/components/camera/LiveCameraPipeline.module.css'

const { navigator } = globalThis
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
  showGrid: boolean
  showFallbacks: boolean
  showHistogramCanvas: boolean
  stream: MediaStream | undefined
  onLog: (msg: string) => void
  onQuadDetection?: (quads: DetectedQuad[], meta: { frameId: number }) => void
  /** When set, draw 2D reprojection overlay and report live metrics. */
  liveCalibration?: () => { k: CameraIntrinsics; layout: TargetLayout } | undefined
  onReprojectionFrame?: (m: { rms: number; tagCount: number; tiltDeg: number; dist: number } | null) => void
  /** Called when snapshot button is pressed - passes current tagged quads. */
  onQuadSnapshotRequest?: () => void
  /** Extra controls (camera select, mode buttons, …). */
  toolbar?: JSX.Element
}

function drawReprojectionOverlay(
  canvas: HTMLCanvasElement,
  w: number,
  h: number,
  quads: DetectedQuad[],
  k: CameraIntrinsics,
  layout: TargetLayout,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) {
    return
  }
  if (canvas.width !== w) {
    canvas.width = w
  }
  if (canvas.height !== h) {
    canvas.height = h
  }
  ctx.clearRect(0, 0, w, h)
  const built = buildReprojectionDrawOps(layout, k, quads, w, h)
  if (!built) {
    return undefined
  }
  for (const op of built.ops) {
    if (op.t === 'ring') {
      ctx.strokeStyle = op.color
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(op.c.x, op.c.y, op.r, 0, 2 * Math.PI)
      ctx.stroke()
    } else if (op.t === 'dot') {
      ctx.fillStyle = op.color
      ctx.beginPath()
      ctx.arc(op.c.x, op.c.y, op.r, 0, 2 * Math.PI)
      ctx.fill()
    } else {
      ctx.strokeStyle = op.color
      ctx.lineWidth = op.w
      ctx.beginPath()
      ctx.moveTo(op.a.x, op.a.y)
      ctx.lineTo(op.b.x, op.b.y)
      ctx.stroke()
    }
  }
  return built
}

export function LiveCameraPipeline(props: LiveCameraPipelineProps) {
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [overlayCanvas, setOverlayCanvas] = createSignal<HTMLCanvasElement>()
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
    showGrid: props.showGrid,
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
    pipelineInteraction().onLog?.(msg)
  }

  const gpu = createMemo(async () => {
    try {
      log('GPU init...')
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
      pip.extentBuffer
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
          pip.frameSlotPool.swapDisplaySlot(slot)
          presentGridFrame(gNow, pip, slot)

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
          if (true) {
            const ovl = overlayCanvas()
            if (ovl && liveCalib) {
              const br = drawReprojectionOverlay(ovl, width, height, tagged, liveCalib.k, liveCalib.layout)
              if (typeof pi.onReprojectionFrame === 'function') {
                if (br) {
                  pi.onReprojectionFrame({
                    rms: br.rms,
                    tagCount: br.tagCount,
                    tiltDeg: cameraTiltDegFromR(br.R),
                    dist: cameraDistanceFromT(br.t),
                  })
                } else {
                  pi.onReprojectionFrame(null)
                }
              }
            } else if (typeof pi.onReprojectionFrame === 'function') {
              pi.onReprojectionFrame(null)
            }
          }
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
        const pi = pipelineInteraction()
        const dm = pi.displayMode
        const enc = gpuNow.device.createCommandEncoder({ label: 'camera frame' })

        if (dm === 'grid' && pi.showGrid) {
          // Grid mode: acquire a slot, run compute with copies pinned into it,
          // submit, then kick off async detection. Canvas is NOT repainted here;
          // presentGridFrame does that after detection resolves.
          const slot = pip.frameSlotPool.acquireFreeSlot()
          if (slot !== undefined) {
            runCompute(enc, gpuNow, pip, video, threshold(), slot)
            gpuNow.device.queue.submit([enc.finish()])
            if (true) {
              scheduleQuadDetection(slot, pi.showFallbacks)
            }
          }
          // If no slot is free, skip this frame entirely (backpressure).
        } else {
          // Non-grid modes: compute + present synchronously as before.
          runCompute(enc, gpuNow, pip, video, threshold())
          presentFrame(enc, gpuNow, pip, dm as NonGridDisplayMode, (_err) => {})
          gpuNow.device.queue.submit([enc.finish()])
          if (dm === 'debug') {
            if (true) {
              scheduleExtentRead()
            }
          }
        }

        // TODO: read using Uint32Array directly
        if (true) {
          void pip.histogramBuffer.read().then((bins) => {
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
          <canvas ref={setCanvasElement} class={styles.feedCanvas} />
          <Show when={props.displayMode === 'grid' && props.showGrid}>
            <canvas ref={setOverlayCanvas} class={styles.reprojOverlay} />
          </Show>
          <Show when={props.displayMode === 'debug'}>
            <QuadCandidateOverlay bboxes={bboxes()} scale={scale()} />
          </Show>
          <Show when={props.displayMode === 'grid' && props.showGrid}>
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
