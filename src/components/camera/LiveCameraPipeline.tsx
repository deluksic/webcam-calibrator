import type { JSX } from 'solid-js'
import { For, Show, createMemo, createSignal, onCleanup } from 'solid-js'

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
  trackSize: { width: number; height: number } | undefined
  onLog: (msg: string) => void
  onQuadDetection?: (quads: DetectedQuad[], meta: { frameId: number }) => void
  /** Extra controls (camera select, mode buttons, …). */
  toolbar?: JSX.Element
}

export function LiveCameraPipeline(props: LiveCameraPipelineProps) {
  const [canvasEl, setCanvasEl] = createSignal<HTMLCanvasElement>()
  const [histCanvasEl, setHistCanvasEl] = createSignal<HTMLCanvasElement>()

  const [threshold, setThreshold] = createSignal(0, { ownedWrite: true })
  const [bboxes, setBboxes] = createSignal<Bbox[]>([], { ownedWrite: true })
  const [gridOverlayQuads, setGridOverlayQuads] = createSignal<DetectedQuad[]>([], {
    ownedWrite: true,
  })

  const canvasSize = createElementSize(canvasEl)
  const scale = createMemo(() => {
    const canvasSize_ = canvasSize()
    const { trackSize } = props
    if (!canvasSize_ || !trackSize) {
      return { x: 0, y: 0 }
    }
    return { x: canvasSize_.width / trackSize.width, y: canvasSize_.height / trackSize.height }
  })

  const log = (msg: string) => {
    props.onLog?.(msg)
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
    const size = props.trackSize
    console.log('Init pipeline', size)
    if (!size) {
      return undefined
    }

    const g = gpu()
    const canvas = canvasEl()
    const histCanvas = histCanvasEl()
    const video = document.createElement('video')
    video.muted = true
    let disposed = false
    let frameLoop: ReturnType<typeof createFrameLoop> | undefined

    onCleanup(() => {
      disposed = true
      frameLoop?.dispose()
      video.pause()
      video.srcObject = null
      log('Pipeline cleanup')
    })

    const stream = props.stream
    if (!g || !canvas || !histCanvas || !stream) {
      log('Pipeline: missing deps')
      return undefined
    }

    histCanvas.width = 512
    histCanvas.height = 120

    video.srcObject = stream
    video.play().catch(() => {})
    if (disposed) {
      return undefined
    }

    const vw = video.videoWidth > 0 ? video.videoWidth : max(1, size.width)
    const vh = video.videoHeight > 0 ? video.videoHeight : max(1, size.height)
    canvas.width = vw
    canvas.height = vh

    log('Creating pipeline...')
    const pip = createCameraPipeline(g, canvas, histCanvas, vw, vh, navigator.gpu.getPreferredCanvasFormat())
    log(`Pipeline created ${vw}x${vh}`)

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
          props.onQuadDetection?.(tagged, { frameId: slot.frameId })
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
      onPrime: () => log('First frame presented'),
      onFrame: () => {
        if (disposed) {
          return
        }
        const gpuNow = gpu()
        if (!gpuNow) {
          return
        }
        const dm = props.displayMode
        const enc = gpuNow.device.createCommandEncoder({ label: 'camera frame' })

        if (dm === 'grid' && props.showGrid) {
          // Grid mode: acquire a slot, run compute with copies pinned into it,
          // submit, then kick off async detection. Canvas is NOT repainted here;
          // presentGridFrame does that after detection resolves.
          const slot = pip.frameSlotPool.acquireFreeSlot()
          if (slot !== undefined) {
            runCompute(enc, gpuNow, pip, video, threshold(), slot)
            gpuNow.device.queue.submit([enc.finish()])
            scheduleQuadDetection(slot, props.showFallbacks)
          }
          // If no slot is free, skip this frame entirely (backpressure).
        } else {
          // Non-grid modes: compute + present synchronously as before.
          runCompute(enc, gpuNow, pip, video, threshold())
          presentFrame(enc, gpuNow, pip, dm as NonGridDisplayMode, (_err) => {})
          gpuNow.device.queue.submit([enc.finish()])
          if (dm === 'debug') {
            scheduleExtentRead()
          }
        }

        // TODO: read using Uint32Array directly
        void pip.histogramBuffer.read().then((bins) => {
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
            Camera Feed — {props.trackSize?.width ?? '-'}×{props.trackSize?.height ?? '-'}
          </span>
          {props.toolbar}
        </div>
        <div class={styles.feedContainer}>
          <canvas ref={setCanvasEl} class={styles.feedCanvas} />
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
          <canvas ref={setHistCanvasEl} class={styles.histogramCanvas} style={{ width: '512px', height: '120px' }} />
          <div class={styles.histogramInfo}>
            <span class={styles.thresholdLabel}>{(THRESHOLD_PERCENTILE * 100).toFixed(0)}th Percentile Threshold</span>
            <span class={styles.thresholdValue}>
              {(threshold() * 255).toFixed(1)} / 255 <span>({(threshold() * 100).toFixed(1)}%)</span>
            </span>
          </div>
        </Show>
        <Show when={!props.showHistogramCanvas}>
          <canvas ref={setHistCanvasEl} class={styles.histogramHidden} width={512} height={120} aria-hidden="true" />
        </Show>
      </div>
    </div>
  )
}
