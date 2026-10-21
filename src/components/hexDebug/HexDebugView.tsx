import type { Component } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, onCleanup, untrack } from 'solid-js'
import type { TgpuRoot } from 'typegpu'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { createHexDebugPipeline, type HexDebugPipeline } from '@/gpu/hexDebug/hexDebugPipeline'
import { initGPU } from '@/gpu/init'
import type { HexDebugComponent, HexDebugTuning } from '@/lib/hexDebug/types'
import { createFrameLoop } from '@/utils/createFrameLoop'

import styles from '@/components/hexDebug/HexDebugView.module.css'

const DEFAULT_TUNING = {
  baseTolerance: 0.02,
  z: 1.5,
  maxTolerance: 0.2,
  minArea: 1,
  maxArea: 100,
  iterations: 12,
} as const

function median(values: readonly number[]): number {
  if (values.length === 0) {
    return 0
  }
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2
}

export const HexDebugView: Component = () => {
  const cam = useCameraStream()
  const [videoElement, setVideoElement] = createSignal<HTMLVideoElement>()
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [root, setRoot] = createSignal<TgpuRoot>()
  const [pipeline, setPipeline] = createSignal<HexDebugPipeline>()
  const [error, setError] = createSignal<string>()
  const [videoSize, setVideoSize] = createSignal<{ width: number; height: number }>()
  const [components, setComponents] = createSignal<HexDebugComponent[]>([])
  const [labels, setLabels] = createSignal<Uint32Array>()
  const [viewMode, setViewMode] = createSignal<'gray' | 'components' | 'boxes'>('gray')
  const [processingMs, setProcessingMs] = createSignal(0)
  const [fps, setFps] = createSignal(0)
  const [paused, setPaused] = createSignal(false)
  const [iterations, setIterations] = createSignal<number>(DEFAULT_TUNING.iterations)
  const [baseTolerance, setBaseTolerance] = createSignal<number>(DEFAULT_TUNING.baseTolerance)
  const [zFactor, setZFactor] = createSignal<number>(DEFAULT_TUNING.z)
  const [maxTolerance, setMaxTolerance] = createSignal<number>(DEFAULT_TUNING.maxTolerance)
  const [minArea, setMinArea] = createSignal<number>(DEFAULT_TUNING.minArea)
  const [maxArea, setMaxArea] = createSignal<number>(DEFAULT_TUNING.maxArea)
  const [expected, setExpected] = createSignal(476)

  let disposed = false
  let processing = false
  let lastFrameAt = 0

  const tuning = createMemo<HexDebugTuning>(() => ({
    baseTolerance: baseTolerance(),
    z: zFactor(),
    maxTolerance: maxTolerance(),
    minArea: minArea(),
    maxArea: maxArea(),
    iterations: iterations(),
  }))

  void initGPU()
    .then((gpuRoot) => {
      if (!disposed) {
        setRoot(gpuRoot)
      }
    })
    .catch((cause: unknown) => {
      if (!disposed) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    })
  onCleanup(() => {
    disposed = true
  })

  createEffect(cam.stream, (stream) => {
    const video = videoElement()
    if (!video || !stream) {
      return
    }
    video.srcObject = stream
    void video.play().catch((cause: unknown) => {
      if (!disposed) {
        setError(cause instanceof Error ? cause.message : String(cause))
      }
    })
    return () => {
      video.srcObject = null
    }
  })

  createEffect(videoElement, (video) => {
    if (!video) {
      return
    }
    const updateSize = (): void => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        setVideoSize({ width: video.videoWidth, height: video.videoHeight })
      }
    }
    updateSize()
    video.addEventListener('loadedmetadata', updateSize)
    video.addEventListener('resize', updateSize)
    return () => {
      video.removeEventListener('loadedmetadata', updateSize)
      video.removeEventListener('resize', updateSize)
    }
  })

  createEffect(
    () => ({
      gpuRoot: root(),
      size: videoSize(),
    }),
    ({ gpuRoot, size }) => {
      if (!gpuRoot || !size || size.width < 8 || size.height < 8) {
        return
      }
      const nextPipeline = createHexDebugPipeline(gpuRoot, size.width, size.height, untrack(tuning))
      setPipeline(nextPipeline)
      return () => {
        nextPipeline.destroy()
      }
    },
  )

  const labelColor = (label: number): [number, number, number] => {
    let hash = (label * 2654435761) >>> 0
    hash ^= hash >>> 16
    hash = Math.imul(hash, 2246822507) >>> 0
    hash ^= hash >>> 13
    return [48 + (hash & 127), 48 + ((hash >>> 8) & 127), 48 + ((hash >>> 16) & 127)]
  }

  const drawLabelMap = (ctx: CanvasRenderingContext2D, labelMap: Uint32Array, width: number, height: number): void => {
    const image = ctx.createImageData(width, height)
    const pixels = image.data
    for (let index = 0; index < width * height; index += 1) {
      const [red, green, blue] = labelColor(labelMap[index] ?? 0)
      const offset = index * 4
      pixels[offset] = red
      pixels[offset + 1] = green
      pixels[offset + 2] = blue
      pixels[offset + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }

  const drawGrayVideo = (
    ctx: CanvasRenderingContext2D,
    video: HTMLVideoElement,
    width: number,
    height: number,
  ): void => {
    ctx.save()
    ctx.filter = 'grayscale(1)'
    ctx.drawImage(video, 0, 0, width, height)
    ctx.restore()
  }

  const drawBoxes = (ctx: CanvasRenderingContext2D, list: readonly HexDebugComponent[]): void => {
    for (const component of list) {
      ctx.strokeStyle = component.color === 0 ? '#f97316' : '#22d3ee'
      ctx.lineWidth = 1.5
      ctx.strokeRect(
        component.bbox.minX,
        component.bbox.minY,
        component.bbox.maxX - component.bbox.minX + 1,
        component.bbox.maxY - component.bbox.minY + 1,
      )
    }
  }

  const processFrame = (video: HTMLVideoElement, size: { width: number; height: number }): void => {
    const canvas = canvasElement()
    const gpuRoot = root()
    const currentPipeline = pipeline()
    if (!canvas || !gpuRoot || !currentPipeline) {
      return
    }
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width
      canvas.height = size.height
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }
    const mode = viewMode()
    if (mode === 'components') {
      const currentLabels = labels()
      if (currentLabels && currentLabels.length === size.width * size.height) {
        drawLabelMap(ctx, currentLabels, size.width, size.height)
      } else {
        drawGrayVideo(ctx, video, canvas.width, canvas.height)
      }
    } else {
      drawGrayVideo(ctx, video, canvas.width, canvas.height)
      if (mode === 'boxes') {
        drawBoxes(ctx, components())
      }
    }
    if (paused() || processing) {
      return
    }
    processing = true
    const start = Date.now()
    const wantsLabels = viewMode() === 'components'
    const enc = gpuRoot.device.createCommandEncoder({ label: 'hex-debug-frame' })
    currentPipeline.encode(enc, video, tuning())
    gpuRoot.device.queue.submit([enc.finish()])
    void Promise.all([
      currentPipeline.readComponents(),
      wantsLabels ? currentPipeline.readLabels() : Promise.resolve(undefined),
    ])
      .then(([frame, labelMap]) => {
        if (disposed || pipeline() !== currentPipeline) {
          return
        }
        const now = Date.now()
        if (lastFrameAt > 0) {
          const instant = 1000 / Math.max(1, now - lastFrameAt)
          setFps((value) => (value === 0 ? instant : value * 0.85 + instant * 0.15))
        }
        lastFrameAt = now
        setProcessingMs(now - start)
        setComponents(frame.components)
        if (labelMap) {
          setLabels(labelMap)
          if (viewMode() === 'components') {
            drawLabelMap(ctx, labelMap, size.width, size.height)
          }
        } else if (viewMode() === 'boxes') {
          drawGrayVideo(ctx, video, canvas.width, canvas.height)
          drawBoxes(ctx, frame.components)
        } else if (viewMode() === 'gray') {
          drawGrayVideo(ctx, video, canvas.width, canvas.height)
        }
      })
      .catch((cause: unknown) => {
        if (!disposed) {
          setError(cause instanceof Error ? cause.message : String(cause))
        }
      })
      .finally(() => {
        processing = false
      })
  }

  createEffect(
    () => ({
      video: videoElement(),
      size: videoSize(),
      activePipeline: pipeline(),
    }),
    ({ video, size, activePipeline }) => {
      if (!video || !size || !activePipeline) {
        return
      }
      const loop = createFrameLoop({
        video,
        onFrame: () => {
          if (!paused()) {
            processFrame(video, size)
          }
        },
      })
      return () => loop.dispose()
    },
  )

  const darkCount = createMemo(() => components().filter((component) => component.color === 0).length)
  const lightCount = createMemo(() => components().filter((component) => component.color === 1).length)
  const medianArea = createMemo(() => median(components().map((component) => component.area)))
  const detectedRatio = createMemo(() => (expected() > 0 ? components().length / expected() : 0))

  return (
    <div class={styles.root}>
      <div class={styles.display}>
        <canvas ref={setCanvasElement} class={styles.canvas} aria-label="Hex Debug component overlay" />
        <video ref={setVideoElement} class={styles.videoSource} muted playsinline />
        <Show when={error()}>{(message) => <div class={styles.overlayError}>{message()}</div>}</Show>
      </div>

      <aside class={styles.controls}>
        <h1 class={styles.title}>Hex Debug</h1>
        <p class={styles.note}>
          Gray → per-pixel variance-aware connected components. Uniform components merge strictly; high-variance
          components merge more freely.
        </p>

        <div class={styles.cameraControls}>
          <CameraStreamSelects />
        </div>

        <div class={styles.field}>
          <span class={styles.label}>View</span>
          <div class={styles.buttonRow}>
            <button
              type="button"
              class={viewMode() === 'gray' ? styles.activeButton : styles.button}
              onClick={() => setViewMode('gray')}
            >
              Gray
            </button>
            <button
              type="button"
              class={viewMode() === 'components' ? styles.activeButton : styles.button}
              onClick={() => setViewMode('components')}
            >
              Components
            </button>
            <button
              type="button"
              class={viewMode() === 'boxes' ? styles.activeButton : styles.button}
              onClick={() => setViewMode('boxes')}
            >
              Boxes
            </button>
          </div>
        </div>

        <label class={styles.field}>
          <span class={styles.label}>Merge iterations: {iterations()}</span>
          <input
            class={styles.range}
            type="range"
            min="2"
            max="24"
            step="1"
            value={iterations()}
            onInput={(event) => setIterations(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Base tolerance: {baseTolerance().toFixed(3)}</span>
          <input
            class={styles.range}
            type="range"
            min="0"
            max="0.2"
            step="0.001"
            value={baseTolerance()}
            onInput={(event) => setBaseTolerance(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Variance multiplier z: {zFactor().toFixed(2)}</span>
          <input
            class={styles.range}
            type="range"
            min="0"
            max="4"
            step="0.05"
            value={zFactor()}
            onInput={(event) => setZFactor(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Max tolerance: {maxTolerance().toFixed(3)}</span>
          <input
            class={styles.range}
            type="range"
            min="0.02"
            max="0.4"
            step="0.005"
            value={maxTolerance()}
            onInput={(event) => setMaxTolerance(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Minimum area: {minArea()} px</span>
          <input
            class={styles.range}
            type="range"
            min="1"
            max="50"
            step="1"
            value={minArea()}
            onInput={(event) => {
              const next = Number(event.currentTarget.value)
              setMinArea(next)
              if (maxArea() < next) {
                setMaxArea(next)
              }
            }}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Maximum area: {maxArea()} px</span>
          <input
            class={styles.range}
            type="range"
            min="1"
            max="100"
            step="1"
            value={maxArea()}
            onInput={(event) => setMaxArea(Math.max(minArea(), Number(event.currentTarget.value)))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Expected markers</span>
          <input
            class={styles.numberInput}
            type="number"
            min="1"
            value={expected()}
            onInput={(event) => setExpected(Math.max(1, Number(event.currentTarget.value)))}
          />
        </label>

        <div class={styles.buttonRow}>
          <button type="button" class={styles.primaryButton} onClick={() => setPaused((value) => !value)}>
            {paused() ? 'Resume' : 'Pause'}
          </button>
        </div>

        <dl class={styles.metrics}>
          <div>
            <dt>resolution</dt>
            <dd>{videoSize() ? `${videoSize()!.width} × ${videoSize()!.height}` : '—'}</dd>
          </div>
          <div>
            <dt>components</dt>
            <dd>{components().length}</dd>
          </div>
          <div>
            <dt>dark / light</dt>
            <dd>
              {darkCount()} / {lightCount()}
            </dd>
          </div>
          <div>
            <dt>detected / expected</dt>
            <dd>{(detectedRatio() * 100).toFixed(1)}%</dd>
          </div>
          <div>
            <dt>median area</dt>
            <dd>{medianArea().toFixed(1)} px</dd>
          </div>
          <div>
            <dt>regions</dt>
            <dd>1×1 pixels</dd>
          </div>
          <div>
            <dt>iterations</dt>
            <dd>{iterations()}</dd>
          </div>
          <div>
            <dt>process</dt>
            <dd>{processingMs()} ms</dd>
          </div>
          <div>
            <dt>fps</dt>
            <dd>{fps().toFixed(1)}</dd>
          </div>
        </dl>

        <p class={styles.note}>
          Metrics use the min/max area filter. The Components view shows every labeled component; the gray view shows
          the untouched camera image.
        </p>
      </aside>
    </div>
  )
}
