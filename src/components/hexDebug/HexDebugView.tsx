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

const PRESENTATION_FORMAT = globalThis.navigator.gpu?.getPreferredCanvasFormat() ?? 'bgra8unorm'

const DEFAULT_TUNING = {
  baseTolerance: 0.05,
  z: 1.5,
  maxTolerance: 0.2,
  minArea: 1,
  maxArea: 100,
  minFillPercent: 60,
  minContrast: 0.15,
  maxRingVariance: 0.01,
  ringMargin: 4,
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
  const [canvasContext, setCanvasContext] = createSignal<ReturnType<TgpuRoot['configureContext']>>()
  const [viewMode, setViewMode] = createSignal<'gray' | 'components' | 'boxes' | 'doh' | 'nms'>('gray')
  const [processingMs, setProcessingMs] = createSignal(0)
  const [fps, setFps] = createSignal(0)
  const [paused, setPaused] = createSignal(false)
  const [iterations, setIterations] = createSignal<number>(DEFAULT_TUNING.iterations)
  const [baseTolerance, setBaseTolerance] = createSignal<number>(DEFAULT_TUNING.baseTolerance)
  const [zFactor, setZFactor] = createSignal<number>(DEFAULT_TUNING.z)
  const [maxTolerance, setMaxTolerance] = createSignal<number>(DEFAULT_TUNING.maxTolerance)
  const [minArea, setMinArea] = createSignal<number>(DEFAULT_TUNING.minArea)
  const [maxArea, setMaxArea] = createSignal<number>(DEFAULT_TUNING.maxArea)
  const [minFillPercent, setMinFillPercent] = createSignal<number>(DEFAULT_TUNING.minFillPercent)
  const [minContrast, setMinContrast] = createSignal<number>(DEFAULT_TUNING.minContrast)
  const [maxRingVariance, setMaxRingVariance] = createSignal<number>(DEFAULT_TUNING.maxRingVariance)
  const [ringMargin, setRingMargin] = createSignal<number>(DEFAULT_TUNING.ringMargin)
  const [showRejections, setShowRejections] = createSignal(false)
  const [nmsThreshold, setNmsThreshold] = createSignal(0.08)
  const [expected, setExpected] = createSignal(420)

  let disposed = false
  let processing = false
  let metricsPending = false
  let lastFrameAt = 0

  const tuning = createMemo<HexDebugTuning>(() => ({
    baseTolerance: baseTolerance(),
    z: zFactor(),
    maxTolerance: maxTolerance(),
    minArea: minArea(),
    maxArea: maxArea(),
    minFillPercent: minFillPercent(),
    minContrast: minContrast(),
    maxRingVariance: maxRingVariance(),
    ringMargin: ringMargin(),
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
      const nextPipeline = createHexDebugPipeline(
        gpuRoot,
        size.width,
        size.height,
        untrack(tuning),
        PRESENTATION_FORMAT,
      )
      setPipeline(nextPipeline)
      return () => {
        nextPipeline.destroy()
      }
    },
  )

  createEffect(
    () => ({
      activePipeline: pipeline(),
      threshold: nmsThreshold(),
    }),
    ({ activePipeline, threshold }) => {
      activePipeline?.setNmsThreshold(threshold)
    },
  )

  createEffect(
    () => ({
      gpuRoot: root(),
      canvas: canvasElement(),
    }),
    ({ gpuRoot, canvas }) => {
      if (!gpuRoot || !canvas) {
        return
      }
      const context = gpuRoot.configureContext({ canvas, alphaMode: 'opaque', format: PRESENTATION_FORMAT })
      setCanvasContext(context)
      return () => {
        setCanvasContext(undefined)
      }
    },
  )

  const processFrame = (video: HTMLVideoElement, size: { width: number; height: number }): void => {
    const canvas = canvasElement()
    const gpuRoot = root()
    const currentPipeline = pipeline()
    const context = canvasContext()
    if (!canvas || !gpuRoot || !currentPipeline || !context) {
      return
    }
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width
      canvas.height = size.height
    }
    if (paused() || processing) {
      return
    }
    processing = true
    const start = Date.now()
    const mode = viewMode()
    try {
      const enc = gpuRoot.device.createCommandEncoder({ label: 'hex-debug-frame' })
      currentPipeline.encode(enc, video, tuning(), { doh: mode === 'doh' || mode === 'nms' })
      currentPipeline.render(enc, context.getCurrentTexture().createView(), mode, components().length, showRejections())
      gpuRoot.device.queue.submit([enc.finish()])
    } catch (cause: unknown) {
      processing = false
      setError(cause instanceof Error ? cause.message : String(cause))
      return
    }
    processing = false
    if (metricsPending) {
      return
    }
    metricsPending = true
    void currentPipeline
      .readComponents()
      .then((frame) => {
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
      })
      .catch((cause: unknown) => {
        if (!disposed) {
          setError(cause instanceof Error ? cause.message : String(cause))
        }
      })
      .finally(() => {
        metricsPending = false
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

  const acceptedComponents = createMemo(() => components().filter((component) => component.rejection === 'accepted'))
  const darkCount = createMemo(() => acceptedComponents().filter((component) => component.color === 0).length)
  const lightCount = createMemo(() => acceptedComponents().filter((component) => component.color === 1).length)
  const medianArea = createMemo(() => median(acceptedComponents().map((component) => component.area)))
  const detectedRatio = createMemo(() => (expected() > 0 ? acceptedComponents().length / expected() : 0))

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
            <button
              type="button"
              class={viewMode() === 'doh' ? styles.activeButton : styles.button}
              onClick={() => setViewMode('doh')}
            >
              DoH
            </button>
            <button
              type="button"
              class={viewMode() === 'nms' ? styles.activeButton : styles.button}
              onClick={() => setViewMode('nms')}
            >
              NMS
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
          <span class={styles.label}>Minimum fill: {minFillPercent()}%</span>
          <input
            class={styles.range}
            type="range"
            min="0"
            max="100"
            step="1"
            value={minFillPercent()}
            onInput={(event) => setMinFillPercent(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Ring margin: {ringMargin()} px</span>
          <input
            class={styles.range}
            type="range"
            min="1"
            max="10"
            step="1"
            value={ringMargin()}
            onInput={(event) => setRingMargin(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Minimum contrast: {minContrast().toFixed(3)}</span>
          <input
            class={styles.range}
            type="range"
            min="0"
            max="0.5"
            step="0.005"
            value={minContrast()}
            onInput={(event) => setMinContrast(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Max ring variance: {maxRingVariance().toFixed(3)}</span>
          <input
            class={styles.range}
            type="range"
            min="0"
            max="0.2"
            step="0.001"
            value={maxRingVariance()}
            onInput={(event) => setMaxRingVariance(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>NMS threshold: {(nmsThreshold() * 100).toFixed(1)}% max</span>
          <input
            class={styles.range}
            type="range"
            min="0.01"
            max="0.5"
            step="0.005"
            value={nmsThreshold()}
            onInput={(event) => setNmsThreshold(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.toggle}>
          <input
            type="checkbox"
            checked={showRejections()}
            onChange={(event) => setShowRejections(event.currentTarget.checked)}
          />
          Show rejections
        </label>

        <div class={styles.legend}>
          <span class={styles.label}>Rejection colors</span>
          <div>
            <i style={{ background: '#22c55e' }} /> accepted
          </div>
          <div>
            <i style={{ background: '#f97316' }} /> fill / too thin
          </div>
          <div>
            <i style={{ background: '#ef4444' }} /> ring not uniform
          </div>
          <div>
            <i style={{ background: '#3b82f6' }} /> low contrast
          </div>
          <div>
            <i style={{ background: '#eab308' }} /> no ring
          </div>
        </div>

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
            <dt>accepted / candidates</dt>
            <dd>
              {acceptedComponents().length} / {components().length}
            </dd>
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
