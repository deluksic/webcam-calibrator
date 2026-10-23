import type { Component } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { initSharedGPUForOnnx } from '@/gpu/init'
import type { RoiComponentPoint } from '@/gpu/roiLab/roiComponentsPipeline'
import type { RoiGraphData } from '@/gpu/roiLab/roiGraphPipeline'
import {
  ROI_MODEL_WIDTH,
  createRoiLabPipeline,
  type RoiHistogramUpdate,
  type RoiLabPipeline,
  type RoiStepTimings,
} from '@/gpu/roiLab/roiLabPipeline'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { computeFingerprints, createIdentityIndexesByDepth } from '@/lib/pointIdentity/identity'
import {
  createLocalIcpOptions,
  MAX_FINGERPRINT_DEPTH,
  refineObservedIdentityWithLocalIcp,
} from '@/lib/pointIdentity/localIcpRefiner'
import {
  createColorIdentityIndex,
  IDENTITY_AMBIGUOUS,
  IDENTITY_CONFLICT,
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  IDENTITY_UNRESOLVED,
  matchObservedIdentity,
} from '@/lib/pointIdentity/match'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import { ROI_HISTOGRAM_BINS } from '@/lib/roiHistogram'
import { createFrameLoop } from '@/utils/createFrameLoop'

import styles from '@/components/roiLab/RoiLabView.module.css'

const { performance } = globalThis
const QUERY = new URLSearchParams(globalThis.location.search)
const MODEL_URL = `${import.meta.env.BASE_URL}models/roi_tiny_no_concat.onnx`
const DEMO_IMAGE_URL = QUERY.get('image') ?? `${import.meta.env.BASE_URL}demo/roi_target.png`
const DEMO_MODE = QUERY.has('demo')
const [DEMO_WIDTH, DEMO_HEIGHT] = (() => {
  const [width, height] = (QUERY.get('demoSize') ?? '640x480').split('x').map(Number)
  return [width || 640, height || 480]
})()

const MIN_ZOOM = 1
const MAX_ZOOM = 12
const WHEEL_SENSITIVITY = 0.0018
const IDENTITY_DEFAULTS = {
  columns: Math.min(14, Math.max(2, Number(QUERY.get('columns')) || 8)),
  pointsPerEdge: Math.min(6, Math.max(1, Number(QUERY.get('points')) || 4)),
  colorSeed: Math.max(0, Math.round(Number(QUERY.get('seed')) || 1)),
  depth: Math.min(
    MAX_FINGERPRINT_DEPTH,
    Math.max(1, Number(QUERY.get('depth')) || MAX_FINGERPRINT_DEPTH),
  ),
} as const

type RoiViewMode = 'blobs' | 'components' | 'labels' | 'identity' | 'mask' | 'input' | 'camera'

type ViewOption = {
  mode: RoiViewMode
  label: string
  hint: string
}

/** Ordered by the pipeline: camera -> model input -> mask -> blobs -> components -> identity. */
const VIEW_OPTIONS: readonly ViewOption[] = [
  { mode: 'camera', label: 'Camera', hint: 'Raw camera frame' },
  { mode: 'input', label: 'Input', hint: 'Normalized grayscale fed to the ROI model' },
  { mode: 'mask', label: 'Mask', hint: 'Raw ROI mask prediction' },
  { mode: 'blobs', label: 'Blobs', hint: 'Histogram-segmented black and white blobs inside the mask' },
  { mode: 'labels', label: 'Labels', hint: 'Connected components colored randomly' },
  { mode: 'components', label: 'Centroids', hint: 'Binary components and their centroids' },
  { mode: 'identity', label: 'Identity', hint: 'Observed graph with stable design identities' },
]

const VIEW_MODE_INDEX: Record<RoiViewMode, 1 | 2 | 3 | 4 | 6 | 7 | 8> = {
  mask: 1,
  input: 2,
  camera: 3,
  blobs: 4,
  components: 6,
  labels: 7,
  identity: 8,
}

function parseViewMode(value: string | null): RoiViewMode {
  const match = VIEW_OPTIONS.find((option) => option.mode === value)
  return match?.mode ?? 'identity'
}

const INITIAL_VIEW: RoiViewMode = parseViewMode(QUERY.get('view'))

function formatPercent(pixels: number, total: number): string {
  return total > 0 ? `${((pixels / total) * 100).toFixed(1)}%` : '0%'
}

function drawRoiHistogram(canvas: HTMLCanvasElement, update: RoiHistogramUpdate): void {
  const context = canvas.getContext('2d')
  if (!context) {
    return
  }
  const width = canvas.width
  const height = canvas.height
  const plotHeight = height - 16
  context.clearRect(0, 0, width, height)
  context.fillStyle = '#050506'
  context.fillRect(0, 0, width, height)

  let maxCount = 1
  for (const count of update.counts) {
    maxCount = Math.max(maxCount, count)
  }
  const logMax = Math.log2(maxCount + 1)
  const barWidth = width / ROI_HISTOGRAM_BINS
  const { analysis } = update
  for (let bin = 0; bin < ROI_HISTOGRAM_BINS; bin++) {
    const count = update.counts[bin] ?? 0
    const barHeight = (Math.log2(count + 1) / logMax) * plotHeight
    let color = '#394150'
    if (analysis.valid && bin < analysis.blackCut) {
      color = '#3f7dff'
    } else if (analysis.valid && bin >= analysis.whiteCut) {
      color = '#ffae2b'
    }
    context.fillStyle = color
    context.fillRect(bin * barWidth, plotHeight - barHeight, Math.max(1, barWidth), barHeight)
  }

  const drawCut = (bin: number, color: string) => {
    const x = Math.round((bin / ROI_HISTOGRAM_BINS) * width) + 0.5
    context.strokeStyle = color
    context.lineWidth = 1
    context.beginPath()
    context.moveTo(x, 0)
    context.lineTo(x, plotHeight)
    context.stroke()
  }
  if (analysis.valid && analysis.blackCut > 0) {
    drawCut(analysis.blackCut, '#8ab4ff')
  }
  if (analysis.valid && analysis.whiteCut < ROI_HISTOGRAM_BINS) {
    drawCut(analysis.whiteCut, '#ffd27a')
  }

  context.fillStyle = '#8b93a1'
  context.font = '10px monospace'
  context.fillText('0', 2, height - 3)
  context.fillText('255', width - 20, height - 3)
}

type IdentityMetrics = {
  total: number
  resolved: number
  ambiguous: number
  unresolved: number
  conflicts: number
  edges: number
  components: number
  recovered: number
  frameId: number
}

type IdentityPerf = {
  readbackMs: number
  identityMs: number
  frameId: number
}

export const RoiLabView: Component = () => {
  const cam = useCameraStream()
  const [videoElement, setVideoElement] = createSignal<HTMLVideoElement>()
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [videoSize, setVideoSize] = createSignal<{ width: number; height: number } | undefined>(undefined, {
    equals: (a, b) => a?.width === b?.width && a?.height === b?.height,
  })
  const [error, setError] = createSignal<string>()
  const [ready, setReady] = createSignal(false)
  const [phase, setPhase] = createSignal('loading')
  const [roiPipeline, setRoiPipeline] = createSignal<RoiLabPipeline>()
  const [viewMode, setViewMode] = createSignal<RoiViewMode>(INITIAL_VIEW)
  const [paused, setPaused] = createSignal(false)
  const [zoom, setZoom] = createSignal(1)
  const [panX, setPanX] = createSignal(0)
  const [panY, setPanY] = createSignal(0)
  const [bufferStats, setBufferStats] = createSignal('')
  const [inferenceMs, setInferenceMs] = createSignal(0)
  const [totalMs, setTotalMs] = createSignal(0)
  const [fps, setFps] = createSignal(0)
  const [histogramData, setHistogramData] = createSignal<RoiHistogramUpdate>()
  const [histogramCanvasElement, setHistogramCanvasElement] = createSignal<HTMLCanvasElement>()
  const [identityColumns, setIdentityColumns] = createSignal(IDENTITY_DEFAULTS.columns)
  const [identityPointsPerEdge, setIdentityPointsPerEdge] = createSignal(IDENTITY_DEFAULTS.pointsPerEdge)
  const [identityColorSeed, setIdentityColorSeed] = createSignal(IDENTITY_DEFAULTS.colorSeed)
  const [identityDepth, setIdentityDepth] = createSignal(IDENTITY_DEFAULTS.depth)
  const [identityRecovery, setIdentityRecovery] = createSignal(true)
  const [identityMetrics, setIdentityMetrics] = createSignal<IdentityMetrics>()
  const [identityPerf, setIdentityPerf] = createSignal<IdentityPerf>()
  const [stepTimings, setStepTimings] = createSignal<RoiStepTimings>()
  const [seedScanRange, setSeedScanRange] = createSignal(512)
  const [seedScan, setSeedScan] = createSignal<{
    scanning: boolean
    progress: number
    bestSeed?: number
    bestResolved?: number
  }>()
  let lastObservedPoints: readonly RoiComponentPoint[] = []

  let disposed = false

  onCleanup(() => {
    disposed = true
  })

  const identityDesign = createMemo(() => {
    const pointsPerEdge = identityPointsPerEdge()
    const pattern = generateHexPattern({
      columns: identityColumns(),
      pointsPerEdge,
    })
    const colors = randomBinaryColors(pattern.points.length, identityColorSeed())
    const graph = buildProximityGraph(pattern.points, { pointsPerEdge })
    const indexesByDepth = createIdentityIndexesByDepth(
      graph,
      colors,
      MAX_FINGERPRINT_DEPTH,
    )
    return { pattern, colors, graph, indexesByDepth, pointsPerEdge }
  })

  const identityIndex = createMemo(
    () => identityDesign().indexesByDepth[identityDepth() - 1]!,
  )
  const identityLooseIndex = createMemo(() =>
    createColorIdentityIndex(identityDesign().graph, identityDesign().colors, identityDepth()),
  )

  const runIdentity = (points: readonly RoiComponentPoint[], frameId: number): RoiGraphData => {
    lastObservedPoints = points
    const design = identityDesign()
    const depth = identityDepth()
    const observedPoints = points.map((point, index) => ({
      x: point.x,
      y: point.y,
      index,
    }))
    const observedColors = points.map((point) => point.color)
    const observedGraph = buildProximityGraph(observedPoints, {
      pointsPerEdge: design.pointsPerEdge,
    })
    const baselineFingerprints = computeFingerprints(observedGraph, observedColors, depth)
    const baseline = matchObservedIdentity(
      identityIndex(),
      observedGraph,
      observedColors,
      depth,
      identityRecovery()
        ? { fingerprints: baselineFingerprints }
        : { looseByKey: identityLooseIndex(), fingerprints: baselineFingerprints },
    )

    let finalPoints = baseline.points
    if (identityRecovery()) {
      const recoveryFingerprints =
        depth === MAX_FINGERPRINT_DEPTH
          ? baselineFingerprints
          : computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
      finalPoints = refineObservedIdentityWithLocalIcp({
        designPattern: design.pattern,
        designColors: design.colors,
        designGraph: design.graph,
        designIndexesByDepth: design.indexesByDepth,
        observedGraph,
        observedColors,
        observedFingerprints: recoveryFingerprints,
        baseline,
        options: createLocalIcpOptions({
          identityDepth: depth,
          maxProofDepth: MAX_FINGERPRINT_DEPTH,
          minProofPower: 7,
          hopRadius: 8,
        }),
      }).points
    }

    let resolved = 0
    let ambiguous = 0
    let unresolved = 0
    let conflicts = 0
    let recovered = 0
    for (const point of finalPoints) {
      switch (point.status) {
        case IDENTITY_RESOLVED:
          resolved += 1
          break
        case IDENTITY_AMBIGUOUS:
          ambiguous += 1
          break
        case IDENTITY_UNRESOLVED:
          unresolved += 1
          break
        case IDENTITY_CONFLICT:
          conflicts += 1
          break
        case IDENTITY_RECOVERED:
          recovered += 1
          break
      }
    }
    setIdentityMetrics({
      total: finalPoints.length,
      resolved,
      ambiguous,
      unresolved,
      conflicts,
      edges: observedGraph.edges.length,
      components: observedGraph.stats.components,
      recovered,
      frameId,
    })
    return {
      points: finalPoints.map((point) => ({
        x: point.x,
        y: point.y,
        designId: point.designId ?? 0,
        status: point.status,
      })),
      edges: observedGraph.edges.map(([a, b]) => ({ a, b })),
    }
  }

  const scanSeeds = async (): Promise<void> => {
    const points = lastObservedPoints
    if (points.length === 0 || seedScan()?.scanning === true) {
      return
    }
    const design = identityDesign()
    const depth = identityDepth()
    const observedPoints = points.map((point, index) => ({ x: point.x, y: point.y, index }))
    const observedColors = points.map((point) => point.color)
    const observedGraph = buildProximityGraph(observedPoints, { pointsPerEdge: design.pointsPerEdge })
    const observedFingerprints = computeFingerprints(observedGraph, observedColors, depth)
    const designIndex = identityIndex()
    const limit = Math.min(10000, Math.max(0, Math.round(seedScanRange())))
    let bestSeed: number | undefined
    let bestScore = Number.NEGATIVE_INFINITY
    let bestResolved = 0
    setSeedScan({ scanning: true, progress: 0 })
    try {
      for (let seed = 0; seed <= limit; seed += 1) {
        const designColors = randomBinaryColors(design.pattern.points.length, seed)
        const looseIndex = createColorIdentityIndex(design.graph, designColors, depth)
        const matched = matchObservedIdentity(designIndex, observedGraph, observedColors, depth, {
          looseByKey: looseIndex,
          fingerprints: observedFingerprints,
        })
        const score =
          matched.summary.resolved * 100 -
          matched.summary.conflicts * 200 -
          matched.summary.ambiguous * 2 -
          matched.summary.unresolved
        if (score > bestScore) {
          bestScore = score
          bestSeed = seed
          bestResolved = matched.summary.resolved
        }
        if (seed % 16 === 0) {
          setSeedScan({ scanning: true, progress: (seed + 1) / (limit + 1) })
          await new Promise<void>((resolve) => setTimeout(resolve, 0))
        }
      }
    } finally {
      setSeedScan({
        scanning: false,
        progress: 1,
        bestSeed,
        bestResolved,
      })
    }
    if (bestSeed !== undefined && bestResolved > 0) {
      setIdentityColorSeed(bestSeed)
    }
  }

  createEffect(
    () => ({ pipeline: roiPipeline(), mode: viewMode() }),
    ({ pipeline, mode }) => {
      pipeline?.setViewMode(VIEW_MODE_INDEX[mode])
    },
  )

  createEffect(
    () => ({ pipeline: roiPipeline(), paused: paused() }),
    ({ pipeline, paused }) => {
      pipeline?.setPaused(paused)
    },
  )

  createEffect(
    () => ({
      pipeline: roiPipeline(),
      zoom: zoom(),
      panX: panX(),
      panY: panY(),
    }),
    ({ pipeline, zoom, panX, panY }) => {
      pipeline?.setViewport(zoom, panX, panY)
    },
  )

  createEffect(
    () => ({ update: histogramData(), canvas: histogramCanvasElement() }),
    ({ update, canvas }) => {
      if (!update || !canvas) {
        return
      }
      drawRoiHistogram(canvas, update)
    },
  )

  // Reusing a WebGPU canvas across a camera resolution change leaves the swap
  // chain/context stale, and the first 720p frame comes out black. A reload is
  // cheap here and starts the page directly at the requested resolution.
  createEffect(
    () => cam.selectedResolution(),
    (resolution, previous) => {
      if (DEMO_MODE || previous === undefined || resolution === previous) {
        return
      }
      const params = new URLSearchParams(globalThis.location.search)
      params.set('resolution', resolution)
      globalThis.location.search = params.toString()
    },
  )

  createEffect(cam.stream, (stream) => {
    const video = videoElement()
    if (DEMO_MODE || !video || !stream) {
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
    if (DEMO_MODE) {
      let cancelled = false
      let frameHandle = 0
      let stream: MediaStream | undefined
      const canvas = document.createElement('canvas')
      canvas.width = DEMO_WIDTH
      canvas.height = DEMO_HEIGHT
      const context2d = canvas.getContext('2d')
      const image = new Image()
      image.src = DEMO_IMAGE_URL
      const draw = (): void => {
        if (cancelled) {
          return
        }
        context2d?.drawImage(image, 0, 0, DEMO_WIDTH, DEMO_HEIGHT)
        frameHandle = requestAnimationFrame(draw)
      }
      image.onload = () => {
        if (cancelled) {
          return
        }
        console.log('[roi-lab] demo image loaded', image.naturalWidth, image.naturalHeight)
        stream = canvas.captureStream(30)
        console.log('[roi-lab] demo stream', Boolean(stream))
        video.srcObject = stream
        setVideoSize({ width: DEMO_WIDTH, height: DEMO_HEIGHT })
        void video.play().catch((cause) => {
          console.error('[roi-lab] demo video play failed', cause)
        })
        draw()
      }
      image.onerror = () => {
        console.error('[roi-lab] demo image failed', DEMO_IMAGE_URL)
      }
      return () => {
        cancelled = true
        cancelAnimationFrame(frameHandle)
        stream?.getTracks().forEach((track) => track.stop())
        video.srcObject = null
      }
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
      video: videoElement(),
      canvas: canvasElement(),
      size: videoSize(),
    }),
    ({ video, canvas, size }) => {
      if (!video || !canvas || !size || size.width < 8 || size.height < 8) {
        return
      }

      let cancelled = false
      let loop: ReturnType<typeof createFrameLoop> | undefined
      let pipeline: RoiLabPipeline | undefined
      let inFlight: Promise<void> | undefined
      let busy = false
      let lastFrameAt = 0

      setZoom(1)
      setPanX(0)
      setPanY(0)
      const dpr = Math.max(1, globalThis.devicePixelRatio || 1)
      canvas.width = Math.max(1, Math.round(size.width * dpr))
      canvas.height = Math.max(1, Math.round(size.height * dpr))
      canvas.style.width = `${size.width}px`
      canvas.style.height = `${size.height}px`
      // Keep the trained pixel scale (256 px wide), but let the height follow
      // the camera aspect so a 16:9 frame is not center-cropped to 4:3.
      // The decoder upsamples with scale factors 4 and 2. Its feature sizes
      // only line up when the input height is a multiple of 16 (e.g. 108 would
      // give encoder 27 rows but 24 upsampled rows). Round up so scale-factor
      // Resize stays exact without reintroducing dynamic-size Concat nodes.
      const rawModelHeight = Math.round((ROI_MODEL_WIDTH * size.height) / size.width)
      const modelHeight = Math.min(224, Math.max(96, Math.ceil(rawModelHeight / 16) * 16))
      setError(undefined)
      setReady(false)
      setPhase('starting')

      void (async () => {
        try {
          console.log('[roi-lab] init shared GPU')
          const shared = await initSharedGPUForOnnx()
          if (cancelled) {
            return
          }
          pipeline = await createRoiLabPipeline(
            shared.root,
            shared.device,
            canvas,
            size.width,
            size.height,
            modelHeight,
            MODEL_URL,
            (message) => {
              console.log('[roi-lab]', message)
              setPhase(message)
            },
            (update) => {
              if (!cancelled && !disposed && !paused()) {
                setHistogramData(update)
              }
            },
            (points, frameId) => {
              if (cancelled || disposed) {
                return undefined
              }
              return runIdentity(points, frameId)
            },
          )
          console.log('[roi-lab] pipeline ready')
          setRoiPipeline(pipeline)
          if (cancelled) {
            pipeline.destroy()
            return
          }
          setReady(true)
          setPhase('running')
          loop = createFrameLoop({
            video,
            onFrame: () => {
              if (busy || paused() || !pipeline || video.readyState < 2) {
                return
              }
              busy = true
              const frame = pipeline.processFrame(video)
              inFlight = frame
                .then((timings) => {
                  if (cancelled || disposed) {
                    return
                  }
                  const now = performance.now()
                  if (lastFrameAt > 0) {
                    setFps(1000 / Math.max(1, now - lastFrameAt))
                  }
                  lastFrameAt = now
                  setInferenceMs(timings.inferenceMs)
                  setTotalMs(timings.totalMs)
                  setIdentityPerf({
                    readbackMs: timings.readbackMs,
                    identityMs: timings.identityMs,
                    frameId: timings.frameId,
                  })
                  setStepTimings(timings.steps)
                })
                .catch((cause: unknown) => {
                  if (!cancelled && !disposed) {
                    setError(cause instanceof Error ? cause.message : String(cause))
                  }
                })
                .finally(() => {
                  busy = false
                  inFlight = undefined
                })
            },
          })
        } catch (cause: unknown) {
          console.error('[roi-lab] pipeline setup failed', cause)
          if (!cancelled) {
            setError(cause instanceof Error ? cause.message : String(cause))
            setPhase('error')
          }
        }
      })()

      return () => {
        cancelled = true
        loop?.dispose()
        setRoiPipeline(undefined)
        const oldPipeline = pipeline
        pipeline = undefined
        if (inFlight) {
          // An in-flight processFrame may still be using the session and
          // textures; destroy only after it settles.
          void inFlight.finally(() => oldPipeline?.destroy())
        } else {
          oldPipeline?.destroy()
        }
      }
    },
  )

  const zoomAt = (factor: number, anchorX: number, anchorY: number): void => {
    const current = zoom()
    const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current * factor))
    if (Math.abs(next - current) < 1e-6) {
      return
    }
    if (next <= MIN_ZOOM + 1e-4) {
      setZoom(MIN_ZOOM)
      setPanX(0)
      setPanY(0)
      return
    }
    const ratio = next / current
    setPanX(anchorX - (anchorX - panX()) * ratio)
    setPanY(anchorY - (anchorY - panY()) * ratio)
    setZoom(next)
  }

  const zoomFromCenter = (factor: number): void => {
    const size = videoSize()
    if (!size) {
      return
    }
    zoomAt(factor, size.width / 2, size.height / 2)
  }

  const resetViewport = (): void => {
    setZoom(1)
    setPanX(0)
    setPanY(0)
  }

  const handleWheel = (event: WheelEvent): void => {
    event.preventDefault()
    const canvas = canvasElement()
    const size = videoSize()
    if (!canvas || !size) {
      return
    }
    const rect = canvas.getBoundingClientRect()
    if (rect.width <= 0 || rect.height <= 0) {
      return
    }
    const anchorX = ((event.clientX - rect.left) / rect.width) * size.width
    const anchorY = ((event.clientY - rect.top) / rect.height) * size.height
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY
    const factor = Math.exp(-delta * WHEEL_SENSITIVITY)
    if (!Number.isFinite(factor) || factor <= 0) {
      return
    }
    zoomAt(factor, anchorX, anchorY)
  }

  return (
    <div class={styles.root}>
      <div class={styles.display}>
        <video ref={setVideoElement} class={styles.videoSource} muted playsinline />
        <canvas ref={setCanvasElement} class={styles.canvas} onWheel={handleWheel} />
        <div class={styles.zoomHud}>
          <button type="button" class={styles.zoomButton} aria-label="Zoom out" onClick={() => zoomFromCenter(1 / 1.5)}>
            −
          </button>
          <span class={styles.zoomValue}>{zoom().toFixed(1)}×</span>
          <button type="button" class={styles.zoomButton} aria-label="Zoom in" onClick={() => zoomFromCenter(1.5)}>
            +
          </button>
          <button type="button" class={styles.zoomReset} onClick={resetViewport}>
            Reset
          </button>
        </div>
        <Show when={paused()}>
          <div class={styles.pausedBadge}>Paused</div>
        </Show>
        <Show when={error()}>{(message) => <div class={styles.overlayError}>{message()}</div>}</Show>
      </div>
      <aside class={styles.controls}>
        <h1 class={styles.title}>ROI Lab</h1>
        <p class={styles.note}>
          Histogram-driven blob segmentation inside the target mask. The tiny U-Net is a 74.0k-parameter ONNX model at
          256 px wide, with the height following the camera aspect. A full-resolution raw-luma histogram inside the mask
          picks black/gray/white cuts. The component path applies a color-aware 2×2 opening, pointer-jump
          connected-component labeling, centroid extraction, and close-centroid NMS. Labels shows one random color per
          component; Centroids shows their centers; Identity shows the observed graph and stable design-ID colors.
        </p>
        <Show when={DEMO_MODE}>
          <p class={styles.note}>
            Demo source: a static target-on-office composite is streamed through a canvas. Add
            <code>?demo=1</code> before the hash to use it without a camera.
          </p>
        </Show>
        <CameraStreamSelects />
        <div class={styles.viewSection}>
          <div class={styles.viewHeader}>
            <span class={styles.viewLabel}>Pipeline view</span>
            <button
              type="button"
              class={[styles.pauseButton, paused() && styles.pauseButtonActive]}
              aria-pressed={paused() ? 'true' : 'false'}
              onClick={() => {
                setPaused((value) => !value)
              }}
            >
              {paused() ? 'Resume' : 'Pause'}
            </button>
          </div>
          <div class={styles.viewBar} role="group" aria-label="Pipeline view">
            {VIEW_OPTIONS.map((option) => (
              <button
                type="button"
                class={[styles.viewButton, viewMode() === option.mode && styles.viewButtonActive]}
                aria-pressed={viewMode() === option.mode ? 'true' : 'false'}
                title={option.hint}
                onClick={() => {
                  setViewMode(option.mode)
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
          <p class={styles.viewHint}>
            Identity: design IDs · Labels: colors · Centroids: centers.
          </p>
        </div>
        <div class={styles.identityPanel}>
          <div class={styles.identityHeader}>
            <span>Identity target</span>
            <span class={styles.identityPointCount}>{identityDesign().pattern.stats.points} design pts</span>
          </div>
          <div class={styles.identityGrid}>
            <label class={styles.identityField}>
              Columns
              <input
                type="number"
                min="2"
                max="14"
                value={identityColumns()}
                onChange={(event) => {
                  setIdentityColumns(Math.min(14, Math.max(2, Math.round(Number(event.currentTarget.value) || 8))))
                }}
              />
            </label>
            <label class={styles.identityField}>
              Points/edge
              <input
                type="number"
                min="1"
                max="6"
                value={identityPointsPerEdge()}
                onChange={(event) => {
                  setIdentityPointsPerEdge(Math.min(6, Math.max(1, Math.round(Number(event.currentTarget.value) || 4))))
                }}
              />
            </label>
            <label class={styles.identityField}>
              Color seed
              <input
                type="number"
                min="0"
                max="9999"
                value={identityColorSeed()}
                onChange={(event) => {
                  setIdentityColorSeed(Math.max(0, Math.round(Number(event.currentTarget.value) || 0)))
                }}
              />
            </label>
            <label class={styles.identityField}>
              Depth
              <input
                type="number"
                min="1"
                max={MAX_FINGERPRINT_DEPTH}
                value={identityDepth()}
                onChange={(event) => {
                  setIdentityDepth(
                    Math.min(
                      MAX_FINGERPRINT_DEPTH,
                      Math.max(1, Math.round(Number(event.currentTarget.value) || MAX_FINGERPRINT_DEPTH)),
                    ),
                  )
                }}
              />
            </label>
          </div>
          <label class={styles.identityToggle}>
            <input
              type="checkbox"
              checked={identityRecovery()}
              onChange={(event) => setIdentityRecovery(event.currentTarget.checked)}
            />
            Local homography recovery
          </label>
          <div class={styles.seedScan}>
            <label class={styles.identityField}>
              Scan 0…
              <input
                type="number"
                min="0"
                max="10000"
                value={seedScanRange()}
                onChange={(event) => {
                  setSeedScanRange(Math.min(10000, Math.max(0, Math.round(Number(event.currentTarget.value) || 0))))
                }}
              />
            </label>
            <button
              type="button"
              class={styles.seedScanButton}
              disabled={seedScan()?.scanning === true}
              onClick={() => {
                void scanSeeds()
              }}
            >
              {seedScan()?.scanning === true ? 'Scanning…' : 'Find seed'}
            </button>
            <Show when={seedScan()}>
              {(scan) => (
                <span class={styles.seedScanStatus}>
                  {scan().scanning
                    ? `${Math.round(scan().progress * 100)}%`
                    : scan().bestSeed === undefined
                      ? 'no matches'
                      : `best ${scan().bestSeed} · ${scan().bestResolved} resolved`}
                </span>
              )}
            </Show>
          </div>
          <Show
            when={identityMetrics()}
            fallback={<p class={styles.identityHint}>Waiting for the first identity frame…</p>}
          >
            {(metrics) => (
              <dl class={styles.identityStats}>
                <div>
                  <dt>frame</dt>
                  <dd>{metrics().frameId}</dd>
                </div>
                <div>
                  <dt>points</dt>
                  <dd>{metrics().total}</dd>
                </div>
                <div>
                  <dt>edges</dt>
                  <dd>{metrics().edges}</dd>
                </div>
                <div>
                  <dt>components</dt>
                  <dd>{metrics().components}</dd>
                </div>
                <div>
                  <dt>resolved</dt>
                  <dd>{metrics().resolved}</dd>
                </div>
                <div>
                  <dt>ambiguous</dt>
                  <dd>{metrics().ambiguous}</dd>
                </div>
                <div>
                  <dt>unresolved</dt>
                  <dd>{metrics().unresolved}</dd>
                </div>
                <div>
                  <dt>recovered</dt>
                  <dd>{metrics().recovered}</dd>
                </div>
                <div>
                  <dt>conflicts</dt>
                  <dd>{metrics().conflicts}</dd>
                </div>
                <Show when={identityPerf()}>
                  {(perf) => (
                    <>
                      <div>
                        <dt>readback</dt>
                        <dd>{perf().readbackMs.toFixed(1)} ms</dd>
                      </div>
                      <div>
                        <dt>identity</dt>
                        <dd>{perf().identityMs.toFixed(1)} ms</dd>
                      </div>
                    </>
                  )}
                </Show>
              </dl>
            )}
          </Show>
        </div>
        <div class={styles.histogramPanel}>
          <div class={styles.histogramHeader}>
            <span>Masked luma histogram</span>
            <Show when={histogramData()}>
              {(update) => (
                <span class={styles.histogramCount}>{update().analysis.maskedPixels.toLocaleString()} px</span>
              )}
            </Show>
          </div>
          <canvas ref={setHistogramCanvasElement} class={styles.histogramCanvas} width={300} height={110} />
          <Show
            when={histogramData()}
            fallback={<p class={styles.histogramHint}>Waiting for the first histogram sample…</p>}
          >
            {(update) => (
              <>
                <div class={styles.histogramLegend}>
                  <span>
                    <i class={[styles.swatch, styles.swatchBlack]} />
                    {formatPercent(update().analysis.blackPixels, update().analysis.maskedPixels)} black
                  </span>
                  <span>
                    <i class={[styles.swatch, styles.swatchWhite]} />
                    {formatPercent(update().analysis.whitePixels, update().analysis.maskedPixels)} white
                  </span>
                  <span class={styles.histogramCuts}>
                    {update().analysis.valid
                      ? `cuts ${update().analysis.blackCut} / ${update().analysis.whiteCut}`
                      : 'no black/gray/white split'}
                  </span>
                </div>
                <p class={styles.histogramStatus}>
                  {update().analysis.valid
                    ? update().analysis.blackCut === 0
                      ? 'Black class off: no dark population separated from the gray target.'
                      : update().analysis.whiteCut === ROI_HISTOGRAM_BINS
                        ? 'White class off: no bright population separated from the gray target.'
                        : 'Three-level split: black / gray / white.'
                    : 'No split: the gray target is not dominant, or the cuts sit too close to it.'}
                </p>
              </>
            )}
          </Show>
        </div>
        <button
          class={styles.button}
          onClick={() => {
            const active = roiPipeline()
            if (!active) {
              return
            }
            setBufferStats('reading…')
            void active
              .dumpStats()
              .then((stats) =>
                setBufferStats(
                  `input  min ${stats.input.min.toFixed(3)}  mean ${stats.input.mean.toFixed(3)}  max ${stats.input.max.toFixed(3)}\n` +
                    `output min ${stats.output.min.toFixed(3)}  mean ${stats.output.mean.toFixed(3)}  max ${stats.output.max.toFixed(3)}`,
                ),
              )
              .catch((cause: unknown) => setBufferStats(cause instanceof Error ? cause.message : String(cause)))
          }}
        >
          Read input/output buffers
        </button>
        <Show when={bufferStats()}>
          <pre class={styles.stats}>{bufferStats()}</pre>
        </Show>
        <dl class={styles.metrics}>
          <dt>Model</dt>
          <dd>tiny ROI U-Net</dd>
          <dt>Input</dt>
          <dd>256 × height</dd>
          <dt>ONNX</dt>
          <dd>297 KiB</dd>
          <Show when={stepTimings()}>
            {(timings) => (
              <>
                <dt>Stats readback</dt>
                <dd>
                  {timings().imageStatsMs.toFixed(2)} ms · age {timings().statsAgeFrames}
                  {timings().statsRefreshed ? ' · refreshed' : ''}
                </dd>
                <dt>Preprocess</dt>
                <dd>{timings().preprocessMs.toFixed(2)} ms</dd>
                <dt>Queue wait</dt>
                <dd>{timings().queueWaitMs.toFixed(2)} ms</dd>
                <dt>Camera copy</dt>
                <dd>{timings().cameraCopyMs.toFixed(2)} ms</dd>
                <dt>Mask</dt>
                <dd>{timings().maskMs.toFixed(2)} ms</dd>
                <dt>Background</dt>
                <dd>{timings().backgroundMs.toFixed(2)} ms</dd>
                <dt>Components</dt>
                <dd>{timings().componentsEncodeMs.toFixed(2)} ms</dd>
                <dt>Debug input</dt>
                <dd>{timings().debugInputMs.toFixed(2)} ms</dd>
                <dt>Readback</dt>
                <dd>
                  {timings().readbackMs.toFixed(2)} ms · {timings().readbackComponents} components ·{' '}
                  {(timings().readbackBytes / 1024).toFixed(1)} KiB
                  {timings().readbackTruncated ? ' · truncated at cap' : ''}
                </dd>
                <dt>Identity</dt>
                <dd>{timings().identityMs.toFixed(2)} ms</dd>
                <dt>Render</dt>
                <dd>{timings().renderMs.toFixed(2)} ms</dd>
                <dt>Inference run</dt>
                <dd>{timings().inferenceRunMs.toFixed(2)} ms</dd>
                <dt>Inference drain</dt>
                <dd>{timings().inferenceDrainMs.toFixed(2)} ms</dd>
                <dt>Frame drain</dt>
                <dd>{timings().frameDrainMs.toFixed(2)} ms</dd>
                <dt>Histogram submit</dt>
                <dd>{timings().histogramMs.toFixed(2)} ms</dd>
                <dt>Accounted / total</dt>
                <dd>
                  {timings().accountedMs.toFixed(2)} / {timings().totalMs.toFixed(2)} ms
                </dd>
                <dt>Overhead</dt>
                <dd>{timings().overheadMs.toFixed(2)} ms</dd>
                <dt>GPU preprocess</dt>
                <dd>{timings().gpuMs.preprocess?.toFixed(2) ?? '—'} ms</dd>
                <dt>GPU copy</dt>
                <dd>{timings().gpuMs.cameraCopy?.toFixed(2) ?? '—'} ms</dd>
                <dt>GPU mask</dt>
                <dd>{timings().gpuMs.mask?.toFixed(2) ?? '—'} ms</dd>
                <dt>GPU bg blur</dt>
                <dd>
                  {(
                    (timings().gpuMs.backgroundWeight ?? 0) +
                    (timings().gpuMs.backgroundHorizontal ?? 0) +
                    (timings().gpuMs.backgroundVertical ?? 0) +
                    (timings().gpuMs.backgroundTexture ?? 0)
                  ).toFixed(2)}{' '}
                  ms
                </dd>
              </>
            )}
          </Show>
          <dt>Inference</dt>
          <dd>{inferenceMs().toFixed(2)} ms</dd>
          <dt>Total</dt>
          <dd>{totalMs().toFixed(2)} ms</dd>
          <dt>Process FPS</dt>
          <dd>{fps().toFixed(0)}</dd>
          <dt>Status</dt>
          <dd>{error() ? `error: ${phase()}` : paused() ? 'paused' : ready() ? 'running' : phase()}</dd>
        </dl>
        <p class={styles.status}>
          The 74.0k-parameter GPU-compatible model runs on ONNX Runtime WebGPU using shared GPU buffers for input and
          output. Input is 256 px wide with dynamic height (e.g. 256×192 for 4:3, 256×144 for 16:9), so the full camera
          frame is kept.
        </p>
      </aside>
    </div>
  )
}
