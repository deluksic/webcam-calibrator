import type { Component } from 'solid-js'
import { Show, createEffect, createMemo, createSignal, onCleanup } from 'solid-js'

import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { CameraStreamSelects } from '@/components/camera/CameraStreamSelects'
import { PeakInspector } from '@/components/roiLab/PeakInspector'
import { initSharedGPUForOnnx } from '@/gpu/init'
import type { RoiGraphData } from '@/gpu/roiLab/roiGraphPipeline'
import {
  ROI_MODEL_WIDTH,
  createRoiLabPipeline,
  type RoiHistogramUpdate,
  type RoiLabPipeline,
  type RoiPeakFrameAnalysis,
  type RoiStepTimings,
} from '@/gpu/roiLab/roiLabPipeline'
import type { RoiDetectedPoint } from '@/gpu/roiLab/roiPeakTypes'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { computeFingerprints, createIdentityIndexesByDepth } from '@/lib/pointIdentity/identity'
import type { IdentityDesignSpec, IdentityMetrics } from '@/lib/pointIdentity/identityPipeline'
import { MAX_FINGERPRINT_DEPTH } from '@/lib/pointIdentity/localIcpRefiner'
import { createColorIdentityIndex, matchObservedIdentity } from '@/lib/pointIdentity/match'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import { ROI_HISTOGRAM_BINS } from '@/lib/roiHistogram'
import { pointerToFrame } from '@/lib/roiLabPointer'
import { describeRefinedPeak } from '@/lib/roiNeighbourFilter'
import { createFrameLoop } from '@/utils/createFrameLoop'
import { configureIdentityWorker, disposeIdentityWorker, runIdentityInWorker } from '@/workers/identityClient'

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
  depth: Math.min(MAX_FINGERPRINT_DEPTH, Math.max(1, Number(QUERY.get('depth')) || MAX_FINGERPRINT_DEPTH)),
} as const

type RoiViewMode = 'blobs' | 'peaks' | 'identity' | 'mask' | 'input' | 'camera'

type ViewOption = {
  mode: RoiViewMode
  label: string
  hint: string
}

/** Ordered by the pipeline: camera -> model input -> mask -> blobs -> peaks -> identity. */
const VIEW_OPTIONS: readonly ViewOption[] = [
  { mode: 'camera', label: 'Camera', hint: 'Raw camera frame' },
  { mode: 'input', label: 'Input', hint: 'Normalized grayscale fed to the ROI model' },
  { mode: 'mask', label: 'Mask', hint: 'Raw ROI mask prediction' },
  { mode: 'blobs', label: 'Blobs', hint: 'Histogram-segmented black and white blobs inside the mask' },
  { mode: 'peaks', label: 'Peaks', hint: 'Refined 5×5 cubic peaks, before the worker 4σ neighbour filter' },
  { mode: 'identity', label: 'Identity', hint: 'Observed graph with stable design identities' },
]

const VIEW_MODE_INDEX: Record<RoiViewMode, 1 | 2 | 3 | 4 | 6 | 8> = {
  mask: 1,
  input: 2,
  camera: 3,
  blobs: 4,
  peaks: 6,
  identity: 8,
}

function parseViewMode(value: string | null): RoiViewMode {
  const match = VIEW_OPTIONS.find((option) => option.mode === value)
  return match?.mode ?? 'identity'
}

const INITIAL_VIEW: RoiViewMode = parseViewMode(QUERY.get('view'))

/** `mapAsync` rejects with a device-lost error when mobile Chrome loses the GPU device. */
function isGpuLossMessage(message: string): boolean {
  return /device.*lost|mapAsync/i.test(message)
}

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
  /** Bumped when the GPU device is lost so the pipeline effect rebuilds. */
  const [pipelineRetryNonce, setPipelineRetryNonce] = createSignal(0)
  let gpuLossRestarts = 0
  const [viewMode, setViewMode] = createSignal<RoiViewMode>(INITIAL_VIEW)
  const [paused, setPaused] = createSignal(false)
  const [zoom, setZoom] = createSignal(1)
  const [panX, setPanX] = createSignal(0)
  const [panY, setPanY] = createSignal(0)
  const [bufferStats, setBufferStats] = createSignal('')
  const [inferenceFallbackMs, setInferenceFallbackMs] = createSignal(0)
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
  const [peakAnalysis, setPeakAnalysis] = createSignal<RoiPeakFrameAnalysis>()
  const [selectedPeak, setSelectedPeak] = createSignal<{ x: number; y: number }>()
  const [seedScanRange, setSeedScanRange] = createSignal(512)
  const [seedScan, setSeedScan] = createSignal<{
    scanning: boolean
    progress: number
    bestSeed?: number
    bestResolved?: number
  }>()
  let lastObservedPoints: readonly RoiDetectedPoint[] = []

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
    const indexesByDepth = createIdentityIndexesByDepth(graph, colors, MAX_FINGERPRINT_DEPTH)
    return { pattern, colors, graph, indexesByDepth, pointsPerEdge }
  })

  const identityIndex = createMemo(() => identityDesign().indexesByDepth[identityDepth() - 1]!)

  /** Design inputs the worker needs; rebuilt there only when one of these changes. */
  const identitySpec = createMemo<IdentityDesignSpec>(() => ({
    columns: identityColumns(),
    pointsPerEdge: identityPointsPerEdge(),
    colorSeed: identityColorSeed(),
  }))

  // Build the worker's design (lattice, indexes, ~60 ms) on the settings change
  // instead of on the next frame that happens to need it.
  createEffect(identitySpec, (spec) => {
    configureIdentityWorker(spec)
  })

  onCleanup(() => {
    disposeIdentityWorker()
  })

  const runIdentity = async (
    points: readonly RoiDetectedPoint[],
    frameId: number,
  ): Promise<RoiGraphData | undefined> => {
    lastObservedPoints = points
    try {
      // Awaited by the pipeline before it renders, so the overlay always
      // belongs to the frame it was computed from.
      const result = await runIdentityInWorker(identitySpec(), {
        points,
        depth: identityDepth(),
        recovery: identityRecovery(),
        frameId,
      })
      if (disposed) {
        return undefined
      }
      setIdentityMetrics(result.metrics)
      return result.graph
    } catch (cause: unknown) {
      console.error('[roi-lab] identity worker failed', cause)
      return undefined
    }
  }

  const nearestPeak = (
    analysis: RoiPeakFrameAnalysis,
    target: { x: number; y: number },
    maxDistance = 24,
  ): { index: number; peak: RoiPeakFrameAnalysis['peaks'][number] } | undefined => {
    let bestIndex = -1
    let bestDistanceSq = maxDistance * maxDistance
    for (let index = 0; index < analysis.peaks.length; index += 1) {
      const peak = analysis.peaks[index]!
      const dx = peak.x - target.x
      const dy = peak.y - target.y
      const distanceSq = dx * dx + dy * dy
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq
        bestIndex = index
      }
    }
    return bestIndex >= 0 ? { index: bestIndex, peak: analysis.peaks[bestIndex]! } : undefined
  }

  const selectedPeakDetail = createMemo(() => {
    const analysis = peakAnalysis()
    const target = selectedPeak()
    if (!analysis || !target) {
      return undefined
    }
    const nearest = nearestPeak(analysis, target)
    return nearest ? describeRefinedPeak(analysis.peaks, nearest.index) : undefined
  })

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
    () => ({ pipeline: roiPipeline(), selected: selectedPeak() }),
    ({ pipeline, selected }) => {
      pipeline?.setSelectedPeak(selected)
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

  createEffect(
    () => ({ stream: cam.stream(), video: videoElement() }),
    ({ stream, video }) => {
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
    },
  )

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
      retry: pipelineRetryNonce(),
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
      let gpuLossHandled = false

      const restartAfterGpuLoss = (cause: unknown): void => {
        if (cancelled || disposed || gpuLossHandled) {
          return
        }
        gpuLossHandled = true
        if (gpuLossRestarts >= 3) {
          setError('WebGPU device was lost repeatedly. Reload the page to continue.')
          setPhase('error')
          return
        }
        gpuLossRestarts += 1
        setError(cause instanceof Error ? cause.message : String(cause))
        setReady(false)
        setPhase('GPU lost; restarting')
        // Let the shared-GPU cache clear its lost-device promise first, then
        // rebuild the pipeline against a fresh device.
        setTimeout(() => {
          if (!cancelled && !disposed) {
            setPipelineRetryNonce((value) => value + 1)
          }
        }, 0)
      }

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
          void shared.device.lost.then((info) => {
            restartAfterGpuLoss(
              new Error(`WebGPU device lost (${info.reason}${info.message ? `: ${info.message}` : ''})`),
            )
          })
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
                  setInferenceFallbackMs(timings.steps.inferenceRunMs)
                  setTotalMs(timings.totalMs)
                  setIdentityPerf({
                    readbackMs: timings.readbackMs,
                    identityMs: timings.identityMs,
                    frameId: timings.frameId,
                  })
                  setStepTimings(timings.steps)
                  const analysis = pipeline?.getPresentedPeakAnalysis()
                  setPeakAnalysis(analysis)
                  const target = selectedPeak()
                  if (analysis && target) {
                    const nearest = nearestPeak(analysis, target)
                    if (!nearest) {
                      setSelectedPeak(undefined)
                    } else if (
                      Math.abs(nearest.peak.x - target.x) > 0.01 ||
                      Math.abs(nearest.peak.y - target.y) > 0.01
                    ) {
                      setSelectedPeak({ x: nearest.peak.x, y: nearest.peak.y })
                    }
                  }
                  gpuLossRestarts = 0
                })
                .catch((cause: unknown) => {
                  if (cancelled || disposed) {
                    return
                  }
                  const message = cause instanceof Error ? cause.message : String(cause)
                  if (isGpuLossMessage(message)) {
                    restartAfterGpuLoss(cause)
                  } else {
                    setError(message)
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
            const message = cause instanceof Error ? cause.message : String(cause)
            if (isGpuLossMessage(message)) {
              restartAfterGpuLoss(cause)
            } else {
              setError(message)
              setPhase('error')
            }
          }
        }
      })()

      return () => {
        cancelled = true
        loop?.dispose()
        setRoiPipeline(undefined)
        const oldPipeline = pipeline
        pipeline = undefined
        const destroyOldPipeline = () => {
          try {
            oldPipeline?.destroy()
          } catch (cause: unknown) {
            // Teardown of a lost device can fail; reinitialization is already
            // handled by the shared GPU cache.
            console.error('[roi-lab] pipeline teardown failed', cause)
          }
        }
        if (inFlight) {
          // An in-flight processFrame may still be using the session and
          // textures; destroy only after it settles.
          void inFlight.finally(destroyOldPipeline)
        } else {
          destroyOldPipeline()
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

  const pointerToCanvas = (event: { clientX: number; clientY: number }): ReturnType<typeof pointerToFrame> => {
    const canvas = canvasElement()
    const size = videoSize()
    if (!canvas || !size) {
      return undefined
    }
    const rect = canvas.getBoundingClientRect()
    return pointerToFrame(event.clientX, event.clientY, rect, size.width, size.height, zoom(), panX(), panY())
  }

  const handleWheel = (event: WheelEvent): void => {
    event.preventDefault()
    const point = pointerToCanvas(event)
    if (!point) {
      return
    }
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY
    const factor = Math.exp(-delta * WHEEL_SENSITIVITY)
    if (!Number.isFinite(factor) || factor <= 0) {
      return
    }
    zoomAt(factor, point.canvasX, point.canvasY)
  }

  const handlePointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || viewMode() !== 'peaks') {
      return
    }
    const analysis = peakAnalysis()
    const point = pointerToCanvas(event)
    if (!analysis || analysis.peaks.length === 0 || !point) {
      return
    }
    const threshold = Math.max(4, 5 / zoom())
    let bestIndex = -1
    let bestDistanceSq = threshold * threshold
    for (let index = 0; index < analysis.peaks.length; index += 1) {
      const peak = analysis.peaks[index]!
      const dx = peak.x - point.frameX
      const dy = peak.y - point.frameY
      const distanceSq = dx * dx + dy * dy
      if (distanceSq < bestDistanceSq) {
        bestDistanceSq = distanceSq
        bestIndex = index
      }
    }
    setSelectedPeak(
      bestIndex >= 0
        ? {
            x: analysis.peaks[bestIndex]!.x,
            y: analysis.peaks[bestIndex]!.y,
          }
        : undefined,
    )
  }

  return (
    <div class={styles.root}>
      <div class={styles.display}>
        <video ref={setVideoElement} class={styles.videoSource} muted playsinline />
        <canvas
          ref={setCanvasElement}
          class={styles.canvas}
          style={{ cursor: viewMode() === 'peaks' ? 'crosshair' : 'zoom-in' }}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
        />
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
          Histogram-driven peak detection inside the target mask. The tiny U-Net is a 74.0k-parameter ONNX model at 256
          px wide, with the height following the camera aspect. A full-resolution histogram of the blurred image picks
          black/gray/white cuts, then 3×3 extrema are refined with a 5×5 cubic fit and a definite-Hessian test. Peaks
          shows the refined candidates before the worker 4σ same-polarity neighbour filter; Identity shows the observed
          graph and stable design-ID colors after that filter.
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
            Identity: design IDs · Peaks: refined candidates before the worker filter; click one to inspect.
          </p>
        </div>
        <PeakInspector detail={selectedPeakDetail()} />
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
                  <dt>observed</dt>
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
                <dt>Stats</dt>
                <dd>{timings().statsMs.toFixed(2)} ms</dd>
                <dt>CPU encode</dt>
                <dd>
                  {(
                    timings().preprocessMs +
                    timings().cameraCopyMs +
                    timings().maskMs +
                    timings().backgroundMs +
                    timings().detectionEncodeMs +
                    timings().debugInputMs +
                    timings().renderMs +
                    timings().histogramMs
                  ).toFixed(2)}{' '}
                  ms
                </dd>
                <dt>Readback wait</dt>
                <dd>
                  {timings().readbackMs.toFixed(2)} ms · {(timings().readbackBytes / 1024).toFixed(1)} KiB
                  {timings().readbackTruncated ? ' · truncated' : ''}
                </dd>
                <dt>Neighbour filter</dt>
                <dd>{timings().neighbourFilterMs.toFixed(2)} ms</dd>
                <dt>Identity wait</dt>
                <dd>
                  {timings().identityMs.toFixed(2)} ms · lag {timings().identityLagFrames} frame
                  {timings().identityLagFrames === 1 ? '' : 's'}
                </dd>
                <dt>Inference submit</dt>
                <dd>{timings().inferenceRunMs.toFixed(2)} ms</dd>
                <dt>Accounted / total</dt>
                <dd>
                  {timings().accountedMs.toFixed(2)} / {timings().totalMs.toFixed(2)} ms
                </dd>
                <dt>Overhead</dt>
                <dd>{timings().overheadMs.toFixed(2)} ms</dd>
                <dt>GPU preprocess</dt>
                <dd>{timings().gpuMs.preprocess?.toFixed(2) ?? '—'} ms</dd>
                <dt>GPU background</dt>
                <dd>{timings().gpuMs.background?.toFixed(2) ?? '—'} ms</dd>
                <dt>GPU detection</dt>
                <dd>{timings().gpuMs.detection?.toFixed(2) ?? '—'} ms</dd>
                <dt>GPU present</dt>
                <dd>{((timings().gpuMs.cameraCopy ?? 0) + (timings().gpuMs.render ?? 0)).toFixed(2)} ms</dd>
              </>
            )}
          </Show>
          <dt>Inference</dt>
          <dd>{stepTimings()?.gpuMs.inference?.toFixed(2) ?? inferenceFallbackMs().toFixed(2)} ms</dd>
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
