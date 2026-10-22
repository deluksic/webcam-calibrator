import type { Component } from 'solid-js'
import { Show, createMemo, createSignal, createTrackedEffect, onCleanup } from 'solid-js'
import type { TgpuRoot } from 'typegpu'

import { createHexTargetRenderer, type HexTargetRenderer } from '@/gpu/hexTargetRenderer'
import { initGPU } from '@/gpu/init'
import { buildHexTargetLayout } from '@/lib/hexBlobTarget'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { createIdentityIndex } from '@/lib/pointIdentity/identity'
import { buildProximityGraph } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import { createElementSize } from '@/utils/createElementSize'

import styles from '@/components/HexTargetView.module.css'

const { navigator } = globalThis

const DEFAULTS = {
  columns: 7,
  pointsPerEdge: 4,
  markerScale: 0.45,
  colorSeed: 1,
} as const

const TARGET_ASPECT = 4 / 3
const FALLBACK_RESOLUTION = { width: 1280, height: 960 }

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export const HexTargetView: Component = () => {
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [stageElement, setStageElement] = createSignal<HTMLDivElement>()
  const [root, setRoot] = createSignal<TgpuRoot>()
  const [renderer, setRenderer] = createSignal<HexTargetRenderer>()
  const [columns, setColumns] = createSignal<number>(DEFAULTS.columns)
  const [pointsPerEdge, setPointsPerEdge] = createSignal<number>(DEFAULTS.pointsPerEdge)
  const [markerScale, setMarkerScale] = createSignal<number>(DEFAULTS.markerScale)
  const [colorSeed, setColorSeed] = createSignal<number>(DEFAULTS.colorSeed)
  const [depth, setDepth] = createSignal(5)
  const [topologyColumns, setTopologyColumns] = createSignal<number>(DEFAULTS.columns)
  const [topologyPointsPerEdge, setTopologyPointsPerEdge] = createSignal<number>(DEFAULTS.pointsPerEdge)
  const [error, setError] = createSignal<string>()
  const [pixelRatio, setPixelRatio] = createSignal(window.devicePixelRatio || 1)

  const stageSize = createElementSize(stageElement)
  const availableStage = createMemo(() => {
    const size = stageSize()
    if (!size) {
      return undefined
    }
    const ratio = pixelRatio()
    const expectedWidth = size.width * ratio
    const expectedHeight = size.height * ratio
    return {
      widthPX: Math.abs(size.widthPX - expectedWidth) > 1 ? Math.round(expectedWidth) : size.widthPX,
      heightPX: Math.abs(size.heightPX - expectedHeight) > 1 ? Math.round(expectedHeight) : size.heightPX,
    }
  })
  const resolution = createMemo(() => {
    const stage = availableStage()
    if (!stage || stage.widthPX < 16 || stage.heightPX < 16) {
      return FALLBACK_RESOLUTION
    }
    const widthPX = Math.max(640, Math.min(stage.widthPX, Math.floor(stage.heightPX * TARGET_ASPECT)))
    const heightPX = Math.max(480, Math.floor(widthPX / TARGET_ASPECT))
    return { width: widthPX, height: heightPX }
  })
  const layout = createMemo(() =>
    buildHexTargetLayout({
      widthPx: resolution().width,
      heightPx: resolution().height,
      columns: columns(),
      pointsPerEdge: pointsPerEdge(),
      markerScale: markerScale(),
      colorSeed: colorSeed(),
    }),
  )

  // Topology depends only on the lattice, not on resolution, marker scale, or
  // color seed. Debounce slider changes so dragging does not rebuild it on
  // every intermediate column/points value.
  createTrackedEffect(() => {
    const nextColumns = columns()
    const nextPointsPerEdge = pointsPerEdge()
    const timer = setTimeout(() => {
      setTopologyColumns(nextColumns)
      setTopologyPointsPerEdge(nextPointsPerEdge)
    }, 180)
    return () => clearTimeout(timer)
  })
  const designPattern = createMemo(() =>
    generateHexPattern({ columns: topologyColumns(), pointsPerEdge: topologyPointsPerEdge() }),
  )
  const designColors = createMemo(() => randomBinaryColors(designPattern().points.length, colorSeed()))
  const designGraph = createMemo(() =>
    buildProximityGraph(designPattern().points, { pointsPerEdge: topologyPointsPerEdge() }),
  )
  const fingerprintIndex = createMemo(() => createIdentityIndex(designGraph(), designColors(), depth()))
  const uniqueness = createMemo(() => {
    const index = fingerprintIndex()
    const total = index.fingerprints.length
    return {
      uniquePercent: total === 0 ? 0 : (total - index.collidingPointCount) / total,
      colliding: index.collidingPointCount,
    }
  })

  let disposed = false
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
  const updatePixelRatio = (): void => {
    setPixelRatio(window.devicePixelRatio || 1)
  }
  window.addEventListener('resize', updatePixelRatio)
  onCleanup(() => {
    disposed = true
    window.removeEventListener('resize', updatePixelRatio)
  })

  createTrackedEffect(() => {
    const gpuRoot = root()
    const canvas = canvasElement()
    if (!gpuRoot || !canvas) {
      return
    }
    const format = navigator.gpu.getPreferredCanvasFormat()
    const targetRenderer = createHexTargetRenderer(gpuRoot, canvas, format)
    setRenderer(targetRenderer)
    return () => {
      targetRenderer.destroy()
    }
  })

  createTrackedEffect(() => {
    const targetRenderer = renderer()
    const nextLayout = layout()
    if (!targetRenderer) {
      return
    }
    targetRenderer.resize(nextLayout.widthPx, nextLayout.heightPx)
    targetRenderer.setLayout(nextLayout)
    targetRenderer.render()
  })

  const shuffleColors = (): void => {
    setColorSeed((seed) => seed + 1)
  }

  const enterFullscreen = (): void => {
    const stage = stageElement()
    if (!stage) {
      return
    }
    void stage.requestFullscreen().catch((cause: unknown) => {
      setError(cause instanceof Error ? cause.message : String(cause))
    })
  }

  const cssTargetSize = createMemo(() => ({
    width: resolution().width / pixelRatio(),
    height: resolution().height / pixelRatio(),
  }))
  const accuracyStatus = createMemo(() => {
    const nextLayout = layout()
    return `${nextLayout.widthPx} × ${nextLayout.heightPx} target px · spacing ${nextLayout.spacingPx} · side ${nextLayout.sidePx}`
  })

  const downloadDefinition = (): void => {
    const nextLayout = layout()
    const payload = {
      version: 1,
      resolution: { widthPx: nextLayout.widthPx, heightPx: nextLayout.heightPx },
      padding: { xPx: nextLayout.paddingXPx, yPx: nextLayout.paddingYPx },
      columns: nextLayout.columns,
      pointsPerEdge: nextLayout.pointsPerEdge,
      markerScale: markerScale(),
      colorSeed: colorSeed(),
      quads: nextLayout.quads,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `hex-target-${nextLayout.widthPx}x${nextLayout.heightPx}-seed${colorSeed()}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div class={styles.root}>
      <div class={styles.display} ref={setStageElement}>
        <canvas
          ref={setCanvasElement}
          class={styles.canvas}
          style={{
            width: `${cssTargetSize().width}px`,
            height: `${cssTargetSize().height}px`,
          }}
          aria-label="Hexagon target of black and white squares"
        />
        <Show when={error()}>{(message) => <div class={styles.canvasError}>{message()}</div>}</Show>
      </div>

      <aside class={styles.controls}>
        <h1 class={styles.title}>Hex Target</h1>
        <p class={styles.note}>
          A gray screen with isolated black and white squares. The target runs at least 640×480, then uses the
          maximum 4:3 resolution that fits the available stage. CSS only centers it; it is never stretched or
          resampled.
        </p>

        <label class={styles.field}>
          <span class={styles.label}>Columns: {columns()}</span>
          <input
            class={styles.range}
            type="range"
            min="4"
            max="16"
            step="1"
            value={columns()}
            onInput={(event) => setColumns(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Points per edge: {pointsPerEdge()}</span>
          <input
            class={styles.range}
            type="range"
            min="4"
            max="6"
            step="1"
            value={pointsPerEdge()}
            onInput={(event) => setPointsPerEdge(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Marker scale: {markerScale().toFixed(2)}</span>
          <input
            class={styles.range}
            type="range"
            min="0.25"
            max="0.65"
            step="0.01"
            value={markerScale()}
            onInput={(event) => setMarkerScale(Number(event.currentTarget.value))}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Hop depth: {depth()}</span>
          <input
            class={styles.range}
            type="range"
            min="1"
            max="6"
            step="1"
            value={depth()}
            onInput={(event) => setDepth(Number(event.currentTarget.value))}
          />
        </label>

        <div class={styles.seedRow}>
          <span class={styles.label}>Color seed: {colorSeed()}</span>
          <button type="button" class={styles.button} onClick={shuffleColors}>
            Shuffle colors
          </button>
        </div>

        <button type="button" class={styles.primaryButton} onClick={enterFullscreen}>
          Fullscreen target
        </button>
        <button type="button" class={styles.button} onClick={downloadDefinition}>
          Download target JSON
        </button>

        <p class={styles.status}>{accuracyStatus()}</p>
        <p class={styles.status}>
          {layout().quads.length} markers · {percent(uniqueness().uniquePercent)} unique fingerprints ·{' '}
          {uniqueness().colliding} colliding points · depth {depth()}
        </p>
        <Show when={error()}>
          <p class={styles.error}>GPU unavailable: {error()}</p>
        </Show>

        <p class={styles.footnote}>
          The canvas backing store is the selected target resolution, centered at 1:1 device pixels. Use 100% page zoom
          for an exact display.
        </p>
      </aside>
    </div>
  )
}
