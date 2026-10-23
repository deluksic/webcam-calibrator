import qrcode from 'qrcode-generator'
import type { Component } from 'solid-js'
import { Show, createMemo, createSignal, createTrackedEffect, onCleanup } from 'solid-js'
import type { TgpuRoot } from 'typegpu'

import { createHexTargetRenderer, type HexTargetRenderer } from '@/gpu/hexTargetRenderer'
import { initGPU } from '@/gpu/init'
import { buildHexTargetLayout } from '@/lib/hexBlobTarget'
import {
  buildCalibrationUrl,
  normalizeHexTargetConfig,
  readHexTargetConfigFromLocation,
  type HexTargetConfig,
} from '@/lib/hexTarget/config'
import { createHexTargetDesign } from '@/lib/hexTarget/target'
import { createElementSize } from '@/utils/createElementSize'

import styles from '@/components/HexTargetView.module.css'

const { navigator } = globalThis

const TARGET_ASPECT = 4 / 3
const FALLBACK_RESOLUTION = { width: 1280, height: 960 }

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export const HexTargetView: Component = () => {
  const [config, setConfig] = createSignal<HexTargetConfig>(readHexTargetConfigFromLocation())
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const [stageElement, setStageElement] = createSignal<HTMLDivElement>()
  const [root, setRoot] = createSignal<TgpuRoot>()
  const [renderer, setRenderer] = createSignal<HexTargetRenderer>()
  const [error, setError] = createSignal<string>()
  const [pixelRatio, setPixelRatio] = createSignal(window.devicePixelRatio || 1)

  const patchConfig = (patch: Partial<HexTargetConfig>): void => {
    setConfig((current) => normalizeHexTargetConfig({ ...current, ...patch }))
  }

  const model = createMemo(() => createHexTargetDesign(config()))
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
  const autoResolution = createMemo(() => {
    const stage = availableStage()
    if (!stage || stage.widthPX < 16 || stage.heightPX < 16) {
      return FALLBACK_RESOLUTION
    }
    const widthPX = Math.max(1, Math.min(stage.widthPX, Math.floor(stage.heightPX * TARGET_ASPECT)))
    const heightPX = Math.max(1, Math.floor(widthPX / TARGET_ASPECT))
    return { width: widthPX, height: heightPX }
  })
  const resolution = createMemo(() => {
    const current = config()
    if (!current.widthPx || !current.heightPx) {
      return autoResolution()
    }
    const stage = availableStage()
    if (!stage || stage.widthPX < 16 || stage.heightPX < 16) {
      return { width: current.widthPx, height: current.heightPx }
    }
    // Never scale the target: clamp the requested pixel size to the physical
    // display area while preserving the requested aspect ratio.
    const scale = Math.min(1, stage.widthPX / current.widthPx, stage.heightPX / current.heightPx)
    return {
      width: Math.max(1, Math.floor(current.widthPx * scale)),
      height: Math.max(1, Math.floor(current.heightPx * scale)),
    }
  })
  const layout = createMemo(() =>
    buildHexTargetLayout({
      widthPx: resolution().width,
      heightPx: resolution().height,
      columns: config().columns,
      rows: config().rows,
      pointsPerEdge: config().pointsPerEdge,
      markerScale: config().markerScale,
      colorSeed: config().colorSeed,
    }),
  )
  const uniqueness = createMemo(() => {
    const depth = Math.min(config().identityDepth, model().indexesByDepth.length)
    const index = model().indexesByDepth[Math.max(0, depth - 1)]!
    const total = index.fingerprints.length
    return {
      uniquePercent: total === 0 ? 0 : (total - index.collidingPointCount) / total,
      colliding: index.collidingPointCount,
    }
  })
  const calibrationUrl = createMemo(() =>
    buildCalibrationUrl(config(), { widthPx: resolution().width, heightPx: resolution().height }),
  )
  const qrDataUrl = createMemo(() => {
    const qr = qrcode(0, 'M')
    qr.addData(calibrationUrl())
    qr.make()
    return qr.createDataURL(4, 8)
  })

  let disposed = false
  let gpuRetries = 0
  const startGPU = (): void => {
    void initGPU()
      .then((gpuRoot) => {
        if (disposed) {
          return
        }
        setRoot(gpuRoot)
        // A lost device makes every subsequent getCurrentTexture/render a
        // no-op, which looks exactly like "the target did not render". Drop the
        // renderer and reinitialize on a fresh shared device.
        void gpuRoot.device.lost.then((info) => {
          if (disposed) {
            return
          }
          setError(`GPU device lost (${info.reason}); reinitializing…`)
          setRenderer(undefined)
          setRoot(undefined)
          if (gpuRetries < 3) {
            gpuRetries += 1
            // The shared-GPU cache clears its own lost-device handler first.
            setTimeout(startGPU, 0)
          }
        })
      })
      .catch((cause: unknown) => {
        if (disposed) {
          return
        }
        setError(cause instanceof Error ? cause.message : String(cause))
        if (gpuRetries < 3) {
          gpuRetries += 1
          setTimeout(startGPU, 300)
        }
      })
  }
  startGPU()
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
    let targetRenderer: HexTargetRenderer
    try {
      const format = navigator.gpu.getPreferredCanvasFormat()
      targetRenderer = createHexTargetRenderer(gpuRoot, canvas, format)
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : String(cause))
      return
    }
    setError(undefined)
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
    const draw = (): void => {
      targetRenderer.resize(nextLayout.widthPx, nextLayout.heightPx)
      targetRenderer.setLayout(nextLayout)
      targetRenderer.render()
      // Only a successful draw clears the retry budget; a device that is lost
      // again before the next paint should still hit the cap.
      gpuRetries = 0
    }
    draw()
    // The first frame can land before the browser has composited the freshly
    // mounted canvas after a route change. Draw once more after layout.
    const frame = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frame)
  })

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
  // What the user asked for, before the "fit the display" clamp. The inputs
  // must show this value, otherwise editing one axis writes the already
  // clamped other axis back into the config and silently changes the aspect.
  const requestedWidth = createMemo(() => config().widthPx ?? resolution().width)
  const requestedHeight = createMemo(() => config().heightPx ?? resolution().height)
  const accuracyStatus = createMemo(() => {
    const nextLayout = layout()
    const current = config()
    const clamped =
      (current.widthPx !== undefined && current.widthPx !== nextLayout.widthPx) ||
      (current.heightPx !== undefined && current.heightPx !== nextLayout.heightPx)
    return `${current.columns} × ${current.rows} cells · ${nextLayout.widthPx} × ${nextLayout.heightPx} device px${clamped ? ' (clamped to display)' : ''} · spacing ${nextLayout.spacingPx} · side ${nextLayout.sidePx}`
  })

  const downloadDefinition = (): void => {
    const nextLayout = layout()
    const current = config()
    const payload = {
      version: 1,
      kind: 'hex-target',
      parameters: { ...current, widthPx: nextLayout.widthPx, heightPx: nextLayout.heightPx },
      resolution: { widthPx: nextLayout.widthPx, heightPx: nextLayout.heightPx },
      padding: { xPx: nextLayout.paddingXPx, yPx: nextLayout.paddingYPx },
      quads: nextLayout.quads,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `hex-target-${current.columns}x${current.rows}-${nextLayout.widthPx}x${nextLayout.heightPx}.json`
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
          Edit the target, then scan the QR code with the camera device. Every setting—including the target
          resolution—is encoded in the link so the calibration page rebuilds exactly this target.
        </p>

        <div class={styles.qrSection}>
          <span class={styles.label}>Open Calibrate on your phone</span>
          <a class={styles.qrLink} href={calibrationUrl()}>
            <img class={styles.qr} src={qrDataUrl()} alt="QR code linking to this target configuration on Calibrate" />
          </a>
          <a class={styles.qrUrl} href={calibrationUrl()}>
            {calibrationUrl()}
          </a>
        </div>

        <button type="button" class={styles.primaryButton} onClick={enterFullscreen}>
          Fullscreen target
        </button>

        <div class={styles.field}>
          <span class={styles.label}>Target layout</span>
          <div class={styles.row}>
            <label class={styles.fieldLabel}>
              Columns
              <input
                class={styles.numberInput}
                type="number"
                min="2"
                max="24"
                value={config().columns}
                onChange={(event) => patchConfig({ columns: Number(event.currentTarget.value) })}
              />
            </label>
            <label class={styles.fieldLabel}>
              Rows
              <input
                class={styles.numberInput}
                type="number"
                min="2"
                max="24"
                value={config().rows}
                onChange={(event) => patchConfig({ rows: Number(event.currentTarget.value) })}
              />
            </label>
          </div>
        </div>

        <label class={styles.field}>
          <span class={styles.label}>Points per cell edge</span>
          <input
            class={styles.numberInput}
            type="number"
            min="1"
            max="8"
            step="1"
            value={config().pointsPerEdge}
            onChange={(event) => patchConfig({ pointsPerEdge: Number(event.currentTarget.value) })}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Marker scale</span>
          <input
            class={styles.numberInput}
            type="number"
            min="0.2"
            max="0.7"
            step="0.01"
            value={config().markerScale}
            onChange={(event) => patchConfig({ markerScale: Number(event.currentTarget.value) })}
          />
        </label>

        <label class={styles.field}>
          <span class={styles.label}>Color seed</span>
          <input
            class={styles.numberInput}
            type="number"
            min="0"
            max="999999"
            value={config().colorSeed}
            onChange={(event) => patchConfig({ colorSeed: Number(event.currentTarget.value) })}
          />
        </label>

        <div class={styles.field}>
          <span class={styles.label}>Target resolution</span>
          <div class={styles.row}>
            <label class={styles.fieldLabel}>
              Width
              <input
                class={styles.numberInput}
                type="number"
                min="320"
                max="8192"
                value={requestedWidth()}
                onChange={(event) =>
                  patchConfig({
                    widthPx: Number(event.currentTarget.value),
                    heightPx: requestedHeight(),
                  })
                }
              />
            </label>
            <label class={styles.fieldLabel}>
              Height
              <input
                class={styles.numberInput}
                type="number"
                min="240"
                max="8192"
                value={requestedHeight()}
                onChange={(event) =>
                  patchConfig({
                    widthPx: requestedWidth(),
                    heightPx: Number(event.currentTarget.value),
                  })
                }
              />
            </label>
          </div>
          <button
            type="button"
            class={styles.button}
            onClick={() => patchConfig({ widthPx: undefined, heightPx: undefined })}
          >
            Use window-fit resolution
          </button>
        </div>

        <p class={styles.status}>{accuracyStatus()}</p>
        <p class={styles.status}>
          {layout().quads.length} markers · {percent(uniqueness().uniquePercent)} unique fingerprints ·{' '}
          {uniqueness().colliding} colliding points · depth {config().identityDepth}
        </p>
        <Show when={error()}>
          <p class={styles.error}>GPU unavailable: {error()}</p>
        </Show>

        <button type="button" class={styles.button} onClick={downloadDefinition}>
          Download target JSON
        </button>

        <p class={styles.footnote}>
          The canvas backing store is the selected target resolution, centered at 1:1 device pixels. Use 100% page zoom
          for an exact display.
        </p>
      </aside>
    </div>
  )
}
