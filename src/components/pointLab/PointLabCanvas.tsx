import type { Component } from 'solid-js'
import { createSignal, createTrackedEffect } from 'solid-js'

import type {
  BinaryColor,
  IdentityEvaluation,
  MatchOutcome,
  PatternPoint,
  ProximityGraph,
  Vec2,
} from '@/lib/pointIdentity/types'
import { createElementSize } from '@/utils/createElementSize'

import styles from '@/components/pointLab/PointLabView.module.css'

type PointLabCanvasProps = {
  projected: readonly PatternPoint[]
  /** Keeps the viewport anchored to the full field while points are cropped. */
  framePoints?: readonly PatternPoint[]
  colors: readonly BinaryColor[]
  graph: ProximityGraph
  evaluation: IdentityEvaluation
  showGraph: boolean
  zoomToCursor: boolean
  visibleEdgeCount: number
  selectedIndex: number | undefined
  onSelect: (index: number | undefined) => void
}

type ScreenPoint = {
  index: number
  x: number
  y: number
}

type ScreenTransform = {
  points: ScreenPoint[]
}

const PADDING = 44
const HIT_RADIUS = 14
const MAX_ZOOM = 12
const WHEEL_SENSITIVITY = 0.0018

function projectionBounds(
  projected: readonly PatternPoint[],
): { minX: number; maxX: number; minY: number; maxY: number } | undefined {
  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY

  const include = (point: Vec2): void => {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }

  for (const point of projected) {
    include(point)
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return undefined
  }
  return { minX, maxX, minY, maxY }
}

function outcomeColor(outcome: MatchOutcome): string {
  switch (outcome.kind) {
    case 'correct':
      return '#34d399'
    case 'false-match':
      return '#f87171'
    case 'ambiguous':
    case 'unresolved':
      return '#fbbf24'
  }
}

function ringRadius(outcome: MatchOutcome): number {
  switch (outcome.kind) {
    case 'correct':
      return 5.4
    case 'false-match':
      return 7.2
    case 'ambiguous':
    case 'unresolved':
      return 6.2
  }
}

function ringWidth(outcome: MatchOutcome): number {
  switch (outcome.kind) {
    case 'correct':
      return 1
    case 'false-match':
      return 2
    case 'ambiguous':
    case 'unresolved':
      return 1.4
  }
}

export const PointLabCanvas: Component<PointLabCanvasProps> = (props) => {
  const [canvasElement, setCanvasElement] = createSignal<HTMLCanvasElement>()
  const elementSize = createElementSize(canvasElement)
  const [zoom, setZoom] = createSignal(1)
  const [panX, setPanX] = createSignal(0)
  const [panY, setPanY] = createSignal(0)
  let lastTransform: ScreenTransform | undefined

  const zoomAt = (factor: number, anchorX: number, anchorY: number): void => {
    const current = zoom()
    const next = Math.min(MAX_ZOOM, Math.max(1, current * factor))
    if (Math.abs(next - current) < 1e-6) {
      return
    }
    if (next <= 1.0001) {
      setZoom(1)
      setPanX(0)
      setPanY(0)
      return
    }
    const ratio = next / current
    setPanX(anchorX - (anchorX - panX()) * ratio)
    setPanY(anchorY - (anchorY - panY()) * ratio)
    setZoom(next)
  }

  const resetViewport = (): void => {
    setZoom(1)
    setPanX(0)
    setPanY(0)
  }

  const zoomFromCenter = (factor: number): void => {
    const size = elementSize()
    if (size === undefined) {
      return
    }
    zoomAt(factor, size.width / 2, size.height / 2)
  }

  const handleWheel = (event: WheelEvent): void => {
    if (!props.zoomToCursor) {
      return
    }
    event.preventDefault()
    const canvas = canvasElement()
    if (canvas === undefined) {
      return
    }
    const rect = canvas.getBoundingClientRect()
    const anchorX = event.clientX - rect.left
    const anchorY = event.clientY - rect.top
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY
    const factor = Math.exp(-delta * WHEEL_SENSITIVITY)
    if (!Number.isFinite(factor) || factor <= 0) {
      return
    }
    zoomAt(factor, anchorX, anchorY)
  }

  const draw = (): void => {
    const canvas = canvasElement()
    const size = elementSize()
    if (canvas === undefined || size === undefined || size.width < 8 || size.height < 8) {
      return
    }

    const pixelRatioX = size.widthPX / Math.max(size.width, 1)
    const pixelRatioY = size.heightPX / Math.max(size.height, 1)
    if (canvas.width !== size.widthPX || canvas.height !== size.heightPX) {
      canvas.width = size.widthPX
      canvas.height = size.heightPX
    }

    const context = canvas.getContext('2d')
    if (context === null) {
      return
    }

    context.setTransform(pixelRatioX, 0, 0, pixelRatioY, 0, 0)
    context.clearRect(0, 0, size.width, size.height)
    context.fillStyle = '#0b0c10'
    context.fillRect(0, 0, size.width, size.height)

    if (props.projected.length === 0) {
      lastTransform = undefined
      context.fillStyle = '#9898a4'
      context.font = '13px system-ui, sans-serif'
      context.textAlign = 'center'
      context.fillText('No points match the current lattice filters.', size.width / 2, size.height / 2)
      return
    }

    const bounds = projectionBounds(props.framePoints ?? props.projected)
    if (bounds === undefined) {
      lastTransform = undefined
      return
    }

    const rangeX = bounds.maxX - bounds.minX
    const rangeY = bounds.maxY - bounds.minY
    const availableWidth = Math.max(size.width - PADDING * 2, 1)
    const availableHeight = Math.max(size.height - PADDING * 2, 1)
    const scale = Math.max(
      0.01,
      Math.min(
        rangeX > 1e-9 ? availableWidth / rangeX : Number.POSITIVE_INFINITY,
        rangeY > 1e-9 ? availableHeight / rangeY : Number.POSITIVE_INFINITY,
      ),
    )
    const safeScale = Number.isFinite(scale) ? scale : 1
    const centerX = (bounds.minX + bounds.maxX) / 2
    const centerY = (bounds.minY + bounds.maxY) / 2
    const offsetX = size.width / 2 - centerX * safeScale
    const offsetY = size.height / 2 - centerY * safeScale

    const viewZoom = zoom()
    const viewPanX = panX()
    const viewPanY = panY()
    const toScreen = (point: Vec2): Vec2 => ({
      x: (point.x * safeScale + offsetX) * viewZoom + viewPanX,
      y: (point.y * safeScale + offsetY) * viewZoom + viewPanY,
    })

    const screenPoints: ScreenPoint[] = props.projected.map((point) => {
      const screen = toScreen(point)
      return { index: point.index, x: screen.x, y: screen.y }
    })
    const screenByIndex = new Map<number, ScreenPoint>()
    for (const point of screenPoints) {
      screenByIndex.set(point.index, point)
    }
    const screenAtPosition = (position: number): ScreenPoint | undefined => {
      const id = props.graph.points[position]?.index
      return id === undefined ? undefined : screenByIndex.get(id)
    }
    lastTransform = { points: screenPoints }

    context.save()
    context.beginPath()
    context.rect(0, 0, size.width, size.height)
    context.clip()

    const totalEdges = props.graph.edges.length
    const visibleEdgeCount = Math.max(0, Math.min(props.visibleEdgeCount, totalEdges))
    const visibleEdges = props.graph.edges.slice(0, visibleEdgeCount)

    if (props.showGraph) {
      context.strokeStyle = 'rgba(160, 168, 190, 0.38)'
      context.lineWidth = 1.1
      context.beginPath()
      for (const [a, b] of visibleEdges) {
        const start = screenAtPosition(a)
        const end = screenAtPosition(b)
        if (start === undefined || end === undefined) {
          continue
        }
        context.moveTo(start.x, start.y)
        context.lineTo(end.x, end.y)
      }
      context.stroke()
    }

    const selectedIndex = props.selectedIndex
    if (selectedIndex !== undefined) {
      context.strokeStyle = 'rgba(91, 138, 240, 0.85)'
      context.lineWidth = 1.5
      context.beginPath()
      for (const [a, b] of visibleEdges) {
        const firstId = props.graph.points[a]?.index
        const secondId = props.graph.points[b]?.index
        if (firstId !== selectedIndex && secondId !== selectedIndex) {
          continue
        }
        const start = screenAtPosition(a)
        const end = screenAtPosition(b)
        if (start === undefined || end === undefined) {
          continue
        }
        context.moveTo(start.x, start.y)
        context.lineTo(end.x, end.y)
      }
      context.stroke()
    }

    if (props.showGraph && visibleEdgeCount > 0 && visibleEdgeCount < totalEdges) {
      const latest = visibleEdges[visibleEdgeCount - 1]
      if (latest !== undefined) {
        const start = screenAtPosition(latest[0])
        const end = screenAtPosition(latest[1])
        if (start !== undefined && end !== undefined) {
          context.beginPath()
          context.moveTo(start.x, start.y)
          context.lineTo(end.x, end.y)
          context.strokeStyle = 'rgba(10, 10, 11, 0.92)'
          context.lineWidth = 6
          context.stroke()
          context.strokeStyle = '#3ddc84'
          context.lineWidth = 3.6
          context.stroke()
        }
      }
    }

    const outcomeByIndex = new Map<number, MatchOutcome>()
    for (const item of props.evaluation.perPoint) {
      outcomeByIndex.set(item.index, item.outcome)
    }

    // Binary color is the detected mark; the ring carries the decode verdict.
    for (const point of screenPoints) {
      const color = props.colors[point.index] ?? 0
      context.beginPath()
      context.arc(point.x, point.y, 3.4, 0, Math.PI * 2)
      context.fillStyle = color === 1 ? '#f1eee6' : '#26262f'
      context.fill()
      context.strokeStyle = color === 1 ? 'rgba(241, 238, 230, 0.9)' : 'rgba(152, 152, 164, 0.65)'
      context.lineWidth = 1
      context.stroke()
    }

    for (const point of screenPoints) {
      const outcome = outcomeByIndex.get(point.index)
      if (outcome === undefined || outcome.kind === 'correct') {
        continue
      }
      context.beginPath()
      context.arc(point.x, point.y, ringRadius(outcome), 0, Math.PI * 2)
      context.strokeStyle = outcomeColor(outcome)
      context.globalAlpha = 0.95
      context.lineWidth = ringWidth(outcome)
      context.stroke()
      context.globalAlpha = 1
    }

    if (selectedIndex !== undefined) {
      const selected = screenByIndex.get(selectedIndex)
      if (selected !== undefined) {
        context.beginPath()
        context.arc(selected.x, selected.y, 10, 0, Math.PI * 2)
        context.strokeStyle = '#ffffff'
        context.lineWidth = 1.5
        context.stroke()

        const label = `#${selectedIndex}`
        context.font = '600 11px ui-monospace, monospace'
        context.textAlign = 'left'
        const width = context.measureText(label).width
        const labelX = selected.x + 13
        const labelY = selected.y - 11
        context.fillStyle = 'rgba(10, 10, 11, 0.88)'
        context.fillRect(labelX - 4, labelY - 10, width + 8, 15)
        context.fillStyle = '#f0f0f5'
        context.fillText(label, labelX, labelY)
      }
    }

    context.restore()
  }

  createTrackedEffect(() => {
    draw()
  })

  const handlePointerDown = (event: PointerEvent): void => {
    const transform = lastTransform
    const canvas = canvasElement()
    if (transform === undefined || canvas === undefined) {
      props.onSelect(undefined)
      return
    }
    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    let nearest: number | undefined
    let nearestDistance = HIT_RADIUS * HIT_RADIUS
    for (const point of transform.points) {
      const dx = point.x - x
      const dy = point.y - y
      const distanceSquared = dx * dx + dy * dy
      if (distanceSquared <= nearestDistance) {
        nearestDistance = distanceSquared
        nearest = point.index
      }
    }
    props.onSelect(nearest)
  }

  return (
    <div class={styles.canvasFrame}>
      <canvas
        ref={setCanvasElement}
        class={`${styles.canvas} ${props.zoomToCursor ? styles.canvasZoomable : ''}`}
        role="img"
        aria-label="Projected hexagonal point field. Click a point to inspect its identity, wheel to zoom."
        onPointerDown={handlePointerDown}
        onWheel={handleWheel}
      />
      <div class={styles.zoomHud}>
        <button type="button" class={styles.zoomButton} onClick={() => zoomFromCenter(1 / 1.5)} aria-label="Zoom out">
          −
        </button>
        <span class={styles.zoomValue}>{zoom().toFixed(1)}×</span>
        <button type="button" class={styles.zoomButton} onClick={() => zoomFromCenter(1.5)} aria-label="Zoom in">
          +
        </button>
        <button type="button" class={styles.zoomReset} onClick={resetViewport}>
          Reset
        </button>
      </div>
    </div>
  )
}
