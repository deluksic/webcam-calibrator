import type {
  HexPattern,
  HexPatternSpec,
  PatternGuide,
  PatternPoint,
  PatternStats,
  Vec2,
} from '@/lib/pointIdentity/types'

/**
 * The target is a flat-top honeycomb: a field of hexagonal cells whose shared
 * edges are sampled at half-step offsets. We build the cell boundary directly
 * from the six corners of each hexagon; there is no triangulation, no Voronoi
 * diagram and no random displacement.
 */

export const DESIGN_WIDTH = 720
export const DESIGN_HEIGHT = 540

const COLUMN_PITCH = 1.5
const ROW_PITCH = Math.sqrt(3)
const CORNER_COUNT = 6
const DUPLICATE_DISTANCE = 0.5

const CORNER_ANGLES = Array.from({ length: CORNER_COUNT }, (_, index) => (index * Math.PI) / 3)

type Cell = Vec2 & {
  column: number
  row: number
}

type Edge = {
  a: Vec2
  b: Vec2
}

function distanceSquared(a: Vec2, b: Vec2): number {
  const dx = a.x - b.x
  const dy = a.y - b.y
  return dx * dx + dy * dy
}

function vertexKey(point: Vec2): string {
  return `${point.x.toFixed(4)}:${point.y.toFixed(4)}`
}

function edgeKey(a: Vec2, b: Vec2): string {
  const first = vertexKey(a)
  const second = vertexKey(b)
  return first < second ? `${first}|${second}` : `${second}|${first}`
}

function removeNearDuplicates(points: readonly Vec2[]): Vec2[] {
  const kept: Vec2[] = []
  const minDistanceSquared = DUPLICATE_DISTANCE * DUPLICATE_DISTANCE
  for (const point of points) {
    let duplicate = false
    for (const other of kept) {
      if (distanceSquared(point, other) < minDistanceSquared) {
        duplicate = true
        break
      }
    }
    if (!duplicate) {
      kept.push(point)
    }
  }
  return kept
}

function buildCells(columns: number, rows: number, radius: number): Cell[] {
  const columnPitch = COLUMN_PITCH * radius
  const rowPitch = ROW_PITCH * radius
  const cells: Cell[] = []

  for (let column = 0; column < columns; column += 1) {
    const columnOffset = column % 2 === 0 ? 0 : rowPitch / 2
    for (let row = 0; row < rows; row += 1) {
      cells.push({
        x: column * columnPitch,
        y: row * rowPitch + columnOffset,
        column,
        row,
      })
    }
  }

  return cells
}

function buildCellEdges(cells: readonly Cell[], radius: number): Edge[] {
  const edgesByKey = new Map<string, Edge>()

  for (const cell of cells) {
    const corners = CORNER_ANGLES.map((angle) => ({
      x: cell.x + Math.cos(angle) * radius,
      y: cell.y + Math.sin(angle) * radius,
    }))
    for (let corner = 0; corner < CORNER_COUNT; corner += 1) {
      const a = corners[corner]!
      const b = corners[(corner + 1) % CORNER_COUNT]!
      const key = edgeKey(a, b)
      if (!edgesByKey.has(key)) {
        edgesByKey.set(key, { a, b })
      }
    }
  }

  return [...edgesByKey.values()]
}

export function generateHexPattern(spec: HexPatternSpec): HexPattern {
  const columns = Math.max(2, Math.round(spec.columns))
  const pointsPerEdge = Math.max(1, Math.round(spec.pointsPerEdge))
  const widthRadius = DESIGN_WIDTH / (COLUMN_PITCH * (columns - 1) + 2)
  const requestedRows =
    spec.rows === undefined ? undefined : Math.max(2, Math.round(spec.rows))
  const rows =
    requestedRows ?? Math.max(2, Math.round((DESIGN_HEIGHT - 2 * widthRadius) / (ROW_PITCH * widthRadius)))
  const heightRadius = DESIGN_HEIGHT / (ROW_PITCH * (rows - 1) + ROW_PITCH / 2 + 2)
  const radius = requestedRows === undefined ? widthRadius : Math.min(widthRadius, heightRadius)
  const cells = buildCells(columns, rows, radius)
  const candidateEdges = buildCellEdges(cells, radius)

  const guides: PatternGuide[] = []
  const rawPoints: Vec2[] = []

  for (const edge of candidateEdges) {
    guides.push({ a: { x: edge.a.x, y: edge.a.y }, b: { x: edge.b.x, y: edge.b.y } })
    const dx = edge.b.x - edge.a.x
    const dy = edge.b.y - edge.a.y
    for (let step = 0; step < pointsPerEdge; step += 1) {
      const t = (step + 0.5) / pointsPerEdge
      rawPoints.push({ x: edge.a.x + dx * t, y: edge.a.y + dy * t })
    }
  }

  const deduped = removeNearDuplicates(rawPoints)

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

  for (const point of deduped) {
    include(point)
  }
  for (const guide of guides) {
    include(guide.a)
    include(guide.b)
  }

  const hasBounds = Number.isFinite(minX) && Number.isFinite(minY)
  const centerX = hasBounds ? (minX + maxX) / 2 : 0
  const centerY = hasBounds ? (minY + maxY) / 2 : 0

  const points: PatternPoint[] = deduped
    .map((point) => ({ x: point.x - centerX, y: point.y - centerY }))
    .sort((a, b) => (a.y === b.y ? a.x - b.x : a.y - b.y))
    .map((point, index) => ({ ...point, index }))

  const centeredGuides = guides.map((guide) => ({
    a: { x: guide.a.x - centerX, y: guide.a.y - centerY },
    b: { x: guide.b.x - centerX, y: guide.b.y - centerY },
  }))

  const bounds = hasBounds
    ? {
        minX: minX - centerX,
        maxX: maxX - centerX,
        minY: minY - centerY,
        maxY: maxY - centerY,
      }
    : { minX: -1, maxX: 1, minY: -1, maxY: 1 }

  const stats: PatternStats = {
    cells: cells.length,
    rows,
    candidateEdges: candidateEdges.length,
    sampledEdges: guides.length,
    rawPoints: rawPoints.length,
    points: points.length,
  }

  return { points, guides: centeredGuides, stats, bounds }
}
