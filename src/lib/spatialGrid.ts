/**
 * Uniform-grid point index for k-nearest queries in the plane.
 *
 * Built once per point set, then queried per point. Cells are linked lists in
 * flat typed arrays (no per-cell allocation, no Map), so a build is O(n) and a
 * query visits only the rings it needs.
 *
 * The query loop lives at the call site because the caller owns the "is this
 * candidate closer than my current k-th?" rule; this module only supplies the
 * geometry — see `forEachRingCandidate` in `roiNeighbourFilter.ts`.
 */

export type PointGrid = {
  /** Cell edge length, in the same units as the coordinates. */
  cellSize: number
  minX: number
  minY: number
  cols: number
  rows: number
  /** Coordinates of the candidate points, indexed by *local* index. */
  xs: Float64Array
  ys: Float64Array
  /** First local index in each cell, or -1. */
  head: Int32Array
  /** Next local index in the same cell, or -1. */
  next: Int32Array
  count: number
}

/** Cells beyond this are not worth the allocation; such inputs fall back to a
 *  linear scan, which is also what tiny point sets use. */
const MAX_CELLS = 1 << 20
/** Below this, a linear scan is faster than building a grid. */
const MIN_POINTS_FOR_GRID = 64

export type BuildGridOptions = {
  minPoints?: number
}

/** Builds a grid over `(xs[i], ys[i])`, or returns undefined when a linear scan
 *  is the better choice (few points, collinear/degenerate extent, oversized
 *  grid). Callers must handle undefined by scanning linearly. */
export function buildPointGrid(
  xs: Float64Array,
  ys: Float64Array,
  count: number,
  options: BuildGridOptions = {},
): PointGrid | undefined {
  const minPoints = options.minPoints ?? MIN_POINTS_FOR_GRID
  if (count < minPoints) {
    return undefined
  }

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (let index = 0; index < count; index += 1) {
    const x = xs[index]!
    const y = ys[index]!
    if (x < minX) {
      minX = x
    }
    if (x > maxX) {
      maxX = x
    }
    if (y < minY) {
      minY = y
    }
    if (y > maxY) {
      maxY = y
    }
  }

  const width = maxX - minX
  const height = maxY - minY
  if (!(width > 0) || !(height > 0)) {
    // All points share an axis: every distance ties, so a grid cannot prune.
    return undefined
  }

  // Aim for about one point per cell; that keeps the first ring cheap for
  // uniformly spread points and lets dense clusters expand outward gracefully.
  const cellSize = Math.max(Math.sqrt((width * height) / count), 1e-6)
  const cols = Math.max(1, Math.ceil(width / cellSize) + 1)
  const rows = Math.max(1, Math.ceil(height / cellSize) + 1)
  if (cols * rows > MAX_CELLS) {
    return undefined
  }

  const head = new Int32Array(cols * rows).fill(-1)
  const next = new Int32Array(count).fill(-1)
  for (let index = 0; index < count; index += 1) {
    const col = Math.min(cols - 1, Math.max(0, Math.floor((xs[index]! - minX) / cellSize)))
    const row = Math.min(rows - 1, Math.max(0, Math.floor((ys[index]! - minY) / cellSize)))
    const cell = row * cols + col
    next[index] = head[cell]!
    head[cell] = index
  }

  return { cellSize, minX, minY, cols, rows, xs, ys, head, next, count }
}

/** Cell coordinates holding `(x, y)`, clamped into the grid. */
export function cellOf(grid: PointGrid, x: number, y: number): { row: number; col: number } {
  const col = Math.min(
    grid.cols - 1,
    Math.max(0, Math.floor((x - grid.minX) / grid.cellSize)),
  )
  const row = Math.min(
    grid.rows - 1,
    Math.max(0, Math.floor((y - grid.minY) / grid.cellSize)),
  )
  return { row, col }
}

/** Lower bound on the distance from a point in its own cell to any point in a
 *  cell at Chebyshev ring `ring` or beyond.
 *
 *  A point sits somewhere inside its cell, so reaching ring `ring` means
 *  crossing at least `ring - 1` full cells. Conservative by construction: it
 *  never cuts the search before a genuinely closer point could still appear. */
export function minDistanceToRing(grid: PointGrid, ring: number): number {
  return Math.max(0, ring - 1) * grid.cellSize
}

/**
 * Visits candidate points in expanding rings around `(x, y)`, stopping as soon
 * as the caller's k-th distance has settled.
 *
 * `kthDistance` returns the current k-th smallest distance, or `Infinity` while
 * fewer than k candidates are known — so callers get the "keep going until we
 * have k, then until the next ring cannot beat the k-th" behaviour for free.
 * `onCandidate` receives a *local* index (the same index space the grid was
 * built over) plus the raw deltas, so each caller keeps its own distance
 * arithmetic (Euclidean in the graph, squared in the ROI filter). Deliberately
 * no `Math.hypot`: it is overflow-safe but markedly slower.
 *
 * The bound is conservative: after ring `r`, every unseen point lies in ring
 * `r + 1` or beyond and is at least `r * cellSize` away, so stopping once the
 * k-th distance is below that cannot drop a point that belongs in the top k.
 */
export function scanNearestCells(
  grid: PointGrid,
  x: number,
  y: number,
  skip: number,
  kthDistance: () => number,
  onCandidate: (index: number, dx: number, dy: number) => void,
): void {
  // Computed inline: this runs once per query, so a {row, col} object here is
  // pure allocation churn.
  const row = Math.min(grid.rows - 1, Math.max(0, Math.floor((y - grid.minY) / grid.cellSize)))
  const col = Math.min(grid.cols - 1, Math.max(0, Math.floor((x - grid.minX) / grid.cellSize)))

  const visitCell = (scanRow: number, scanCol: number): void => {
    if (scanRow < 0 || scanRow >= grid.rows || scanCol < 0 || scanCol >= grid.cols) {
      return
    }
    for (
      let index = grid.head[scanRow * grid.cols + scanCol]!;
      index !== -1;
      index = grid.next[index]!
    ) {
      if (index === skip) {
        continue
      }
      onCandidate(index, grid.xs[index]! - x, grid.ys[index]! - y)
    }
  }

  const maxRing = Math.max(grid.cols, grid.rows)
  for (let ring = 0; ring <= maxRing; ring += 1) {
    if (ring === 0) {
      visitCell(row, col)
    } else {
      for (let offset = -ring; offset <= ring; offset += 1) {
        visitCell(row + offset, col - ring)
        visitCell(row + offset, col + ring)
      }
      for (let offset = -ring + 1; offset <= ring - 1; offset += 1) {
        visitCell(row - ring, col + offset)
        visitCell(row + ring, col + offset)
      }
    }
    if (kthDistance() < minDistanceToRing(grid, ring + 1)) {
      return
    }
    // Everything in the grid has been scanned; no ring can add candidates.
    if (
      row - ring <= 0 &&
      row + ring >= grid.rows - 1 &&
      col - ring <= 0 &&
      col + ring >= grid.cols - 1
    ) {
      return
    }
  }
}
