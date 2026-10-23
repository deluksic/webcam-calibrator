import type { BinaryColor, PatternPoint, Vec2 } from '@/lib/pointIdentity/types'

export type DesignLatticeMetrics = {
  /** Median nearest-neighbor distance between design points. */
  medianNearest: number
  /** Minimum nearest-neighbor distance; the matching gate must stay below half this. */
  minNearest: number
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    return 0
  }
  const sorted = [...values].sort((first, second) => first - second)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2
}

export function computeDesignLatticeMetrics(points: readonly PatternPoint[]): DesignLatticeMetrics {
  if (points.length < 2) {
    return { medianNearest: 0, minNearest: 0 }
  }
  const nearestDistances: number[] = []
  for (let first = 0; first < points.length; first += 1) {
    const origin = points[first]!
    let nearest = Number.POSITIVE_INFINITY
    for (let second = 0; second < points.length; second += 1) {
      if (first === second) {
        continue
      }
      const candidate = points[second]!
      nearest = Math.min(nearest, Math.hypot(origin.x - candidate.x, origin.y - candidate.y))
    }
    if (Number.isFinite(nearest)) {
      nearestDistances.push(nearest)
    }
  }
  return {
    medianNearest: median(nearestDistances),
    minNearest: nearestDistances.length === 0 ? 0 : Math.min(...nearestDistances),
  }
}

export function createDesignPositionMap(points: readonly PatternPoint[]): Map<number, number> {
  const result = new Map<number, number>()
  for (let position = 0; position < points.length; position += 1) {
    result.set(points[position]!.index, position)
  }
  return result
}

export type DesignGrid = {
  cellSize: number
  /** Positions (array indices into the original design point array) within radius. */
  within: (point: Vec2, radius: number) => number[]
  nearest: (point: Vec2, radius: number, color?: BinaryColor) => number | undefined
}

function cellKey(x: number, y: number): string {
  return `${x}:${y}`
}

export function createDesignGrid(
  points: readonly PatternPoint[],
  colors: readonly BinaryColor[],
  cellSize: number,
): DesignGrid {
  const safeCellSize = Math.max(1e-6, cellSize)
  const buckets = new Map<string, number[]>()
  for (let position = 0; position < points.length; position += 1) {
    const point = points[position]!
    const key = cellKey(Math.floor(point.x / safeCellSize), Math.floor(point.y / safeCellSize))
    const bucket = buckets.get(key)
    if (bucket === undefined) {
      buckets.set(key, [position])
    } else {
      bucket.push(position)
    }
  }

  const within = (point: Vec2, radius: number): number[] => {
    const cellRadius = Math.max(1, Math.ceil(radius / safeCellSize))
    const centerX = Math.floor(point.x / safeCellSize)
    const centerY = Math.floor(point.y / safeCellSize)
    const result: number[] = []
    for (let cellY = centerY - cellRadius; cellY <= centerY + cellRadius; cellY += 1) {
      for (let cellX = centerX - cellRadius; cellX <= centerX + cellRadius; cellX += 1) {
        const bucket = buckets.get(cellKey(cellX, cellY))
        if (bucket === undefined) {
          continue
        }
        for (const position of bucket) {
          const candidate = points[position]!
          if (Math.hypot(candidate.x - point.x, candidate.y - point.y) <= radius) {
            result.push(position)
          }
        }
      }
    }
    return result
  }

  const nearest = (point: Vec2, radius: number, color?: BinaryColor): number | undefined => {
    const candidates = within(point, radius)
    let best: number | undefined
    let bestDistance = Number.POSITIVE_INFINITY
    for (const position of candidates) {
      if (color !== undefined && (colors[position] ?? 0) !== color) {
        continue
      }
      const candidate = points[position]!
      const distance = Math.hypot(candidate.x - point.x, candidate.y - point.y)
      if (
        distance < bestDistance ||
        (distance === bestDistance && best !== undefined && position < best)
      ) {
        bestDistance = distance
        best = position
      }
    }
    return best
  }

  return { cellSize: safeCellSize, within, nearest }
}
