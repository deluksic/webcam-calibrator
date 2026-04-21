import { length } from '@/lib/geometry'
import type { StripCorners } from '@/tests/shared/types'

const { max } = Math

export function cornerErrorStats(
  gt: StripCorners,
  test: StripCorners,
): {
  perCornerPx: number[]
  maxPx: number
  rmsePx: number
} {
  const perCornerPx: number[] = []
  let sumSq = 0
  for (let i = 0; i < 4; i++) {
    const d = length(test[i].x - gt[i].x, test[i].y - gt[i].y)
    perCornerPx.push(d)
    sumSq += d * d
  }
  return {
    perCornerPx,
    maxPx: max(...perCornerPx),
    rmsePx: sqrt(sumSq / 4),
  }
}

/** Rounded for snapshot stability. */
export function roundCornerStats(
  s: ReturnType<typeof cornerErrorStats>,
  decimals = 4,
): Record<string, number | number[]> {
  const f = (x: number) => Number(x.toFixed(decimals))
  return {
    perCornerPx: s.perCornerPx.map(f),
    maxPx: f(s.maxPx),
    rmsePx: f(s.rmsePx),
  }
}
