import type { Corners } from '@/lib/geometry'
import { length } from '@/lib/geometry'

const { max, sqrt } = Math

export function cornerErrorStats(
  gt: Corners,
  test: Corners,
): {
  perCornerPx: number[]
  maxPx: number
  rmsePx: number
} {
  const [gt0, gt1, gt2, gt3] = gt
  const [t0, t1, t2, t3] = test
  const perCornerPx = [
    length(t0.x - gt0.x, t0.y - gt0.y),
    length(t1.x - gt1.x, t1.y - gt1.y),
    length(t2.x - gt2.x, t2.y - gt2.y),
    length(t3.x - gt3.x, t3.y - gt3.y),
  ]
  const sumSq = perCornerPx.reduce((s, d) => s + d * d, 0)
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
