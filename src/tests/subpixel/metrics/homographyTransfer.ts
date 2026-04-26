import type { Mat3, Point } from '@/lib/geometry'
import { applyHomography, length } from '@/lib/geometry'

const { max, sqrt } = Math

function dist(a: Point, b: Point): number {
  return length(a.x - b.x, a.y - b.y)
}

/**
 * Sample `gridN`×`gridN` points in unit square [0,1]² (inclusive edges) and compare image positions.
 */
export function homographyTransferStats(H_gt: Mat3, H_test: Mat3, gridN = 9): { rmsePx: number; maxPx: number } {
  const n = max(2, gridN)
  let sumSq = 0
  let maxD = 0
  let count = 0
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1)
      const v = j / (n - 1)
      const p0 = applyHomography(H_gt, u, v)
      const p1 = applyHomography(H_test, u, v)
      const d = dist(p0, p1)
      sumSq += d * d
      maxD = max(maxD, d)
      count++
    }
  }
  return { rmsePx: sqrt(sumSq / count), maxPx: maxD }
}

export function roundTransferStats(
  s: ReturnType<typeof homographyTransferStats>,
  decimals = 4,
): { rmsePx: number; maxPx: number } {
  const f = (x: number) => Number(x.toFixed(decimals))
  return { rmsePx: f(s.rmsePx), maxPx: f(s.maxPx) }
}
