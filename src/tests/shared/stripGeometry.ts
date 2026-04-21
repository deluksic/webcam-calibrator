import type { Point } from '@/lib/geometry'
import { length } from '@/lib/geometry'
import type { StripCorners } from '@/tests/shared/types'

const { max, cos, sin } = Math

function dist(a: Point, b: Point): number {
  return length(a.x - b.x, a.y - b.y)
}

/** Longest edge of quad TL, TR, BL, BR (triangle-strip order). */
export function quadMaxEdgePx(strip: StripCorners): number {
  const [tl, tr, bl, br] = strip
  const e0 = dist(tl, tr)
  const e1 = dist(tr, br)
  const e2 = dist(br, bl)
  const e3 = dist(bl, tl)
  return max(e0, e1, e2, e3)
}

function stripCentroid(strip: StripCorners): Point {
  let x = 0
  let y = 0
  for (const p of strip) {
    x += p.x
    y += p.y
  }
  return { x: x / 4, y: y / 4 }
}

/** Rotate each corner around centroid by `rad` (CCW). */
export function rotateStripAroundCentroid(strip: StripCorners, rad: number): StripCorners {
  const c = stripCentroid(strip)
  const cosTheta = cos(rad)
  const sinTheta = sin(rad)
  return strip.map((p) => {
    const dx = p.x - c.x
    const dy = p.y - c.y
    return {
      x: c.x + dx * cosTheta - dy * sinTheta,
      y: c.y + dx * sinTheta + dy * cosTheta,
    }
  }) as StripCorners
}

/** Uniform scale about centroid so longest edge ≤ `maxEdgePx`. */
export function scaleStripToMaxEdgePx(strip: StripCorners, maxEdgePx: number): StripCorners {
  const m = quadMaxEdgePx(strip)
  if (m < 1e-12) {
    return strip
  }
  const s = maxEdgePx / m
  const c = stripCentroid(strip)
  return strip.map((p) => ({
    x: c.x + (p.x - c.x) * s,
    y: c.y + (p.y - c.y) * s,
  })) as StripCorners
}
