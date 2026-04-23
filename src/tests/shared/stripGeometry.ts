import type { Corners, Point } from '@/lib/geometry'
import { length } from '@/lib/geometry'

const { max, cos, sin } = Math

function dist(a: Point, b: Point): number {
  return length(a.x - b.x, a.y - b.y)
}

/** Longest edge of quad TL, TR, BL, BR (triangle-strip order). */
export function quadMaxEdgePx(strip: Corners): number {
  const [tl, tr, bl, br] = strip
  const e0 = dist(tl, tr)
  const e1 = dist(tr, br)
  const e2 = dist(br, bl)
  const e3 = dist(bl, tl)
  return max(e0, e1, e2, e3)
}

function stripCentroid(strip: Corners): Point {
  let x = 0
  let y = 0
  for (const p of strip) {
    x += p.x
    y += p.y
  }
  return { x: x / 4, y: y / 4 }
}

/** Rotate each corner around centroid by `rad` (CCW). */
export function rotateStripAroundCentroid(strip: Corners, rad: number): Corners {
  const c = stripCentroid(strip)
  const cosTheta = cos(rad)
  const sinTheta = sin(rad)
  const rot = (p: Point): Point => ({
    x: c.x + (p.x - c.x) * cosTheta - (p.y - c.y) * sinTheta,
    y: c.y + (p.x - c.x) * sinTheta + (p.y - c.y) * cosTheta,
  })
  return [rot(strip[0]), rot(strip[1]), rot(strip[2]), rot(strip[3])]
}

/** Uniform scale about centroid so longest edge ≤ `maxEdgePx`. */
export function scaleStripToMaxEdgePx(strip: Corners, maxEdgePx: number): Corners {
  const m = quadMaxEdgePx(strip)
  if (m < 1e-12) {
    return strip
  }
  const s = maxEdgePx / m
  const c = stripCentroid(strip)
  const scale = (p: Point): Point => ({
    x: c.x + (p.x - c.x) * s,
    y: c.y + (p.y - c.y) * s,
  })
  return [scale(strip[0]), scale(strip[1]), scale(strip[2]), scale(strip[3])]
}
