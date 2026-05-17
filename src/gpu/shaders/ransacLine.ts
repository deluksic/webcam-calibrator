import { d, std, tgpu } from 'typegpu'
import { abs } from 'typegpu/std'

export const RANSAC_RESERVOIR_CAP = 32

/** Count reservoir points within perpendicular distance of the line through P0 with normal n. */
export const scoreSegmentOnPoints = tgpu.fn(
  [d.u32, d.arrayOf(d.vec2f, RANSAC_RESERVOIR_CAP), d.vec2f, d.vec2f, d.f32],
  d.u32,
)((pointCount, points, P0, n, inlierDistPx) => {
  'use gpu'
  let score = d.u32(0)
  for (const k of tgpu.unroll(std.range(0, RANSAC_RESERVOIR_CAP - 1))) {
    if (k < pointCount) {
      const p = points[k]!
      const relX = p.x - P0.x
      const relY = p.y - P0.y
      const s = relX * n.x + relY * n.y
      if (abs(s) < inlierDistPx) {
        score = score + d.u32(1)
      }
    }
  }
  return score
})
