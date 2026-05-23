import { d, tgpu } from 'typegpu'
import { abs, length } from 'typegpu/std'

import { LINE_INTERSECT_DET_EPS } from '@/gpu/lineFitThresholds'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'

export const LineIntersectResult = d.struct({
  point: d.vec2f,
  ok: d.u32,
})

/** Infinite line n·p = d with unit n, plus segment midpoint and length. */
export const LineNormalD = d.struct({
  nx: d.f32,
  ny: d.f32,
  d: d.f32,
  mx: d.f32,
  my: d.f32,
  span: d.f32,
})

export const lineNormalFromEdge = tgpu.fn([EdgeLineEntry], LineNormalD)((line) => {
  'use gpu'
  const dx = line.p1x - line.p0x
  const dy = line.p1y - line.p0y
  return LineNormalD({
    nx: line.sumGx,
    ny: line.sumGy,
    d: line.nDotMean,
    mx: (line.p0x + line.p1x) * d.f32(0.5),
    my: (line.p0y + line.p1y) * d.f32(0.5),
    span: length(d.vec2f(dx, dy)),
  })
})

/** Intersection of n1·p = d1 and n2·p = d2. */
export const lineIntersectNormal = tgpu.fn(
  [d.vec2f, d.f32, d.vec2f, d.f32],
  LineIntersectResult,
)((n1, d1, n2, d2) => {
  'use gpu'
  const det = n1.x * n2.y - n2.x * n1.y
  if (abs(det) < d.f32(LINE_INTERSECT_DET_EPS)) {
    return LineIntersectResult({ point: d.vec2f(0, 0), ok: d.u32(0) })
  }
  const invDet = d.f32(1) / det
  const x = (n2.y * d1 - n1.y * d2) * invDet
  const y = (n1.x * d2 - n2.x * d1) * invDet
  return LineIntersectResult({ point: d.vec2f(x, y), ok: d.u32(1) })
})
