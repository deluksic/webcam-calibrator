import { d, tgpu } from 'typegpu'
import { abs } from 'typegpu/std'

import { LINE_INTERSECT_DET_EPS } from '@/gpu/lineFitThresholds'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'

export const LineIntersectResult = d.struct({
  point: d.vec2f,
  ok: d.u32,
})

/** Infinite line n·p = d with unit n. */
export const LineNormalD = d.struct({
  nx: d.f32,
  ny: d.f32,
  d: d.f32,
})

export const lineNormalFromEdge = tgpu.fn([EdgeLineEntry], LineNormalD)((line) => {
  'use gpu'
  return LineNormalD({
    nx: line.sumGx,
    ny: line.sumGy,
    d: line.nDotMean,
  })
})

/** Intersection of n1·p = d1 and n2·p = d2. */
export const lineIntersectNormal = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
  LineIntersectResult,
)((n1x, n1y, d1, n2x, n2y, d2) => {
  'use gpu'
  const det = n1x * n2y - n2x * n1y
  if (abs(det) < d.f32(LINE_INTERSECT_DET_EPS)) {
    return LineIntersectResult({ point: d.vec2f(0, 0), ok: d.u32(0) })
  }
  const invDet = d.f32(1) / det
  const x = (n2y * d1 - n1y * d2) * invDet
  const y = (n1x * d2 - n2x * d1) * invDet
  return LineIntersectResult({ point: d.vec2f(x, y), ok: d.u32(1) })
})
