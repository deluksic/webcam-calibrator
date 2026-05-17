import { d, tgpu } from 'typegpu'
import { abs, dot, max, select, sqrt } from 'typegpu/std'

export const PCA_ISOTROPY_MAX = 0.1
export const LINE_EXTREMA_MIN_SPAN_PX = 5

export const LineSegmentEndpoints = d.struct({
  p0: d.vec2f,
  p1: d.vec2f,
})

/** Segment endpoints for unit normal n and t along dir = (n.y, -n.x). */
export const lineSegmentEndpoints = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32],
  LineSegmentEndpoints,
)((nx, ny, nDotMean, tMin, tMax) => {
  'use gpu'
  return LineSegmentEndpoints({
    p0: d.vec2f(nDotMean * nx + tMin * ny, nDotMean * ny - tMin * nx),
    p1: d.vec2f(nDotMean * nx + tMax * ny, nDotMean * ny - tMax * nx),
  })
})

/** dot(p, dir) for the same (n, t) frame as {@link lineSegmentEndpoints}. */
export const lineDirDot = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32],
  d.f32,
)((px, py, nx, ny) => {
  'use gpu'
  return px * ny - py * nx
})

/** p = n·nDot + dir·t with unit n ⟂ dir. */
export const linePointFromNT = tgpu.fn(
  [d.vec2f, d.f32, d.vec2f, d.f32],
  d.vec2f,
)((n, nDot, dir, t) => {
  'use gpu'
  return d.vec2f(n.x * nDot + dir.x * t, n.y * nDot + dir.y * t)
})

export const LineSegmentGeom = d.struct({
  n: d.vec2f,
  dir: d.vec2f,
  span: d.f32,
  ok: d.u32,
})

/** Unit frame from endpoints: dir = (P1−P0)/|·|, n ⟂ dir, span = |P1−P0|. Along-edge t is dot(p−P0, dir) in [0, span]. */
export const lineFromSegment = tgpu.fn(
  [d.vec2f, d.vec2f],
  LineSegmentGeom,
)((P0, P1) => {
  'use gpu'
  const seg = d.vec2f(P1 - P0)
  const span = sqrt(dot(seg, seg))
  const invSpan = d.f32(1) / max(span, d.f32(1e-8))
  const dir = d.vec2f(seg * invSpan)
  const n = d.vec2f(-dir.y, dir.x)
  const ok = select(d.u32(0), d.u32(1), span >= d.f32(LINE_EXTREMA_MIN_SPAN_PX))
  return LineSegmentGeom({
    n: d.vec2f(n),
    dir: d.vec2f(dir),
    span,
    ok,
  })
})

export const LineExtentAlongDir = d.struct({
  tMin: d.f32,
  tMax: d.f32,
  meanT: d.f32,
})

/** Along-edge extent from inlier moments projected on unit dir. */
export const lineExtentAlongDir = tgpu.fn(
  [d.u32, d.f32, d.f32, d.f32, d.f32, d.f32, d.vec2f, d.f32],
  LineExtentAlongDir,
)((count, sumX, sumY, sumXX, sumXY, sumYY, dir, minHalfSpan) => {
  'use gpu'
  const invN = d.f32(1) / d.f32(count)
  const cx = sumX * invN
  const cy = sumY * invN
  const sxx = sumXX * invN - cx * cx
  const syy = sumYY * invN - cy * cy
  const sxy = sumXY * invN - cx * cy
  const meanT = dir.x * cx + dir.y * cy
  const varT = dir.x * dir.x * sxx + dir.y * dir.y * syy + d.f32(2) * dir.x * dir.y * sxy
  const stdT = sqrt(max(varT, d.f32(0)))
  const half = max(stdT * d.f32(2.5), minHalfSpan)
  return LineExtentAlongDir({
    tMin: meanT - half,
    tMax: meanT + half,
    meanT,
  })
})

/** TLS normal agrees with segment normal (|dot| >= cosMaxAngle). */
export const tlsAgreesWithNormal = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32],
  d.u32,
)((tlsNx, tlsNy, refNx, refNy, cosMaxAngle) => {
  'use gpu'
  const ddot = abs(tlsNx * refNx + tlsNy * refNy)
  return select(d.u32(0), d.u32(1), ddot >= cosMaxAngle)
})

export const TlsNormalResult = d.struct({
  nx: d.f32,
  ny: d.f32,
  lamMin: d.f32,
  lamMax: d.f32,
  ok: d.u32,
})

/** Orthogonal LS normal = eigenvector of smaller covariance eigenvalue. */
export const tlsNormalFromMoments = tgpu.fn(
  [d.u32, d.f32, d.f32, d.f32, d.f32, d.f32],
  TlsNormalResult,
)((count, sumX, sumY, sumXX, sumXY, sumYY) => {
  'use gpu'
  if (count < d.u32(2)) {
    return TlsNormalResult({
      nx: d.f32(0),
      ny: d.f32(0),
      lamMin: d.f32(0),
      lamMax: d.f32(0),
      ok: d.u32(0),
    })
  }
  const invN = d.f32(1) / d.f32(count)
  const cx = sumX * invN
  const cy = sumY * invN
  const sxx = sumXX * invN - cx * cx
  const syy = sumYY * invN - cy * cy
  const sxy = sumXY * invN - cx * cy

  const tr = sxx + syy
  const det = sxx * syy - sxy * sxy
  const disc = max(d.f32(0), tr * tr - d.f32(4) * det)
  const root = sqrt(disc)
  const lamMax = (tr + root) * d.f32(0.5)
  const lamMin = (tr - root) * d.f32(0.5)

  if (lamMax < d.f32(1e-10)) {
    return TlsNormalResult({ nx: d.f32(0), ny: d.f32(0), lamMin, lamMax, ok: d.u32(0) })
  }
  if (lamMin / lamMax > d.f32(PCA_ISOTROPY_MAX)) {
    return TlsNormalResult({ nx: d.f32(0), ny: d.f32(0), lamMin, lamMax, ok: d.u32(0) })
  }

  let nx = sxy
  let ny = lamMin - sxx
  let len = sqrt(nx * nx + ny * ny)
  if (len < d.f32(1e-10)) {
    nx = lamMin - syy
    ny = sxy
    len = sqrt(nx * nx + ny * ny)
  }
  if (len < d.f32(1e-10)) {
    return TlsNormalResult({ nx: d.f32(0), ny: d.f32(0), lamMin, lamMax, ok: d.u32(0) })
  }
  return TlsNormalResult({
    nx: nx / len,
    ny: ny / len,
    lamMin,
    lamMax,
    ok: d.u32(1),
  })
})
