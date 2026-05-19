import { d, tgpu } from 'typegpu'
import { abs, max, select, sqrt } from 'typegpu/std'

export const PCA_ISOTROPY_MAX = 0.15

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
