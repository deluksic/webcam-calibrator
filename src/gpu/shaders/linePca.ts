import { d, tgpu } from 'typegpu'
import { abs, dot, length, max, select, sqrt } from 'typegpu/std'

export const PCA_ISOTROPY_MAX = 0.15
/** sxx / syy below this → treat as vertical edge in image (constant x). */
export const PCA_AXIS_VAR_FRAC = 1e-4

export const LineSegmentEndpoints = d.struct({
  p0: d.vec2f,
  p1: d.vec2f,
})

/** Segment endpoints for unit normal n and t along dir = (n.y, -n.x). */
export const lineSegmentEndpoints = tgpu.fn(
  [d.vec2f, d.f32, d.f32, d.f32],
  LineSegmentEndpoints,
)((n, nDotMean, tMin, tMax) => {
  'use gpu'
  return LineSegmentEndpoints({
    p0: d.vec2f(nDotMean * n.x + tMin * n.y, nDotMean * n.y - tMin * n.x),
    p1: d.vec2f(nDotMean * n.x + tMax * n.y, nDotMean * n.y - tMax * n.x),
  })
})

/** dot(p, dir) for the same (n, t) frame as {@link lineSegmentEndpoints}. */
export const lineDirDot = tgpu.fn(
  [d.vec2f, d.vec2f],
  d.f32,
)((p, n) => {
  'use gpu'
  return p.x * n.y - p.y * n.x
})

/** TLS normal agrees with segment normal (|dot| >= cosMaxAngle). */
export const tlsAgreesWithNormal = tgpu.fn(
  [d.f32, d.f32, d.f32, d.f32, d.f32],
  d.u32,
)((tlsNx, tlsNy, refNx, refNy, cosMaxAngle) => {
  'use gpu'
  const ddot = abs(dot(d.vec2f(tlsNx, tlsNy), d.vec2f(refNx, refNy)))
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

  // Colinear sets (e.g. every pixel has the same x): lamMin ≈ 0 and the generic
  // eigenvector solve can return an along-edge normal, which fails the inlier gate.
  if (syy > d.f32(1e-10) && sxx <= syy * d.f32(PCA_AXIS_VAR_FRAC)) {
    return TlsNormalResult({
      nx: d.f32(1),
      ny: d.f32(0),
      lamMin: sxx,
      lamMax: syy,
      ok: d.u32(1),
    })
  }
  if (sxx > d.f32(1e-10) && syy <= sxx * d.f32(PCA_AXIS_VAR_FRAC)) {
    return TlsNormalResult({
      nx: d.f32(0),
      ny: d.f32(1),
      lamMin: syy,
      lamMax: sxx,
      ok: d.u32(1),
    })
  }

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
  let len = length(d.vec2f(nx, ny))
  if (len < d.f32(1e-10)) {
    nx = lamMin - syy
    ny = sxy
    len = length(d.vec2f(nx, ny))
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
