// Homography UV mapping for decode: inverse 3×3 of `computeHomography`'s strip corners → unit-square (u, v).
import type { Mat3 } from '@/lib/geometry'

const { abs } = Math

/** Full inverse of a 3×3 row-major matrix; returns row-major 9-tuple or `undefined` if singular. */
export function invertMat3RowMajor(m: Mat3): Mat3 | undefined {
  const [a, b, c, d, e, f, g, h, i] = m

  const ei_minus_fh = e * i - f * h
  const di_minus_fg = d * i - f * g
  const dh_minus_eg = d * h - e * g
  const bi_minus_ch = b * i - c * h
  const ai_minus_cg = a * i - c * g
  const ah_minus_bg = a * h - b * g
  const bf_minus_ce = b * f - c * e
  const af_minus_cd = a * f - c * d
  const ae_minus_bd = a * e - b * d

  const det = a * ei_minus_fh - b * di_minus_fg + c * dh_minus_eg
  if (abs(det) < 1e-14) {
    return undefined
  }

  const invDet = 1 / det
  return [
    ei_minus_fh * invDet,
    -bi_minus_ch * invDet,
    bf_minus_ce * invDet,
    -di_minus_fg * invDet,
    ai_minus_cg * invDet,
    -af_minus_cd * invDet,
    dh_minus_eg * invDet,
    -ah_minus_bg * invDet,
    ae_minus_bd * invDet,
  ]
}

/**
 * Map image pixel (x, y) to unit-square (u, v) using the inverse of the homography from `computeHomography`.
 * Corners must be **TL, TR, BL, BR** (same order as `computeHomography`).
 */
export function imagePixelToUnitSquareUv(
  homography: Mat3,
  x: number,
  y: number,
): { u: number; v: number; inside: boolean } {
  const inv = invertMat3RowMajor(homography)
  if (!inv) {
    return { u: 0, v: 0, inside: false }
  }

  const [i0, i1, i2, i3, i4, i5, i6, i7, i8] = inv
  const xh = i0 * x + i1 * y + i2
  const yh = i3 * x + i4 * y + i5
  const wh = i6 * x + i7 * y + i8
  if (abs(wh) < 1e-12) {
    return { u: 0, v: 0, inside: false }
  }

  const u = xh / wh
  const v = yh / wh
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1
  return { u, v, inside }
}
