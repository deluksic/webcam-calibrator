// Homography UV mapping for decode: inverse 3×3 of `computeHomography`’s strip corners → unit-square (u, v).
const { abs } = Math

/** 3×3 row-major [m0..m8] = [r0c0, r0c1, r0c2, r1c0, ...] */
function mat3FromHomography8(h: Float32Array): number[] {
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1]
}

/** Full inverse of a 3×3 row-major matrix; returns row-major 9 floats or null if singular. */
export function invertMat3RowMajor(m: number[]): Float32Array | null {
  const a = m[0],
    b = m[1],
    c = m[2]
  const d = m[3],
    e = m[4],
    f = m[5]
  const g = m[6],
    h = m[7],
    i = m[8]

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
  if (abs(det) < 1e-14) return null

  const invDet = 1 / det
  return new Float32Array([
    ei_minus_fh * invDet,
    -bi_minus_ch * invDet,
    bf_minus_ce * invDet,
    -di_minus_fg * invDet,
    ai_minus_cg * invDet,
    -af_minus_cd * invDet,
    dh_minus_eg * invDet,
    -ah_minus_bg * invDet,
    ae_minus_bd * invDet,
  ])
}

/**
 * Map image pixel (x, y) to unit-square (u, v) using the inverse of `computeHomography`’s 8-parameter map.
 * Corners must be **TL, TR, BL, BR** (same order as `computeHomography`).
 */
export function imagePixelToUnitSquareUv(
  homography8: Float32Array,
  x: number,
  y: number,
): { u: number; v: number; inside: boolean } {
  const M = mat3FromHomography8(homography8)
  const inv = invertMat3RowMajor(M)
  if (!inv) return { u: 0, v: 0, inside: false }

  const xh = inv[0] * x + inv[1] * y + inv[2]
  const yh = inv[3] * x + inv[4] * y + inv[5]
  const wh = inv[6] * x + inv[7] * y + inv[8]
  if (abs(wh) < 1e-12) return { u: 0, v: 0, inside: false }

  const u = xh / wh
  const v = yh / wh
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1
  return { u, v, inside }
}
