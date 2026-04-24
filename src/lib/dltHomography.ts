// Homogeneous DLT for plane (X,Y) -> image (u,v), 9 DoF, Hartley normalization.

import type { Mat3, Point } from '@/lib/geometry'
import { solveHomogeneousNullVector } from '@/lib/jacobiEigenSym'
import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'

const { sqrt, abs } = Math

export interface PlanePoint {
  x: number
  y: number
}

export interface Correspondence {
  plane: PlanePoint
  image: Point
}

type Mat3Mut = [number, number, number, number, number, number, number, number, number]

/** 2D similarity+translation, row-major; (x,y,1) -> normalized (x',y',1). */
function hartley2(pts: ReadonlyArray<{ x: number; y: number }>, n: number): Mat3 {
  if (n === 0) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  }
  let cx = 0
  let cy = 0
  for (let i = 0; i < n; i++) {
    cx += pts[i]!.x
    cy += pts[i]!.y
  }
  cx /= n
  cy /= n
  let r2 = 0
  for (let i = 0; i < n; i++) {
    const dx = pts[i]!.x - cx
    const dy = pts[i]!.y - cy
    r2 += dx * dx + dy * dy
  }
  const r = sqrt((r2 / n) + 1e-18)
  const s = sqrt(2) / r
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1]
}

function matMul3(a: Mat3, b: Mat3): Mat3Mut {
  const a00 = a[0]!, a01 = a[1]!, a02 = a[2]!
  const a10 = a[3]!, a11 = a[4]!, a12 = a[5]!
  const a20 = a[6]!, a21 = a[7]!, a22 = a[8]!
  const b00 = b[0]!, b01 = b[1]!, b02 = b[2]!
  const b10 = b[3]!, b11 = b[4]!, b12 = b[5]!
  const b20 = b[6]!, b21 = b[7]!, b22 = b[8]!
  return [
    a00 * b00 + a01 * b10 + a02 * b20,
    a00 * b01 + a01 * b11 + a02 * b21,
    a00 * b02 + a01 * b12 + a02 * b22,
    a10 * b00 + a11 * b10 + a12 * b20,
    a10 * b01 + a11 * b11 + a12 * b21,
    a10 * b02 + a11 * b12 + a12 * b22,
    a20 * b00 + a21 * b10 + a22 * b20,
    a20 * b01 + a21 * b11 + a22 * b22,
    a20 * b02 + a21 * b12 + a22 * b22,
  ]
}

function hVecToMat3(h: Float64Array): Mat3Mut {
  return [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!, h[8]!]
}

/**
 * DLT: plane XY -> image UV, H maps (X,Y) in projective to image (s·u, s·v, s) with w normalized.
 * Returns H with h22 = 1, or undefined.
 */
export function solveHomographyDLT(pairs: ReadonlyArray<Correspondence>): Mat3 | undefined {
  const m = pairs.length
  if (m < 4) {
    return undefined
  }

  const rows = 2 * m
  const a = new Float64Array(rows * 9)

  // Raw DLT constraint (no Hartley normalization):
  // For normalized coordinates Xn=[X,Y,1], Un=[u,v,1]:
  // u = (h0*X + h1*Y + h2) / (h6*X + h7*Y + h8)
  // Rearranging: u*(h6*X + h7*Y + h8) - (h0*X + h1*Y + h2) = 0
  //              = -h0*X - h1*Y - h2 + h6*X*u + h7*Y*u + h8*u = 0

  for (let r = 0; r < m; r++) {
    const pl = pairs[r]!.plane
    const im = pairs[r]!.image

    const row = r * 18

    // Row for u coordinate
    a[row + 0] = -pl.x
    a[row + 1] = -pl.y
    a[row + 2] = -1
    a[row + 6] = pl.x * im.x
    a[row + 7] = pl.y * im.x
    a[row + 8] = im.x

    // Row for v coordinate
    a[row + 9 + 3] = -pl.x
    a[row + 9 + 4] = -pl.y
    a[row + 9 + 5] = -1
    a[row + 9 + 6] = pl.x * im.y
    a[row + 9 + 7] = pl.y * im.y
    a[row + 9 + 8] = im.y
  }

  const h = solveHomogeneousNullVector(a, rows, 9)
  if (!h) {
    return undefined
  }

  const hN = hVecToMat3(h)

  // Normalize: divide by h[8] to ensure h[8] = 1
  const hN_norm = hN.map((v, i) => i === 8 ? v : v / h[8]!)

  // Check if h[8] is valid
  if (abs(hN_norm[8]!) < 1e-12) {
    return undefined
  }

  return hN_norm
}