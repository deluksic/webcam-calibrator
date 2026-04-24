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
  const a00 = a[0]!,
    a01 = a[1]!,
    a02 = a[2]!
  const a10 = a[3]!,
    a11 = a[4]!,
    a12 = a[5]!
  const a20 = a[6]!,
    a21 = a[7]!,
    a22 = a[8]!
  const b00 = b[0]!,
    b01 = b[1]!,
    b02 = b[2]!
  const b10 = b[3]!,
    b11 = b[4]!,
    b12 = b[5]!
  const b20 = b[6]!,
    b21 = b[7]!,
    b22 = b[8]!
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
  const pPlane: { x: number; y: number }[] = []
  const pIm: { x: number; y: number }[] = []
  for (const c of pairs) {
    pPlane.push(c.plane)
    pIm.push(c.image)
  }
  const tP = hartley2(pPlane, m) as Mat3
  const tI = hartley2(pIm, m) as Mat3
  const tInvI = invertMat3RowMajor(tI)
  if (!tInvI) {
    return undefined
  }

  const rows = 2 * m
  const a = new Float64Array(rows * 9)
  for (let r = 0; r < m; r++) {
    const pl = pPlane[r]!
    const im = pIm[r]!
    const Xn = tP[0]! * pl.x + tP[1]! * pl.y + tP[2]!
    const Yn = tP[3]! * pl.x + tP[4]! * pl.y + tP[5]!
    const Un = tI[0]! * im.x + tI[1]! * im.y + tI[2]!
    const Vn = tI[3]! * im.x + tI[4]! * im.y + tI[5]!
    const row0 = 2 * r
    a[row0 * 9 + 0] = -Xn
    a[row0 * 9 + 1] = -Yn
    a[row0 * 9 + 2] = -1
    a[row0 * 9 + 3] = 0
    a[row0 * 9 + 4] = 0
    a[row0 * 9 + 5] = 0
    a[row0 * 9 + 6] = Un * Xn
    a[row0 * 9 + 7] = Un * Yn
    a[row0 * 9 + 8] = Un
    const row1 = 2 * r + 1
    a[row1 * 9 + 0] = 0
    a[row1 * 9 + 1] = 0
    a[row1 * 9 + 2] = 0
    a[row1 * 9 + 3] = -Xn
    a[row1 * 9 + 4] = -Yn
    a[row1 * 9 + 5] = -1
    a[row1 * 9 + 6] = Vn * Xn
    a[row1 * 9 + 7] = Vn * Yn
    a[row1 * 9 + 8] = Vn
  }

  const h = solveHomogeneousNullVector(a, rows, 9)
  const hN = hVecToMat3(h)
  const comb = matMul3(matMul3(tInvI, hN), tP)
  const w8 = comb[8]!
  if (abs(w8) < 1e-12) {
    return undefined
  }
  for (let i = 0; i < 9; i++) {
    comb[i]! /= w8
  }
  return comb
}
