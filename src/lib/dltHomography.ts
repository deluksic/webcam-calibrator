import type { Mat3, Point } from '@/lib/geometry'
import { solveHomogeneousNullVector } from '@/lib/jacobiEigenSym'

const { abs } = Math

export interface PlanePoint {
  x: number
  y: number
}

export interface Correspondence {
  plane: PlanePoint
  image: Point
}

type Mat3Mut = [number, number, number, number, number, number, number, number, number]

function hVecToMat3(h: Float64Array): Mat3Mut {
  return [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!, h[8]!]
}

export function solveHomographyDLT(pairs: ReadonlyArray<Correspondence>): Mat3 | undefined {
  const m = pairs.length
  if (m < 4) {
    return undefined
  }

  const rows = 2 * m
  const a = new Float64Array(rows * 9)

  for (let r = 0; r < m; r++) {
    const pl = pairs[r]!.plane
    const im = pairs[r]!.image

    const row = r * 18

    // Row for u: -X, -Y, -1, 0, 0, 0, X*u, Y*u, u
    a[row + 0] = -pl.x
    a[row + 1] = -pl.y
    a[row + 2] = -1
    a[row + 6] = pl.x * im.x
    a[row + 7] = pl.y * im.x
    a[row + 8] = im.x

    // Row for v: -X, -Y, -1, 0, 0, 0, X*v, Y*v, v
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
  const h_norm = hN.map((v, i) => i === 8 ? v : v / h[8]!)

  // Check if h[8] is valid
  if (abs(h_norm[8]!) < 1e-12) {
    return undefined
  }

  return h_norm
}