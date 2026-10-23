import type { Vec2 } from '@/lib/pointIdentity/types'

/** 2D affine transform: x' = a*x + b*y + c, y' = d*x + e*y + f. */
export type Affine2D = {
  a: number
  b: number
  c: number
  d: number
  e: number
  f: number
}

export type AffineCorrespondence = {
  source: Vec2
  target: Vec2
  weight?: number
}

function solve3(matrix: number[][], rhs: number[]): number[] | undefined {
  const augmented = matrix.map((row, index) => [...row, rhs[index]!])
  for (let column = 0; column < 3; column += 1) {
    let pivot = column
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row]![column]!) > Math.abs(augmented[pivot]![column]!)) {
        pivot = row
      }
    }
    if (Math.abs(augmented[pivot]![column]!) < 1e-12) {
      return undefined
    }
    ;[augmented[column], augmented[pivot]] = [augmented[pivot]!, augmented[column]!]
    const divisor = augmented[column]![column]!
    for (let index = column; index <= 3; index += 1) {
      augmented[column]![index]! /= divisor
    }
    for (let row = 0; row < 3; row += 1) {
      if (row === column) {
        continue
      }
      const factor = augmented[row]![column]!
      if (factor === 0) {
        continue
      }
      for (let index = column; index <= 3; index += 1) {
        augmented[row]![index]! -= factor * augmented[column]![index]!
      }
    }
  }
  return [augmented[0]![3]!, augmented[1]![3]!, augmented[2]![3]!]
}

export function fitAffine2D(correspondences: readonly AffineCorrespondence[]): Affine2D | undefined {
  if (correspondences.length < 4) {
    return undefined
  }

  // Hartley-style normalization keeps small local neighborhoods and
  // hundred-pixel image coordinates well conditioned. Raw-coordinate normal
  // equations can produce garbage affines on one-sided crop edges.
  let sourceMeanX = 0
  let sourceMeanY = 0
  let targetMeanX = 0
  let targetMeanY = 0
  for (const correspondence of correspondences) {
    sourceMeanX += correspondence.source.x
    sourceMeanY += correspondence.source.y
    targetMeanX += correspondence.target.x
    targetMeanY += correspondence.target.y
  }
  const count = correspondences.length
  sourceMeanX /= count
  sourceMeanY /= count
  targetMeanX /= count
  targetMeanY /= count

  let sourceScale = 0
  let targetScale = 0
  for (const correspondence of correspondences) {
    sourceScale += Math.hypot(
      correspondence.source.x - sourceMeanX,
      correspondence.source.y - sourceMeanY,
    )
    targetScale += Math.hypot(
      correspondence.target.x - targetMeanX,
      correspondence.target.y - targetMeanY,
    )
  }
  sourceScale /= count
  targetScale /= count
  if (!(sourceScale > 1e-9) || !(targetScale > 1e-9)) {
    return undefined
  }

  const normal = Array.from({ length: 3 }, () => [0, 0, 0])
  const rightX = [0, 0, 0]
  const rightY = [0, 0, 0]
  for (const correspondence of correspondences) {
    const weight = Math.max(1e-6, correspondence.weight ?? 1)
    const row = [
      (correspondence.source.x - sourceMeanX) / sourceScale,
      (correspondence.source.y - sourceMeanY) / sourceScale,
      1,
    ]
    const targetX = (correspondence.target.x - targetMeanX) / targetScale
    const targetY = (correspondence.target.y - targetMeanY) / targetScale
    for (let first = 0; first < 3; first += 1) {
      for (let second = 0; second < 3; second += 1) {
        normal[first]![second]! += weight * row[first]! * row[second]!
      }
      rightX[first]! += weight * row[first]! * targetX
      rightY[first]! += weight * row[first]! * targetY
    }
  }
  const x = solve3(normal, rightX)
  const y = solve3(normal, rightY)
  if (x === undefined || y === undefined) {
    return undefined
  }
  const determinant = x[0]! * y[1]! - x[1]! * y[0]!
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-8) {
    return undefined
  }

  // Denormalize: source u=(x-meanX)/scale, target X=(x'-meanX')/scale'.
  const a = (targetScale * x[0]!) / sourceScale
  const b = (targetScale * x[1]!) / sourceScale
  const c = targetMeanX + targetScale * x[2]! - a * sourceMeanX - b * sourceMeanY
  const d = (targetScale * y[0]!) / sourceScale
  const e = (targetScale * y[1]!) / sourceScale
  const f = targetMeanY + targetScale * y[2]! - d * sourceMeanX - e * sourceMeanY
  return { a, b, c, d, e, f }
}

export function applyAffine(affine: Affine2D, point: Vec2): Vec2 {
  return {
    x: affine.a * point.x + affine.b * point.y + affine.c,
    y: affine.d * point.x + affine.e * point.y + affine.f,
  }
}

export function invertAffine(affine: Affine2D): Affine2D | undefined {
  const determinant = affine.a * affine.e - affine.b * affine.d
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) {
    return undefined
  }
  const inverse = 1 / determinant
  return {
    a: affine.e * inverse,
    b: -affine.b * inverse,
    c: (affine.b * affine.f - affine.e * affine.c) * inverse,
    d: -affine.d * inverse,
    e: affine.a * inverse,
    f: (affine.d * affine.c - affine.a * affine.f) * inverse,
  }
}

export function affineDeterminant(affine: Affine2D): number {
  return affine.a * affine.e - affine.b * affine.d
}

/** Smaller is better. Used to reject nearly singular local affine fits. */
export function affineCondition(affine: Affine2D): number {
  const determinant = Math.abs(affineDeterminant(affine))
  const norm = Math.hypot(affine.a, affine.b, affine.d, affine.e)
  return norm === 0 ? Number.POSITIVE_INFINITY : norm / Math.max(determinant, 1e-12)
}

/**
 * A valid local registration is finite, orientation-preserving, and not
 * near-collinear. Mirroring is physically impossible here; a negative
 * determinant means the fit reflected the lattice. A high condition number
 * means the affine is nearly singular and will splatter predictions.
 */
export function isAffineUsable(affine: Affine2D, maxCondition = 50): boolean {
  const values = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f]
  if (!values.every((value) => Number.isFinite(value))) {
    return false
  }
  const determinant = affineDeterminant(affine)
  if (!(determinant > 0)) {
    return false
  }
  return affineCondition(affine) <= maxCondition
}
