// 2D geometry utilities for quad fitting

const { sqrt, abs, max, min } = Math

export function length(x: number, y: number): number {
  return sqrt(x * x + y * y)
}

export type Point = {
  x: number
  y: number
}

/** Quad corners in image / triangle-strip order: TL, TR, BL, BR. */
export type Corners = [tl: Point, tr: Point, bl: Point, br: Point]

/** 3×3 row-major homography matrix `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`. */
export type Mat3 = readonly [number, number, number, number, number, number, number, number, number]

export type Line = {
  a: number
  b: number
  c: number // ax + by + c = 0
}

/**
 * Compute line coefficients from two points.
 * Returns null if points are coincident.
 */
export function lineFromPoints(p1: Point, p2: Point): Line | null {
  const dx = p2.x - p1.x
  const dy = p2.y - p1.y
  if (abs(dx) < 1e-10 && abs(dy) < 1e-10) {
    return null // coincident points
  }
  // Normalize: a² + b² = 1
  const len = sqrt(dx * dx + dy * dy)
  return {
    a: -dy / len,
    b: dx / len,
    c: -(p1.x * (-dy / len) + p1.y * (dx / len)),
  }
}

/**
 * Find intersection of two lines.
 * Returns null if lines are parallel.
 */
export function lineIntersection(l1: Line, l2: Line): Point | null {
  const det = l1.a * l2.b - l1.b * l2.a
  if (abs(det) < 1e-10) {
    return null // parallel or coincident
  }
  // Lines are a x + b y + c = 0 (same as a x + b y = −c). Cramer gives:
  // x = (b1*c2 − c1*b2) / det,  y = (c1*a2 − a1*c2) / det
  return {
    x: (l1.b * l2.c - l1.c * l2.b) / det,
    y: (l1.c * l2.a - l1.a * l2.c) / det,
  }
}

/**
 * Compute distance from point to line.
 */
export function pointLineDistance(p: Point, l: Line): number {
  return abs(l.a * p.x + l.b * p.y + l.c)
}

/**
 * Subdivide a line segment into n+1 points (including endpoints).
 * Returns array of n+1 points.
 */
export function subdivideSegment(p1: Point, p2: Point, divisions: number): Point[] {
  const points: Point[] = []
  for (let i = 0; i <= divisions; i++) {
    const t = i / divisions
    points.push({
      x: p1.x + t * (p2.x - p1.x),
      y: p1.y + t * (p2.y - p1.y),
    })
  }
  return points
}

/**
 * Check if four points form a valid quad (roughly).
 * Returns aspect ratio of the quad.
 */
export function quadAspectRatio(corners: Corners): number {
  const [tl, tr, bl, br] = corners

  // side lengths
  const dists = [
    length(tr.x - tl.x, tr.y - tl.y),
    length(br.x - tr.x, br.y - tr.y),
    length(br.x - bl.x, br.y - bl.y),
    length(bl.x - tl.x, bl.y - tl.y),
  ]

  // Use max/min for aspect ratio check
  const maxD = max(...dists)
  const minD = min(...dists)
  return maxD / minD
}

/**
 * Check if two line segments are roughly parallel.
 */
export function areParallel(l1: Line, l2: Line, threshold: number = 0.1): boolean {
  // Parallel if normals are aligned
  const dot = abs(l1.a * l2.a + l1.b * l2.b)
  return dot > 1 - threshold
}

/**
 * Compute a 3x3 homography H that maps the unit square corners to 4 source points.
 *
 * Source points (dst in unit square → src in image space):
 *   (0,0) → src[0] (top-left)
 *   (1,0) → src[1] (top-right)
 *   (0,1) → src[2] (bottom-left)
 *   (1,1) → src[3] (bottom-right)
 *
 * The resulting H has h33=1 and maps (u,v) → (x,y) as:
 *   w = h7*u + h8*v + 1
 *   x = (h1*u + h2*v + h3) / w
 *   y = (h4*u + h5*v + h6) / w
 *
 * On success: 8 values `[h1, h2, h3, h4, h5, h6, h7, h8]`. Returns `undefined` if the 8×8 system is
 * singular (degenerate / collinear corners, or wrong point count).
 */
export function tryComputeHomography(src: Corners): Mat3 | undefined {
  const N = 8

  // Unit-square destination corners (TL, TR, BL, BR) paired with source points.
  const correspondences: readonly (readonly [u: number, v: number, p: Point])[] = [
    [0, 0, src[0]],
    [1, 0, src[1]],
    [0, 1, src[2]],
    [1, 1, src[3]],
  ]

  // Build rows of the 8x8 linear system Ah = b directly as literals.
  // Per correspondence (u,v) → (x,y):
  //   row 2i : [u, v, 1, 0, 0, 0, -u*x, -v*x] · h = x
  //   row 2i+1: [0, 0, 0, u, v, 1, -u*y, -v*y] · h = y
  const A: number[][] = []
  const b: number[] = []
  for (const [u, v, p] of correspondences) {
    A.push([u, v, 1, 0, 0, 0, -u * p.x, -v * p.x])
    A.push([0, 0, 0, u, v, 1, -u * p.y, -v * p.y])
    b.push(p.x, p.y)
  }

  // Gaussian elimination with partial pivoting
  for (let col = 0; col < N; col++) {
    // Find pivot
    let maxRow = col
    for (let row = col + 1; row < N; row++) {
      if (abs(A[row]![col]!) > abs(A[maxRow]![col]!)) {
        maxRow = row
      }
    }

    // Swap rows
    ;[A[col], A[maxRow]] = [A[maxRow]!, A[col]!]
    ;[b[col], b[maxRow]] = [b[maxRow]!, b[col]!]

    const pivot = A[col]![col]!
    if (abs(pivot) < 1e-10) {
      return undefined
    }

    // Normalize pivot row
    const pivotRow = A[col]!
    for (let j = col; j < N; j++) {
      pivotRow[j]! /= pivot
    }
    b[col]! /= pivot

    // Eliminate
    for (let row = 0; row < N; row++) {
      if (row !== col) {
        const rowA = A[row]!
        const factor = rowA[col]!
        if (factor !== 0) {
          for (let j = col; j < N; j++) {
            rowA[j]! -= factor * pivotRow[j]!
          }
          b[row]! -= factor * b[col]!
        }
      }
    }
  }

  return [b[0]!, b[1]!, b[2]!, b[3]!, b[4]!, b[5]!, b[6]!, b[7]!, 1]
}

/**
 * Same as {@link tryComputeHomography}, but throws if the quad does not admit a unique homography.
 */
export function computeHomography(src: Corners): Mat3 {
  const h = tryComputeHomography(src)
  if (!h) {
    throw new Error('Singular homography matrix')
  }
  return h
}

/**
 * Apply homography to a unit-square point (u, v).
 * Returns {x, y} in image coordinates.
 */
export function applyHomography(h: Mat3, u: number, v: number): Point {
  const [h0, h1, h2, h3, h4, h5, h6, h7, h8] = h
  const w = h6 * u + h7 * v + h8
  const x = (h0 * u + h1 * v + h2) / w
  const y = (h3 * u + h4 * v + h5) / w
  return { x, y }
}

/**
 * Compute projective weights for perspective-correct quad rendering.
 * Maps unit square corners to detected quad via homography.
 * Corner order: TL, TR, BL, BR
 * Returns [w0, w1, w2, w3] — weights for TL, TR, BL, BR
 *
 * For homography H: (x,y) = (h1*u + h2*v + h3) / w, where w = h7*u + h8*v + 1
 * Weights at corners: w0=1, w1=h7+1, w2=h8+1, w3=h7+h8+1
 */
export function computeProjectiveWeights(corners: Corners): [number, number, number, number] {
  const x0 = corners[0].x,
    y0 = corners[0].y // TL
  const x1 = corners[1].x,
    y1 = corners[1].y // TR
  const x2 = corners[2].x,
    y2 = corners[2].y // BL
  const x3 = corners[3].x,
    y3 = corners[3].y // BR

  // Solve for h7, h8 from corner correspondences
  // Using: (x3-x1)*h7 + (x3-x2)*h8 = x1 + x2 - x0 - x3
  //        (y3-y1)*h7 + (y3-y2)*h8 = y1 + y2 - y0 - y3
  const A = x3 - x1
  const B = x3 - x2
  const C = x1 + x2 - x0 - x3
  const D = y3 - y1
  const E = y3 - y2
  const F = y1 + y2 - y0 - y3

  const det = A * E - B * D

  if (abs(det) < 1e-10) {
    // Degenerate case - parallelogram or axis-aligned
    return [1, 1, 1, 1]
  }

  const h7 = (C * E - B * F) / det
  const h8 = (A * F - C * D) / det

  // Weights at unit square corners
  const w0 = 1
  const w1 = h7 + 1
  const w2 = h8 + 1
  const w3 = h7 + h8 + 1

  return [w0, w1, w2, w3]
}

/**
 * Verify projective weights: blend corners by weights and check unit square maps.
 * Returns blended corner positions — should equal [0,0],[1,0],[0,1],[1,1].
 */
export function verifyProjectiveWeights(corners: Corners): {
  blended: Corners
  errors: [number, number, number, number]
} {
  const [w0, w1, w2, w3] = computeProjectiveWeights(corners)

  // Compute what corners would be if the unit square maps through these weights
  const _wSum = w0 + w1 + w2 + w3

  // At u=0,v=0: w = w0/wSum → blended = (0 * w0 + 0 * w2 + 0 * w1 + 0 * w3) / wSum = 0 → OK
  // At u=1,v=0: w = w1/wSum → blended = (1 * w1 + 0 * w0 + ... ) / wSum = w1/wSum
  // We want blended = 1 at TR, 0 at TL/BL, and w1/wSum to normalize to 1...

  const [c0, c1, c2, c3] = corners
  const blended: Corners = [
    { x: c0.x / w0, y: c0.y / w0 },
    { x: c1.x / w1, y: c1.y / w1 },
    { x: c2.x / w2, y: c2.y / w2 },
    { x: c3.x / w3, y: c3.y / w3 },
  ]

  const errors: [number, number, number, number] = [
    sqrt(blended[0].x ** 2 + blended[0].y ** 2),                          // expected (0,0)
    sqrt((blended[1].x - 1) ** 2 + blended[1].y ** 2),                    // expected (1,0)
    sqrt(blended[2].x ** 2 + (blended[2].y - 1) ** 2),                    // expected (0,1)
    sqrt((blended[3].x - 1) ** 2 + (blended[3].y - 1) ** 2),              // expected (1,1)
  ]

  return { blended, errors }
}
