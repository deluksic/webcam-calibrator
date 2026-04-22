// Corner detection: label-filtered edge extraction, Sobel-gradient clustering (cosine),
// RANSAC + PCA line fit per cluster, and line intersection for robust quad corners.

import type { Corners, Point } from '@/lib/geometry'
import { length } from '@/lib/geometry'
import { hasExactlyFourElements } from '@/utils/assertArray'

const { cos, sin, PI, floor, sqrt, max, min, abs, sign } = Math

export interface CornerDebugInfo {
  /** Failure code: 0 = success, or bitmask of failure reasons */
  failureCode: number
  /** Number of edge pixels extracted */
  edgePixelCount: number
  /** Minimum R² among the 4 fitted lines */
  minR2: number
  /** Number of valid line-line intersections found */
  intersectionCount: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Label-filtered edge pixel extraction
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Extract edge pixels belonging to a specific region label within a bounding box.
 * Ignores edge pixels whose label doesn't match the region's label — prevents
 * contamination from neighboring regions whose bounding boxes overlap.
 */
export function extractLabeledEdgePixels(
  sobelData: Float32Array,
  labelData: Uint32Array,
  width: number,
  regionLabel: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): LabeledEdgePixel[] {
  const pixels: LabeledEdgePixel[] = []
  const EPS = 1e-6

  const x0 = floor(minX)
  const y0 = floor(minY)
  const x1 = floor(maxX)
  const y1 = floor(maxY)

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      // Only include pixels belonging to this region
      if (labelData[y * width + x] !== regionLabel) {
        continue
      }

      const idx = y * width + x
      const gx = sobelData[idx * 2]
      const gy = sobelData[idx * 2 + 1]
      const mag = sqrt(gx * gx + gy * gy)
      if (mag < EPS) {
        continue
      }

      pixels.push({ x: x + 0.5, y: y + 0.5, gx, gy, magnitude: mag })
    }
  }

  return pixels
}

/** One edge sample: position, raw Sobel gradient (gx, gy), magnitude ‖(gx,gy)‖. */
export interface LabeledEdgePixel {
  x: number
  y: number
  gx: number
  gy: number
  magnitude: number
}

function dot2(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return a.x * b.x + a.y * b.y
}

/** Directed cosine dissimilarity: 1 − cos θ = 1 − (u·v)/(‖u‖‖v‖). No unit pre-normalize required. */
function cosineDissimilarity(u: { x: number; y: number }, v: { x: number; y: number }): number {
  const mu = sqrt(u.x * u.x + u.y * u.y)
  const mv = sqrt(v.x * v.x + v.y * v.y)
  if (mu < 1e-14 || mv < 1e-14) {
    return 1
  }
  return 1 - dot2(u, v) / (mu * mv)
}

/**
 * K-means into 4 clusters on raw gradient vectors (gx, gy).
 * Cost = 1 − cos θ; centroids = normalized sum of member gradients (no sign flips).
 */
function kMeansGradientDirections(pixels: LabeledEdgePixel[], k: number = 4, maxRestarts: number = 3): Int32Array {
  const n = pixels.length
  if (n < k) {
    return new Int32Array(n)
  }

  let bestAssignments: Int32Array | undefined = undefined
  let bestTotalCost = Infinity

  for (let restart = 0; restart < maxRestarts; restart++) {
    const centroids: { x: number; y: number }[] = []
    const spacing = (2 * PI) / k
    const base = (restart / maxRestarts) * spacing
    for (let i = 0; i < k; i++) {
      const t = i * spacing + base
      centroids.push({ x: cos(t), y: sin(t) })
    }

    let assignments = new Int32Array(n)
    let converged = false

    for (let iter = 0; iter < 30 && !converged; iter++) {
      converged = true

      for (let i = 0; i < n; i++) {
        const v = { x: pixels[i].gx, y: pixels[i].gy }
        let bestCluster = 0
        let bestCost = Infinity
        for (let c = 0; c < k; c++) {
          const cost = cosineDissimilarity(v, centroids[c])
          if (cost < bestCost) {
            bestCost = cost
            bestCluster = c
          }
        }
        if (assignments[i] !== bestCluster) {
          converged = false
        }
        assignments[i] = bestCluster
      }
      if (converged) {
        break
      }

      for (let c = 0; c < k; c++) {
        let sx = 0
        let sy = 0
        for (let i = 0; i < n; i++) {
          if (assignments[i] !== c) {
            continue
          }
          sx += pixels[i].gx
          sy += pixels[i].gy
        }
        const len = length(sx, sy)
        if (len > 1e-8) {
          centroids[c] = { x: sx / len, y: sy / len }
        }
      }
    }

    let totalCost = 0
    for (let i = 0; i < n; i++) {
      const v = { x: pixels[i].gx, y: pixels[i].gy }
      totalCost += cosineDissimilarity(v, centroids[assignments[i]])
    }

    if (totalCost < bestTotalCost) {
      bestTotalCost = totalCost
      bestAssignments = Int32Array.from(assignments)
    }
  }

  return bestAssignments!
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Orthogonal least-squares line fit + R² metric
// ─────────────────────────────────────────────────────────────────────────────

export interface LineFit {
  /** Unit vector along the fitted edge */
  dir: { x: number; y: number }
  /** Unit normal (line: normal·(x,y) = d) */
  normal: { x: number; y: number }
  /** Distance from origin along normal (signed) */
  d: number
  /** R² — fraction of variance explained (0..1). Higher = better fit. */
  r2: number
  /** Number of inliers used for the fit */
  count: number
}

/**
 * Unit normal from 2D point scatter: eigenvector for the smaller covariance eigenvalue.
 * Returns undefined if scatter is degenerate or too isotropic for a line.
 */
function normalFromInlierScatter(points: { x: number; y: number }[]): { nx: number; ny: number } | undefined {
  const n = points.length
  if (n < 2) {
    return undefined
  }

  let cx = 0
  let cy = 0
  for (const p of points) {
    cx += p.x
    cy += p.y
  }
  cx /= n
  cy /= n

  let sxx = 0
  let syy = 0
  let sxy = 0
  for (const p of points) {
    const dx = p.x - cx
    const dy = p.y - cy
    sxx += dx * dx
    syy += dy * dy
    sxy += dx * dy
  }

  const tr = sxx + syy
  const det = sxx * syy - sxy * sxy
  const disc = max(0, tr * tr - 4 * det)
  const root = sqrt(disc)
  const lamMax = (tr + root) * 0.5
  const lamMin = (tr - root) * 0.5

  if (lamMax < 1e-10) {
    return undefined
  }
  if (lamMin / lamMax > 0.15) {
    return undefined
  }

  let nx = sxy
  let ny = lamMin - sxx
  let len = length(nx, ny)
  if (len < 1e-10) {
    nx = lamMin - syy
    ny = sxy
    len = length(nx, ny)
  }
  if (len < 1e-10) {
    return undefined
  }
  return { nx: nx / len, ny: ny / len }
}

/**
 * RANSAC line fit on (x,y), then PCA on inliers for the normal. undefined if PCA rejects the inlier set.
 */
function fitLine(points: { x: number; y: number }[], seed: number = 42): LineFit | undefined {
  if (points.length < 3) {
    return undefined
  }

  const n = points.length
  const ITER = 50
  const THRESH = 3.0

  let rng = (seed * 1664525 + 1013904223) >>> 0
  const rand = () => {
    rng = (rng * 1664525 + 1013904223) >>> 0
    return rng / 0xffffffff
  }
  const randInt = (max: number) => floor(rand() * max)

  let bestNx = 0,
    bestNy = 0,
    bestD = 0,
    bestInliers = 0

  for (let iter = 0; iter < ITER; iter++) {
    const i1 = randInt(n)
    let i2 = randInt(n)
    while (i2 === i1) {
      i2 = randInt(n)
    }

    const p1 = points[i1],
      p2 = points[i2]
    const dx = p2.x - p1.x,
      dy = p2.y - p1.y
    const len = sqrt(dx * dx + dy * dy)
    if (len < 1) {
      continue
    }

    const nx = -dy / len
    const ny = dx / len
    const d = nx * p1.x + ny * p1.y

    let inliers = 0
    for (let i = 0; i < n; i++) {
      const dist = abs(nx * points[i].x + ny * points[i].y - d)
      if (dist < THRESH) {
        inliers++
      }
    }

    if (inliers > bestInliers) {
      bestInliers = inliers
      bestNx = nx
      bestNy = ny
      bestD = d
    }
  }

  if (bestInliers < 3) {
    return undefined
  }

  const inlierPoints: { x: number; y: number }[] = []
  for (let i = 0; i < n; i++) {
    const dist = abs(bestNx * points[i].x + bestNy * points[i].y - bestD)
    if (dist < THRESH) {
      inlierPoints.push(points[i])
    }
  }

  let cx = 0,
    cy = 0
  for (const p of inlierPoints) {
    cx += p.x
    cy += p.y
  }
  cx /= inlierPoints.length
  cy /= inlierPoints.length

  const refined = normalFromInlierScatter(inlierPoints)
  if (!refined) {
    return undefined
  }

  const gx = refined.nx
  const gy = refined.ny
  const dRefined = gx * cx + gy * cy
  const r2 = inlierPoints.length / n

  return {
    dir: { x: -gy, y: gx },
    normal: { x: gx, y: gy },
    d: dRefined,
    r2,
    count: inlierPoints.length,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Line intersection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Intersection of two lines given in normal form: n1·(x,y) = d1, n2·(x,y) = d2.
 * Solves: [n1x n1y; n2x n2y] * [x;y] = [d1; d2]
 */
function lineIntersection(l1: LineFit, l2: LineFit): Point | undefined {
  const det = l1.normal.x * l2.normal.y - l2.normal.x * l1.normal.y
  if (abs(det) < 1e-10) {
    return undefined
  } // parallel or coincident lines
  const invDet = 1 / det
  const x = (l2.normal.y * l1.d - l1.normal.y * l2.d) * invDet
  const y = (l1.normal.x * l2.d - l2.normal.x * l1.d) * invDet
  return { x, y }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5: Plausibility checks on corners
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Allowed distance outside the region extent bbox (min/max from labeling).
 * Must match line–line intersection clipping so corners are not accepted in step 4
 * then rejected in plausibility solely for sitting just past the bbox.
 */
function extentBBoxSlack(minX: number, minY: number, maxX: number, maxY: number): number {
  return max(6, 0.5 * max(maxX - minX, maxY - minY))
}

/** Signed area of a polygon (positive = CCW, negative = CW). */
function signedArea(pts: Point[]): number {
  let a = 0
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length
    a += pts[i].x * pts[j].y - pts[j].x * pts[i].y
  }
  return a / 2
}

/** Z-component of (b − a) × (c − b): turn at corner b along a → b → c. */
function crossTurn(a: Point, b: Point, c: Point): number {
  return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
}

/**
 * True iff these four points in cyclic order form a simple strictly convex quad
 * (every turn same sign as shoelace area — no bow-tie, no collinear vertex).
 */
function isStrictConvexQuadCycle(order: Corners): boolean {
  const area = signedArea(order)
  if (abs(area) < 1e-12) {
    return false
  }
  const s = sign(area)
  for (let i = 0; i < 4; i++) {
    const a = order[i]!
    const b = order[(i + 1) % 4]!
    const c = order[(i + 2) % 4]!
    const t = crossTurn(a, b, c)
    if (abs(t) < 1e-10) {
      return false
    }
    if (sign(t) !== s) {
      return false
    }
  }
  return true
}

function allPermutationsFour(pts: Corners): Corners[] {
  const out: Corners[] = []
  const indices = [0, 1, 2, 3] as const
  for (const a of indices) {
    for (const b of indices) {
      if (b === a) {
        continue
      }
      for (const c of indices) {
        if (c === a || c === b) {
          continue
        }
        for (const d of indices) {
          if (d === a || d === b || d === c) {
            continue
          }
          out.push([pts[a], pts[b], pts[c], pts[d]])
        }
      }
    }
  }
  return out
}

/**
 * One CCW convex boundary order of the four points (positive signed area, strict convexity).
 * Picks the cyclic order with largest signed area among valid permutations.
 */
function findConvexCCWCycle(pts: Corners): Corners | undefined {
  let best: Corners | undefined = undefined
  let bestArea = 0
  for (const ordered of allPermutationsFour(pts)) {
    const area = signedArea(ordered)
    if (area <= 1e-10) {
      continue
    }
    if (!isStrictConvexQuadCycle(ordered)) {
      continue
    }
    if (area > bestArea) {
      bestArea = area
      best = [ordered[0], ordered[1], ordered[2], ordered[3]]
    }
  }
  return best
}

export function rotateRing(ring: Corners, k: number): Corners {
  const indices = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]][k % 4]
  return [ring[indices[0]]!, ring[indices[1]]!, ring[indices[2]]!, ring[indices[3]]!]
}

/**
 * Plausibility checks on 4 corners:
 * - All 4 corners defined
 * - Quad is convex (signed area must have consistent sign)
 * - All R² values above threshold (lines fit well)
 * - Corners within region extent bbox ± same slack as intersection clipping (half max span, min 6px)
 * - Min/max edge lengths > 0 and edge length ratio reasonable
 */
function plausibilityCheck(
  corners: Corners,
  lines: LineFit[],
  assignments: Int32Array,
  pixels: LabeledEdgePixel[],
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  minR2: number = 0.8,
): boolean {
  if (lines.some((l) => l.r2 < minR2)) {
    return false
  }

  // Must be convex — all corners must turn the same direction
  const area = signedArea(corners)
  // Use relative tolerance: if |area| < 1e-4 * max(|x|,|y|)^2, shape is degenerate
  const scale = max(...corners.map((c) => max(abs(c.x), abs(c.y))))
  if (abs(area) < 1e-4 * scale * scale) {
    return false
  }

  const margin = extentBBoxSlack(minX, minY, maxX, maxY)
  for (const c of corners) {
    if (c.x < minX - margin || c.x > maxX + margin) {
      return false
    }
    if (c.y < minY - margin || c.y > maxY + margin) {
      return false
    }
  }

  // Compute edge lengths and check ratios
  const edges: number[] = []
  for (let i = 0; i < 4; i++) {
    const c1 = corners[i]
    const c2 = corners[(i + 1) % 4]
    edges.push(length(c2.x - c1.x, c2.y - c1.y))
  }

  // All edges must be non-zero
  if (edges.some((e) => e < 2)) {
    return false
  }

  // Opposite edges should have similar lengths (parallelogram check)
  // Relaxed for oblique/foreshortened quads: allow up to 3x ratio
  const ratio01 = edges[0] / edges[2]
  const ratio23 = edges[1] / edges[3]
  if (ratio01 < 0.33 || ratio01 > 3.0 || ratio23 < 0.33 || ratio23 > 3.0) {
    return false
  }

  // Each cluster should have a reasonable number of inliers
  // Reduced from 5 to 3 for sparse edge regions (oblique tags)
  const clusterCounts = Array.from({ length: 4 }, () => 0)
  for (let i = 0; i < assignments.length; i++) {
    clusterCounts[assignments[i]]++
  }
  if (clusterCounts.some((c) => c < 3)) {
    return false
  }

  return true
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Result of corner detection: corners + debug info for GPU overlay.
 */
export interface CornerResult {
  /** 4 corners ordered [TL, TR, BL, BR], or `undefined` if detection failed */
  corners: Corners | undefined
  debug: CornerDebugInfo
}

// Failure codes (bitmask — can combine multiple failures). Order matches pipeline stages.
export const FAIL_INSUFFICIENT_EDGES = 1 << 0 // not enough edge pixels (first gate)
export const FAIL_ASPECT_RATIO = 1 << 1 // reserved — not set by findCornersFromEdgesWithDebug today
export const FAIL_LINE_FIT_FAILED = 1 << 2 // undefined line from RANSAC (after clustering)
export const FAIL_PLAUSIBILITY = 1 << 3 // no convex cycle, no valid TL..BR rotation, or plausibility checks
export const FAIL_NO_INTERSECTIONS = 1 << 4 // <4 valid line-line intersections

/**
 * Find quad corners using:
 * 1. Label-filtered Sobel edge extraction
 * 2. K-means clustering of raw Sobel gradients into 4 groups (1 − cos θ)
 * 3. Orthogonal least-squares line fit per cluster (with R² metric)
 * 4. Intersection of all line pairs (non-parallel) → up to 6 points, deduped to 4 corners
 * 5. Convex cyclic order (strict turns) + rotation so edges match fitted lines, then plausibility (R², bounds, edge ratios)
 *
 * Returns corners ordered [TL, TR, BL, BR] for `computeHomography` / triangle strip and `buildTagGrid`.
 *
 * Returns `undefined` if detection fails.
 */
export function findCornersFromEdges(
  sobelData: Float32Array,
  labelData: Uint32Array,
  width: number,
  regionLabel: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  minR2: number = 0.7,
  minEdgePixels: number = 12,
  seed: number = 42,
): Corners | undefined {
  const result = findCornersFromEdgesWithDebug(
    sobelData,
    labelData,
    width,
    regionLabel,
    minX,
    minY,
    maxX,
    maxY,
    minR2,
    minEdgePixels,
    seed,
  )
  return result.corners
}

/**
 * Same as findCornersFromEdges but returns corners + debug info for GPU overlay.
 */
export function findCornersFromEdgesWithDebug(
  sobelData: Float32Array,
  labelData: Uint32Array,
  width: number,
  regionLabel: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  minR2: number = 0.7,
  minEdgePixels: number = 12,
  seed: number = 42,
): CornerResult {
  let failureCode = 0
  let edgePixelCount = 0
  let minR2Seen = 1.0
  let intersectionCount = 0

  // Step 1: Extract label-filtered edge pixels
  const pixels = extractLabeledEdgePixels(sobelData, labelData, width, regionLabel, minX, minY, maxX, maxY)
  edgePixelCount = pixels.length

  if (pixels.length < minEdgePixels) {
    failureCode |= FAIL_INSUFFICIENT_EDGES
    return {
      corners: undefined,
      debug: { failureCode, edgePixelCount, minR2: 0, intersectionCount: 0 },
    }
  }

  // Step 2: Cluster by gradient direction (cosine on raw gx, gy)
  const assignments = kMeansGradientDirections(pixels, 4, 3)

  // Step 3: Fit lines per cluster
  const lines: (LineFit | undefined)[] = []
  for (let c = 0; c < 4; c++) {
    const clusterPoints: { x: number; y: number }[] = []
    for (let i = 0; i < pixels.length; i++) {
      if (assignments[i] === c) {
        clusterPoints.push({ x: pixels[i].x, y: pixels[i].y })
      }
    }
    const line = fitLine(clusterPoints, seed + c)
    lines.push(line)
    if (line) {
      minR2Seen = min(minR2Seen, line.r2)
    }
    if (!line) {
      failureCode |= FAIL_LINE_FIT_FAILED
    }
  }

  // Step 4: Intersect all pairs of non-undefined fitted lines.
  const intersectionSlack = extentBBoxSlack(minX, minY, maxX, maxY)

  const rawIntersections: Point[] = []
  for (let i = 0; i < 4; i++) {
    if (!lines[i]) {
      continue
    }
    for (let j = i + 1; j < 4; j++) {
      if (!lines[j]) {
        continue
      }
      const p = lineIntersection(lines[i]!, lines[j]!)
      if (!p) {
        continue
      }
      if (p.x < minX - intersectionSlack || p.x > maxX + intersectionSlack) {
        continue
      }
      if (p.y < minY - intersectionSlack || p.y > maxY + intersectionSlack) {
        continue
      }
      rawIntersections.push(p)
      intersectionCount++
    }
  }

  if (rawIntersections.length < 4) {
    failureCode |= FAIL_NO_INTERSECTIONS
    return {
      corners: undefined,
      debug: {
        failureCode,
        edgePixelCount,
        minR2: minR2Seen,
        intersectionCount,
      },
    }
  }

  // Deduplicate: intersections that are very close together belong to the same corner.
  const deduped: Point[] = []
  for (const p of rawIntersections) {
    const tooClose = deduped.some((c) => length(p.x - c.x, p.y - c.y) < 5)
    if (!tooClose) {
      deduped.push(p)
    }
  }
  if (!hasExactlyFourElements(deduped)) {
    failureCode |= FAIL_NO_INTERSECTIONS
    return {
      corners: undefined,
      debug: {
        failureCode,
        edgePixelCount,
        minR2: minR2Seen,
        intersectionCount,
      },
    }
  }

  // Step 5: Convex boundary order (consistent turn signs), then label corners by rotation.
  const cycle = findConvexCCWCycle(deduped)
  if (!cycle) {
    failureCode |= FAIL_PLAUSIBILITY
    return {
      corners: undefined,
      debug: {
        failureCode,
        edgePixelCount,
        minR2: minR2Seen,
        intersectionCount,
      },
    }
  }

  const linesNonundefined = lines.filter((l) => l !== undefined)
  let labeled: Corners | undefined = undefined
  for (let k = 0; k < 4; k++) {
    const c = rotateRing(cycle, k)
    if (plausibilityCheck(c, linesNonundefined, assignments, pixels, minX, minY, maxX, maxY, minR2)) {
      labeled = c
      break
    }
  }
  if (!labeled) {
    failureCode |= FAIL_PLAUSIBILITY
    return {
      corners: undefined,
      debug: {
        failureCode,
        edgePixelCount,
        minR2: minR2Seen,
        intersectionCount,
      },
    }
  }
  const [tl, tr, br, bl] = labeled

  return {
    corners: [tl, tr, bl, br],
    debug: { failureCode, edgePixelCount, minR2: minR2Seen, intersectionCount },
  }
}
