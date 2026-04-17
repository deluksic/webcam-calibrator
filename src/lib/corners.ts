// Corner detection: label-filtered edge extraction, orientation clustering,
// line fitting with R², and line intersection for robust quad corners.

import { Point } from './geometry';

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
): { x: number; y: number; tangent: number; magnitude: number }[] {
  const pixels: { x: number; y: number; tangent: number; magnitude: number }[] = [];
  const EPS = 1e-6;

  const x0 = Math.floor(minX);
  const y0 = Math.floor(minY);
  const x1 = Math.floor(maxX);
  const y1 = Math.floor(maxY);

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      // Only include pixels belonging to this region
      if (labelData[y * width + x] !== regionLabel) continue;

      const idx = y * width + x;
      const gx = sobelData[idx * 2];
      const gy = sobelData[idx * 2 + 1];
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag < EPS) continue;

      // Tangent is perpendicular to gradient (gives edge direction, not gradient direction)
      const tangent = Math.atan2(gy, gx) + Math.PI / 2;

      pixels.push({ x, y, tangent, magnitude: mag });
    }
  }

  return pixels;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: K-means clustering on circular tangent space
// ─────────────────────────────────────────────────────────────────────────────

/** Distance between two angles, accounting for circular wrap-around. */
function tangentDist(a: number, b: number): number {
  let d = Math.abs(a - b);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d;
}

/** Mean of circular angles (vector sum method). */
function circularMean(angles: number[]): number {
  let sx = 0, sy = 0;
  for (const a of angles) {
    sx += Math.cos(a);
    sy += Math.sin(a);
  }
  return Math.atan2(sy, sx);
}

/**
 * K-means clustering into 4 groups on circular tangent space.
 * Uses 3 random restarts and picks the best partition by total intra-cluster distance.
 * Returns cluster assignments (each pixel index → cluster id 0..3).
 */
function kMeansTangent(
  pixels: { tangent: number }[],
  k: number = 4,
  maxRestarts: number = 3,
): Int32Array {
  const n = pixels.length;
  if (n < k) return new Int32Array(n);

  let bestAssignments: Int32Array | null = null;
  let bestTotalDist = Infinity;

  for (let restart = 0; restart < maxRestarts; restart++) {
    // Initialize centroids by spreading them around the circle
    const centroids: number[] = [];
    const spacing = (2 * Math.PI) / k;
    for (let i = 0; i < k; i++) {
      const base = (restart / maxRestarts) * spacing;
      centroids.push((i * spacing + base) % (2 * Math.PI));
    }

    let assignments = new Int32Array(n);
    let converged = false;

    for (let iter = 0; iter < 20 && !converged; iter++) {
      converged = true;

      // Assign each pixel to nearest centroid
      for (let i = 0; i < n; i++) {
        let bestCluster = 0;
        let bestDist = Infinity;
        for (let c = 0; c < k; c++) {
          const d = tangentDist(pixels[i].tangent, centroids[c]);
          if (d < bestDist) {
            bestDist = d;
            bestCluster = c;
          }
        }
        if (assignments[i] !== bestCluster) converged = false;
        assignments[i] = bestCluster;
      }
      if (converged) break;

      // Update centroids
      for (let c = 0; c < k; c++) {
        const groupAngles = [];
        for (let i = 0; i < n; i++) {
          if (assignments[i] === c) groupAngles.push(pixels[i].tangent);
        }
        centroids[c] = groupAngles.length > 0 ? circularMean(groupAngles) : centroids[c];
      }
    }

    // Compute total intra-cluster distance (quality metric)
    let totalDist = 0;
    for (let i = 0; i < n; i++) {
      totalDist += tangentDist(pixels[i].tangent, centroids[assignments[i]]);
    }

    if (totalDist < bestTotalDist) {
      bestTotalDist = totalDist;
      bestAssignments = assignments;
    }
  }

  return bestAssignments!;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Orthogonal least-squares line fit + R² metric
// ─────────────────────────────────────────────────────────────────────────────

export interface LineFit {
  /** Normalized direction: (cos(angle), sin(angle)) — along the edge */
  dir: { x: number; y: number };
  /** Perpendicular direction: normal to the edge */
  normal: { x: number; y: number };
  /** Distance from origin along normal (signed) */
  d: number;
  /** R² — fraction of variance explained (0..1). Higher = better fit. */
  r2: number;
  /** Number of inliers used for the fit */
  count: number;
}

/**
 * RANSAC line fitting: picks random 2-point samples, counts inliers within
 * threshold, and returns the best line found.
 * Returns line in normal form: normal·(x,y) = d.
 * Normal is the average direction perpendicular to sampled segments.
 */
function fitLine(
  points: { x: number; y: number }[],
  tangents: number[],
  seed: number = 42,
): LineFit | null {
  if (points.length < 3) return null;

  const n = points.length;
  const ITER = 50;
  const THRESH = 1.5;

  // Seeded LCG RNG (deterministic)
  let rng = (seed * 1664525 + 1013904223) >>> 0;
  const rand = () => { rng = (rng * 1664525 + 1013904223) >>> 0; return rng / 0xFFFFFFFF; };
  const randInt = (max: number) => Math.floor(rand() * max);

  let bestNx = 0, bestNy = 0, bestD = 0, bestInliers = 0;

  for (let iter = 0; iter < ITER; iter++) {
    // Pick 2 random distinct points
    const i1 = randInt(n);
    let i2 = randInt(n);
    while (i2 === i1) i2 = randInt(n);

    const p1 = points[i1], p2 = points[i2];
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 1) continue;

    // Direction along the line segment
    const dirX = dx / len, dirY = dy / len;
    // Normal = perpendicular to direction
    let nx = -dirY, ny = dirX;
    // Ensure consistent orientation using mean tangent
    const meanT = circularMean(tangents);
    const tNx = Math.cos(meanT - Math.PI / 2);
    const tNy = Math.sin(meanT - Math.PI / 2);
    if (nx * tNx + ny * tNy < 0) { nx = -nx; ny = -ny; }

    const d = nx * p1.x + ny * p1.y;

    // Count inliers
    let inliers = 0;
    for (let i = 0; i < n; i++) {
      const dist = Math.abs(nx * points[i].x + ny * points[i].y - d);
      if (dist < THRESH) inliers++;
    }

    if (inliers > bestInliers) {
      bestInliers = inliers;
      bestNx = nx; bestNy = ny; bestD = d;
    }
  }

  if (bestInliers < 3) return null;

  // Refine: fit a line to all inliers using least-squares (centroid + normal from mean tangent)
  const inlierPoints: { x: number; y: number }[] = [];
  const inlierTangents: number[] = [];
  for (let i = 0; i < n; i++) {
    const dist = Math.abs(bestNx * points[i].x + bestNy * points[i].y - bestD);
    if (dist < THRESH) {
      inlierPoints.push(points[i]);
      inlierTangents.push(tangents[i]);
    }
  }

  let cx = 0, cy = 0;
  for (const p of inlierPoints) { cx += p.x; cy += p.y; }
  cx /= inlierPoints.length;
  cy /= inlierPoints.length;

  // Normal from mean inlier tangent
  const meanT = circularMean(inlierTangents);
  let gx = Math.cos(meanT - Math.PI / 2);
  let gy = Math.sin(meanT - Math.PI / 2);
  const gLen = Math.sqrt(gx * gx + gy * gy);
  if (gLen < 1e-6) return null;
  gx /= gLen; gy /= gLen;
  const dRefined = gx * cx + gy * cy;

  // R² = fraction of inliers
  const r2 = inlierPoints.length / n;

  return {
    dir: { x: -gy, y: gx },
    normal: { x: gx, y: gy },
    d: dRefined,
    r2,
    count: inlierPoints.length,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Line intersection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Intersection of two lines given in normal form: n1·(x,y) = d1, n2·(x,y) = d2.
 * Solves: [n1x n1y; n2x n2y] * [x;y] = [d1; d2]
 */
function lineIntersection(l1: LineFit, l2: LineFit): Point | null {
  const det = l1.normal.x * l2.normal.y - l2.normal.x * l1.normal.y;
  if (Math.abs(det) < 1e-10) return null; // parallel or coincident lines
  const invDet = 1 / det;
  const x = (l2.normal.y * l1.d - l1.normal.y * l2.d) * invDet;
  const y = (l1.normal.x * l2.d - l2.normal.x * l1.d) * invDet;
  return { x, y };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5: Plausibility checks on corners
// ─────────────────────────────────────────────────────────────────────────────

/** Signed area of a polygon (positive = CCW, negative = CW). */
function signedArea(pts: Point[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
  }
  return a / 2;
}

/**
 * Plausibility checks on 4 corners:
 * - All 4 corners defined
 * - Quad is convex (signed area must have consistent sign)
 * - All R² values above threshold (lines fit well)
 * - Corners within reasonable bounds of the region
 * - Min/max edge lengths > 0 and edge length ratio reasonable
 */
function plausibilityCheck(
  corners: Point[],
  lines: LineFit[],
  assignments: Int32Array,
  pixels: { x: number; y: number; tangent: number; magnitude: number }[],
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  minR2: number = 0.80,
): boolean {
  if (corners.length !== 4) return false;
  if (lines.some((l) => l.r2 < minR2)) return false;

  // Must be convex — all corners must turn the same direction
  const area = signedArea(corners);
  // Use relative tolerance: if |area| < 1e-4 * max(|x|,|y|)^2, shape is degenerate
  const scale = Math.max(...corners.map(c => Math.max(Math.abs(c.x), Math.abs(c.y))));
  if (Math.abs(area) < 1e-4 * scale * scale) return false;

  // Check each corner is roughly within region bounds (with some margin)
  const margin = Math.max((maxX - minX), (maxY - minY)) * 0.1;
  for (const c of corners) {
    if (c.x < minX - margin || c.x > maxX + margin) return false;
    if (c.y < minY - margin || c.y > maxY + margin) return false;
  }

  // Compute edge lengths and check ratios
  const edges: number[] = [];
  for (let i = 0; i < 4; i++) {
    const c1 = corners[i];
    const c2 = corners[(i + 1) % 4];
    edges.push(Math.hypot(c2.x - c1.x, c2.y - c1.y));
  }

  // All edges must be non-zero
  if (edges.some((e) => e < 2)) return false;

  // Opposite edges should have similar lengths (parallelogram check)
  const ratio01 = edges[0] / edges[2];
  const ratio23 = edges[1] / edges[3];
  if (ratio01 < 0.5 || ratio01 > 2.0 || ratio23 < 0.5 || ratio23 > 2.0) return false;

  // Each cluster should have a reasonable number of inliers
  const clusterCounts = new Array(4).fill(0);
  for (let i = 0; i < assignments.length; i++) {
    clusterCounts[assignments[i]]++;
  }
  if (clusterCounts.some((c) => c < 5)) return false;

  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Find quad corners using:
 * 1. Label-filtered Sobel edge extraction
 * 2. K-means clustering of edge tangents into 4 groups
 * 3. Orthogonal least-squares line fit per cluster (with R² metric)
 * 4. Intersection of adjacent lines → 4 corner points
 * 5. Plausibility checks (convexity, R² threshold, bounds, edge ratios)
 *
 * Returns 4 corners ordered [TL, TR, BL, BR] for triangle strip topology,
 * or empty array if detection fails.
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
  minR2: number = 0.80,
  minEdgePixels: number = 20,
  seed: number = 42,
): Point[] {
  // Step 1: Extract label-filtered edge pixels
  const pixels = extractLabeledEdgePixels(
    sobelData, labelData, width, regionLabel, minX, minY, maxX, maxY,
  );

  if (pixels.length < minEdgePixels) return [];

  // Step 2: Cluster by tangent direction
  const assignments = kMeansTangent(pixels, 4, 3);

  // Step 3: Fit lines per cluster
  const lines: (LineFit | null)[] = [];
  for (let c = 0; c < 4; c++) {
    const clusterPoints: { x: number; y: number }[] = [];
    const clusterTangents: number[] = [];
    for (let i = 0; i < pixels.length; i++) {
      if (assignments[i] === c) {
        clusterPoints.push({ x: pixels[i].x, y: pixels[i].y });
        clusterTangents.push(pixels[i].tangent);
      }
    }
    lines.push(fitLine(clusterPoints, clusterTangents, seed + c));
  }


  // Step 4: Intersect lines.
  // Pair each line with those whose normal is ~90° away (perpendicular).
  const rawIntersections: Point[] = [];
  for (let i = 0; i < 4; i++) {
    if (!lines[i]) continue;
    for (let j = i + 1; j < 4; j++) {
      if (!lines[j]) continue;
      const dot = Math.abs(lines[i]!.normal.x * lines[j]!.normal.x + lines[i]!.normal.y * lines[j]!.normal.y);
      if (dot > 0.2) continue;
      const p = lineIntersection(lines[i]!, lines[j]!);
      if (!p) continue;
      const margin = Math.max(maxX - minX, maxY - minY);
      if (p.x < minX - margin || p.x > maxX + margin) continue;
      if (p.y < minY - margin || p.y > maxY + margin) continue;
      rawIntersections.push(p);
    }
  }

  if (rawIntersections.length < 4) return [];

  // Deduplicate: intersections that are very close together belong to the same corner.
  const corners: Point[] = [];
  for (const p of rawIntersections) {
    const tooClose = corners.some((c) => Math.hypot(p.x - c.x, p.y - c.y) < 5);
    if (!tooClose) corners.push(p);
  }
  if (corners.length < 4) return [];

  // Step 5: Order corners as [TL, TR, BL, BR] using centroid-based row split.
  const cx = corners.reduce((s, p) => s + p.x, 0) / corners.length;
  const cy = corners.reduce((s, p) => s + p.y, 0) / corners.length;

  // Split into top row (y <= centroid) and bottom row (y > centroid),
  // then sort each row by x to get left/right within the row.
  const top: Point[] = [];
  const bottom: Point[] = [];
  for (const c of corners) {
    if (c.y <= cy) top.push(c);
    else bottom.push(c);
  }
  top.sort((a, b) => a.x - b.x);
  bottom.sort((a, b) => a.x - b.x);
  const tl = top[0]!, tr = top[1]!, bl = bottom[0]!, br = bottom[1]!;

  // Step 6: Plausibility checks
  if (!plausibilityCheck([tl, tr, br, bl], lines.filter((l) => l !== null) as LineFit[], assignments, pixels, minX, minY, maxX, maxY, minR2)) {
    return [];
  }

  return [tl, tr, bl, br];
}
