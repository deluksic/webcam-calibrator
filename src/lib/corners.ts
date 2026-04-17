// Corner detection from edge pixels using local edge orientation changes.
// Pixels where local edge orientations vary significantly among neighbors
// are classified as corners. Results clustered into 4 groups.

import { Point } from './geometry';

// EdgePixel stored as flat array: [x, y, tangent, magnitude] per pixel
export type EdgePixelArray = Float32Array; // length = n * 4

/**
 * Extract edge pixels from Sobel data within a bounding box.
 * Returns flat array: [x, y, tangent, magnitude, x, y, tangent, magnitude, ...]
 */
export function extractEdgePixelsFromBbox(
  sobelData: Float32Array,
  width: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
): EdgePixelArray {
  const pixels: number[] = [];
  const EPS = 1e-6;

  for (let y = Math.floor(minY); y <= Math.floor(maxY); y++) {
    for (let x = Math.floor(minX); x <= Math.floor(maxX); x++) {
      const idx = y * width + x;

      const gx = sobelData[idx * 2];
      const gy = sobelData[idx * 2 + 1];
      const mag = Math.sqrt(gx * gx + gy * gy);

      if (mag < EPS) continue;

      // Tangent is perpendicular to gradient
      const tangent = Math.atan2(gy, gx) + Math.PI / 2;

      pixels.push(x, y, tangent, mag);
    }
  }

  return new Float32Array(pixels);
}

/**
 * Get pixel from flat array.
 */
export function getPixel(pixels: EdgePixelArray, i: number): { x: number; y: number; tangent: number; magnitude: number } {
  const idx = i * 4;
  return {
    x: pixels[idx],
    y: pixels[idx + 1],
    tangent: pixels[idx + 2],
    magnitude: pixels[idx + 3],
  };
}

/**
 * Find corner candidates by checking neighbor orientation differences.
 * A pixel is a corner if its neighbors have significantly different edge orientations.
 */
export function findCornerCandidates(
  pixels: EdgePixelArray,
  cornerThreshold: number = 0.5, // normalized difference threshold
  minNeighborCount: number = 4,
): { x: number; y: number; diff: number }[] {
  const n = pixels.length / 4;
  if (n < 9) return []; // need 3x3 neighborhood

  const candidates: { x: number; y: number; diff: number }[] = [];
  const maxMag = findMaxMagnitude(pixels);
  const minMagThreshold = maxMag * 0.3;

  for (let i = 0; i < n; i++) {
    const px = pixels[i * 4];
    const py = pixels[i * 4 + 1];
    const tangent = pixels[i * 4 + 2];
    const mag = pixels[i * 4 + 3];

    // Only consider strong edge pixels
    if (mag < minMagThreshold) continue;

    // Find neighbors (within ~5 pixel radius)
    const neighbors: number[] = [];
    const neighborRadius = 5;

    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const nx = pixels[j * 4];
      const ny = pixels[j * 4 + 1];
      const dist = Math.hypot(nx - px, ny - py);
      if (dist <= neighborRadius) {
        neighbors.push(j);
      }
    }

    if (neighbors.length < minNeighborCount) continue;

    // Compute orientation difference with each neighbor
    let totalDiff = 0;
    let maxDiff = 0;

    for (const ni of neighbors) {
      const neighborTangent = pixels[ni * 4 + 2];
      let diff = Math.abs(tangent - neighborTangent);
      if (diff > Math.PI) diff = 2 * Math.PI - diff;
      totalDiff += diff;
      maxDiff = Math.max(maxDiff, diff);
    }

    const avgDiff = totalDiff / neighbors.length;

    // A corner is where local orientation varies significantly
    // Use both average and max difference
    if (maxDiff > cornerThreshold * Math.PI || avgDiff > cornerThreshold * Math.PI * 0.5) {
      candidates.push({ x: px, y: py, diff: maxDiff });
    }
  }

  return candidates;
}

/**
 * Find max magnitude in pixel array.
 */
export function findMaxMagnitude(pixels: EdgePixelArray): number {
  let max = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] > max) max = pixels[i + 3];
  }
  return max;
}

/**
 * Cluster corner candidates into 4 groups based on position.
 * Returns cluster centers ordered roughly clockwise from top-left.
 */
export function clusterCorners(
  candidates: { x: number; y: number; diff: number }[],
): Point[] {
  if (candidates.length < 4) return [];

  // Sort by diff (strength) descending
  const sorted = [...candidates].sort((a, b) => b.diff - a.diff);

  // Simple clustering: divide into 4 quadrants based on centroid
  let sumX = 0, sumY = 0;
  for (const c of candidates) {
    sumX += c.x;
    sumY += c.y;
  }
  const cx = sumX / candidates.length;
  const cy = sumY / candidates.length;

  // Assign candidates to quadrants
  const quadrants: { x: number; y: number; diff: number }[][] = [[], [], [], []];
  for (const c of candidates) {
    const isLeft = c.x < cx;
    const isTop = c.y < cy;
    if (isLeft && isTop) quadrants[0].push(c);      // top-left
    else if (!isLeft && isTop) quadrants[1].push(c); // top-right
    else if (!isLeft && !isTop) quadrants[2].push(c); // bottom-right
    else quadrants[3].push(c);                      // bottom-left
  }

  // Get strongest candidate in each quadrant
  const centers: Point[] = [];
  for (const quad of quadrants) {
    if (quad.length === 0) {
      // No candidates in this quadrant - estimate from overall bounds
      continue;
    }
    // Pick the candidate with highest diff (strongest corner)
    const strongest = quad.reduce((best, c) => c.diff > best.diff ? c : best, quad[0]);
    centers.push({ x: strongest.x, y: strongest.y });
  }

  return centers;
}

/**
 * Order 4 corners as [TL, TR, BR, BL].
 * Sort by y to get top/bottom rows, then by x within each row for left/right.
 */
export function orderCornersClockwise(corners: Point[]): Point[] {
  if (corners.length !== 4) return corners;

  const sorted = [...corners].sort((a, b) => a.y - b.y);
  const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x); // TL, TR
  const bottom = sorted.slice(2, 4).sort((a, b) => a.x - b.x); // BL, BR

  return [top[0], top[1], bottom[1], bottom[0]];
}

/**
 * Full pipeline: find corners from edge pixel array.
 * Returns 4 corner points ordered clockwise.
 */
export function findCornersFromEdges(
  pixels: EdgePixelArray,
  cornerThreshold: number = 0.5,
  minNeighborCount: number = 4,
): Point[] {
  const candidates = findCornerCandidates(pixels, cornerThreshold, minNeighborCount);

  if (candidates.length < 4) {
    return [];
  }

  const clustered = clusterCorners(candidates);

  if (clustered.length !== 4) {
    return [];
  }

  const ordered = orderCornersClockwise(clustered);

  return ordered;
}

/**
 * Get corner as Point (for compatibility).
 */
export function getCornerPoint(cornerIdx: number, pixels: EdgePixelArray): Point {
  return {
    x: pixels[cornerIdx * 4],
    y: pixels[cornerIdx * 4 + 1],
  };
}