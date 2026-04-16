// Corner detection from edge pixels using tangent direction changes
// Uses typed arrays for performance

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
  edgeMask?: Uint8Array,
): EdgePixelArray {
  const pixels: number[] = [];
  const EPS = 1e-6;

  for (let y = Math.floor(minY); y <= Math.floor(maxY); y++) {
    for (let x = Math.floor(minX); x <= Math.floor(maxX); x++) {
      const idx = y * width + x;

      if (edgeMask && edgeMask[idx] === 0) continue;

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
 * Sort pixels by angle around centroid.
 * Returns index array for sorted order.
 */
export function orderPixelsAlongContour(pixels: EdgePixelArray): Uint16Array {
  const n = pixels.length / 4;
  if (n <= 1) return new Uint16Array([0]);

  // Find centroid
  let sumX = 0, sumY = 0;
  for (let i = 0; i < n; i++) {
    sumX += pixels[i * 4];
    sumY += pixels[i * 4 + 1];
  }
  const cx = sumX / n;
  const cy = sumY / n;

  // Create index array
  const indices = new Uint16Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  // Sort by angle
  indices.sort((a, b) => {
    const angleA = Math.atan2(pixels[a * 4 + 1] - cy, pixels[a * 4] - cx);
    const angleB = Math.atan2(pixels[b * 4 + 1] - cy, pixels[b * 4] - cx);
    return angleA - angleB;
  });

  return indices;
}

/**
 * Detect corners by finding sharp turns in tangent direction.
 * Returns indices into the sorted pixel array.
 */
export function detectCorners(
  pixels: EdgePixelArray,
  sortedIndices: Uint16Array,
  angleThreshold: number = Math.PI / 3,
  magnitudeWeight: number = 0.5,
  windowSize: number = 5,
): { idx: number; angle: number }[] {
  const n = sortedIndices.length;
  if (n < windowSize * 2 + 1) return [];

  const corners: { idx: number; angle: number }[] = [];
  const maxMag = findMaxMagnitude(pixels);

  for (let i = windowSize; i < n - windowSize; i++) {
    const pixelIdx = sortedIndices[i];
    const mag = pixels[pixelIdx * 4 + 3];

    // Skip low magnitude pixels
    if (mag < maxMag * magnitudeWeight) continue;

    // Compute average tangent before and after
    let tanBefore = 0, tanAfter = 0;

    for (let j = i - windowSize; j < i; j++) {
      const pxIdx = sortedIndices[j];
      tanBefore += pixels[pxIdx * 4 + 2];
    }
    for (let j = i + 1; j <= i + windowSize; j++) {
      const pxIdx = sortedIndices[j];
      tanAfter += pixels[pxIdx * 4 + 2];
    }

    const avgBefore = tanBefore / windowSize;
    const avgAfter = tanAfter / windowSize;

    // Compute angular difference (handling wrap-around)
    let diff = avgAfter - avgBefore;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    const absDiff = Math.abs(diff);

    if (absDiff > angleThreshold) {
      corners.push({ idx: pixelIdx, angle: absDiff });
    }
  }

  return corners;
}

/**
 * Cluster corners by proximity, keep strongest per cluster.
 */
export function clusterCorners(
  corners: { idx: number; angle: number }[],
  pixels: EdgePixelArray,
  minDist: number = 20,
): number[] {
  if (corners.length === 0) return [];

  // Sort by angle (strength) descending
  const sorted = [...corners].sort((a, b) => b.angle - a.angle);
  const kept: number[] = [];

  for (const corner of sorted) {
    const px = pixels[corner.idx * 4];
    const py = pixels[corner.idx * 4 + 1];

    // Check if too close to any kept corner
    let tooClose = false;
    for (const ki of kept) {
      const kx = pixels[ki * 4];
      const ky = pixels[ki * 4 + 1];
      const dist = Math.hypot(px - kx, py - ky);
      if (dist < minDist) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      kept.push(corner.idx);
    }
  }

  return kept;
}

/**
 * Select best 4 corners forming a quad.
 * Returns indices into pixel array, ordered clockwise.
 */
export function selectBestQuadCorners(
  cornerIndices: number[],
  pixels: EdgePixelArray,
  expectedAspect: number = 1.0,
): number[] {
  if (cornerIndices.length < 4) return [];

  let bestQuad: number[] = [];
  let bestScore = -1;

  const n = cornerIndices.length;

  // Try all 4-element combinations
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        for (let l = k + 1; l < n; l++) {
          const quad = [cornerIndices[i], cornerIndices[j], cornerIndices[k], cornerIndices[l]];
          const score = quadScore(quad, pixels, expectedAspect);
          if (score > bestScore) {
            bestScore = score;
            bestQuad = quad;
          }
        }
      }
    }
  }

  if (bestQuad.length === 4) {
    return orderCornersClockwise(bestQuad, pixels);
  }
  return [];
}

/**
 * Score a quadrilateral based on regularity.
 */
function quadScore(cornerIndices: number[], pixels: EdgePixelArray, expectedAspect: number): number {
  const n = cornerIndices.length;
  if (n !== 4) return 0;

  // Get points
  const pts = cornerIndices.map(ci => ({
    x: pixels[ci * 4],
    y: pixels[ci * 4 + 1],
  }));

  // Compute side lengths
  const dists = [
    Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y),
    Math.hypot(pts[2].x - pts[1].x, pts[2].y - pts[1].y),
    Math.hypot(pts[3].x - pts[2].x, pts[3].y - pts[2].y),
    Math.hypot(pts[0].x - pts[3].x, pts[0].y - pts[3].y),
  ];

  // Score based on uniform side lengths
  const avg = dists.reduce((a, b) => a + b, 0) / 4;
  const variance = dists.reduce((s, d) => s + (d - avg) * (d - avg), 0) / 4;
  const sideScore = 1 / (1 + Math.sqrt(variance) / avg);

  // Score based on aspect ratio
  const maxD = Math.max(...dists);
  const minD = Math.min(...dists);
  const aspectScore = expectedAspect / (maxD / minD + 0.001);

  return sideScore * aspectScore;
}

/**
 * Order 4 corners clockwise from top-left.
 */
export function orderCornersClockwise(cornerIndices: number[], pixels: EdgePixelArray): number[] {
  if (cornerIndices.length !== 4) return cornerIndices;

  const pts = cornerIndices.map(ci => ({
    x: pixels[ci * 4],
    y: pixels[ci * 4 + 1],
  }));

  // Find centroid
  let cx = 0, cy = 0;
  for (const p of pts) {
    cx += p.x;
    cy += p.y;
  }
  cx /= 4;
  cy /= 4;

  // Sort by angle from centroid
  const indexed = cornerIndices.map((ci, i) => ({ ci, angle: Math.atan2(pts[i].y - cy, pts[i].x - cx) }));
  indexed.sort((a, b) => a.angle - b.angle);

  return indexed.map(x => x.ci);
}

/**
 * Get corner as Point.
 */
export function getCornerPoint(cornerIdx: number, pixels: EdgePixelArray): Point {
  return {
    x: pixels[cornerIdx * 4],
    y: pixels[cornerIdx * 4 + 1],
  };
}

/**
 * Full pipeline: find corners from edge pixel array.
 * Returns corner indices (into pixel array).
 */
export function findCornersFromEdges(
  pixels: EdgePixelArray,
  angleThreshold: number = Math.PI / 3,
  clusterDist: number = 20,
): number[] {
  const sortedIndices = orderPixelsAlongContour(pixels);
  const rawCorners = detectCorners(pixels, sortedIndices, angleThreshold);
  const clustered = clusterCorners(rawCorners, pixels, clusterDist);
  return selectBestQuadCorners(clustered, pixels);
}