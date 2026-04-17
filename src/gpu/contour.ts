// Contour detection: CPU region extraction + quad fitting (labels from GPU pointer-jump).

import { buildTagGrid, decodeTagPattern } from '../lib/grid';
import { findCornersFromEdges } from '../lib/corners';
import { Point } from '../lib/geometry';
import type { TagPattern } from '../lib/tag36h11';

export const COMPONENT_LABEL_INVALID = 0xFFFFFFFF;

export interface DetectedQuad {
  corners: Point[]; // 4 corner points
  label: number;
  count: number;
  aspectRatio: number;
  area: number;
  gridCells: ReturnType<typeof buildTagGrid> | null;
  pattern: TagPattern | null;
  hasCorners: boolean; // true if detected via corner finding, false if fallback bbox
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side processing: Extract quads from labeled image
// ──────────────────────────────────────���──────────────────────────────────────

export interface RegionData {
  label: number;
  rootLabel: number;
  count: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  pixels: [number, number][];
}

export function extractRegions(
  labelData: Uint32Array,
  width: number,
  height: number,
  _edgeData: Float32Array,
): RegionData[] {
  const regions = new Map<number, RegionData>();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const label = labelData[idx];

      if (label === COMPONENT_LABEL_INVALID) continue;

      if (!regions.has(label)) {
        regions.set(label, {
          label,
          rootLabel: label, // raw pixel-index label IS the root
          count: 0,
          minX: x,
          minY: y,
          maxX: x,
          maxY: y,
          pixels: [],
        });
      }

      const region = regions.get(label)!;
      region.count++;
      region.minX = Math.min(region.minX, x);
      region.minY = Math.min(region.minY, y);
      region.maxX = Math.max(region.maxX, x);
      region.maxY = Math.max(region.maxY, y);

      if (region.pixels.length < 500) {
        region.pixels.push([x, y]);
      }
    }
  }

  const result = [...regions.values()];
  return result;
}

export function fitQuadToRegion(region: RegionData): [number, number][] | null {
  const { minX, minY, maxX, maxY } = region;

  const boundingBoxWidth = maxX - minX;
  const boundingBoxHeight = maxY - minY;

  const aspectRatio = boundingBoxWidth / boundingBoxHeight;
  if (aspectRatio < 0.5 || aspectRatio > 2.0) {
    return null;
  }

  let leftEdge = maxX;
  let rightEdge = minX;
  let topEdge = maxY;
  let bottomEdge = minY;

  for (const [px, py] of region.pixels) {
    if (px - minX < leftEdge - minX) leftEdge = px;
    if (maxX - px < maxX - rightEdge) rightEdge = px;
    if (py - minY < topEdge - minY) topEdge = py;
    if (maxY - py < maxY - bottomEdge) bottomEdge = py;
  }

  return [
    [leftEdge, topEdge],
    [rightEdge, topEdge],
    [rightEdge, bottomEdge],
    [leftEdge, bottomEdge],
  ];
}

export function validateAndFilterQuads(
  regions: RegionData[],
  sobelData: Float32Array,
  labelData: Uint32Array,
  width: number,
  minArea: number = 400,
  maxArea: number = 200000,
): DetectedQuad[] {
  const quads: DetectedQuad[] = [];

  for (const region of regions) {
    const w = region.maxX - region.minX;
    const h = region.maxY - region.minY;
    const area = w * h;

    if (area < minArea) {
      continue;
    }
    if (area > maxArea) {
      continue;
    }

    const aspectRatio = w / h;
    if (aspectRatio < 0.3 || aspectRatio > 3.5) {
      continue;
    }

    const perimeter = 2 * (w + h);
    // Oblique/far tags have fewer edge pixels — relax density threshold
    const edgeDensity = region.count / perimeter;
    if (edgeDensity < 0.2 || edgeDensity > 10) {
      continue;
    }


    // Try to detect real corners via line intersection
    const detectedCorners = findCornersFromEdges(
      sobelData,
      labelData,
      width,
      region.label,
      region.minX,
      region.minY,
      region.maxX,
      region.maxY,
    );
    const corners: [Point, Point, Point, Point] =
      detectedCorners.length === 4
        ? [detectedCorners[0], detectedCorners[1], detectedCorners[2], detectedCorners[3]]
        : [
            { x: region.minX, y: region.minY },
            { x: region.maxX, y: region.minY },
            { x: region.minX, y: region.maxY },
            { x: region.maxX, y: region.maxY },
          ];
    const tagGrid = buildTagGrid(corners);
    if (!tagGrid || !tagGrid.cells || tagGrid.cells.length === 0) {
      continue;
    }
    quads.push({
      corners,
      label: region.label,
      count: region.count,
      aspectRatio,
      area,
      gridCells: tagGrid,
      pattern: null,
      hasCorners: detectedCorners.length === 4,
    });
  }

  return quads;
}

export function filterNestedQuads(quads: DetectedQuad[]): DetectedQuad[] {
  return quads.filter((candidate) => {
    for (const other of quads) {
      if (other === candidate) continue;
      // Discard candidate if it is fully contained inside another box
      if (
        other.corners[0].x <= candidate.corners[0].x &&
        other.corners[2].x >= candidate.corners[2].x &&
        other.corners[0].y <= candidate.corners[0].y &&
        other.corners[2].y >= candidate.corners[2].y
      ) {
        return false;
      }
    }
    return true;
  });
}
