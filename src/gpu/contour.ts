// Contour detection: CPU region extraction + quad fitting (labels from GPU pointer-jump).

import { buildTagGrid, decodeTagPattern } from '../lib/grid';
import { extractEdgePixelsFromBbox, findCornersFromEdges } from '../lib/corners';
import { Point } from '../lib/geometry';
import type { TagPattern } from '../lib/tag36h11';

export const COMPONENT_LABEL_INVALID = 0xFFFFFFFF;

export interface DetectedQuad {
  corners: Point[]; // 4 corner points
  label: number;
  count: number;
  aspectRatio: number;
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
    if (aspectRatio < 0.6 || aspectRatio > 1.7) {
      continue;
    }

    const perimeter = 2 * (w + h);
    const edgeDensity = region.count / perimeter;
    if (edgeDensity < 0.5 || edgeDensity > 5) {
      continue;
    }

    console.log('[validateAndFilterQuads] PASS:', {
      label: region.label,
      area,
      aspectRatio,
      edgeDensity,
    });

    const bboxCorners: [Point, Point, Point, Point] = [
      { x: region.minX, y: region.minY }, // TL
      { x: region.maxX, y: region.minY }, // TR
      { x: region.minX, y: region.maxY }, // BL
      { x: region.maxX, y: region.maxY }, // BR
    ];
    const tagGrid = buildTagGrid(bboxCorners);
    if (!tagGrid || !tagGrid.cells || tagGrid.cells.length === 0) {
      continue;
    }
    quads.push({
      corners: bboxCorners,
      label: region.label,
      count: region.count,
      aspectRatio,
      gridCells: tagGrid,
      pattern: null,
      hasCorners: false,
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
