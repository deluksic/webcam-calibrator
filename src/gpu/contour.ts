// Contour detection: CPU region extraction + quad fitting (labels from GPU pointer-jump).

import { buildTagGrid, decodeTagPattern } from '../lib/grid';
import { extractEdgePixelsFromBbox, findCornersFromEdges, getCornerPoint } from '../lib/corners';
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
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU-side processing: Extract quads from labeled image
// ──────────────────────────────────────���──────────────────────────────────────

export interface RegionData {
  label: number;
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
): Map<number, RegionData> {
  const regions = new Map<number, RegionData>();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const label = labelData[idx];

      if (label === COMPONENT_LABEL_INVALID) continue;

      if (!regions.has(label)) {
        regions.set(label, {
          label,
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

  return regions;
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
  regions: Map<number, RegionData>,
  sobelData: Float32Array,
  width: number,
  minArea: number = 400,
  maxArea: number = 200000,
): DetectedQuad[] {
  const quads: DetectedQuad[] = [];

  for (const [, region] of regions) {
    const w = region.maxX - region.minX;
    const h = region.maxY - region.minY;
    const area = w * h;

    if (area < minArea || area > maxArea) continue;

    const aspectRatio = w / h;
    if (aspectRatio < 0.6 || aspectRatio > 1.7) continue;

    const perimeter = 2 * (w + h);
    const edgeDensity = region.count / perimeter;
    if (edgeDensity < 0.5 || edgeDensity > 5) continue;

    // Extract edge pixels from bbox and detect corners via tangent turns
    const edgePixels = extractEdgePixelsFromBbox(
      sobelData,
      width,
      region.minX,
      region.minY,
      region.maxX,
      region.maxY,
    );

    const cornerIndices = findCornersFromEdges(edgePixels);
    if (cornerIndices.length < 4) {
      // Fallback to bounding box corners
      const corners = fitQuadToRegion(region);
      if (!corners) continue;

      const tagGrid = buildTagGrid([
        { x: corners[0][0], y: corners[0][1] },
        { x: corners[1][0], y: corners[1][1] },
        { x: corners[2][0], y: corners[2][1] },
        { x: corners[3][0], y: corners[3][1] },
      ] as [Point, Point, Point, Point]);

      quads.push({
        corners: corners.map(c => ({ x: c[0], y: c[1] })),
        label: region.label,
        count: region.count,
        aspectRatio,
        gridCells: tagGrid,
        pattern: null,
      });
      continue;
    }

    // Convert corner indices to points
    const corners = cornerIndices.map(ci => getCornerPoint(ci, edgePixels));

    // Build perspective-correct grid and decode pattern
    const grid = buildTagGrid(corners as [Point, Point, Point, Point]);
    const pattern = decodeTagPattern(grid, sobelData, width);

    quads.push({
      corners,
      label: region.label,
      count: region.count,
      aspectRatio,
      gridCells: grid,
      pattern,
    });
  }

  return quads;
}
