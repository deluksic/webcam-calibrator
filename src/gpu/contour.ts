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

  return [...regions.values()];
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

  let okAreaAR = 0, okEdgeDensity = 0, okCorners = 0, okBbox = 0, okFailed = 0;

  for (const region of regions) {
    const w = region.maxX - region.minX;
    const h = region.maxY - region.minY;
    const area = w * h;

    if (area < minArea || area > maxArea) { okFailed++; continue; }

    const aspectRatio = w / h;
    if (aspectRatio < 0.6 || aspectRatio > 1.7) { okFailed++; continue; }
    okAreaAR++;

    const perimeter = 2 * (w + h);
    const edgeDensity = region.count / perimeter;
    if (edgeDensity < 0.5 || edgeDensity > 5) { okFailed++; continue; }
    okEdgeDensity++;

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
      if (!corners) { okFailed++; continue; }

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
        hasCorners: false, // fallback - bbox only
      });
      okBbox++;
      continue;
    }
    okCorners++;

    // Convert corner indices to points
    const corners = cornerIndices.map(ci => getCornerPoint(ci, edgePixels));

    // Check if corners are suspiciously far from the region's extent bounds.
    // If corner detection picked noise points, the quad will be far from the bbox.
    // Fall back to bbox corners in that case.
    const bboxMinX = region.minX, bboxMinY = region.minY;
    const bboxMaxX = region.maxX, bboxMaxY = region.maxY;
    const cx = corners.reduce((s, c) => s + c.x, 0) / 4;
    const cy = corners.reduce((s, c) => s + c.y, 0) / 4;
    const bboxCx = (bboxMinX + bboxMaxX) / 2;
    const bboxCy = (bboxMinY + bboxMaxY) / 2;
    const dist = Math.hypot(cx - bboxCx, cy - bboxCy);
    const bboxHalfW = (bboxMaxX - bboxMinX) / 2;
    const bboxHalfH = (bboxMaxY - bboxMinY) / 2;
    const bboxRadius = Math.hypot(bboxHalfW, bboxHalfH);
    if (dist > bboxRadius * 0.5) {
      console.log(`[validateQuads] corners centroid (${cx.toFixed(0)},${cy.toFixed(0)}) too far from bbox centroid (${bboxCx.toFixed(0)},${bboxCy.toFixed(0)}) — using bbox fallback`);
      const bboxCorners: [Point, Point, Point, Point] = [
        { x: bboxMinX, y: bboxMinY },
        { x: bboxMaxX, y: bboxMinY },
        { x: bboxMaxX, y: bboxMaxY },
        { x: bboxMinX, y: bboxMaxY },
      ];
      const tagGrid = buildTagGrid(bboxCorners);
      quads.push({
        corners: bboxCorners,
        label: region.label,
        count: region.count,
        aspectRatio,
        gridCells: tagGrid,
        pattern: null,
        hasCorners: false,
      });
      okBbox++;
      continue;
    }

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
      hasCorners: true, // detected via corner finding
    });
  }

  console.log(`[validateQuads] regions=${regions.length} okAreaAR=${okAreaAR} okEdgeDensity=${okEdgeDensity} okCorners=${okCorners} okBbox=${okBbox} failed=${okFailed}`);

  return quads;
}
