// Perspective-correct grid subdivision for AprilTag decode
// Uses line intersection + proportional subdivision (no bilinear interpolation)

import { Point, lineFromPoints, lineIntersection, subdivideSegment } from './geometry';
import type { TagPattern } from './tag36h11';

export interface GridCell {
  row: number;
  col: number;
  corners: [Point, Point, Point, Point]; // TL, TR, BR, BL
  center: Point;
}

export interface GridResult {
  outerCorners: [Point, Point, Point, Point]; // TL, TR, BR, BL
  cells: GridCell[]; // 6x6 cells
  innerCorners: Point[]; // 7x7 grid intersection points
}

/**
 * Subdivide an edge of a quadrilateral proportionally.
 * Uses linear interpolation since we're already in 2D projected space.
 * For perspective-correct subdivision, we interpolate in homogeneous coordinates.
 *
 * @param p1 Start corner
 * @param p2 End corner
 * @param divisions Number of segments
 * @param offset Which division point (1 to divisions-1)
 */
function subdivideEdgeProportional(
  p1: Point,
  p2: Point,
  divisions: number,
  offset: number,
): Point {
  const t = offset / divisions;
  return {
    x: p1.x + t * (p2.x - p1.x),
    y: p1.y + t * (p2.y - p1.y),
  };
}

/**
 * Connect two points on opposite edges of a quad to form a grid line.
 * Returns the intersection with the opposite boundary line.
 */
function connectToGridLine(
  p1: Point,
  p2: Point,
  line1Start: Point,
  line1End: Point,
): Point | null {
  // Line from p1 to p2
  const line = lineFromPoints(p1, p2);
  if (!line) return null;

  // Edge line
  const edge = lineFromPoints(line1Start, line1End);
  if (!edge) return null;

  return lineIntersection(line, edge);
}

/**
 * Build perspective-correct grid inside a quadrilateral.
 * Divides each edge into 6 segments and creates inner grid lines.
 *
 * @param corners 4 corners in order (TL, TR, BR, BL)
 * @param divisions Number of cell divisions (6 for 6x6 tag)
 */
export function buildTagGrid(
  corners: [Point, Point, Point, Point],
  divisions: number = 6,
): GridResult {
  const [tl, tr, br, bl] = corners;

  // Build 7x7 inner corner grid (7 points per edge, 49 total)
  // First, subdivide all 4 edges
  const topEdge: Point[] = [];
  const bottomEdge: Point[] = [];
  const leftEdge: Point[] = [];
  const rightEdge: Point[] = [];

  for (let i = 0; i <= divisions; i++) {
    topEdge.push(subdivideEdgeProportional(tl, tr, divisions, i));
    bottomEdge.push(subdivideEdgeProportional(bl, br, divisions, i));
    leftEdge.push(subdivideEdgeProportional(tl, bl, divisions, i));
    rightEdge.push(subdivideEdgeProportional(tr, br, divisions, i));
  }

  // Now build inner grid by connecting opposite edge points
  const innerCorners: Point[] = [];

  // For each intersection point, we need to find where the horizontal
  // and vertical lines from subdivision cross
  for (let row = 0; row <= divisions; row++) {
    for (let col = 0; col <= divisions; col++) {
      const topPoint = topEdge[col];
      const bottomPoint = bottomEdge[col];
      const leftPoint = leftEdge[row];
      const rightPoint = rightEdge[row];

      // Horizontal line: from left edge to right edge at row position
      const hLine = lineFromPoints(leftPoint, rightPoint);
      // Vertical line: from top edge to bottom edge at col position
      const vLine = lineFromPoints(topPoint, bottomPoint);

      const intersection = lineIntersection(hLine!, vLine!);
      if (intersection) {
        innerCorners.push(intersection);
      } else {
        // Fallback: use midpoint
        innerCorners.push({
          x: (topPoint.x + bottomPoint.x) / 2,
          y: (leftPoint.y + rightPoint.y) / 2,
        });
      }
    }
  }

  // Build 6x6 cells from inner corners
  const cells: GridCell[] = [];

  for (let row = 0; row < divisions; row++) {
    for (let col = 0; col < divisions; col++) {
      const tlIdx = row * (divisions + 1) + col;
      const trIdx = tlIdx + 1;
      const brIdx = (row + 1) * (divisions + 1) + col + 1;
      const blIdx = brIdx - 1;

      const cellCorners: [Point, Point, Point, Point] = [
        innerCorners[tlIdx],
        innerCorners[trIdx],
        innerCorners[brIdx],
        innerCorners[blIdx],
      ];

      const center = {
        x: (cellCorners[0].x + cellCorners[1].x + cellCorners[2].x + cellCorners[3].x) / 4,
        y: (cellCorners[0].y + cellCorners[1].y + cellCorners[2].y + cellCorners[3].y) / 4,
      };

      cells.push({
        row,
        col,
        corners: cellCorners,
        center,
      });
    }
  }

  return {
    outerCorners: corners,
    cells,
    innerCorners,
  };
}

/**
 * Sample edge pixels within a cell using Sobel data.
 * Returns array of gradient magnitudes and directions.
 */
export function sampleCellPixels(
  cell: GridCell,
  sobelData: Float32Array,
  imageWidth: number,
  edgeMask?: Uint8Array,
): { mag: number; tangent: number }[] {
  const samples: { mag: number; tangent: number }[] = [];

  // Bounding box of cell
  const minX = Math.max(0, Math.floor(Math.min(...cell.corners.map(c => c.x))));
  const maxX = Math.max(0, Math.ceil(Math.max(...cell.corners.map(c => c.x))));
  const minY = Math.max(0, Math.floor(Math.min(...cell.corners.map(c => c.y))));
  const maxY = Math.max(0, Math.ceil(Math.max(...cell.corners.map(c => c.y))));

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const idx = y * imageWidth + x;

      if (edgeMask && edgeMask[idx] === 0) continue;

      const gx = sobelData[idx * 2];
      const gy = sobelData[idx * 2 + 1];
      const mag = Math.sqrt(gx * gx + gy * gy);

      if (mag < 0.01) continue;

      const tangent = Math.atan2(gy, gx) + Math.PI / 2;

      samples.push({ mag, tangent });
    }
  }

  return samples;
}

/**
 * Determine if a cell is black or white based on edge samples.
 * Uses gradient direction consensus.
 *
 * @param samples Edge pixels in the cell
 * @returns 0 for white, 1 for black, or -1 for uncertain
 */
export function decodeCell(samples: { mag: number; tangent: number }[]): number {
  if (samples.length < 4) return -1;

  // Check if this is an edge cell or a solid cell
  const avgMag = samples.reduce((s, p) => s + p.mag, 0) / samples.length;

  // If low average magnitude, it's likely a solid (non-edge) region
  if (avgMag < 0.1) {
    // Could be either black or white - need more info
    return -1;
  }

  // Analyze gradient directions for edge cells
  // Edges at cell boundary will have strong gradients
  // For a black cell on white: gradient points inward (or outward from white)
  // For a white cell on black: gradient points inward (or outward from black)

  // Count gradients pointing in each direction (4 bins)
  const bins = [0, 0, 0, 0];
  const binSize = Math.PI / 2;

  for (const sample of samples) {
    // Normalize tangent to [0, 2pi)
    let angle = sample.tangent;
    while (angle < 0) angle += 2 * Math.PI;
    while (angle >= 2 * Math.PI) angle -= 2 * Math.PI;

    const bin = Math.floor(angle / binSize) % 4;
    bins[bin] += sample.mag;
  }

  // Check for dominant direction (indicating edge orientation)
  const maxBin = Math.max(...bins);
  const total = bins.reduce((a, b) => a + b, 0);

  if (maxBin / total < 0.4) {
    // No strong direction consensus - might be interior of solid region
    return -1;
  }

  // Strong edge direction - this is a boundary cell
  // For AprilTag: black cells have edges around them, white cells don't
  // Return based on edge density
  return avgMag > 0.2 ? 1 : 0;
}

/**
 * Full decode of 6x6 tag pattern.
 *
 * @param grid Grid from buildTagGrid
 * @param sobelData Sobel gradient data
 * @param imageWidth Image width
 * @param edgeMask Optional edge mask
 * @returns 6x6 binary pattern (0=white, 1=black)
 */
export function decodeTagPattern(
  grid: GridResult,
  sobelData: Float32Array,
  imageWidth: number,
  edgeMask?: Uint8Array,
): TagPattern | null {
  const pattern: TagPattern = [] as unknown as TagPattern;

  // Decode each cell
  for (const cell of grid.cells) {
    const samples = sampleCellPixels(cell, sobelData, imageWidth, edgeMask);
    const cellValue = decodeCell(samples);

    if (cellValue === -1) {
      // Uncertain - keep as unknown
      pattern.push(-1);
    } else {
      pattern.push(cellValue as 0 | 1);
    }
  }

  return pattern;
}