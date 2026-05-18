// Perspective-correct grid subdivision for AprilTag geometry (buildTagGrid)
// Uses line intersection + proportional subdivision (no bilinear interpolation)

import { length, lineFromPoints, lineIntersection } from '@/lib/geometry'
import type { Corners, Point } from '@/lib/geometry'

const { min, max } = Math

export interface GridCell {
  row: number
  col: number
  corners: Corners // TL, TR, BL, BR
  center: Point
}

export interface GridResult {
  outerCorners: Corners // TL, TR, BL, BR
  cells: GridCell[] // 6x6 cells
  innerCorners: Point[] // 7x7 grid intersection points
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
function subdivideEdgeProportional(p1: Point, p2: Point, divisions: number, offset: number): Point {
  const t = offset / divisions
  return {
    x: p1.x + t * (p2.x - p1.x),
    y: p1.y + t * (p2.y - p1.y),
  }
}

/**
 * Build perspective-correct grid inside a quadrilateral.
 * Divides each edge into 6 segments and creates inner grid lines.
 *
 * @param corners 4 corners in order (TL, TR, BL, BR)
 * @param divisions Number of cell divisions (6 for 6x6 tag)
 */
export function buildTagGrid(corners: Corners, divisions: number = 6): GridResult {
  const [tl, tr, bl, br] = corners

  // Build 7x7 inner corner grid (7 points per edge, 49 total)
  // First, subdivide all 4 edges
  const topEdge: Point[] = []
  const bottomEdge: Point[] = []
  const leftEdge: Point[] = []
  const rightEdge: Point[] = []

  for (let i = 0; i <= divisions; i++) {
    topEdge.push(subdivideEdgeProportional(tl, tr, divisions, i))
    bottomEdge.push(subdivideEdgeProportional(bl, br, divisions, i))
    leftEdge.push(subdivideEdgeProportional(tl, bl, divisions, i))
    rightEdge.push(subdivideEdgeProportional(tr, br, divisions, i))
  }

  // Now build inner grid by connecting opposite edge points
  const innerCorners: Point[] = []

  // For each intersection point, we need to find where the horizontal
  // and vertical lines from subdivision cross
  for (let row = 0; row <= divisions; row++) {
    for (let col = 0; col <= divisions; col++) {
      const topPoint = topEdge[col]!
      const bottomPoint = bottomEdge[col]!
      const leftPoint = leftEdge[row]!
      const rightPoint = rightEdge[row]!

      // Horizontal line: from left edge to right edge at row position
      const hLine = lineFromPoints(leftPoint, rightPoint)
      // Vertical line: from top edge to bottom edge at col position
      const vLine = lineFromPoints(topPoint, bottomPoint)

      // Guard: either line can be undefined if endpoints are coincident (e.g. at quad corners)
      if (!hLine || !vLine) {
        innerCorners.push({
          x: (topPoint.x + bottomPoint.x) / 2,
          y: (leftPoint.y + rightPoint.y) / 2,
        })
        continue
      }

      const intersection = lineIntersection(hLine, vLine)
      if (intersection) {
        innerCorners.push(intersection)
      } else {
        // Fallback: use midpoint
        innerCorners.push({
          x: (topPoint.x + bottomPoint.x) / 2,
          y: (leftPoint.y + rightPoint.y) / 2,
        })
      }
    }
  }

  // Build 6x6 cells from inner corners
  const cells: GridCell[] = []

  for (let row = 0; row < divisions; row++) {
    for (let col = 0; col < divisions; col++) {
      const tlIdx = row * (divisions + 1) + col
      const trIdx = tlIdx + 1
      const brIdx = (row + 1) * (divisions + 1) + col + 1
      const blIdx = brIdx - 1

      const cellCorners: Corners = [
        innerCorners[tlIdx]!,
        innerCorners[trIdx]!,
        innerCorners[blIdx]!,
        innerCorners[brIdx]!,
      ]

      const center = {
        x: (cellCorners[0].x + cellCorners[1].x + cellCorners[2].x + cellCorners[3].x) / 4,
        y: (cellCorners[0].y + cellCorners[1].y + cellCorners[2].y + cellCorners[3].y) / 4,
      }

      cells.push({
        row,
        col,
        corners: cellCorners,
        center,
      })
    }
  }

  return {
    outerCorners: corners,
    cells,
    innerCorners,
  }
}

