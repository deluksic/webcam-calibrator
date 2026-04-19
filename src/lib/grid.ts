// Perspective-correct grid subdivision for AprilTag decode
// Uses line intersection + proportional subdivision (no bilinear interpolation)

import { imagePixelToUnitSquareUv } from './aprilTagRaycast';
import { Point, computeHomography, lineFromPoints, lineIntersection, subdivideSegment } from './geometry';
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

      // Guard: either line can be null if endpoints are coincident (e.g. at quad corners)
      if (!hLine || !vLine) {
        innerCorners.push({
          x: (topPoint.x + bottomPoint.x) / 2,
          y: (leftPoint.y + rightPoint.y) / 2,
        });
        continue;
      }

      const intersection = lineIntersection(hLine, vLine);
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

/** Map cell UV in [0,1]² (TL origin) to image; `cell.corners` are TL, TR, BR, BL. */
export function cellUvToImage(cell: GridCell, u: number, v: number): Point {
  const [tl, tr, br, bl] = cell.corners;
  return {
    x: (1 - u) * (1 - v) * tl.x + u * (1 - v) * tr.x + u * v * br.x + (1 - u) * v * bl.x,
    y: (1 - u) * (1 - v) * tl.y + u * (1 - v) * tr.y + u * v * br.y + (1 - u) * v * bl.y,
  };
}

/**
 * Pull image-space gradient (gx, gy) = (∂I/∂x, ∂I/∂y) back to cell UV via the bilinear quad map.
 * (∂I/∂u, ∂I/∂v) = Jᵀ (gx, gy) with J = [[∂x/∂u, ∂x/∂v], [∂y/∂u, ∂y/∂v]].
 */
export function imageGradToUvGrad(
  cell: GridCell,
  u: number,
  v: number,
  gx: number,
  gy: number,
): { gu: number; gv: number } {
  const [tl, tr, br, bl] = cell.corners;
  const dxdU = -(1 - v) * tl.x + (1 - v) * tr.x + v * br.x - v * bl.x;
  const dxdV = -(1 - u) * tl.x - u * tr.x + u * br.x + (1 - u) * bl.x;
  const dydU = -(1 - v) * tl.y + (1 - v) * tr.y + v * br.y - v * bl.y;
  const dydV = -(1 - u) * tl.y - u * tr.y + u * br.y + (1 - u) * bl.y;
  const gu = gx * dxdU + gy * dydU;
  const gv = gx * dxdV + gy * dydV;
  return { gu, gv };
}

export type CellSobelSample = {
  mag: number;
  tangent: number;
  gx: number;
  gy: number;
  u: number;
  v: number;
};

/** Min |UV − center|² so we ignore the flat interior where radial direction is ill-defined. */
const DECODE_RADIAL_MIN2 = 0.015 * 0.015;
/** Dot-product deadband in UV-gradient space (noise). */
const DECODE_DOT_EPS = 1e-8;
/**
 * Minimum count of directional votes (pos + neg) before calling black/white vs **no signal** (`-1`).
 * Ties with at least this many votes return **`-2`** (ambiguous, not “missing edges”).
 */
const DECODE_MIN_VOTE_TOTAL = 3;

/**
 * Classify a 6×6 cell: **0 = white**, **1 = black**, **-1 = no/weak signal**, **-2 = tie** (enough votes).
 * Edge-only: pull `(gx, gy)` to UV, dot with outward radial from cell center `(u−½, v−½)`;
 * **unweighted** ±1 vote per sample (GPU/NMS is assumed to have already filtered edges).
 */
export function decodeCell(cell: GridCell, samples: CellSobelSample[]): 0 | 1 | -1 | -2 {
  const n = samples.length;
  if (n < 9) return -1;

  let pos = 0;
  let neg = 0;
  for (const s of samples) {
    if (s.mag <= 1e-12) continue;
    const ru = s.u - 0.5;
    const rv = s.v - 0.5;
    const r2 = ru * ru + rv * rv;
    if (r2 < DECODE_RADIAL_MIN2) continue;

    const { gu, gv } = imageGradToUvGrad(cell, s.u, s.v, s.gx, s.gy);
    const dot = gu * ru + gv * rv;
    if (dot > DECODE_DOT_EPS) pos += 1;
    else if (dot < -DECODE_DOT_EPS) neg += 1;
  }

  const sum = pos + neg;
  if (sum < DECODE_MIN_VOTE_TOTAL) return -1;
  if (pos > neg) return 1;
  if (neg > pos) return 0;
  return -2;
}

/** AprilTag unit square is an 8×8 module grid (black border ring + inner 6×6 data). */
const TAG_MODULES = 8;
/** 10% of one module side in tag UV — pixels within this distance of a module cell vote for it. */
const TAU_MODULE_UV = 0.1 / TAG_MODULES;
/** Skip votes when radial offset from module center in tag UV is below this (ill-conditioned). */
const TAG_MODULE_RADIAL_MIN2 = (0.01 / TAG_MODULES) ** 2;

/**
 * Jacobian ∂(x,y)/∂(u,v) for `applyHomography` (same 8-parameter `h` as `computeHomography`).
 */
function jacobianImageWrtTagUv(h: Float32Array, u: number, v: number) {
  const h0 = h[0],
    h1 = h[1],
    h2 = h[2];
  const h3 = h[3],
    h4 = h[4],
    h5 = h[5];
  const h6 = h[6],
    h7 = h[7];
  const xh = h0 * u + h1 * v + h2;
  const yh = h3 * u + h4 * v + h5;
  const wh = h6 * u + h7 * v + 1;
  const wh2 = wh * wh;
  const xu = (h0 * wh - xh * h6) / wh2;
  const xv = (h1 * wh - xh * h7) / wh2;
  const yu = (h3 * wh - yh * h6) / wh2;
  const yv = (h4 * wh - yh * h7) / wh2;
  return { xu, xv, yu, yv };
}

/** Tag-space gradient (∂I/∂u, ∂I/∂v) from image Sobel and homography: [gu,gv] = Jᵀ [gx,gy]. */
export function imageSobelToTagGradient(
  h: Float32Array,
  u: number,
  v: number,
  gx: number,
  gy: number,
): { gu: number; gv: number } {
  const { xu, xv, yu, yv } = jacobianImageWrtTagUv(h, u, v);
  return {
    gu: gx * xu + gy * yu,
    gv: gx * xv + gy * yv,
  };
}

/** Euclidean distance from (u,v) to the closed axis-aligned rectangle [u0,u1]×[v0,v1]. */
export function distPointToClosedRectUv(
  u: number,
  v: number,
  u0: number,
  u1: number,
  v0: number,
  v1: number,
): number {
  const cu = Math.min(u1, Math.max(u0, u));
  const cv = Math.min(v1, Math.max(v0, v));
  return Math.hypot(u - cu, v - cv);
}

function classifyModuleFromPosNeg(pos: number, neg: number): 0 | 1 | -1 | -2 {
  const sum = pos + neg;
  if (sum < DECODE_MIN_VOTE_TOTAL) return -1;
  if (pos > neg) return 1;
  if (neg > pos) return 0;
  return -2;
}

/**
 * One pass: only **`-1`** (weak votes). If all known cardinal neighbors agree on `0`/`1`, adopt that
 * color. **`-2`** (tie with enough votes) is skipped — neighbor homogeneity is a poor stand-in for
 * conflicting directional evidence, and dictionary decode already treats **`-2`** as an unknown bit.
 */
export function fillUnknownNeighbors6(pattern: TagPattern): void {
  const idx = (r: number, c: number) => r * 6 + c;
  const next = [...pattern] as TagPattern;
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      if (pattern[idx(r, c)] !== -1) continue;
      const vals: number[] = [];
      if (r > 0) {
        const v = pattern[idx(r - 1, c)];
        if (v === 0 || v === 1) vals.push(v);
      }
      if (r < 5) {
        const v = pattern[idx(r + 1, c)];
        if (v === 0 || v === 1) vals.push(v);
      }
      if (c > 0) {
        const v = pattern[idx(r, c - 1)];
        if (v === 0 || v === 1) vals.push(v);
      }
      if (c < 5) {
        const v = pattern[idx(r, c + 1)];
        if (v === 0 || v === 1) vals.push(v);
      }
      if (vals.length === 0) continue;
      const first = vals[0]!;
      if (vals.every((x) => x === first)) next[idx(r, c)] = first as 0 | 1;
    }
  }
  for (let i = 0; i < 36; i++) pattern[i] = next[i]!;
}

/** Same magnitude floor as `extractLabeledEdgePixels` in `corners.ts`. */
const DECODE_EDGE_MASK_EPS = 1e-6;

/**
 * Pixels where **decodeTagPattern** may sample Sobel: same connected-component `regionLabel`,
 * non‑zero NMS gradient magnitude, within the region bbox expanded by `padPx` (quad samples can
 * sit slightly outside the tight bbox).
 */
export function buildDecodeEdgeMask(
  labelData: Uint32Array,
  sobelData: Float32Array,
  imageWidth: number,
  imageHeight: number,
  regionLabel: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  padPx: number = 32,
): Uint8Array {
  const w = imageWidth;
  const h = imageHeight;
  const mask = new Uint8Array(w * h);
  const x0 = Math.max(0, Math.floor(minX) - padPx);
  const y0 = Math.max(0, Math.floor(minY) - padPx);
  const x1 = Math.min(w - 1, Math.ceil(maxX) + padPx);
  const y1 = Math.min(h - 1, Math.ceil(maxY) + padPx);
  const eps2 = DECODE_EDGE_MASK_EPS * DECODE_EDGE_MASK_EPS;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = y * w + x;
      if (labelData[idx] !== regionLabel) continue;
      const gx = sobelData[idx * 2];
      const gy = sobelData[idx * 2 + 1];
      if (gx * gx + gy * gy >= eps2) mask[idx] = 1;
    }
  }
  return mask;
}

/**
 * Full decode of 6×6 tag pattern from Sobel: iterate **image pixels** in the quad bbox,
 * inverse homography → tag UV, push gradient to tag UV, **8×8** modules (1/8 UV) with **τ = 0.1/8**
 * proximity voting (edges/corners vote multiple modules). **Unweighted** ±1 votes per pixel
 * (no `mag²`; no quantile magnitude gate — use `edgeMask` / upstream NMS so only real edges run).
 * Then inner **6×6** + neighbor fill for **`-1`** only (**`-2`** unchanged). Outcomes: `0`/`1`/`-1`/`-2`.
 *
 * @param grid Grid from `buildTagGrid` (uses `outerCorners` TL,TR,BR,BL; homography uses TL,TR,BL,BR strip order).
 * @param edgeMask Optional: skip pixels where mask index is 0.
 */
export function decodeTagPattern(
  grid: GridResult,
  sobelData: Float32Array,
  imageWidth: number,
  edgeMask?: Uint8Array,
  imageHeight?: number,
): TagPattern | null {
  const imageH =
    imageHeight !== undefined
      ? imageHeight
      : Math.floor(sobelData.length / (2 * imageWidth));
  const oc = grid.outerCorners;
  const strip: [Point, Point, Point, Point] = [oc[0], oc[1], oc[3], oc[2]];
  const h = computeHomography([...strip]);

  let x0 = Math.min(oc[0].x, oc[1].x, oc[2].x, oc[3].x);
  let y0 = Math.min(oc[0].y, oc[1].y, oc[2].y, oc[3].y);
  let x1 = Math.max(oc[0].x, oc[1].x, oc[2].x, oc[3].x);
  let y1 = Math.max(oc[0].y, oc[1].y, oc[2].y, oc[3].y);
  const ix0 = Math.max(0, Math.floor(x0));
  const iy0 = Math.max(0, Math.floor(y0));
  const ix1 = Math.min(imageWidth - 1, Math.ceil(x1));
  const iy1 = Math.min(imageH - 1, Math.ceil(y1));

  const posM = new Float64Array(64);
  const negM = new Float64Array(64);

  for (let iy = iy0; iy <= iy1; iy++) {
    for (let ix = ix0; ix <= ix1; ix++) {
      if (edgeMask && edgeMask[iy * imageWidth + ix] === 0) continue;
      const { u, v, inside } = imagePixelToUnitSquareUv(h, ix + 0.5, iy + 0.5);
      if (!inside) continue;
      const gx = sobelData[(iy * imageWidth + ix) * 2];
      const gy = sobelData[(iy * imageWidth + ix) * 2 + 1];
      const mag = Math.hypot(gx, gy);
      if (mag <= 1e-12) continue;

      const { gu, gv } = imageSobelToTagGradient(h, u, v, gx, gy);

      // Each edge pixel votes into every 8×8 module whose closed cell lies within `TAU_MODULE_UV`
      // (distance-to-rect). **Unweighted** ±1 per pixel; sign from radial dot gu*ru + gv*rv in tag UV.
      for (let my = 0; my < TAG_MODULES; my++) {
        for (let mx = 0; mx < TAG_MODULES; mx++) {
          const u0 = mx / TAG_MODULES;
          const u1 = (mx + 1) / TAG_MODULES;
          const v0 = my / TAG_MODULES;
          const v1 = (my + 1) / TAG_MODULES;
          const d = distPointToClosedRectUv(u, v, u0, u1, v0, v1);
          if (d > TAU_MODULE_UV) continue;

          const cu = (mx + 0.5) / TAG_MODULES;
          const cv = (my + 0.5) / TAG_MODULES;
          const ru = u - cu;
          const rv = v - cv;
          if (ru * ru + rv * rv < TAG_MODULE_RADIAL_MIN2) continue;

          const dot = gu * ru + gv * rv;
          const mi = my * TAG_MODULES + mx;
          if (dot > DECODE_DOT_EPS) posM[mi] += 1;
          else if (dot < -DECODE_DOT_EPS) negM[mi] += 1;
        }
      }
    }
  }

  const pattern: TagPattern = [] as unknown as TagPattern;
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const mx = col + 1;
      const my = row + 1;
      const mi = my * TAG_MODULES + mx;
      const cell = classifyModuleFromPosNeg(posM[mi]!, negM[mi]!);
      pattern.push(cell);
    }
  }

  fillUnknownNeighbors6(pattern);
  return pattern;
}