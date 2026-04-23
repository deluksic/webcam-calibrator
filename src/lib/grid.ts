// Perspective-correct grid subdivision for AprilTag decode
// Uses line intersection + proportional subdivision (no bilinear interpolation)

import { imagePixelToUnitSquareUv } from '@/lib/aprilTagRaycast'
import { length, lineFromPoints, lineIntersection, tryComputeHomography } from '@/lib/geometry'
import type { Corners, Mat3, Point } from '@/lib/geometry'
import type { TagPattern } from '@/lib/tag36h11'

const { min, max, abs, floor, ceil, round } = Math

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

      // Guard: either line can be null if endpoints are coincident (e.g. at quad corners)
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

/** Map cell UV in [0,1]² (TL origin) to image; `cell.corners` are TL, TR, BL, BR. */
export function cellUvToImage(cell: GridCell, u: number, v: number): Point {
  const [tl, tr, bl, br] = cell.corners
  return {
    x: (1 - u) * (1 - v) * tl.x + u * (1 - v) * tr.x + (1 - u) * v * bl.x + u * v * br.x,
    y: (1 - u) * (1 - v) * tl.y + u * (1 - v) * tr.y + (1 - u) * v * bl.y + u * v * br.y,
  }
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
  const [tl, tr, bl, br] = cell.corners
  const dxdU = -(1 - v) * tl.x + (1 - v) * tr.x - v * bl.x + v * br.x
  const dxdV = -(1 - u) * tl.x - u * tr.x + (1 - u) * bl.x + u * br.x
  const dydU = -(1 - v) * tl.y + (1 - v) * tr.y - v * bl.y + v * br.y
  const dydV = -(1 - u) * tl.y - u * tr.y + (1 - u) * bl.y + u * br.y
  const gu = gx * dxdU + gy * dydU
  const gv = gx * dxdV + gy * dydV
  return { gu, gv }
}

/** Dot-product deadband in UV-gradient space (noise). */
const DECODE_DOT_EPS = 1e-8
/**
 * Minimum directional vote total (`white+black`) is `max(2, round(this × shortest quad edge px))` so
 * the bar tracks on-screen tag size under homography (distance / angle). Floor **2** matches the old
 * fixed threshold for ~100px edges at 2%.
 */
export const DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE = 0.02

/** AprilTag unit square is an 8×8 module grid (black border ring + inner 6×6 data). */
const TAG_MODULES = 8
/** 10% of one module side in tag UV — proximity gate for edge-aligned votes. */
const TAU_MODULE_UV = 0.1 / TAG_MODULES
/** Floor for shortest quad edge (px) so `2 / L_min` stays finite on degenerate quads. */
const MIN_QUAD_EDGE_EPS_PX = 1e-6

/**
 * Jacobian **J** = ∂(x,y)/∂(u,v) for `applyHomography` / homography 8-vector `h`.
 *
 * Forward map (tag UV → image pixels), with **w = h₆u + h₇v + 1**:
 *
 * - **x(u,v) = (h₀u + h₁v + h₂) / w**, **y(u,v) = (h₃u + h₄v + h₅) / w**
 *
 * Quotient rule (same as in code):
 *
 * - **∂x/∂u = (h₀·w − x·h₆) / w²**, **∂x/∂v = (h₁·w − x·h₇) / w²**
 * - **∂y/∂u = (h₃·w − y·h₆) / w²**, **∂y/∂v = (h₄·w − y·h₇) / w²**
 *
 * Image Sobel gives **gₓ = ∂I/∂x**, **gᵧ = ∂I/∂y**. Composed intensity **I(u,v) = I_image(x(u,v), y(u,v))**
 * obeys the chain rule
 *
 * - **∂I/∂u = gₓ ∂x/∂u + gᵧ ∂y/∂u**, **∂I/∂v = gₓ ∂x/∂v + gᵧ ∂y/∂v**
 *
 * i.e. **[gᵤ, gᵥ]ᵀ = Jᵀ [gₓ, gᵧ]ᵀ** with **Jᵢⱼ = ∂(x,y)ᵢ/∂(u,v)ⱼ** in row-(x,y), column-(u,v) order.
 *
 * **Not** the same as mapping a displacement with **J⁻¹**: gradients are **covectors** (1-forms); they
 * **pull back** with **Jᵀ**. Tangents to the tag in UV push forward with **J** (columns **(∂x/∂u, ∂y/∂u)**
 * and **(∂x/∂v, ∂y/∂v)** in image). Sobel estimates the cotangent **dI** in image; tag-side **dI** uses **Jᵀ**.
 */
function jacobianImageWrtTagUv(h: Mat3, u: number, v: number) {
  const [h0, h1, h2, h3, h4, h5, h6, h7] = h
  const xh = h0 * u + h1 * v + h2
  const yh = h3 * u + h4 * v + h5
  const wh = h6 * u + h7 * v + 1
  const wh2 = wh * wh
  const xu = (h0 * wh - xh * h6) / wh2
  const xv = (h1 * wh - xh * h7) / wh2
  const yu = (h3 * wh - yh * h6) / wh2
  const yv = (h4 * wh - yh * h7) / wh2
  return { xu, xv, yu, yv }
}

/**
 * Tag-space partials **(∂I/∂u, ∂I/∂v)** from image Sobel **(gₓ, gᵧ)** and the same homography as
 * `applyHomography`: **[gᵤ, gᵥ]ᵀ = Jᵀ [gₓ, gᵧ]ᵀ** (see `jacobianImageWrtTagUv`).
 */
export function imageSobelToTagGradient(
  h: Mat3,
  u: number,
  v: number,
  gx: number,
  gy: number,
): { gu: number; gv: number } {
  const { xu, xv, yu, yv } = jacobianImageWrtTagUv(h, u, v)
  return {
    gu: gx * xu + gy * yu,
    gv: gx * xv + gy * yv,
  }
}

function quadEdgeLenPx(a: Point, b: Point): number {
  return length(a.x - b.x, a.y - b.y)
}

/**
 * Shortest side of the outer quad (TL→TR→BR→BL polygon walk) in pixels. Used to coarsely map ~2px to UV
 * (`2 / L_min`) for decode gating alongside `TAU_MODULE_UV`.
 */
export function minQuadEdgeLengthPx(outerCorners: Corners): number {
  const [tl, tr, bl, br] = outerCorners
  const m = min(quadEdgeLenPx(tl, tr), quadEdgeLenPx(tr, br), quadEdgeLenPx(br, bl), quadEdgeLenPx(bl, tl))
  return max(MIN_QUAD_EDGE_EPS_PX, m)
}

/** Primary 8×8 bin from tag UV (floor with clamp on the closed unit square). */
export function primaryModuleFromUv(u: number, v: number): { mx: number; my: number } {
  const mx = min(TAG_MODULES - 1, max(0, floor(u * TAG_MODULES)))
  const my = min(TAG_MODULES - 1, max(0, floor(v * TAG_MODULES)))
  return { mx, my }
}

export type DecodeTriangle = 'top' | 'bottom' | 'left' | 'right'

/** Four wedges meeting at module center `(½,½)` in local `[0,1]²` (center + diagonals). */
export function decodeTriangleFromLocalUv(fu: number, fv: number): DecodeTriangle {
  const du = fu - 0.5
  const dv = fv - 0.5
  const d1 = dv <= du
  const d2 = dv <= -du
  if (d1 && d2) {
    return 'top'
  }
  if (!d1 && !d2) {
    return 'bottom'
  }
  if (du >= 0) {
    return 'right'
  }
  return 'left'
}

/**
 * L∞ distance from `(fu,fv)` in module-local `[0,1]²` to that cell’s boundary, expressed in **tag UV**
 * (one module spans `1/TAG_MODULES` in `u` and `v`). Same as Chebyshev gap `(0.5 − max|local−½|) / 8`.
 */
export function decodeEdgeDistanceUv(fu: number, fv: number): number {
  return (0.5 - max(abs(fu - 0.5), abs(fv - 0.5))) / TAG_MODULES
}

/**
 * Linear indices `my*8+mx` into `whiteModuleCount`/`blackModuleCount` for the two modules sharing the chosen edge.
 * Drops neighbors outside the 8×8 lattice (tag border).
 */
export function decodeVoteModuleIndices(mx: number, my: number, tri: DecodeTriangle): number[] {
  const out: number[] = []
  const push = (x: number, y: number) => {
    if (x < 0 || x >= TAG_MODULES || y < 0 || y >= TAG_MODULES) {
      return
    }
    out.push(y * TAG_MODULES + x)
  }
  switch (tri) {
    case 'top':
      push(mx, my - 1)
      push(mx, my)
      break
    case 'bottom':
      push(mx, my)
      push(mx, my + 1)
      break
    case 'left':
      push(mx - 1, my)
      push(mx, my)
      break
    case 'right':
      push(mx, my)
      push(mx + 1, my)
      break
  }
  return out
}

/**
 * Scalar used in **unit tests** only: one number from the **primary** `(mx,my)` and the shared lattice
 * line (no per‑vote‑bin center). **`decodeTagPattern`** uses per‑bin **radial** dots
 * {@link decodeVoteBinRadialDot} instead (see there vs this edge‑aligned channel).
 */
export function decodeEdgeAlignedDot(
  u: number,
  v: number,
  mx: number,
  my: number,
  tri: DecodeTriangle,
  gu: number,
  gv: number,
): number {
  switch (tri) {
    case 'top':
      return gv * (v - my / TAG_MODULES)
    case 'bottom':
      return gv * (v - (my + 1) / TAG_MODULES)
    case 'left':
      return gu * (u - mx / TAG_MODULES)
    case 'right':
      return gu * (u - (mx + 1) / TAG_MODULES)
  }
}

/**
 * Convention: positive ⇒ vote toward black. Dot of tag UV
 * gradient with outward radial from that bin’s center `(cu, cv)` toward `(u, v)`.
 *
 * **`decodeTagPattern`** accumulates votes with this scalar (see {@link decodeVoteBinEdgeChannelDot} for
 * the older edge‑channel variant, still used in unit tests).
 */
export function decodeVoteBinRadialDot(u: number, v: number, cu: number, cv: number, gu: number, gv: number): number {
  return gu * (u - cu) + gv * (v - cv)
}

/** Euclidean distance from (u,v) to the closed axis-aligned rectangle [u0,u1]×[v0,v1]. */
export function distPointToClosedRectUv(u: number, v: number, u0: number, u1: number, v0: number, v1: number): number {
  const cu = min(u1, max(u0, u))
  const cv = min(v1, max(v0, v))
  return length(u - cu, v - cv)
}

function decodeMinVoteTotalFromShortestEdgePx(lMinPx: number): number {
  return max(2, round(DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE * lMinPx))
}

function classifyModuleFromPosNeg(
  whiteCount: number,
  blackCount: number,
  minVoteTotal: number,
): 0 | 1 | -1 | -2 {
  const sum = whiteCount + blackCount
  if (sum < minVoteTotal) {
    return -1
  }
  if (blackCount > whiteCount) {
    return 0
  }
  if (whiteCount > blackCount) {
    return 1
  }
  return -2
}

/**
 * One pass: only **`-1`** (weak votes). If all known cardinal neighbors agree on `0`/`1`, adopt that
 * color. **`-2`** (tie with enough votes) is skipped — neighbor homogeneity is a poor stand-in for
 * conflicting directional evidence, and dictionary decode already treats **`-2`** as an unknown bit.
 */
export function fillUnknownNeighbors6(pattern: TagPattern): void {
  const idx = (r: number, c: number) => r * 6 + c
  const next = [...pattern] as TagPattern
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      if (pattern[idx(r, c)] !== -1) {
        continue
      }
      const vals: number[] = []
      if (r > 0) {
        const v = pattern[idx(r - 1, c)]
        if (v === 0 || v === 1) {
          vals.push(v)
        }
      }
      if (r < 5) {
        const v = pattern[idx(r + 1, c)]
        if (v === 0 || v === 1) {
          vals.push(v)
        }
      }
      if (c > 0) {
        const v = pattern[idx(r, c - 1)]
        if (v === 0 || v === 1) {
          vals.push(v)
        }
      }
      if (c < 5) {
        const v = pattern[idx(r, c + 1)]
        if (v === 0 || v === 1) {
          vals.push(v)
        }
      }
      if (vals.length === 0) {
        continue
      }
      const first = vals[0]!
      if (vals.every((x) => x === first)) {
        next[idx(r, c)] = first as 0 | 1
      }
    }
  }
  for (let i = 0; i < 36; i++) {
    pattern[i] = next[i]!
  }
}

/** Same magnitude floor as `extractLabeledEdgePixels` in `corners.ts`. */
const DECODE_EDGE_MASK_EPS = 1e-6

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
  const w = imageWidth
  const h = imageHeight
  const mask = new Uint8Array(w * h)
  const x0 = max(0, floor(minX) - padPx)
  const y0 = max(0, floor(minY) - padPx)
  const x1 = min(w - 1, ceil(maxX) + padPx)
  const y1 = min(h - 1, ceil(maxY) + padPx)
  const eps2 = DECODE_EDGE_MASK_EPS * DECODE_EDGE_MASK_EPS
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const idx = y * w + x
      if (labelData[idx] !== regionLabel) {
        continue
      }
      const gx = sobelData[idx * 2]!
      const gy = sobelData[idx * 2 + 1]!
      if (gx * gx + gy * gy >= eps2) {
        mask[idx] = 1
      }
    }
  }
  return mask
}

/**
 * Full decode of 6×6 tag pattern from Sobel: iterate **image pixels** in the quad bbox,
 * inverse homography → tag UV, push gradient to tag UV, **8×8** modules. Each sample maps to a
 * primary module, **center+diagonal** wedge from {@link decodeTriangleFromLocalUv}, then **at most two** bins.
 * Each bin gets **`decodeVoteBinRadialDot`**: **`gᵤ(u−cu)+gᵥ(v−cv)`** (outward radial in tag UV toward the
 * sample). Proximity: **L∞ half-module gap** {@link decodeEdgeDistanceUv} in local `(fu,fv)` must satisfy
 * **`d_edge ≤ max(TAU_MODULE_UV, 2/L_min, 0.5/8)`** (same units as tag UV per module side).
 * Then inner **6×6** + neighbor fill for **`-1`** only
 * (**`-2`** unchanged). Outcomes: `0`/`1`/`-1`/`-2`.
 *
 * @param corners Quad corners
 * @param edgeMask Optional: skip pixels where mask index is 0
 */
export type DecodeTagPatternVoteMaps = {
  pattern: TagPattern
  /** Per 8×8 module index: unweighted count of radial dots voting “toward black”. */
  whiteModuleCount: Uint32Array
  /** Per 8×8 module index: unweighted count voting “toward white”. */
  blackModuleCount: Uint32Array
  /** Same gate as the inner loop: `max(TAU_MODULE_UV, 2/L_min, 0.5/8)` in tag UV. */
  uvProximityMax: number
  /**
   * Minimum `white+black` per cell for black/white vs **`-1`**; `max(2, round(f·L_min))` with
   * **f** = {@link DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE}.
   */
  minVoteTotal: number
}

function decodeTagPatternVoteAccumulation(
  corners: Corners,
  sobelData: Float32Array,
  imageWidth: number,
  imageH: number,
  edgeMask?: Uint8Array,
): {
  whiteModuleCount: Uint32Array
  blackModuleCount: Uint32Array
  uvProximityMax: number
  minVoteTotal: number
} {
  const [tl, tr, bl, br] = corners
  const h = tryComputeHomography(corners)

  const lMin = minQuadEdgeLengthPx(corners)
  const minVoteTotal = decodeMinVoteTotalFromShortestEdgePx(lMin)
  const uvHalfModule = 0.5 / TAG_MODULES
  const uvProximityMax = max(TAU_MODULE_UV, 2 / lMin, uvHalfModule)

  if (!h) {
    return {
      whiteModuleCount: new Uint32Array(64),
      blackModuleCount: new Uint32Array(64),
      uvProximityMax,
      minVoteTotal,
    }
  }

  let x0 = min(tl.x, tr.x, bl.x, br.x)
  let y0 = min(tl.y, tr.y, bl.y, br.y)
  let x1 = max(tl.x, tr.x, bl.x, br.x)
  let y1 = max(tl.y, tr.y, bl.y, br.y)
  const ix0 = max(0, floor(x0))
  const iy0 = max(0, floor(y0))
  const ix1 = min(imageWidth - 1, ceil(x1))
  const iy1 = min(imageH - 1, ceil(y1))

  const whiteModuleCount = new Uint32Array(64)
  const blackModuleCount = new Uint32Array(64)

  for (let iy = iy0; iy <= iy1; iy++) {
    for (let ix = ix0; ix <= ix1; ix++) {
      if (edgeMask && edgeMask[iy * imageWidth + ix] === 0) {
        continue
      }
      const { u, v, inside } = imagePixelToUnitSquareUv(h, ix + 0.5, iy + 0.5)
      if (!inside) {
        continue
      }
      const gx = sobelData[(iy * imageWidth + ix) * 2]!
      const gy = sobelData[(iy * imageWidth + ix) * 2 + 1]!
      const mag = gx * gx + gy * gy
      if (mag <= 1e-12) {
        continue
      }

      const { gu, gv } = imageSobelToTagGradient(h, u, v, gx, gy)

      const { mx, my } = primaryModuleFromUv(u, v)
      const fu = u * TAG_MODULES - mx
      const fv = v * TAG_MODULES - my
      const tri = decodeTriangleFromLocalUv(fu, fv)
      const dEdge = decodeEdgeDistanceUv(fu, fv)
      if (dEdge > uvProximityMax) {
        continue
      }

      for (const mi of decodeVoteModuleIndices(mx, my, tri)) {
        const mx2 = mi % TAG_MODULES
        const my2 = (mi / TAG_MODULES) | 0
        const cu = (mx2 + 0.5) / TAG_MODULES
        const cv = (my2 + 0.5) / TAG_MODULES
        const ru = u - cu
        const rv = v - cv
        if (ru * ru + rv * rv < 1e-10) {
          continue
        }
        const dot = decodeVoteBinRadialDot(u, v, cu, cv, gu, gv)
        if (dot > DECODE_DOT_EPS) {
          blackModuleCount[mi] = (blackModuleCount[mi] ?? 0) + 1
        } else if (dot < -DECODE_DOT_EPS) {
          whiteModuleCount[mi] = (whiteModuleCount[mi] ?? 0) + 1
        }
      }
    }
  }

  return { whiteModuleCount, blackModuleCount, uvProximityMax, minVoteTotal }
}

/**
 * Same decode as {@link decodeTagPattern}, plus raw **whiteModuleCount** / **blackModuleCount** tallies (for tooling / stress forensics).
 * Each contributing pixel adds **±1** to **at most two** 8×8 bins per {@link decodeVoteBinRadialDot}.
 */
export function decodeTagPatternWithVoteMaps(
  corners: Corners,
  sobelData: Float32Array,
  imageWidth: number,
  edgeMask?: Uint8Array,
  imageHeight?: number,
): DecodeTagPatternVoteMaps {
  const imageH = imageHeight !== undefined ? imageHeight : floor(sobelData.length / (2 * imageWidth))
  const { whiteModuleCount, blackModuleCount, uvProximityMax, minVoteTotal } = decodeTagPatternVoteAccumulation(
    corners,
    sobelData,
    imageWidth,
    imageH,
    edgeMask,
  )

  const pattern: TagPattern = [] as unknown as TagPattern
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const mx = col + 1
      const my = row + 1
      const mi = my * TAG_MODULES + mx
      const cell = classifyModuleFromPosNeg(whiteModuleCount[mi]!, blackModuleCount[mi]!, minVoteTotal)
      pattern.push(cell)
    }
  }

  fillUnknownNeighbors6(pattern)
  return { pattern, whiteModuleCount, blackModuleCount, uvProximityMax, minVoteTotal }
}

export function decodeTagPattern(
  corners: Corners,
  sobelData: Float32Array,
  imageWidth: number,
  edgeMask?: Uint8Array,
  imageHeight?: number,
): TagPattern {
  return decodeTagPatternWithVoteMaps(corners, sobelData, imageWidth, edgeMask, imageHeight).pattern
}
