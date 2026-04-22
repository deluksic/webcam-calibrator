import { describe, it, expect } from 'vitest'

import { type Corners, applyHomography, computeHomography } from '@/lib/geometry'
import {
  buildDecodeEdgeMask,
  buildTagGrid,
  decodeEdgeAlignedDot,
  decodeEdgeDistanceUv,
  decodeTriangleFromLocalUv,
  decodeVoteModuleIndices,
  fillUnknownNeighbors6,
  imageSobelToTagGradient,
  minQuadEdgeLengthPx,
  primaryModuleFromUv,
} from '@/lib/grid'
import type { TagPattern } from '@/lib/tag36h11'

const { max, abs } = Math

/** Central FD of J = ∂(x,y)/∂(u,v) for `applyHomography` (column u then column v). */
function jacobianImageWrtTagUvFd(
  h: Float32Array,
  u: number,
  v: number,
  eps = 1e-5,
): { xu: number; xv: number; yu: number; yv: number } {
  const xp = applyHomography(h, u + eps, v)
  const xm = applyHomography(h, u - eps, v)
  const yp = applyHomography(h, u, v + eps)
  const ym = applyHomography(h, u, v - eps)
  return {
    xu: (xp.x - xm.x) / (2 * eps),
    xv: (yp.x - ym.x) / (2 * eps),
    yu: (xp.y - xm.y) / (2 * eps),
    yv: (yp.y - ym.y) / (2 * eps),
  }
}

function tagGradientFromJacobianTranspose(
  j: { xu: number; xv: number; yu: number; yv: number },
  gx: number,
  gy: number,
) {
  return {
    gu: gx * j.xu + gy * j.yu,
    gv: gx * j.xv + gy * j.yv,
  }
}

/** ∂/∂u of a·x(u,v)+b·y(u,v) by central differences (independent check on chain rule). */
function linearImageFieldPartialU(h: Float32Array, u: number, v: number, a: number, b: number, eps = 1e-5): number {
  const pp = applyHomography(h, u + eps, v)
  const pm = applyHomography(h, u - eps, v)
  return (a * (pp.x - pm.x) + b * (pp.y - pm.y)) / (2 * eps)
}

function linearImageFieldPartialV(h: Float32Array, u: number, v: number, a: number, b: number, eps = 1e-5): number {
  const pp = applyHomography(h, u, v + eps)
  const pm = applyHomography(h, u, v - eps)
  return (a * (pp.x - pm.x) + b * (pp.y - pm.y)) / (2 * eps)
}

describe('grid', () => {
  describe('buildTagGrid', () => {
    it('builds 6x6 grid from square corners', () => {
      const corners: Corners = [
        { x: 0, y: 0 }, // TL
        { x: 100, y: 0 }, // TR
        { x: 0, y: 100 }, // BL
        { x: 100, y: 100 }, // BR
      ]

      const grid = buildTagGrid(corners, 6)

      expect(grid.outerCorners).toEqual(corners)
      expect(grid.cells.length).toBe(36) // 6x6
      expect(grid.innerCorners.length).toBe(49) // 7x7
    })

    it('builds grid from perspective quad', () => {
      // Simulate perspective view of square
      const corners: Corners = [
        { x: 10, y: 0 }, // TL (shifted left)
        { x: 90, y: 0 }, // TR (shifted right)
        { x: 0, y: 100 }, // BL
        { x: 100, y: 100 }, // BR
      ]

      const grid = buildTagGrid(corners, 6)

      expect(grid.cells.length).toBe(36)
      // Inner corners should form a grid even with perspective
      expect(grid.innerCorners.length).toBe(49)
    })

    it('has correct number of cells', () => {
      const corners: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]

      const grid = buildTagGrid(corners, 6)

      // Should have 36 cells (6x6)
      expect(grid.cells.length).toBe(36)
    })
  })

  describe('fillUnknownNeighbors6', () => {
    it('fills -1 from unanimous neighbors but leaves -2 unchanged', () => {
      const weak = Array(36).fill(0) as unknown as TagPattern
      // Cell (2,2) idx 14; four cardinals all 1
      weak[8] = weak[20] = weak[13] = weak[15] = 1
      weak[14] = -1
      fillUnknownNeighbors6(weak)
      expect(weak[14]).toBe(1)

      const tie = Array(36).fill(0) as unknown as TagPattern
      tie[8] = tie[20] = tie[13] = tie[15] = 1
      tie[14] = -2
      fillUnknownNeighbors6(tie)
      expect(tie[14]).toBe(-2)
    })
  })

  describe('decode edge voting helpers', () => {
    it('primaryModuleFromUv floors into 0..7', () => {
      expect(primaryModuleFromUv(0, 0)).toEqual({ mx: 0, my: 0 })
      expect(primaryModuleFromUv(0.125, 0.25)).toEqual({ mx: 1, my: 2 })
      expect(primaryModuleFromUv(1, 1)).toEqual({ mx: 7, my: 7 })
    })

    it('minQuadEdgeLengthPx returns shortest side', () => {
      const sq: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]
      expect(minQuadEdgeLengthPx(sq)).toBe(100)
      const thin: Corners = [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 0, y: 100 },
        { x: 10, y: 100 },
      ]
      expect(minQuadEdgeLengthPx(thin)).toBe(10)
    })

    it('decodeTriangleFromLocalUv: center + diagonals', () => {
      expect(decodeTriangleFromLocalUv(0.2, 0.2)).toBe('top')
      expect(decodeTriangleFromLocalUv(0.8, 0.8)).toBe('right')
      expect(decodeTriangleFromLocalUv(0.15, 0.75)).toBe('left')
      expect(decodeTriangleFromLocalUv(0.3, 0.9)).toBe('bottom')
      expect(decodeTriangleFromLocalUv(0.75, 0.25)).toBe('top')
      expect(decodeTriangleFromLocalUv(0.5, 0.5)).toBe('top')
      expect(decodeTriangleFromLocalUv(0.2, 0.5)).toBe('left')
    })

    it('decodeVoteModuleIndices returns two interior neighbors for bottom', () => {
      const idx = decodeVoteModuleIndices(3, 4, 'bottom')
      expect(idx.sort((a, b) => a - b)).toEqual([4 * 8 + 3, 5 * 8 + 3])
    })

    it('decodeEdgeDistanceUv is L∞ gap in local cell; decodeEdgeAlignedDot on bottom edge', () => {
      const mx = 2
      const my = 3
      const u = (mx + 0.5) / 8
      const vBottom = (my + 1) / 8
      const fu = 0.5
      const fv = 1
      expect(decodeEdgeDistanceUv(fu, fv)).toBe(0)
      expect(decodeEdgeAlignedDot(u, vBottom + 0.01, mx, my, 'bottom', 0, 1)).toBeCloseTo(0.01, 5)
    })

    /** L∞ distance from `(fu,fv)` in primary `[0,1]²` to that cell’s boundary, in tag UV. */
    function chebyshevGapPrimaryCellUv(u: number, v: number, mx: number, my: number, tagModules = 8): number {
      const fu = u * tagModules - mx
      const fv = v * tagModules - my
      return (0.5 - max(abs(fu - 0.5), abs(fv - 0.5))) / tagModules
    }

    it('decodeEdgeDistanceUv matches Chebyshev gap in tag UV for all local samples', () => {
      const tag = 8
      const mx = 2
      const my = 3
      for (let k = 1; k < 40; k++) {
        const fv = 0.03 + k * 0.024
        if (fv >= 0.99) {
          break
        }
        const fu = 0.11 + (k % 7) * 0.11
        if (fu >= 0.99) {
          continue
        }
        const u = (mx + fu) / tag
        const v = (my + fv) / tag
        const line = decodeEdgeDistanceUv(fu, fv)
        const cheb = chebyshevGapPrimaryCellUv(u, v, mx, my, tag)
        expect(line).toBeCloseTo(cheb, 12)
      }
    })

    it('imageSobelToTagGradient pulls image Sobel to tag UV via Jᵀ (identity homography)', () => {
      const h = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0])
      const { gu, gv } = imageSobelToTagGradient(h, 0.25, 0.25, 3, -4)
      expect(gu).toBeCloseTo(3, 10)
      expect(gv).toBeCloseTo(-4, 10)
    })

    it('decodeEdgeAlignedDot can disagree with primary-bin radial for the same tag-UV gradient', () => {
      const mx = 3
      const my = 3
      const fu = 0.9
      const fv = 0.55
      const u = (mx + fu) / 8
      const v = (my + fv) / 8
      expect(decodeTriangleFromLocalUv(fu, fv)).toBe('right')
      const tri = 'right' as const
      const gu = 1
      const gv = -0.6
      const edgeDot = decodeEdgeAlignedDot(u, v, mx, my, tri, gu, gv)
      const cu = (mx + 0.5) / 8
      const cv = (my + 0.5) / 8
      const radial = gu * (u - cu) + gv * (v - cv)
      expect(edgeDot).toBeLessThan(0)
      expect(radial).toBeGreaterThan(0)
    })
  })

  describe('imageSobelToTagGradient (Jᵀ pullback)', () => {
    it('matches Jᵀ·g from finite-difference Jacobian on a projective homography', () => {
      const src: Corners = [
        { x: 120, y: 80 },
        { x: 540, y: 100 },
        { x: 100, y: 420 },
        { x: 520, y: 400 },
      ]
      const h = computeHomography(src)
      const probes: [number, number][] = [
        [0.12, 0.37],
        [0.55, 0.55],
        [0.88, 0.22],
      ]
      const grads: [number, number][] = [
        [1, 0],
        [0, 1],
        [0.6, -0.8],
        [-2, 3],
      ]
      for (const [u, v] of probes) {
        const jFd = jacobianImageWrtTagUvFd(h, u, v)
        for (const [gx, gy] of grads) {
          const got = imageSobelToTagGradient(h, u, v, gx, gy)
          const exp = tagGradientFromJacobianTranspose(jFd, gx, gy)
          expect(got.gu).toBeCloseTo(exp.gu, 5)
          expect(got.gv).toBeCloseTo(exp.gv, 5)
        }
      }
    })

    it('agrees with central differences on I(u,v)=a·x(u,v)+b·y(u,v) (chain rule end-to-end)', () => {
      const h = Float32Array.from([2.5, -0.25, 30, 0.1, 1.8, 40, 0.03, -0.02])
      const u = 0.41
      const v = 0.62
      const a = 1.7
      const b = -0.9
      const g = imageSobelToTagGradient(h, u, v, a, b)
      expect(g.gu).toBeCloseTo(linearImageFieldPartialU(h, u, v, a, b), 6)
      expect(g.gv).toBeCloseTo(linearImageFieldPartialV(h, u, v, a, b), 6)
    })

    it('affine (h₆=h₇=0): matches explicit constant Jᵀ for shear + scale + translation', () => {
      const h0 = 2
      const h1 = 0.5
      const h2 = 100
      const h3 = -0.25
      const h4 = 1.5
      const h5 = -20
      const h = Float32Array.from([h0, h1, h2, h3, h4, h5, 0, 0])
      const gx = 0.25
      const gy = 0.75
      for (const u of [0, 0.3, 0.99]) {
        for (const v of [0.1, 0.7]) {
          const { gu, gv } = imageSobelToTagGradient(h, u, v, gx, gy)
          expect(gu).toBeCloseTo(gx * h0 + gy * h3, 12)
          expect(gv).toBeCloseTo(gx * h1 + gy * h4, 12)
        }
      }
    })

    it('90° rotation + uniform scale: image x-axis Sobel maps to tag −v direction', () => {
      const s = 40
      const c = 0
      const si = 1
      const h = Float32Array.from([c * s, -si * s, 0, si * s, c * s, 0, 0, 0])
      const { gu, gv } = imageSobelToTagGradient(h, 0.5, 0.5, 1, 0)
      expect(gu).toBeCloseTo(0, 6)
      expect(gv).toBeCloseTo(-s, 6)
    })
  })

  describe('buildDecodeEdgeMask', () => {
    it('marks only matching label with non-zero Sobel', () => {
      const w = 8
      const h = 4
      const labelData = new Uint32Array(w * h).fill(2)
      labelData[10] = 7
      labelData[11] = 7
      const sobel = new Float32Array(w * h * 2)
      sobel[10 * 2] = 0.1
      sobel[10 * 2 + 1] = 0
      sobel[11 * 2] = 0
      sobel[11 * 2 + 1] = 0
      const mask = buildDecodeEdgeMask(labelData, sobel, w, h, 7, 1, 1, 4, 2, 0)
      expect(mask[10]).toBe(1)
      expect(mask[11]).toBe(0)
      expect(mask[0]).toBe(0)
    })
  })

  describe('grid cell access', () => {
    it('cells are indexed row by row', () => {
      const corners: Corners = [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 0, y: 100 },
        { x: 100, y: 100 },
      ]

      const grid = buildTagGrid(corners, 6)

      // First row, first column
      expect(grid.cells[0]!.row).toBe(0)
      expect(grid.cells[0]!.col).toBe(0)

      // First row, second column
      expect(grid.cells[1]!.row).toBe(0)
      expect(grid.cells[1]!.col).toBe(1)

      // Second row, first column
      expect(grid.cells[6]!.row).toBe(1)
      expect(grid.cells[6]!.col).toBe(0)
    })
  })
})
