import { describe, it, expect } from 'vitest'

import { imagePixelToUnitSquareUv, invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import { applyHomography, computeHomography, type Corners, type Mat3 } from '@/lib/geometry'
import { buildTagGrid, decodeTagPattern } from '@/lib/grid'
import { tag36h11Code, codeToPattern, decodeTag36h11AnyRotation } from '@/lib/tag36h11'
import {
  finiteDifferenceSobelFromIntensity,
  intensityAtTagUv,
  renderAprilTagIntensity,
  renderAprilTagSobelFiniteDifference,
} from '@/tests/utils/syntheticAprilTag'

const { max, min, floor, round, abs } = Math

describe('aprilTagRaycast', () => {
  it('invertMat3RowMajor * M ≈ I', () => {
    const M: Mat3 = [2, 0, 1, 0, 3, 0, 0, 0, 1]
    const inv = invertMat3RowMajor(M)!
    const mul = (A: readonly number[], B: readonly number[]) => {
      const o = Array.from({ length: 9 }, () => 0)
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          for (let k = 0; k < 3; k++) {
            o[r * 3 + c]! += A[r * 3 + k]! * B[k * 3 + c]!
          }
        }
      }
      return o
    }
    const I = mul(M, [...inv])
    expect(I[0]).toBeCloseTo(1, 6)
    expect(I[4]).toBeCloseTo(1, 6)
    expect(I[8]).toBeCloseTo(1, 6)
    expect(I[1]).toBeCloseTo(0, 5)
    expect(I[3]).toBeCloseTo(0, 5)
  })

  it('imagePixelToUnitSquareUv round-trips forward homography on a square', () => {
    const strip: Corners = [
      { x: 20, y: 20 },
      { x: 220, y: 30 },
      { x: 10, y: 200 },
      { x: 210, y: 210 },
    ]
    const h = computeHomography(strip)
    const u0 = 0.37
    const v0 = 0.52
    const p = applyHomography(h, u0, v0)
    const back = imagePixelToUnitSquareUv(h, p.x, p.y)
    expect(back.inside).toBe(true)
    expect(back.u).toBeCloseTo(u0, 5)
    expect(back.v).toBeCloseTo(v0, 5)
  })

  it('intensityAtTagUv: black 1/8 border + inner 6×6 from pattern', () => {
    const pattern = codeToPattern(tag36h11Code(0))
    expect(intensityAtTagUv(0.04, 0.04, pattern)).toBe(0)
    expect(intensityAtTagUv(0.96, 0.96, pattern)).toBe(0)
    const innerU = 3.5 / 8
    const innerV = 3.5 / 8
    const bit = pattern[2 * 6 + 2]
    expect(intensityAtTagUv(innerU, innerV, pattern)).toBe(bit)
    const lastU = 6.5 / 8
    const lastV = 6.5 / 8
    const bitBr = pattern[5 * 6 + 5]
    expect(intensityAtTagUv(lastU, lastV, pattern)).toBe(bitBr)
  })

  it('renderAprilTagIntensity matches UV law at each cell center (forward projection)', () => {
    const pattern = codeToPattern(tag36h11Code(42))
    const strip: Corners = [
      { x: 40, y: 40 },
      { x: 280, y: 45 },
      { x: 35, y: 260 },
      { x: 275, y: 265 },
    ]
    const w = 320
    const h = 320
    const intensity = renderAprilTagIntensity({
      width: w,
      height: h,
      corners: strip,
      pattern,
    })
    const h8 = computeHomography(strip)

    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 6; col++) {
        const u = (col + 1.5) / 8
        const v = (row + 1.5) / 8
        const want = intensityAtTagUv(u, v, pattern)
        const p = applyHomography(h8, u, v)
        const xi = max(0, min(w - 1, round(p.x)))
        const yi = max(0, min(h - 1, round(p.y)))
        const got = intensity[yi * w + xi]
        expect(got).toBe(want)
      }
    }
  })

  it('finiteDifferenceSobelFromIntensity has strong response on a vertical black/white edge', () => {
    const w = 64
    const h = 64
    const I = new Float32Array(w * h).fill(1)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < 32; x++) {
        I[y * w + x] = 0
      }
    }
    const s = finiteDifferenceSobelFromIntensity(I, w, h)
    const cx = 32
    const cy = 32
    const o = (cy * w + cx) * 2
    expect(abs(s[o]!)).toBeGreaterThan(0.2)
  })

  it('buildTagGrid cell centers: UV indices match row/col; raster matches tag law (nearest pixel)', () => {
    const tagId = 7
    const pattern = codeToPattern(tag36h11Code(tagId))
    const size = 360
    const strip: Corners = [
      { x: 20, y: 20 },
      { x: 20 + size, y: 20 },
      { x: 20, y: 20 + size },
      { x: 20 + size, y: 20 + size },
    ]
    const w = 400
    const h = 400
    const intensity = renderAprilTagIntensity({
      width: w,
      height: h,
      corners: strip,
      pattern,
    })
    const grid = buildTagGrid(strip, 6)
    const h8 = computeHomography(strip)

    for (const cell of grid.cells) {
      const { u, v, inside } = imagePixelToUnitSquareUv(h8, cell.center.x, cell.center.y)
      expect(inside).toBe(true)
      const col = min(5, max(0, floor(u * 6 - 1e-9)))
      const row = min(5, max(0, floor(v * 6 - 1e-9)))
      expect(row).toBe(cell.row)
      expect(col).toBe(cell.col)

      const analytical = intensityAtTagUv(u, v, pattern)
      const xi = max(0, min(w - 1, round(cell.center.x)))
      const yi = max(0, min(h - 1, round(cell.center.y)))
      const pix = intensity[yi * w + xi]
      expect(pix).toBeCloseTo(analytical, 4)
    }
  })

  it('decodeTagPattern recovers dictionary id from synthetic raycast + finite-difference Sobel', () => {
    const tagId = 0
    const pattern = codeToPattern(tag36h11Code(tagId))
    const size = 360
    const strip: Corners = [
      { x: 20, y: 20 },
      { x: 20 + size, y: 20 },
      { x: 20, y: 20 + size },
      { x: 20 + size, y: 20 + size },
    ]
    const w = 400
    const h = 400
    const { sobel } = renderAprilTagSobelFiniteDifference(
      { width: w, height: h, corners: strip, pattern, supersample: 4 },
      { gradientScale: 4 },
    )
    const decodedPattern = decodeTagPattern(strip, sobel, w, undefined, h)
    const match = decodeTag36h11AnyRotation(decodedPattern, 8)
    expect(match).not.toBeNull()
    expect(match!.id).toBe(tagId)
  })
})
